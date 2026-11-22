//! moe-engine: run a mixture-of-experts LLM with experts streamed from the SSD.
//!
//!   moe-engine --pack model.moe --prompt-ids 151644,872,... [options]
//!
//! Options
//!   --prompt-ids a,b,c | --prompt-file f   prompt token ids (comma/space separated)
//!   --max-new N            tokens to generate (default 64)
//!   --ram-gb G             hard cap for all engine RAM (default 7): attention weights, KV cache,
//!                          buffers; the expert cache gets the rest
//!   --cache-gb G           expert cache size instead of "the rest" (0 = minimal, strict mode)
//!   --max-context N        longest conversation in tokens (default 4096; reserves KV cache RAM)
//!   --refine               re-predict the next layer after each layer and withdraw stale guesses
//!   --threads N            compute threads (default: all cores)
//!   --io-threads N         parallel SSD reads (default 4)
//!   --max-guess-reads N    at most N guesses read at once (default io-threads - 1)
//!   --lookahead N          prefetch experts N layers ahead (default 2)
//!   --prefill-batch N      prompt tokens processed together, layer by layer (default 512)
//!   --max-cpu-temp C       pause between tokens if the CPU reaches C degrees (default 90; 0 = off)
//!   --max-ssd-temp C       same for the NVMe drives (default 70; 0 = off)
//!   --prefetch-width N     experts predicted for the next layer (default 8; 0 = no prefetch)
//!   --prefetch-width2 N    experts predicted for layers further ahead (default 4)
//!   --mirror p1,p2         identical copies of the pack on other drives; reads are spread over all
//!   --buffered             do not use O_DIRECT / F_NOCACHE
//!   --dump-logits f        write every step's logits (f32) to f   (for verification)
//!   --out-ids f            write prompt + generated ids to f
//!   --quiet                only print the generated text

mod kernels;
mod model;
mod pack;
mod pool;
mod store;
mod thermal;

use std::io::Write;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn parse_ids(s: &str) -> Vec<u32> {
    s.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()).map(|t| t.parse().expect("bad token id")).collect()
}

/// --bench-kernels: time the engine's expert kernels on random blocks laid out exactly
/// like the cache (one huge-page region), without SSD or model. [--threads N]
fn bench_kernels(threads: usize) {
    use kernels::{matvec, QAct, QTensor, Scales};
    let simd = kernels::init();
    let (h, f) = (2048usize, 768usize);
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let q4b = |rows: usize, cols: usize| rows * cols / 2 + rows * (cols / 32) * 2;
    let (ga, ua, da) = (q4b(f, h), q4b(f, h), q4b(h, f));
    let len = (ga + ua + da).div_ceil(4096) * 4096;
    let (base, huge) = store::bench_region(n * len);
    let mut seed = 12345u64;
    let sbits = (0.005f32.to_bits() >> 16) as u16;
    for j in 0..n {
        // Safety: inside the region allocated above.
        let b = unsafe { std::slice::from_raw_parts_mut((base + j * len) as *mut u8, len) };
        for x in b.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (seed >> 56) as u8;
        }
        for (a, rows, cols) in [(0usize, f, h), (ga, f, h), (ga + ua, h, f)] {
            let d = rows * cols / 2;
            for c in b[a + d..a + d + rows * (cols / 32) * 2].chunks_exact_mut(2) {
                c.copy_from_slice(&sbits.to_le_bytes());
            }
        }
    }
    let mut x = QAct::new(h);
    x.set(&(0..h).map(|i| ((i * 37 % 100) as f32 - 50.0) / 50.0).collect::<Vec<_>>());
    let pool = pool::Pool::new(threads);
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    while t0.elapsed().as_secs_f64() < 3.0 {
        pool.run(n, |j| {
            let b = unsafe { std::slice::from_raw_parts((base + j * len) as *const u8, len) };
            let view = |a: usize, rows: usize, cols: usize| {
                let d = rows * cols / 2;
                let sb = &b[a + d..a + d + rows * (cols / 32) * 2];
                let s = unsafe { std::slice::from_raw_parts(sb.as_ptr() as *const u16, sb.len() / 2) };
                QTensor { data: &b[a..a + d], scales: Scales::Bf16(s), rows, cols, q4: true }
            };
            let (mut g, mut u, mut hq, mut y) = (vec![0f32; f], vec![0f32; f], QAct::new(f), vec![0f32; h]);
            matvec(&view(0, f, h), &x, 0, &mut g);
            matvec(&view(ga, f, h), &x, 0, &mut u);
            for i in 0..f {
                g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
            }
            hq.set(&g);
            matvec(&view(ga + ua, h, f), &hq, 0, &mut y);
            std::hint::black_box(&y);
        });
        done += n;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "kernels: {simd} | threads {threads} | {n} experts, huge pages {} | {:.2} GMAC/s ({:.0} experts/s)",
        if huge { "on" } else { "off" },
        done as f64 * 3.0 * (h * f) as f64 / dt / 1e9,
        done as f64 / dt
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench-kernels") {
        let t = arg(&args, "--threads").and_then(|v| v.parse().ok()).unwrap_or(1);
        bench_kernels(t);
        return;
    }
    let flag = |n: &str| args.iter().any(|a| a == n);
    let get = |n: &str, d: &str| arg(&args, n).unwrap_or_else(|| d.to_string());
    let Some(pack_path) = arg(&args, "--pack") else {
        eprintln!("usage: moe-engine --pack model.moe --prompt-ids 1,2,3 [--max-new 64] [--cache-gb 4] ...");
        std::process::exit(1);
    };
    let prompt: Vec<u32> = if let Some(f) = arg(&args, "--prompt-file") {
        parse_ids(&std::fs::read_to_string(f).expect("cannot read prompt file"))
    } else {
        parse_ids(&get("--prompt-ids", ""))
    };
    if prompt.is_empty() {
        eprintln!("no prompt ids given (--prompt-ids or --prompt-file)");
        std::process::exit(1);
    }
    let max_new: usize = get("--max-new", "64").parse().unwrap();
    let ram_gb: f64 = get("--ram-gb", "7").parse().unwrap();
    let max_ctx: usize = get("--max-context", "4096").parse().unwrap();
    let cache_req: Option<f64> = arg(&args, "--cache-gb").map(|v| v.parse().unwrap());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads: usize = get("--threads", &cores.to_string()).parse().unwrap();
    let io_threads: usize = get("--io-threads", "4").parse().unwrap();
    let max_guess: usize = get("--max-guess-reads", &io_threads.saturating_sub(1).max(1).to_string()).parse().unwrap();
    let lookahead: usize = get("--lookahead", "2").parse().unwrap();
    let prefill_batch: usize = get("--prefill-batch", "512").parse::<usize>().unwrap().max(1);
    let mut guard = thermal::Guard::new(get("--max-cpu-temp", "90").parse().unwrap(), get("--max-ssd-temp", "70").parse().unwrap());
    let width: usize = get("--prefetch-width", "8").parse().unwrap();
    let width2: usize = get("--prefetch-width2", "4").parse().unwrap();
    let mut paths = vec![pack_path.clone()];
    if let Some(m) = arg(&args, "--mirror") {
        paths.extend(m.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()));
    }
    let quiet = flag("--quiet");

    let simd = kernels::init();
    let pk = pack::Pack::open(&pack_path).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    let blocks: Vec<(u64, usize)> = pk.experts.iter().map(|&(o, n)| (pk.data_start + o, n)).collect();
    let block_bytes = blocks[0].1;

    // ---- RAM plan under the hard cap
    let cfgu = |k: &str| pk.cfg_usize(k).unwrap_or(0);
    let dense: usize = pk.tensors.iter().filter(|(n, _)| n.as_str() != "embed").map(|(_, t)| t.nbytes).sum();
    let kv = max_ctx * cfgu("num_hidden_layers") * 2 * cfgu("num_key_value_heads") * cfgu("head_dim") * 2;
    let other = 300_000_000usize; // buffers, vocabulary, thread stacks, the program itself
    let budget = (ram_gb * 1e9) as i64 - (dense + kv + other) as i64;
    let cache_bytes = match cache_req {
        Some(g) => (g * 1e9) as usize,
        None => budget.max(0) as usize,
    };
    let max_new = if prompt.len() + max_new > max_ctx {
        if prompt.len() >= max_ctx {
            eprintln!("error: the prompt has {} tokens, more than --max-context {}", prompt.len(), max_ctx);
            std::process::exit(1);
        }
        eprintln!("note: generating at most {} tokens to stay within --max-context {}", max_ctx - prompt.len(), max_ctx);
        max_ctx - prompt.len()
    } else {
        max_new
    };
    // slots needed at once even with no cache: this layer's experts, guesses for the
    // next layers, reads in flight, and some slack
    let min_slots = 2 * 8 + width + width2 * lookahead.saturating_sub(1) + io_threads + 8;
    let min_cache = min_slots * block_bytes.div_ceil(4096) * 4096;
    if (dense + kv + other + cache_bytes.max(min_cache)) as f64 > ram_gb * 1e9 {
        eprintln!(
            "error: does not fit in --ram-gb {ram_gb}: attention/output {:.2} GB + KV for {max_ctx} tokens {:.2} GB + other {:.2} GB + expert cache {:.2} GB",
            dense as f64 / 1e9, kv as f64 / 1e9, other as f64 / 1e9, cache_bytes.max(min_cache) as f64 / 1e9
        );
        eprintln!("       lower --cache-gb or --max-context, or raise --ram-gb");
        std::process::exit(1);
    }
    let store = store::ExpertStore::new(&paths, blocks, cache_bytes, min_slots, io_threads, max_guess, !flag("--buffered"))
        .unwrap_or_else(|e| {
            eprintln!("error opening experts: {e}");
            std::process::exit(1)
        });
    let vocab = pk.vocab().unwrap_or_default();
    let _t_load = Instant::now();
    let mut m = model::Model::load(pk, store.clone(), model::Options { threads, lookahead, prefetch_width: width, prefetch_width2: width2, refine: flag("--refine") })
        .unwrap_or_else(|e| {
            eprintln!("error loading model: {e}");
            std::process::exit(1)
        });
    if !quiet {
        let c = &m.cfg;
        eprintln!(
            "model: {} layers, {} experts (top-{}), hidden {}, vocab {} | kernels: {} | threads {} | io threads {}",
            c.layers, c.experts, c.topk, c.hidden, c.vocab, simd, threads, io_threads
        );
        eprintln!(
            "RAM (cap {:.1} GB): attention/output {:.2} + expert cache {:.2} ({} huge pages) + KV for {} tokens {:.2} + other {:.2} = {:.2} GB | expert block {:.2} MB | SSD reads: {}",
            ram_gb,
            m.dense_bytes as f64 / 1e9,
            store.cache_bytes() as f64 / 1e9,
            if store.huge_pages() { "with" } else { "without" },
            max_ctx,
            kv as f64 / 1e9,
            other as f64 / 1e9,
            (m.dense_bytes + store.cache_bytes() + kv + other) as f64 / 1e9,
            block_bytes as f64 / 1e6,
            if store.direct { "direct (no OS cache)" } else { "buffered (OS cache may hide SSD speed)" }
        );
        eprintln!(
            "prefetch: {} experts for the next layer, {} for up to {} layers ahead | pack copies: {} | prompt batch {}",
            width, width2, lookahead, paths.len(), prefill_batch
        );
        eprintln!(
            "heat guard: pause above CPU {}C / SSD {}C ({})",
            guard.max_cpu, guard.max_ssd, guard.sensors()
        );
    }

    let mut dump = arg(&args, "--dump-logits").map(|f| std::fs::File::create(f).expect("cannot create logits file"));
    let mut ids = prompt.clone();
    let out = std::io::stdout();
    let mut out = out.lock();
    let print_tok = |out: &mut std::io::StdoutLock, t: u32| {
        if let Some(b) = vocab.get(t as usize) {
            let _ = out.write_all(b);
        } else {
            let _ = write!(out, "<{t}>");
        }
        let _ = out.flush();
    };

    // prefill: the prompt in batches, layer by layer (logits only for the last token,
    // unless every step's logits are being dumped for verification)
    let t0 = Instant::now();
    let mut logits = Vec::new();
    for chunk in prompt.chunks(prefill_batch) {
        let out = m.forward_batch(chunk, dump.is_some()).expect("forward failed");
        if let Some(f) = dump.as_mut() {
            for lg in &out {
                let b: Vec<u8> = lg.iter().flat_map(|v| v.to_le_bytes()).collect();
                f.write_all(&b).unwrap();
            }
        }
        logits = out.into_iter().last().unwrap();
        guard.check();
    }
    let t_prefill = t0.elapsed().as_secs_f64();
    let s_pre = snapshot(&m);
    let bytes_pre = store.stats.bytes_read.load(std::sync::atomic::Ordering::Relaxed);

    // decode: greedy
    let t1 = Instant::now();
    let mut generated = 0;
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
            .0 as u32;
        ids.push(next);
        generated += 1;
        print_tok(&mut out, next);
        if m.cfg.eos.contains(&next) || step + 1 == max_new {
            break; // the last token needs no forward pass
        }
        guard.check();
        logits = m.forward(next).expect("forward failed");
        if let Some(f) = dump.as_mut() {
            let b: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&b).unwrap();
        }
    }
    let t_decode = t1.elapsed().as_secs_f64();
    let _ = writeln!(out);
    if let Some(f) = arg(&args, "--out-ids") {
        let s: Vec<String> = ids.iter().map(|v| v.to_string()).collect();
        std::fs::write(f, s.join(",")).unwrap();
    }
    if quiet {
        return;
    }
    let s_end = snapshot(&m);
    let st = &store.stats;
    let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64;
    let (reuse, hits, waits, misses) = (ld(&st.reuse), ld(&st.hits), ld(&st.waits), ld(&st.misses));
    let total = (reuse + hits + waits + misses).max(1.0);
    let decode_steps = (generated as f64 - 1.0).max(1.0); // the last sampled token needs no forward
    let dec = s_end.sub(&s_pre);
    let dec_bytes = (ld(&st.bytes_read) - bytes_pre as f64) / decode_steps;
    eprintln!("\n---- stats");
    eprintln!(
        "prefill: {} tokens in {:.2}s ({:.2} tok/s) | decode: {} tokens in {:.2}s ({:.2} tok/s)",
        prompt.len(),
        t_prefill,
        prompt.len() as f64 / t_prefill,
        generated,
        t_decode,
        decode_steps / t_decode.max(1e-9)
    );
    eprintln!(
        "decode per token: SSD read {:.0} MB ({:.2} GB/s over the decode time)",
        dec_bytes / 1e6,
        dec_bytes * decode_steps / 1e9 / t_decode.max(1e-9)
    );
    eprintln!(
        "decode time per token: attention {:.0} ms, experts {:.0} ms (of which waiting for SSD ~{:.0} ms), routers {:.1} ms, output head {:.0} ms",
        dec.attn / decode_steps * 1e3,
        dec.moe / decode_steps * 1e3,
        dec.stall / decode_steps * 1e3,
        dec.predict / decode_steps * 1e3,
        dec.head / decode_steps * 1e3
    );
    let (dem, unp) = (ld(&st.demanded).max(1.0), ld(&st.unpredicted));
    let _ = misses;
    eprintln!(
        "expert blocks used: {:.0} | from cache {:.1}% | guessed, arrived in time {:.1}% | still being read when needed {:.1}% | not predicted at all {:.1}%",
        total,
        reuse / total * 100.0,
        hits / total * 100.0,
        waits / total * 100.0,
        unp / dem * 100.0
    );
    let rt = ld(&st.read_ns) / 1e9;
    eprintln!(
        "heat: peak CPU {:.0}C, peak SSD {:.0}C, paused {:.1}s",
        guard.peak_cpu, guard.peak_ssd, guard.paused
    );
    eprintln!(
        "SSD: {} reads, {:.2} GB, {:.2} GB/s per read thread while reading | guesses {}: used {:.0}%, withdrawn {:.0}%, dropped late {:.0}% | evictions {} (forced {}) | cache in use {:.2} GB",
        ld(&st.reads),
        ld(&st.bytes_read) / 1e9,
        if rt > 0.0 { ld(&st.bytes_read) / 1e9 / rt } else { 0.0 },
        ld(&st.prefetched),
        ld(&st.prefetch_used) / ld(&st.prefetched).max(1.0) * 100.0,
        ld(&st.cancelled) / ld(&st.prefetched).max(1.0) * 100.0,
        ld(&st.dropped) / ld(&st.prefetched).max(1.0) * 100.0,
        ld(&st.evictions),
        ld(&st.forced),
        store.cached_bytes() as f64 / 1e9
    );
}

struct Snap {
    attn: f64,
    moe: f64,
    head: f64,
    predict: f64,
    stall: f64,
}

impl Snap {
    fn sub(&self, o: &Snap) -> Snap {
        Snap {
            attn: self.attn - o.attn,
            moe: self.moe - o.moe,
            head: self.head - o.head,
            predict: self.predict - o.predict,
            stall: self.stall - o.stall,
        }
    }
}

fn snapshot(m: &model::Model) -> Snap {
    let t = &m.timing;
    Snap {
        attn: t.attn,
        moe: t.moe,
        head: t.head,
        predict: t.predict,
        stall: t.stall,
    }
}
