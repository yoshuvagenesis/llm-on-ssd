//! moe-engine: run a mixture-of-experts LLM with experts streamed from the SSD.
//!
//!   moe-engine --pack model.moe --tokenizer tokenizer.json --serve        chat server (web page + API)
//!   moe-engine --pack model.moe --tokenizer tokenizer.json --prompt "Hi"  one answer in the terminal
//!   moe-engine --pack model.moe --prompt-ids 151644,872,... [options]      token ids in, for tests
//!
//! Chat server
//!   --serve                start the chat server: web page at /, OpenAI-compatible API at /v1
//!   --port N               port (default 8080)        --host A   address to listen on (default 0.0.0.0)
//!   --password P           require this password (sent by the page / API clients as a Bearer token)
//!   --max-answer N         default answer length limit in tokens (default 1024)
//!   --tokenizer f          the model's tokenizer.json (needed for text in: --serve, --prompt)
//!
//! Options
//!   --prompt "text"        one user message (the model's chat format); --think lets it reason first
//!                          (models with no thinking mode at all, per the pack's "no_think" flag,
//!                          get no <think> tags either way, and --think is refused for them)
//!   --bench-prompt N --bench-text f   benchmark prompt of exactly N tokens: "Summarize this text:"
//!                          and the start of text file f, in the chat format (as tools/make_prompt.py)
//!   --prompt-ids a,b,c | --prompt-file f   prompt token ids (comma/space separated)
//!   --perplexity f         measure the model's perplexity on text file f (first --max-tokens
//!                          tokens, default 2048) and exit: a quality check of the whole chain
//!   --tokenize f           print the token ids of text file f and exit (needs --tokenizer;
//!                          --no-special treats "<|im_start|>" etc. as plain text)
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

mod json;
mod kernels;
mod model;
mod pack;
mod pool;
mod server;
mod store;
mod thermal;
mod tokenizer;
mod unicode_data;

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
    let serve = flag("--serve");
    let tok = arg(&args, "--tokenizer").map(|p| {
        let t = tokenizer::Tokenizer::load(&p).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1)
        });
        if let Some(w) = &t.warning {
            eprintln!("warning: {w}");
        }
        t
    });
    if let Some(f) = arg(&args, "--tokenize") {
        let Some(t) = &tok else {
            eprintln!("--tokenize needs --tokenizer tokenizer.json");
            std::process::exit(1)
        };
        let text = std::fs::read_to_string(&f).expect("cannot read text file");
        if flag("--pieces") {
            // debugging: the pre-tokenizer's pieces, one JSON string per line
            for p in tokenizer::pretokenize(&text, t.marks) {
                println!("{}", json::esc(p));
            }
            return;
        }
        if flag("--nfc") {
            // debugging: the NFC-normalized text as one JSON string
            println!("{}", json::esc(&tokenizer::nfc(&text)));
            return;
        }
        let ids = t.encode(&text, !flag("--no-special"));
        println!("{}", ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        return;
    }
    let Some(pack_path) = arg(&args, "--pack") else {
        eprintln!("usage: moe-engine --pack model.moe --tokenizer tokenizer.json --serve");
        eprintln!("       moe-engine --pack model.moe --tokenizer tokenizer.json --prompt \"text\" [--max-new 64]");
        eprintln!("       moe-engine --pack model.moe --prompt-ids 1,2,3 [--max-new 64]");
        std::process::exit(1);
    };
    let prompt_text = arg(&args, "--prompt");
    if prompt_text.is_some() && tok.is_none() {
        eprintln!("--prompt needs --tokenizer tokenizer.json (or use --prompt-ids)");
        std::process::exit(1)
    }
    let bench_n: Option<usize> = arg(&args, "--bench-prompt").map(|v| v.parse().expect("bad --bench-prompt"));
    if bench_n.is_some() && (tok.is_none() || arg(&args, "--bench-text").is_none()) {
        eprintln!("--bench-prompt needs --tokenizer and --bench-text");
        std::process::exit(1)
    }
    let mut prompt: Vec<u32> = if prompt_text.is_some() || bench_n.is_some() {
        vec![0] // built below, once the model's chat format is known
    } else if let Some(f) = arg(&args, "--prompt-file") {
        parse_ids(&std::fs::read_to_string(f).expect("cannot read prompt file"))
    } else {
        parse_ids(&get("--prompt-ids", ""))
    };
    let ppl_file = arg(&args, "--perplexity");
    if ppl_file.is_some() && tok.is_none() {
        eprintln!("--perplexity needs --tokenizer tokenizer.json");
        std::process::exit(1);
    }
    if prompt.is_empty() && !serve && ppl_file.is_none() {
        eprintln!("no prompt given (--serve, --prompt, --prompt-ids or --prompt-file)");
        std::process::exit(1);
    }
    if serve && tok.is_none() {
        eprintln!("--serve needs --tokenizer tokenizer.json (the model's own file)");
        std::process::exit(1);
    }
    let max_new: usize = get("--max-new", "64").parse().unwrap();
    let ram_gb: f64 = get("--ram-gb", "7").parse().unwrap();
    let max_ctx_arg: Option<usize> = arg(&args, "--max-context").map(|v| v.parse().expect("bad --max-context"));
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
    let mcfg = model::Model::config(&pk).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    // default context: Qwen3.6's KV cache is small (only every 4th layer has one), so it
    // gets a longer default conversation for about the same RAM
    let max_ctx = max_ctx_arg.unwrap_or(if mcfg.qwen35 { 16384 } else { 4096 });
    if flag("--think") && mcfg.no_think {
        eprintln!("--think: this pack's model has no thinking mode (no_think=1); drop --think");
        std::process::exit(1);
    }
    if let Some(text) = &prompt_text {
        // one user message in the model's chat format; thinking off unless --think.
        // A model with no thinking mode at all (mcfg.no_think) never gets <think> tags:
        // inserting the empty pair it was never trained on can derail generation entirely.
        let t = tok.as_ref().unwrap();
        prompt = t.encode("<|im_start|>user\n", true);
        prompt.extend(t.encode(if mcfg.think_open { text.trim() } else { text }, false));
        prompt.extend(t.encode("<|im_end|>\n<|im_start|>assistant\n", true));
        if !flag("--think") {
            if !mcfg.no_think {
                prompt.extend(t.encode("<think>\n\n</think>\n\n", true));
            }
        } else if mcfg.think_open {
            prompt.extend(t.encode("<think>\n", true));
        }
    }
    if let Some(n) = bench_n {
        let t = tok.as_ref().unwrap();
        let text = std::fs::read_to_string(arg(&args, "--bench-text").unwrap()).expect("cannot read --bench-text");
        let head = t.encode("<|im_start|>user\nSummarize this text:\n\n", true);
        let tail = t.encode(
            if mcfg.no_think { "<|im_end|>\n<|im_start|>assistant\n" } else { "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" },
            true,
        );
        let body = t.encode(&text, true);
        let need = n.saturating_sub(head.len() + tail.len());
        if need == 0 || body.len() < need {
            eprintln!("--bench-prompt {n}: text too short or N too small");
            std::process::exit(1);
        }
        prompt = head;
        prompt.extend_from_slice(&body[..need]);
        prompt.extend(tail);
    }

    // ---- RAM plan under the hard cap
    let dense: usize = pk.tensors.iter().filter(|(n, _)| n.as_str() != "embed").map(|(_, t)| t.nbytes).sum();
    // KV cache for the attention layers, plus the DeltaNet layers' fixed running state
    // (twice: the live state and one saved snapshot)
    let n_attn = mcfg.attn.iter().filter(|&&a| a).count();
    let state_bytes = (mcfg.layers - n_attn) * mcfg.lin.state_len() * 4 * 2;
    let kv = max_ctx * n_attn * 2 * mcfg.kv_heads * mcfg.head_dim * 2 + state_bytes;
    // buffers, vocabulary, thread stacks, the program itself; the tokenizer's tables
    let other = 300_000_000usize + if tok.is_some() { 100_000_000 } else { 0 };
    let budget = (ram_gb * 1e9) as i64 - (dense + kv + other) as i64;
    let cache_bytes = match cache_req {
        Some(g) => (g * 1e9) as usize,
        None => budget.max(0) as usize,
    };
    let max_new = if serve || ppl_file.is_some() {
        max_new
    } else if prompt.len() + max_new > max_ctx {
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
            "error: does not fit in --ram-gb {ram_gb}: dense weights {:.2} GB + conversation memory for {max_ctx} tokens {:.2} GB + other {:.2} GB + expert cache {:.2} GB",
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
        let kind = if c.qwen35 {
            let na = c.attn.iter().filter(|&&a| a).count();
            format!("Qwen3.5/3.6-MoE, {} layers ({} DeltaNet + {} attention), shared expert", c.layers, c.layers - na, na)
        } else {
            format!("Qwen3-MoE, {} layers", c.layers)
        };
        eprintln!(
            "model: {kind}, {} experts (top-{}), hidden {}, vocab {} | kernels: {} | threads {} | io threads {}",
            c.experts, c.topk, c.hidden, c.vocab, simd, threads, io_threads
        );
        eprintln!(
            "RAM (cap {:.1} GB): dense weights {:.2} + expert cache {:.2} ({} huge pages) + conversation memory for {} tokens {:.2} + other {:.2} = {:.2} GB | expert block {:.2} MB | SSD reads: {}",
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

    if let Some(f) = &ppl_file {
        // perplexity: exp(mean negative log-probability of each next token)
        let text = std::fs::read_to_string(f).expect("cannot read text file");
        let mut ids = tok.as_ref().unwrap().encode(&text, true);
        let n: usize = get("--max-tokens", "2048").parse().expect("bad --max-tokens");
        ids.truncate(n.min(max_ctx));
        let t0 = Instant::now();
        let (mut nll, mut count) = (0f64, 0usize);
        // small chunks: every position's logits are kept until scored (vocab-sized each)
        for (ci, chunk) in ids.chunks(64).enumerate() {
            let out = m.forward_batch(chunk, true).expect("forward failed");
            for (j, lg) in out.iter().enumerate() {
                let i = ci * 64 + j;
                if i + 1 >= ids.len() {
                    break;
                }
                let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + lg.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
                nll += lse - lg[ids[i + 1] as usize] as f64;
                count += 1;
            }
            guard.check();
            eprint!("\r{} / {} tokens", (ci * 64 + chunk.len()).min(ids.len()), ids.len());
        }
        eprintln!();
        println!(
            "perplexity {:.3} over {} tokens of {} ({:.0} s)",
            (nll / count.max(1) as f64).exp(),
            count,
            f,
            t0.elapsed().as_secs_f64()
        );
        return;
    }

    if serve {
        let ram_json = format!(
            "{{\"cap_gb\":{:.2},\"attention_gb\":{:.2},\"cache_gb\":{:.2},\"kv_gb\":{:.2},\"other_gb\":{:.2}}}",
            ram_gb,
            m.dense_bytes as f64 / 1e9,
            store.cache_bytes() as f64 / 1e9,
            kv as f64 / 1e9,
            other as f64 / 1e9
        );
        let model_name = std::path::Path::new(&pack_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llm-on-ssd".into());
        let opts = server::Opts {
            host: get("--host", "0.0.0.0"),
            port: get("--port", "8080").parse().expect("bad --port"),
            password: arg(&args, "--password"),
            max_ctx,
            prefill_batch,
            model_name,
            default_max_tokens: get("--max-answer", "1024").parse().expect("bad --max-answer"),
            ram_json,
            qwen35: m.cfg.qwen35,
        };
        server::run(server::Engine { m, guard, vocab, tok: tok.unwrap() }, opts);
        return;
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
