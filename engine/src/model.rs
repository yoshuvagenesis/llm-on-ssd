//! Qwen3-MoE forward pass, one token at a time.
//! Dense weights (attention, routers, norms, output head) live in RAM.
//! Expert weights come from the ExpertStore (SSD + LRU cache + prefetch).
//! Embedding rows are read from the SSD per token.

use crate::kernels::{axpy_bf16, dot_bf16, matvec, matvec_f32, to_bf16, QAct, QTensor, Scales, QK};
use crate::pack::{Fmt, Pack};
use crate::pool::Pool;
use crate::store::{Block, ExpertStore};
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub topk: usize,
    pub ffn: usize,
    pub eps: f32,
    pub theta: f64,
    pub vocab: usize,
    pub norm_topk: bool,
    pub eos: Vec<u32>,
    pub expert_bf16_scales: bool,
}

enum OwnedScales {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

/// quantized matrix owned in RAM (q8, q4 or q4b)
pub struct QMat {
    data: Vec<u8>,
    scales: OwnedScales,
    rows: usize,
    cols: usize,
    q4: bool,
}

impl QMat {
    fn view(&self) -> QTensor<'_> {
        let scales = match &self.scales {
            OwnedScales::F32(v) => Scales::F32(v),
            OwnedScales::Bf16(v) => Scales::Bf16(v),
        };
        QTensor { data: &self.data, scales, rows: self.rows, cols: self.cols, q4: self.q4 }
    }
    fn bytes(&self) -> usize {
        self.data.len()
            + match &self.scales {
                OwnedScales::F32(v) => v.len() * 4,
                OwnedScales::Bf16(v) => v.len() * 2,
            }
    }
}

struct Layer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    wq: QMat,
    wk: QMat,
    wv: QMat,
    wo: QMat,
    router: Vec<f32>,
}

pub struct Options {
    pub threads: usize,
    pub lookahead: usize,
    pub prefetch_width: usize,  // experts predicted for the next layer
    pub prefetch_width2: usize, // experts predicted for layers further ahead
    pub refine: bool,           // re-predict the next layer after each layer's experts
}

#[derive(Default, Clone)]
pub struct Timing {
    pub attn: f64,
    pub moe: f64,
    pub head: f64,
    pub predict: f64,
    pub stall: f64, // wall-clock time waiting for expert blocks
}

pub struct Model {
    pub cfg: Config,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    lm_head: QMat,
    embed_off: u64,
    pack: Pack,
    pub store: Arc<ExpertStore>,
    pub opt: Options,
    pool: Pool,
    pub dense_bytes: usize,
    pub timing: Timing,
    // state
    kcache: Vec<Vec<u16>>, // bf16
    vcache: Vec<Vec<u16>>,
    guesses: Vec<Vec<usize>>, // experts guessed for each layer during the current token
    pub pos: usize,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn load_f32(p: &Pack, name: &str) -> io::Result<Vec<f32>> {
    let t = p.tensor(name)?;
    if t.fmt != Fmt::F32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{name}: expected f32")));
    }
    Ok(f32s(&p.tensor_bytes(name)?))
}

fn load_q(p: &Pack, name: &str) -> io::Result<QMat> {
    let t = p.tensor(name)?.clone();
    let mut b = p.tensor_bytes(name)?;
    let (n, q4) = match t.fmt {
        Fmt::Q8 => (t.rows * t.cols, false),
        Fmt::Q4 | Fmt::Q4b => (t.rows * t.cols / 2, true),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{name}: expected q8/q4/q4b"))),
    };
    let scales = if t.fmt == Fmt::Q4b {
        OwnedScales::Bf16(b[n..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    } else {
        OwnedScales::F32(f32s(&b[n..]))
    };
    b.truncate(n);
    Ok(QMat { data: b, scales, rows: t.rows, cols: t.cols, q4 })
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r * w[i];
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * r * w[i];
    }
}

/// cos/sin of every rotary frequency at one position (computed once per token)
fn rope_table(pos: usize, hd: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    (0..hd / 2)
        .map(|i| {
            let ang = pos as f64 * (1.0 / theta.powf((2 * i) as f64 / hd as f64));
            (ang.cos() as f32, ang.sin() as f32)
        })
        .unzip()
}

fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

/// Rows [s, s + out.len()) of several matrices stacked on top of each other.
fn matvec_stacked(mats: &[QTensor], x: &QAct, s: usize, out: &mut [f32]) {
    let mut done = 0;
    let mut start = 0;
    for m in mats {
        let end = start + m.rows;
        let g = s + done;
        if done < out.len() && g < end && g >= start {
            let take = (out.len() - done).min(end - g);
            matvec(m, x, g - start, &mut out[done..done + take]);
            done += take;
        }
        start = end;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn topk(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// View the three 4-bit matrices inside an expert block (gate, up, down).
fn expert_views<'a>(buf: &'a Block, h: usize, f: usize, bf16: bool) -> [QTensor<'a>; 3] {
    let b = buf.as_slice();
    let ssz = if bf16 { 2 } else { 4 };
    let q4 = |rows: usize, cols: usize| rows * cols / 2 + rows * (cols / QK) * ssz;
    let (ga, gb) = (0, q4(f, h));
    let (ua, ub) = (gb, gb + q4(f, h));
    let (da, db) = (ub, ub + q4(h, f));
    let mk = |a: usize, e: usize, rows: usize, cols: usize| {
        let d = rows * cols / 2;
        let sb = &b[a + d..e];
        assert!(sb.as_ptr() as usize % ssz == 0 && sb.len() % ssz == 0);
        // Safety: alignment and length checked above.
        let scales = unsafe {
            if bf16 {
                Scales::Bf16(std::slice::from_raw_parts(sb.as_ptr() as *const u16, sb.len() / 2))
            } else {
                Scales::F32(std::slice::from_raw_parts(sb.as_ptr() as *const f32, sb.len() / 4))
            }
        };
        QTensor { data: &b[a..a + d], scales, rows, cols, q4: true }
    };
    [mk(ga, gb, f, h), mk(ua, ub, f, h), mk(da, db, h, f)]
}

impl Model {
    pub fn load(pack: Pack, store: Arc<ExpertStore>, opt: Options) -> io::Result<Model> {
        let eos = pack
            .cfg_str("eos")?
            .split(',')
            .filter_map(|s| s.parse::<i64>().ok())
            .filter(|&v| v >= 0)
            .map(|v| v as u32)
            .collect();
        let cfg = Config {
            hidden: pack.cfg_usize("hidden_size")?,
            layers: pack.cfg_usize("num_hidden_layers")?,
            heads: pack.cfg_usize("num_attention_heads")?,
            kv_heads: pack.cfg_usize("num_key_value_heads")?,
            head_dim: pack.cfg_usize("head_dim")?,
            experts: pack.cfg_usize("num_experts")?,
            topk: pack.cfg_usize("num_experts_per_tok")?,
            ffn: pack.cfg_usize("moe_intermediate_size")?,
            eps: pack.cfg_f64("rms_norm_eps")? as f32,
            theta: pack.cfg_f64("rope_theta")?,
            vocab: pack.cfg_usize("vocab_size")?,
            norm_topk: pack.cfg_usize("norm_topk_prob")? != 0,
            eos,
            expert_bf16_scales: match pack.cfg.get("expert_fmt").map(|s| s.as_str()) {
                None | Some("q4") => false,
                Some("q4b") => true,
                Some(o) => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown expert_fmt {o}"))),
            },
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut dense_bytes = 0;
        for l in 0..cfg.layers {
            let n = |s: &str| format!("L{l}.{s}");
            let ly = Layer {
                ln1: load_f32(&pack, &n("ln1"))?,
                ln2: load_f32(&pack, &n("ln2"))?,
                qn: load_f32(&pack, &n("qn"))?,
                kn: load_f32(&pack, &n("kn"))?,
                wq: load_q(&pack, &n("wq"))?,
                wk: load_q(&pack, &n("wk"))?,
                wv: load_q(&pack, &n("wv"))?,
                wo: load_q(&pack, &n("wo"))?,
                router: load_f32(&pack, &n("router"))?,
            };
            dense_bytes += ly.wq.bytes() + ly.wk.bytes() + ly.wv.bytes() + ly.wo.bytes() + ly.router.len() * 4;
            layers.push(ly);
        }
        let norm = load_f32(&pack, "norm")?;
        let lm_head = load_q(&pack, "lm_head")?;
        dense_bytes += lm_head.bytes();
        let et = pack.tensor("embed")?;
        if et.fmt != Fmt::Bf16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "embed: expected bf16"));
        }
        let embed_off = pack.data_start + et.off;
        let nl = cfg.layers;
        let pool = Pool::new(opt.threads);
        Ok(Model {
            cfg,
            layers,
            norm,
            lm_head,
            embed_off,
            pack,
            store,
            opt,
            pool,
            dense_bytes,
            timing: Timing::default(),
            kcache: vec![Vec::new(); nl],
            vcache: vec![Vec::new(); nl],
            guesses: vec![Vec::new(); nl],
            pos: 0,
        })
    }

    fn embed(&self, tok: u32) -> io::Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let mut b = vec![0u8; h * 2];
        self.pack.file().read_exact_at(&mut b, self.embed_off + tok as u64 * (h as u64 * 2))?;
        Ok(b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect())
    }

    fn probs_topk(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = p.iter().sum();
        p.iter_mut().for_each(|v| *v /= s);
        let idx = topk(&p, k);
        let w = idx.iter().map(|&i| p[i]).collect();
        (idx, w)
    }

    /// Forget the conversation after the first `n` tokens (n = 0 starts fresh). The
    /// KV cache keeps the first n positions, so a new message that repeats them
    /// continues from there instead of reading everything again.
    pub fn truncate(&mut self, n: usize) {
        let n = n.min(self.pos);
        let row = self.cfg.kv_heads * self.cfg.head_dim;
        for l in 0..self.cfg.layers {
            self.kcache[l].truncate(n * row);
            self.vcache[l].truncate(n * row);
            self.guesses[l].clear();
        }
        self.pos = n;
    }

    /// Run one token (decode). Returns the logits for the next token.
    pub fn forward(&mut self, tok: u32) -> io::Result<Vec<f32>> {
        Ok(self.forward_batch(&[tok], false)?.pop().unwrap())
    }

    /// Run several consecutive tokens layer by layer (prompt reading), so every expert is
    /// read once per layer for the whole batch. Per token the math is exactly the same as
    /// in single-token decoding (same kernels, same order), so results are bit-identical.
    /// Returns logits for the last token, or for every token if `all_logits`.
    pub fn forward_batch(&mut self, toks: &[u32], all_logits: bool) -> io::Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let (hdim, nh, nkv, hd, nl) = (c.hidden, c.heads, c.kv_heads, c.head_dim, c.layers);
        let (eps, theta, kk, ff, ne) = (c.eps, c.theta, c.topk, c.ffn, c.experts);
        let bt = toks.len();
        let pool = &self.pool;
        let pos0 = self.pos;
        let bf16 = self.cfg.expert_bf16_scales;
        let la = if self.opt.prefetch_width == 0 { 0 } else { self.opt.lookahead };
        let kvd = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut hs: Vec<Vec<f32>> = toks.iter().map(|&t| self.embed(t)).collect::<io::Result<_>>()?;
        let ropes: Vec<(Vec<f32>, Vec<f32>)> = (0..bt).map(|b| rope_table(pos0 + b, hd, theta)).collect();
        let mut xn = vec![0f32; hdim];
        let mut xa = QAct::new(hdim);
        let mut qkv = vec![0f32; (nh + 2 * nkv) * hd];
        let mut att = vec![0f32; nh * hd];
        let mut atq = QAct::new(nh * hd);
        let mut o = vec![0f32; hdim];
        // per token: router inputs/logits (this layer + guesses), routing, quantized input, expert outputs
        let mut rx: Vec<Vec<f32>> = vec![vec![0f32; hdim]; bt * (1 + la)];
        let mut rlog: Vec<Vec<f32>> = vec![vec![0f32; ne]; bt * (1 + la)];
        let mut xq: Vec<QAct> = (0..bt).map(|_| QAct::new(hdim)).collect();
        let mut outs: Vec<Vec<f32>> = vec![vec![0f32; hdim]; bt * kk];
        let mut last_routes: Vec<(usize, Vec<usize>)> = Vec::new(); // (layer, experts of the last tokens)

        for l in 0..nl {
            self.store.tick();
            let t0 = Instant::now();
            let ly = &self.layers[l];
            let wo = ly.wo.view();
            let stacked = [ly.wq.view(), ly.wk.view(), ly.wv.view()];
            // ---- attention, token by token (each token sees the ones before it)
            for b in 0..bt {
                let h = &mut hs[b];
                let (rc, rs) = (&ropes[b].0, &ropes[b].1);
                rmsnorm(h, &ly.ln1, eps, &mut xn);
                xa.set(&xn);
                pool.chunks(&mut qkv, 16, |r0, out| matvec_stacked(&stacked, &xa, r0, out));
                let (q, kv) = qkv.split_at_mut(nh * hd);
                let (k, v) = kv.split_at_mut(kvd);
                for hh in 0..nh {
                    rmsnorm_inplace(&mut q[hh * hd..(hh + 1) * hd], &ly.qn, eps);
                    rope(&mut q[hh * hd..(hh + 1) * hd], rc, rs);
                }
                for hh in 0..nkv {
                    rmsnorm_inplace(&mut k[hh * hd..(hh + 1) * hd], &ly.kn, eps);
                    rope(&mut k[hh * hd..(hh + 1) * hd], rc, rs);
                }
                // KV cache in bfloat16 (Qwen3's native precision): half the RAM
                self.kcache[l].extend(k.iter().map(|&x| to_bf16(x)));
                self.vcache[l].extend(v.iter().map(|&x| to_bf16(x)));
                let (kc, vc) = (&self.kcache[l], &self.vcache[l]);
                let npos = pos0 + b + 1;
                let qref = &*q;
                pool.chunks(&mut att, hd, |i0, out| {
                    let mut sc = vec![0f32; npos];
                    for (j, oh) in out.chunks_mut(hd).enumerate() {
                        let hh = i0 / hd + j;
                        let g = hh / (nh / nkv);
                        let qh = &qref[hh * hd..(hh + 1) * hd];
                        let mut m = f32::NEG_INFINITY;
                        for p in 0..npos {
                            let s = dot_bf16(qh, &kc[p * kvd + g * hd..p * kvd + (g + 1) * hd]) * scale;
                            sc[p] = s;
                            m = m.max(s);
                        }
                        let mut tot = 0f32;
                        for s in sc.iter_mut() {
                            *s = (*s - m).exp();
                            tot += *s;
                        }
                        oh.iter_mut().for_each(|x| *x = 0.0);
                        for p in 0..npos {
                            axpy_bf16(sc[p] / tot, &vc[p * kvd + g * hd..p * kvd + (g + 1) * hd], oh);
                        }
                    }
                });
                atq.set(&att);
                pool.chunks(&mut o, 16, |r0, out| matvec(&wo, &atq, r0, out));
                for i in 0..hdim {
                    h[i] += o[i];
                }
            }
            let t1 = Instant::now();
            self.timing.attn += (t1 - t0).as_secs_f64();

            // ---- routers for every token: this layer's, plus guesses for the next layers
            let nr = (0..=la).take_while(|&d| l + d < nl).count();
            for b in 0..bt {
                for d in 0..nr {
                    rmsnorm(&hs[b], &self.layers[l + d].ln2, eps, &mut rx[b * (1 + la) + d]);
                }
            }
            {
                let lb: Vec<usize> = rlog.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
                let (layers, rxr) = (&self.layers, &rx);
                pool.run(bt * nr, |t| {
                    let (b, d) = (t / nr, t % nr);
                    let i = b * (1 + la) + d;
                    // Safety: each task writes only its own logits vector.
                    let out = unsafe { std::slice::from_raw_parts_mut(lb[i] as *mut f32, ne) };
                    matvec_f32(&layers[l + d].router, hdim, &rxr[i], out);
                });
            }
            let mut routes: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(bt);
            for b in 0..bt {
                let (idx, mut w) = Self::probs_topk(&rlog[b * (1 + la)], kk);
                if self.cfg.norm_topk {
                    let s: f32 = w.iter().sum();
                    w.iter_mut().for_each(|x| *x /= s);
                }
                xq[b].set(&rx[b * (1 + la)]);
                routes.push((idx, w));
            }
            if bt > 1 {
                let mut es: Vec<usize> = routes[bt.saturating_sub(8)..].iter().flat_map(|(i, _)| i.iter().copied()).collect();
                es.sort_unstable();
                es.dedup();
                last_routes.push((l, es));
            }
            // experts needed now (union over the batch), in order of first use
            let mut order: Vec<usize> = Vec::new();
            let mut users: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
            for (b, (idx, _)) in routes.iter().enumerate() {
                for (j, &e) in idx.iter().enumerate() {
                    if users[e].is_empty() {
                        order.push(e);
                    }
                    users[e].push((b, j));
                }
            }
            for &e in &order {
                self.store.prefetch(l * ne + e, 2, true); // read before any guesses
            }
            // guesses for the next layers (queued behind this layer's reads)
            for d in 1..nr {
                let width = if d == 1 { self.opt.prefetch_width } else { self.opt.prefetch_width2 };
                if width == 0 {
                    break;
                }
                for b in 0..bt {
                    let (pidx, _) = Self::probs_topk(&rlog[b * (1 + la) + d], width.min(ne));
                    for e in pidx {
                        self.store.prefetch((l + d) * ne + e, (d + 1) as u64, false);
                        if bt == 1 && !self.guesses[l + d].contains(&e) {
                            self.guesses[l + d].push(e);
                        }
                    }
                }
            }
            let t2 = Instant::now();
            self.timing.predict += (t2 - t1).as_secs_f64();

            // ---- experts: blocks already in RAM first, then in arrival order; each expert
            // is computed for all of its tokens as soon as its block is there
            order.sort_by_key(|&e| !self.store.is_ready(l * ne + e));
            let compute_ns = AtomicU64::new(0);
            {
                let obase: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
                let (store, users, xq, order) = (&self.store, &users, &xq, &order);
                let cns = &compute_ns;
                pool.run(order.len(), |t| {
                    let e = order[t];
                    let blk = store.get(l * ne + e);
                    let tc = Instant::now();
                    let [wg, wu, wd] = expert_views(&blk, hdim, ff, bf16);
                    let mut g = vec![0f32; ff];
                    let mut u = vec![0f32; ff];
                    let mut hq = QAct::new(ff);
                    for &(b, j) in &users[e] {
                        matvec(&wg, &xq[b], 0, &mut g);
                        matvec(&wu, &xq[b], 0, &mut u);
                        for i in 0..ff {
                            g[i] = silu(g[i]) * u[i];
                        }
                        hq.set(&g);
                        // Safety: each (token, slot) output is written by exactly one task.
                        let y = unsafe { std::slice::from_raw_parts_mut(obase[b * kk + j] as *mut f32, hdim) };
                        matvec(&wd, &hq, 0, y);
                    }
                    cns.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                });
            }
            // fixed summation order per token: identical for any batch size or thread count
            for (b, (idx, w)) in routes.iter().enumerate() {
                let h = &mut hs[b];
                for j in 0..idx.len() {
                    let y = &outs[b * kk + j];
                    for i in 0..hdim {
                        h[i] += w[j] * y[i];
                    }
                }
            }
            // ---- refine the guess for the next layer from the newer hidden state: add what
            // it now expects, withdraw queued guesses it now finds unlikely (decode only)
            if bt == 1 && self.opt.refine && la > 0 && l + 1 < nl {
                let tr = Instant::now();
                let w1 = self.opt.prefetch_width.min(ne);
                let mut xr = vec![0f32; hdim];
                rmsnorm(&hs[0], &self.layers[l + 1].ln2, eps, &mut xr);
                let mut lg = vec![0f32; ne];
                matvec_f32(&self.layers[l + 1].router, hdim, &xr, &mut lg);
                let (keep, _) = Self::probs_topk(&lg, (w1 + 4).min(ne)); // generous: cancel only clearly unlikely
                for &e in &keep[..w1] {
                    self.store.prefetch((l + 1) * ne + e, 2, false);
                }
                for &e in &self.guesses[l + 1] {
                    if !keep.contains(&e) {
                        self.store.cancel_guess((l + 1) * ne + e);
                    }
                }
                self.timing.predict += tr.elapsed().as_secs_f64();
            }
            self.guesses[l].clear();
            let wall = t2.elapsed().as_secs_f64();
            let busy = compute_ns.load(Ordering::Relaxed) as f64 / 1e9 / pool.threads.min(order.len().max(1)) as f64;
            self.timing.moe += wall;
            self.timing.stall += (wall - busy).max(0.0);
            if let Ok(path) = std::env::var("MOE_DEBUG_DUMP") {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
                for h in &hs {
                    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
                    f.write_all(&bytes)?;
                }
            }
        }

        // ---- after reading a prompt: keep the experts of its last tokens, layer 0 last
        // (most recent), because generation starts with them
        if bt > 1 {
            let keys: Vec<usize> = last_routes.iter().rev().flat_map(|(l, es)| es.iter().map(move |&e| l * ne + e)).collect();
            self.store.touch(&keys);
        }

        // ---- output head: only where logits are needed (prompt reading needs just the last)
        let t0 = Instant::now();
        let lm = self.lm_head.view();
        let mut res = Vec::new();
        for b in 0..bt {
            if !all_logits && b + 1 != bt {
                continue;
            }
            rmsnorm(&hs[b], &self.norm, eps, &mut xn);
            xa.set(&xn);
            let mut logits = vec![0f32; self.cfg.vocab];
            pool.chunks(&mut logits, 16, |r0, out| matvec(&lm, &xa, r0, out));
            res.push(logits);
        }
        self.timing.head += t0.elapsed().as_secs_f64();
        self.pos += bt;
        Ok(res)
    }
}
