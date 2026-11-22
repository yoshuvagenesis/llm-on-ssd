//! Step 0 / compute benchmark for the SSD-MoE engine.
//!
//! Measures how fast this CPU runs one MoE "expert FFN"
//! (gate + up + down projections, Qwen3-30B-A3B shapes: 2048 -> 768 -> 2048)
//! using only SSE-family instructions (SSSE3 / SSE4.1), because the N5105 has
//! no AVX / AVX2 / FMA / F16C.
//!
//! Five ways of doing the math, from "multiply" down to "add only":
//!   q8   : multiply 8-bit weights           (activations 8-bit per 32)
//!   q4   : multiply 4-bit weights           (activations 8-bit per 32)
//!   lut4 : LOOK UP 4-bit weights            (activations 6-bit per 32; tables via pshufb)
//!   lut2 : LOOK UP 2-bit weights            (activations 6-bit per 32)
//!   tern : ADD/SUBTRACT only, weights -1/0/+1 (activations 8-bit per token; needs a
//!          model trained this way, e.g. BitNet-style)
//!
//! Every kernel is first checked against a plain scalar reference.
//! "Batch" K = tokens processed against the same weights (speculative verification).
//!
//! Build & run:  cargo build --release && ./target/release/int8bench
//! Options: --threads N  --secs S  --pool-mb MB  --active-gmac G  --formats q4,lut2,...

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const QK: usize = 32; // quantization block
const TILE: usize = 16; // rows per LUT tile (one pshufb lane per row)
const HIDDEN: usize = 2048;
const FFN: usize = 768;
const MACS_PER_EXPERT: f64 = (3 * FFN * HIDDEN) as f64;

// ---------------------------------------------------------------- formats

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Q8,
    Q4,
    Lut4,
    Lut2,
    Tern,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Act {
    Block8,  // int8, one scale per 32 values
    Block6,  // int6 (-31..31), one scale per 32 values (keeps LUT entries in int8)
    Tensor8, // int8, one scale for the whole vector
}

impl Fmt {
    fn parse(s: &str) -> Option<Fmt> {
        Some(match s {
            "q8" => Fmt::Q8,
            "q4" => Fmt::Q4,
            "lut4" => Fmt::Lut4,
            "lut2" => Fmt::Lut2,
            "tern" => Fmt::Tern,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Fmt::Q8 => "q8",
            Fmt::Q4 => "q4",
            Fmt::Lut4 => "lut4",
            Fmt::Lut2 => "lut2",
            Fmt::Tern => "tern",
        }
    }
    /// integer range of the stored weight values
    fn range(self) -> (i32, i32) {
        match self {
            Fmt::Q8 => (-127, 127),
            Fmt::Q4 | Fmt::Lut4 => (-8, 7),
            Fmt::Lut2 => (-2, 1),
            Fmt::Tern => (-1, 1),
        }
    }
    fn act(self) -> Act {
        match self {
            Fmt::Q8 | Fmt::Q4 => Act::Block8,
            Fmt::Lut4 | Fmt::Lut2 => Act::Block6,
            Fmt::Tern => Act::Tensor8,
        }
    }
    /// storage bits per weight including f32 block scales as used here
    fn bits(self) -> f64 {
        match self {
            Fmt::Q8 => 9.0,
            Fmt::Q4 | Fmt::Lut4 => 5.0,
            Fmt::Lut2 => 3.0,
            Fmt::Tern => 2.0,
        }
    }
    fn how(self) -> &'static str {
        match self {
            Fmt::Q8 => "multiply, 8-bit weights",
            Fmt::Q4 => "multiply, 4-bit weights",
            Fmt::Lut4 => "table lookup, 4-bit weights",
            Fmt::Lut2 => "table lookup, 2-bit weights",
            Fmt::Tern => "add/subtract only, -1/0/+1 weights",
        }
    }
    fn lut_bits(self) -> usize {
        match self {
            Fmt::Lut4 => 4,
            Fmt::Lut2 => 2,
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------- data types

struct QMat {
    rows: usize,
    cols: usize,
    fmt: Fmt,
    data: Vec<u8>,
    scales: Vec<f32>,     // [row][block] (Tern: [row])
    lut_scales: Vec<f32>, // LUT only: [tile][block][16]
    logical: Option<Vec<i8>>,
}

impl QMat {
    fn nb(&self) -> usize {
        self.cols / QK
    }
    fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    fn random(rows: usize, cols: usize, fmt: Fmt, rng: &mut Rng, keep: bool) -> Self {
        let (lo, hi) = fmt.range();
        let span = (hi - lo + 1) as u64;
        let w: Vec<i8> = (0..rows * cols).map(|_| (lo + (rng.next() % span) as i32) as i8).collect();
        let ns = if fmt == Fmt::Tern { rows } else { rows * cols / QK };
        let scales = (0..ns).map(|_| 0.002 + 0.004 * rng.unit()).collect();
        Self::from_logical(rows, cols, fmt, w, scales, keep)
    }

    fn from_logical(rows: usize, cols: usize, fmt: Fmt, w: Vec<i8>, scales: Vec<f32>, keep: bool) -> Self {
        assert!(cols % 64 == 0 && rows % TILE == 0);
        let nb = cols / QK;
        let mut lut_scales = Vec::new();
        let data: Vec<u8> = match fmt {
            Fmt::Q8 => w.iter().map(|&v| v as u8).collect(),
            Fmt::Q4 => {
                // per block: byte j = elem j (low nibble) | elem j+16 (high nibble)
                let mut d = vec![0u8; rows * cols / 2];
                for r in 0..rows {
                    for b in 0..nb {
                        for j in 0..QK / 2 {
                            let lo = (w[r * cols + b * QK + j] as i32 + 8) as u8;
                            let hi = (w[r * cols + b * QK + j + 16] as i32 + 8) as u8;
                            d[r * cols / 2 + b * QK / 2 + j] = (lo & 0xF) | (hi << 4);
                        }
                    }
                }
                d
            }
            Fmt::Lut4 | Fmt::Lut2 => {
                // tile (16 rows) x block (32 cols) x bit-plane (MSB first) x pair of 4-col groups:
                // 16 bytes, byte r = index(group 2q, row r) | index(group 2q+1, row r) << 4
                let bits = fmt.lut_bits();
                let off = -fmt.range().0;
                let mut d = Vec::with_capacity(rows * cols * bits / 8);
                for t in 0..rows / TILE {
                    for b in 0..nb {
                        for p in (0..bits).rev() {
                            for q in 0..4 {
                                for r in 0..TILE {
                                    let row = t * TILE + r;
                                    let idx = |g: usize| -> u8 {
                                        let mut x = 0u8;
                                        for j in 0..4 {
                                            let u = (w[row * cols + b * QK + g * 4 + j] as i32 + off) as u8;
                                            x |= ((u >> p) & 1) << j;
                                        }
                                        x
                                    };
                                    d.push(idx(2 * q) | (idx(2 * q + 1) << 4));
                                }
                            }
                        }
                        for r in 0..TILE {
                            lut_scales.push(scales[(t * TILE + r) * nb + b]);
                        }
                    }
                }
                d
            }
            Fmt::Tern => {
                // per row, per 64 cols: byte j holds cols j, j+16, j+32, j+48 at bits 0,2,4,6
                // code 0 -> 0, 1 -> +1, 2 -> -1
                let mut d = vec![0u8; rows * cols / 4];
                for r in 0..rows {
                    for c in 0..cols / 64 {
                        for j in 0..16 {
                            let mut byte = 0u8;
                            for s in 0..4 {
                                let v = w[r * cols + c * 64 + s * 16 + j];
                                let code = match v {
                                    0 => 0u8,
                                    1 => 1,
                                    -1 => 2,
                                    _ => unreachable!(),
                                };
                                byte |= code << (2 * s);
                            }
                            d[r * cols / 4 + c * 16 + j] = byte;
                        }
                    }
                }
                d
            }
        };
        QMat { rows, cols, fmt, data, scales, lut_scales, logical: if keep { Some(w) } else { None } }
    }
}

/// quantized activation vector
#[derive(Clone)]
struct QVec {
    q: Vec<i8>,
    s: Vec<f32>, // per 32 values (for Tensor8 all entries are equal)
}

impl QVec {
    fn zeros(n: usize) -> Self {
        QVec { q: vec![0; n], s: vec![0.0; n / QK] }
    }
}

fn quantize(x: &[f32], act: Act, out: &mut QVec) {
    let qmax = if act == Act::Block6 { 31.0 } else { 127.0 };
    let global = if act == Act::Tensor8 { x.iter().fold(0f32, |m, v| m.max(v.abs())) } else { 0.0 };
    for (b, blk) in x.chunks_exact(QK).enumerate() {
        let amax = if act == Act::Tensor8 { global } else { blk.iter().fold(0f32, |m, v| m.max(v.abs())) };
        let d = amax / qmax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.s[b] = d;
        for (j, v) in blk.iter().enumerate() {
            out.q[b * QK + j] = (v * id).round().clamp(-qmax, qmax) as i8;
        }
    }
}

/// LUT tables for one activation vector: for every group of 4 activations,
/// the 16 possible partial sums (every subset of the 4 values).
struct LutTabs {
    t: Vec<i8>,     // [block][group 0..8][16]
    sums: Vec<i32>, // sum of activations per block (for the weight offset)
}

fn build_tabs(x: &QVec) -> LutTabs {
    let nb = x.q.len() / QK;
    let mut t = vec![0i8; nb * 8 * 16];
    let mut sums = vec![0i32; nb];
    for b in 0..nb {
        for g in 0..8 {
            let a = &x.q[b * QK + g * 4..b * QK + g * 4 + 4];
            let tab = &mut t[(b * 8 + g) * 16..(b * 8 + g + 1) * 16];
            for idx in 1..16usize {
                let low = idx.trailing_zeros() as usize;
                tab[idx] = tab[idx & (idx - 1)] + a[low]; // |sum| <= 4*31 = 124, fits i8
            }
        }
        sums[b] = x.q[b * QK..(b + 1) * QK].iter().map(|&v| v as i32).sum();
    }
    LutTabs { t, sums }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ---------------------------------------------------------------- scalar reference

fn rows_scalar(m: &QMat, xs: &[QVec], out: &mut [f32]) {
    let w = m.logical.as_ref().expect("scalar reference needs logical weights");
    let nb = m.nb();
    for r in 0..m.rows {
        for (k, x) in xs.iter().enumerate() {
            let row = &w[r * m.cols..(r + 1) * m.cols];
            out[k * m.rows + r] = if m.fmt == Fmt::Tern {
                let s: i32 = row.iter().zip(&x.q).map(|(&a, &b)| a as i32 * b as i32).sum();
                s as f32 * m.scales[r] * x.s[0]
            } else {
                let mut acc = 0f32;
                for b in 0..nb {
                    let s: i32 = (0..QK).map(|j| row[b * QK + j] as i32 * x.q[b * QK + j] as i32).sum();
                    acc += s as f32 * (m.scales[r * nb + b] * x.s[b]);
                }
                acc
            };
        }
    }
}

// ---------------------------------------------------------------- SSE kernels

#[cfg(target_arch = "x86_64")]
mod sse {
    use super::{build_tabs, Fmt, LutTabs, QMat, QVec, QK, TILE};
    use std::arch::x86_64::*;

    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn hsum(v: __m128) -> f32 {
        let hi = _mm_movehl_ps(v, v);
        let s = _mm_add_ps(v, hi);
        let s2 = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s2)
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn hsum_i32(v: __m128i) -> i32 {
        let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        _mm_cvtsi128_si32(s)
    }

    /// signed int8 dot of a 32-wide block using the sign trick for _mm_maddubs_epi16
    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn block_dot(w0: __m128i, w1: __m128i, a: *const i8, ones: __m128i) -> __m128i {
        let a0 = _mm_loadu_si128(a as *const __m128i);
        let a1 = _mm_loadu_si128(a.add(16) as *const __m128i);
        let p0 = _mm_maddubs_epi16(_mm_sign_epi8(w0, w0), _mm_sign_epi8(a0, w0));
        let p1 = _mm_maddubs_epi16(_mm_sign_epi8(w1, w1), _mm_sign_epi8(a1, w1));
        _mm_add_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones))
    }

    // ---- multiply kernels (q8, q4)
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn mul_rows<const K: usize>(m: &QMat, xs: &[QVec], out: &mut [f32]) {
        let nb = m.nb();
        let rb = if m.fmt == Fmt::Q8 { m.cols } else { m.cols / 2 };
        let ones = _mm_set1_epi16(1);
        let m4 = _mm_set1_epi8(0x0F);
        let off = _mm_set1_epi8(8);
        let w = m.data.as_ptr();
        let ws = m.scales.as_ptr();
        for r in 0..m.rows {
            let wr = w.add(r * rb);
            let sr = ws.add(r * nb);
            let mut acc = [_mm_setzero_ps(); K];
            for b in 0..nb {
                let (w0, w1) = if m.fmt == Fmt::Q8 {
                    (
                        _mm_loadu_si128(wr.add(b * QK) as *const __m128i),
                        _mm_loadu_si128(wr.add(b * QK + 16) as *const __m128i),
                    )
                } else {
                    let raw = _mm_loadu_si128(wr.add(b * QK / 2) as *const __m128i);
                    (
                        _mm_sub_epi8(_mm_and_si128(raw, m4), off),
                        _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), off),
                    )
                };
                let d = *sr.add(b);
                for k in 0..K {
                    let s = block_dot(w0, w1, xs[k].q.as_ptr().add(b * QK), ones);
                    let sc = _mm_set1_ps(d * xs[k].s[b]);
                    acc[k] = _mm_add_ps(acc[k], _mm_mul_ps(_mm_cvtepi32_ps(s), sc));
                }
            }
            for k in 0..K {
                out[k * m.rows + r] = hsum(acc[k]);
            }
        }
    }

    // ---- lookup-table kernels (lut4, lut2)
    // For each tile of 16 rows and each group of 4 activations, a 4-bit index per row
    // selects one of the 16 precomputed partial sums; _mm_shuffle_epi8 does 16 such
    // lookups (one per row) in one instruction. Bit-planes are combined by doubling.
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn lut_rows<const K: usize, const BITS: usize>(m: &QMat, tabs: &[LutTabs], xs: &[QVec], out: &mut [f32]) {
        let nb = m.nb();
        let off = -m.fmt.range().0;
        let m4 = _mm_set1_epi8(0x0F);
        let data = m.data.as_ptr();
        let lsc = m.lut_scales.as_ptr();
        for t in 0..m.rows / TILE {
            let mut facc = [[_mm_setzero_ps(); 4]; K];
            for b in 0..nb {
                let mut lo = [_mm_setzero_si128(); K];
                let mut hi = [_mm_setzero_si128(); K];
                let base = data.add((t * nb + b) * BITS * 4 * 16);
                for p in 0..BITS {
                    for k in 0..K {
                        lo[k] = _mm_add_epi16(lo[k], lo[k]);
                        hi[k] = _mm_add_epi16(hi[k], hi[k]);
                    }
                    for q in 0..4 {
                        let v = _mm_loadu_si128(base.add((p * 4 + q) * 16) as *const __m128i);
                        let i0 = _mm_and_si128(v, m4);
                        let i1 = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
                        for k in 0..K {
                            let tp = tabs[k].t.as_ptr().add((b * 8 + 2 * q) * 16);
                            let r0 = _mm_shuffle_epi8(_mm_loadu_si128(tp as *const __m128i), i0);
                            let r1 = _mm_shuffle_epi8(_mm_loadu_si128(tp.add(16) as *const __m128i), i1);
                            lo[k] = _mm_add_epi16(lo[k], _mm_add_epi16(_mm_cvtepi8_epi16(r0), _mm_cvtepi8_epi16(r1)));
                            hi[k] = _mm_add_epi16(
                                hi[k],
                                _mm_add_epi16(
                                    _mm_cvtepi8_epi16(_mm_srli_si128(r0, 8)),
                                    _mm_cvtepi8_epi16(_mm_srli_si128(r1, 8)),
                                ),
                            );
                        }
                    }
                }
                let sp = lsc.add((t * nb + b) * TILE);
                for k in 0..K {
                    // stored codes are (weight + off), so subtract off * sum(activations)
                    let corr = _mm_set1_epi16((off * tabs[k].sums[b]) as i16);
                    let l = _mm_sub_epi16(lo[k], corr);
                    let h = _mm_sub_epi16(hi[k], corr);
                    let parts = [
                        _mm_cvtepi16_epi32(l),
                        _mm_cvtepi16_epi32(_mm_srli_si128(l, 8)),
                        _mm_cvtepi16_epi32(h),
                        _mm_cvtepi16_epi32(_mm_srli_si128(h, 8)),
                    ];
                    let sa = _mm_set1_ps(xs[k].s[b]);
                    for i in 0..4 {
                        let ws = _mm_mul_ps(_mm_loadu_ps(sp.add(i * 4)), sa);
                        facc[k][i] = _mm_add_ps(facc[k][i], _mm_mul_ps(_mm_cvtepi32_ps(parts[i]), ws));
                    }
                }
            }
            for k in 0..K {
                for i in 0..4 {
                    _mm_storeu_ps(out.as_mut_ptr().add(k * m.rows + t * TILE + i * 4), facc[k][i]);
                }
            }
        }
    }

    // ---- ternary add-only kernel
    // weights -1/0/+1: _mm_sign_epi8(a, w) gives a, 0 or -a (no multiplication),
    // then bytes are summed. One scale per row, one per token.
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn tern_rows<const K: usize>(m: &QMat, xs: &[QVec], out: &mut [f32]) {
        let map = _mm_setr_epi8(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let m3 = _mm_set1_epi8(3);
        let ones8 = _mm_set1_epi8(1);
        let ones16 = _mm_set1_epi16(1);
        let nchunks = m.cols / 64;
        for r in 0..m.rows {
            let wr = m.data.as_ptr().add(r * nchunks * 16);
            let mut acc32 = [_mm_setzero_si128(); K];
            let mut acc16 = [_mm_setzero_si128(); K];
            for c in 0..nchunks {
                let v = _mm_loadu_si128(wr.add(c * 16) as *const __m128i);
                for s in 0..4 {
                    let code = _mm_and_si128(_mm_srl_epi16(v, _mm_cvtsi32_si128(2 * s as i32)), m3);
                    let wv = _mm_shuffle_epi8(map, code);
                    for k in 0..K {
                        let a = _mm_loadu_si128(xs[k].q.as_ptr().add(c * 64 + s * 16) as *const __m128i);
                        acc16[k] = _mm_add_epi16(acc16[k], _mm_maddubs_epi16(ones8, _mm_sign_epi8(a, wv)));
                    }
                }
                if c % 16 == 15 {
                    // flush int16 sums before they can overflow
                    for k in 0..K {
                        acc32[k] = _mm_add_epi32(acc32[k], _mm_madd_epi16(acc16[k], ones16));
                        acc16[k] = _mm_setzero_si128();
                    }
                }
            }
            for k in 0..K {
                let tot = _mm_add_epi32(acc32[k], _mm_madd_epi16(acc16[k], ones16));
                out[k * m.rows + r] = hsum_i32(tot) as f32 * m.scales[r] * xs[k].s[0];
            }
        }
    }

    macro_rules! by_k {
        ($k:expr, $f:ident, $($args:expr),*) => {
            match $k {
                1 => $f::<1>($($args),*),
                2 => $f::<2>($($args),*),
                4 => $f::<4>($($args),*),
                8 => $f::<8>($($args),*),
                n => panic!("batch {n} not compiled in"),
            }
        };
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn lut4<const K: usize>(m: &QMat, t: &[LutTabs], xs: &[QVec], o: &mut [f32]) {
        lut_rows::<K, 4>(m, t, xs, o)
    }
    #[target_feature(enable = "ssse3,sse4.1")]
    unsafe fn lut2<const K: usize>(m: &QMat, t: &[LutTabs], xs: &[QVec], o: &mut [f32]) {
        lut_rows::<K, 2>(m, t, xs, o)
    }

    pub fn rows(m: &QMat, xs: &[QVec], out: &mut [f32]) {
        // Safety: SSSE3/SSE4.1 verified at startup; shapes checked by caller.
        unsafe {
            match m.fmt {
                Fmt::Q8 | Fmt::Q4 => by_k!(xs.len(), mul_rows, m, xs, out),
                Fmt::Lut4 | Fmt::Lut2 => {
                    let tabs: Vec<LutTabs> = xs.iter().map(build_tabs).collect();
                    if m.fmt == Fmt::Lut4 {
                        by_k!(xs.len(), lut4, m, &tabs, xs, out)
                    } else {
                        by_k!(xs.len(), lut2, m, &tabs, xs, out)
                    }
                }
                Fmt::Tern => by_k!(xs.len(), tern_rows, m, xs, out),
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse,
}

fn matvec(m: &QMat, xs: &[QVec], out: &mut [f32], kern: Kernel) {
    assert!(out.len() >= xs.len() * m.rows);
    for x in xs {
        assert_eq!(x.q.len(), m.cols);
    }
    match kern {
        Kernel::Scalar => rows_scalar(m, xs, out),
        #[cfg(target_arch = "x86_64")]
        Kernel::Sse => sse::rows(m, xs, out),
    }
}

// ---------------------------------------------------------------- expert FFN

struct Expert {
    gate: QMat,
    up: QMat,
    down: QMat,
}

impl Expert {
    fn random(fmt: Fmt, rng: &mut Rng, keep: bool) -> Self {
        Expert {
            gate: QMat::random(FFN, HIDDEN, fmt, rng, keep),
            up: QMat::random(FFN, HIDDEN, fmt, rng, keep),
            down: QMat::random(HIDDEN, FFN, fmt, rng, keep),
        }
    }
    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

struct Scratch {
    g: Vec<f32>,
    u: Vec<f32>,
    hf: Vec<f32>,
    h: Vec<QVec>,
    o: Vec<f32>,
}

impl Scratch {
    fn new(k: usize) -> Self {
        Scratch {
            g: vec![0.0; k * FFN],
            u: vec![0.0; k * FFN],
            hf: vec![0.0; FFN],
            h: (0..k).map(|_| QVec::zeros(FFN)).collect(),
            o: vec![0.0; k * HIDDEN],
        }
    }
}

/// out = down( silu(gate x) * (up x) ) for each of the K token vectors in `xs`.
fn expert_ffn(e: &Expert, xs: &[QVec], sc: &mut Scratch, kern: Kernel) {
    let k = xs.len();
    let act = e.gate.fmt.act();
    matvec(&e.gate, xs, &mut sc.g, kern);
    matvec(&e.up, xs, &mut sc.u, kern);
    for t in 0..k {
        for i in 0..FFN {
            let g = sc.g[t * FFN + i];
            sc.hf[i] = g / (1.0 + (-g).exp()) * sc.u[t * FFN + i];
        }
        quantize(&sc.hf, act, &mut sc.h[t]);
    }
    matvec(&e.down, &sc.h[..k], &mut sc.o, kern);
}

fn random_acts(k: usize, n: usize, act: Act, rng: &mut Rng) -> Vec<QVec> {
    (0..k)
        .map(|_| {
            let x: Vec<f32> = (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect();
            let mut q = QVec::zeros(n);
            quantize(&x, act, &mut q);
            q
        })
        .collect()
}

// ---------------------------------------------------------------- correctness

fn check(fmt: Fmt, kern: Kernel) -> bool {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let e = Expert::random(fmt, &mut rng, true);
    let mut ok = true;
    let mut worst = 0f32;
    for k in [1usize, 2, 4, 8] {
        let xs = random_acts(k, HIDDEN, fmt.act(), &mut rng);
        let mut a = Scratch::new(k);
        let mut b = Scratch::new(k);
        expert_ffn(&e, &xs, &mut a, Kernel::Scalar);
        expert_ffn(&e, &xs, &mut b, kern);
        let scale = a.o.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let err = a.o.iter().zip(&b.o).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
        worst = worst.max(err);
        ok &= err < 1e-3;
    }
    println!(
        "  check vs scalar reference (batch 1,2,4,8): max error {:.1e} {}",
        worst,
        if ok { "OK" } else { "FAIL" }
    );
    ok
}

// ---------------------------------------------------------------- benchmark

fn bench(pool: &Arc<Vec<Expert>>, threads: usize, k: usize, secs: f64, kern: Kernel) -> (f64, f64) {
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(threads + 1));
    let expert_bytes = pool[0].bytes() as f64;
    let act = pool[0].gate.fmt.act();
    let (n, dt) = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let pool = Arc::clone(pool);
                let stop = Arc::clone(&stop);
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let mut rng = Rng(0xdead_beef ^ (tid as u64 + 1) * 0x9E37_79B9);
                    let xs = random_acts(k, HIDDEN, act, &mut rng);
                    let mut sc = Scratch::new(k);
                    let mine: Vec<usize> = (tid..pool.len()).step_by(threads).collect();
                    barrier.wait();
                    let mut n = 0u64;
                    'outer: loop {
                        for &i in &mine {
                            expert_ffn(&pool[i], &xs, &mut sc, kern);
                            n += 1;
                            if stop.load(Ordering::Relaxed) {
                                break 'outer;
                            }
                        }
                    }
                    std::hint::black_box(&sc.o);
                    n
                })
            })
            .collect();
        barrier.wait();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        let n: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        (n, t0.elapsed().as_secs_f64())
    });
    (n as f64 * k as f64 * MACS_PER_EXPERT / dt / 1e9, n as f64 * expert_bytes / dt / 1e9)
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads: usize = arg("--threads", max_threads);
    let secs: f64 = arg("--secs", 3.0);
    let pool_mb: usize = arg("--pool-mb", 256);
    let active_gmac: f64 = arg("--active-gmac", 3.0);
    let fmts: Vec<Fmt> = arg("--formats", "q8,q4,lut4,lut2,tern".to_string())
        .split(',')
        .map(|s| Fmt::parse(s.trim()).unwrap_or_else(|| panic!("unknown format {s}")))
        .collect();

    #[allow(unused_mut)]
    let mut kern = Kernel::Scalar;
    #[cfg(target_arch = "x86_64")]
    {
        let ssse3 = is_x86_feature_detected!("ssse3");
        let sse41 = is_x86_feature_detected!("sse4.1");
        let avx2 = is_x86_feature_detected!("avx2");
        println!("CPU features: ssse3={ssse3} sse4.1={sse41} avx2={avx2} (avx2 is NOT used)");
        if ssse3 && sse41 {
            kern = Kernel::Sse;
        }
    }
    if kern == Kernel::Scalar {
        println!("NOTE: no SIMD kernels for this CPU; scalar-only numbers are not meaningful.");
    }
    println!("threads={threads} secs={secs} pool={pool_mb} MB, expert = 3 matrices 768x2048 / 2048x768\n");

    let mut all_ok = true;
    let mut summary = Vec::new();
    for &fmt in &fmts {
        println!("== {} ({}; {} bits/weight)", fmt.name(), fmt.how(), fmt.bits());
        if kern != Kernel::Scalar {
            all_ok &= check(fmt, kern);
        }
        let mut rng = Rng(42);
        let first = Expert::random(fmt, &mut rng, false);
        let per = first.bytes();
        let n = (pool_mb * 1_000_000 / per).max(2 * threads);
        let mut v = vec![first];
        while v.len() < n {
            v.push(Expert::random(fmt, &mut rng, false));
        }
        let pool = Arc::new(v);
        println!("  pool: {} experts x {:.2} MB = {:.0} MB (larger than CPU cache)", n, per as f64 / 1e6, (n * per) as f64 / 1e6);
        println!("  {:>7} {:>5} {:>8} {:>12} {:>15}", "threads", "batch", "GMAC/s", "weights GB/s", "ceiling tok/s");
        let mut tlist = vec![1];
        if threads > 1 {
            tlist.push(threads);
        }
        let mut row = (0.0, 0.0);
        for &t in &tlist {
            for k in [1usize, 2, 4] {
                let (g, gb) = bench(&pool, t, k, secs, kern);
                println!("  {:>7} {:>5} {:>8.2} {:>12.2} {:>15.2}", t, k, g, gb, g / active_gmac);
                if t == *tlist.last().unwrap() {
                    if k == 1 {
                        row.0 = g;
                    }
                    if k == 4 {
                        row.1 = g;
                    }
                }
            }
        }
        summary.push((fmt, row));
        println!();
    }

    println!("== summary ({} threads)", threads);
    let base = summary.iter().find(|(f, _)| *f == Fmt::Q4).map(|(_, r)| r.0);
    println!("  {:<6} {:<36} {:>5} {:>13} {:>13} {:>8}", "format", "math", "bits", "GMAC/s b=1", "GMAC/s b=4", "vs q4");
    for (fmt, (g1, g4)) in &summary {
        let rel = base.map(|b| format!("{:.2}x", g1 / b)).unwrap_or_default();
        println!("  {:<6} {:<36} {:>5} {:>13.2} {:>13.2} {:>8}", fmt.name(), fmt.how(), fmt.bits(), g1, g4, rel);
    }
    println!(
        "\nceiling tok/s = GMAC/s / {active_gmac} GMAC per token (all ~3B active params of a 30B-A3B model;\n\
         attention over the context not included). lut* use 6-bit activations and tern needs a\n\
         model trained with -1/0/+1 weights: their quality cost is measured separately (tracer).\n"
    );
    if !all_ok {
        println!("WARNING: a correctness check failed; do not trust the numbers above.");
        std::process::exit(1);
    }
}
