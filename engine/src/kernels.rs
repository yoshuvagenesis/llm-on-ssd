//! Matrix-vector kernels. SSE (SSSE3 + SSE4.1) on x86-64, plain Rust elsewhere.
//! Weights: q8 (int8) or q4 (4-bit) with one f32 scale per 32 weights.
//! Activations: int8 with one f32 scale per 32 values.

use std::sync::atomic::{AtomicBool, Ordering};

pub const QK: usize = 32;

static USE_SSE: AtomicBool = AtomicBool::new(false);
static USE_AVX2: AtomicBool = AtomicBool::new(false);

/// Detect CPU features once. Returns a short description for logging.
/// Prefers AVX2 (256-bit, every hot loop does one 32-byte QK block per instruction
/// instead of SSE's two 16-byte halves) and falls back to SSE, then scalar.
pub fn init() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            USE_AVX2.store(true, Ordering::Relaxed);
            return "AVX2";
        }
        if is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1") {
            USE_SSE.store(true, Ordering::Relaxed);
            return "SSE (SSSE3+SSE4.1)";
        }
    }
    "scalar (no SIMD kernels for this CPU yet)"
}

/// Per-block scales: float32, or bfloat16 (stored as raw u16 bits, 2 bytes per block).
#[derive(Clone, Copy)]
pub enum Scales<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

/// A quantized matrix: `data` = q8 bytes or packed q4 nibbles, scales = [rows][cols/32].
#[derive(Clone, Copy)]
pub struct QTensor<'a> {
    pub data: &'a [u8],
    pub scales: Scales<'a>,
    pub rows: usize,
    pub cols: usize,
    pub q4: bool,
}

impl<'a> QTensor<'a> {
    #[inline(always)]
    pub fn scale(&self, i: usize) -> f32 {
        match self.scales {
            Scales::F32(s) => s[i],
            Scales::Bf16(s) => f32::from_bits((s[i] as u32) << 16),
        }
    }
    pub fn row_bytes(&self) -> usize {
        if self.q4 {
            self.cols / 2
        } else {
            self.cols
        }
    }
}

/// int8 activation vector
pub struct QAct {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

impl QAct {
    pub fn new(n: usize) -> QAct {
        QAct { q: vec![0; n], s: vec![0.0; n / QK] }
    }
    pub fn set(&mut self, x: &[f32]) {
        quantize(x, &mut self.q, &mut self.s);
    }
}

pub fn quantize(x: &[f32], q: &mut [i8], s: &mut [f32]) {
    for (b, blk) in x.chunks_exact(QK).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        s[b] = d;
        for (j, v) in blk.iter().enumerate() {
            q[b * QK + j] = (v * id).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// out[i] = row (r0 + i) of W times x, for i in 0..out.len()
pub fn matvec(w: &QTensor, x: &QAct, r0: usize, out: &mut [f32]) {
    debug_assert_eq!(x.q.len(), w.cols);
    debug_assert!(r0 + out.len() <= w.rows);
    #[cfg(target_arch = "x86_64")]
    {
        if USE_AVX2.load(Ordering::Relaxed) {
            // Safety: AVX2 presence checked in init(); bounds checked above.
            unsafe { avx2::matvec(w, x, r0, out) };
            return;
        }
        if USE_SSE.load(Ordering::Relaxed) {
            // Safety: SSSE3/SSE4.1 presence checked in init(); bounds checked above.
            unsafe { sse::matvec(w, x, r0, out) };
            return;
        }
    }
    matvec_scalar(w, x, r0, out)
}

fn matvec_scalar(w: &QTensor, x: &QAct, r0: usize, out: &mut [f32]) {
    let nb = w.cols / QK;
    let rb = w.row_bytes();
    for (i, o) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let wr = &w.data[r * rb..(r + 1) * rb];
        let mut acc = 0f32;
        for b in 0..nb {
            let a = &x.q[b * QK..(b + 1) * QK];
            let mut s: i32 = 0;
            if w.q4 {
                let blk = &wr[b * 16..b * 16 + 16];
                for j in 0..16 {
                    s += ((blk[j] & 0x0F) as i32 - 8) * a[j] as i32 + ((blk[j] >> 4) as i32 - 8) * a[j + 16] as i32;
                }
            } else {
                let blk = &wr[b * QK..(b + 1) * QK];
                for j in 0..QK {
                    s += (blk[j] as i8 as i32) * a[j] as i32;
                }
            }
            acc += s as f32 * (w.scale(r * nb + b) * x.s[b]);
        }
        *o = acc;
    }
}

/// float32 -> bfloat16 bits, round to nearest even (KV cache storage)
#[inline]
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// dot(q, k) with k stored as bf16; n divisible by 8
pub fn dot_bf16(q: &[f32], k: &[u16]) -> f32 {
    debug_assert!(q.len() == k.len() && q.len() % 8 == 0);
    #[cfg(target_arch = "x86_64")]
    {
        if USE_AVX2.load(Ordering::Relaxed) {
            return unsafe { avx2::dot_bf16(q, k) };
        }
        if USE_SSE.load(Ordering::Relaxed) {
            return unsafe { sse::dot_bf16(q, k) };
        }
    }
    q.iter().zip(k).map(|(a, &b)| a * f32::from_bits((b as u32) << 16)).sum()
}

/// out += w * v, with v stored as bf16; n divisible by 8
pub fn axpy_bf16(w: f32, v: &[u16], out: &mut [f32]) {
    debug_assert!(v.len() == out.len() && v.len() % 8 == 0);
    #[cfg(target_arch = "x86_64")]
    {
        if USE_AVX2.load(Ordering::Relaxed) {
            unsafe { avx2::axpy_bf16(w, v, out) };
            return;
        }
        if USE_SSE.load(Ordering::Relaxed) {
            unsafe { sse::axpy_bf16(w, v, out) };
            return;
        }
    }
    for (o, &b) in out.iter_mut().zip(v) {
        *o += w * f32::from_bits((b as u32) << 16);
    }
}

/// float32 matrix [rows x cols] times x (used for the small router matrices)
pub fn matvec_f32(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if USE_AVX2.load(Ordering::Relaxed) && cols % 32 == 0 {
            unsafe { avx2::matvec_f32(w, cols, x, out) };
            return;
        }
        if USE_SSE.load(Ordering::Relaxed) && cols % 16 == 0 {
            unsafe { sse::matvec_f32(w, cols, x, out) };
            return;
        }
    }
    for (r, o) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = [0f32; 8];
        for (c, v) in row.chunks_exact(8).zip(x.chunks_exact(8)) {
            for k in 0..8 {
                acc[k] += c[k] * v[k];
            }
        }
        let mut s: f32 = acc.iter().sum();
        for k in (cols / 8) * 8..cols {
            s += row[k] * x[k];
        }
        *o = s;
    }
}

#[cfg(target_arch = "x86_64")]
mod sse {
    use super::{QAct, QTensor, Scales, QK};
    use std::arch::x86_64::*;

    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn hsum(v: __m128) -> f32 {
        let hi = _mm_movehl_ps(v, v);
        let s = _mm_add_ps(v, hi);
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55)))
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn block_dot(w0: __m128i, w1: __m128i, a: *const i8, ones: __m128i) -> __m128i {
        let a0 = _mm_loadu_si128(a as *const __m128i);
        let a1 = _mm_loadu_si128(a.add(16) as *const __m128i);
        let p0 = _mm_maddubs_epi16(_mm_sign_epi8(w0, w0), _mm_sign_epi8(a0, w0));
        let p1 = _mm_maddubs_epi16(_mm_sign_epi8(w1, w1), _mm_sign_epi8(a1, w1));
        _mm_add_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones))
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    pub unsafe fn dot_bf16(q: &[f32], k: &[u16]) -> f32 {
        let z = _mm_setzero_si128();
        let (mut a0, mut a1) = (_mm_setzero_ps(), _mm_setzero_ps());
        let (qp, kp) = (q.as_ptr(), k.as_ptr());
        let mut i = 0;
        while i < q.len() {
            let kv = _mm_loadu_si128(kp.add(i) as *const __m128i);
            let lo = _mm_castsi128_ps(_mm_unpacklo_epi16(z, kv));
            let hi = _mm_castsi128_ps(_mm_unpackhi_epi16(z, kv));
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(qp.add(i)), lo));
            a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(qp.add(i + 4)), hi));
            i += 8;
        }
        hsum(_mm_add_ps(a0, a1))
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    pub unsafe fn axpy_bf16(w: f32, v: &[u16], out: &mut [f32]) {
        let z = _mm_setzero_si128();
        let wv = _mm_set1_ps(w);
        let (vp, op) = (v.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i < v.len() {
            let x = _mm_loadu_si128(vp.add(i) as *const __m128i);
            let lo = _mm_castsi128_ps(_mm_unpacklo_epi16(z, x));
            let hi = _mm_castsi128_ps(_mm_unpackhi_epi16(z, x));
            _mm_storeu_ps(op.add(i), _mm_add_ps(_mm_loadu_ps(op.add(i)), _mm_mul_ps(wv, lo)));
            _mm_storeu_ps(op.add(i + 4), _mm_add_ps(_mm_loadu_ps(op.add(i + 4)), _mm_mul_ps(wv, hi)));
            i += 8;
        }
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    pub unsafe fn matvec_f32(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
        let xp = x.as_ptr();
        for (r, o) in out.iter_mut().enumerate() {
            let wp = w.as_ptr().add(r * cols);
            let mut a = [_mm_setzero_ps(); 4];
            let mut c = 0;
            while c < cols {
                for k in 0..4 {
                    a[k] = _mm_add_ps(a[k], _mm_mul_ps(_mm_loadu_ps(wp.add(c + 4 * k)), _mm_loadu_ps(xp.add(c + 4 * k))));
                }
                c += 16;
            }
            *o = hsum(_mm_add_ps(_mm_add_ps(a[0], a[1]), _mm_add_ps(a[2], a[3])));
        }
    }

    /// Dispatch once per call on the scale format, so the inner loop has no branches
    /// or bounds checks.
    #[target_feature(enable = "ssse3,sse4.1")]
    pub unsafe fn matvec(w: &QTensor, x: &QAct, r0: usize, out: &mut [f32]) {
        match (w.q4, w.scales) {
            (true, Scales::Bf16(s)) => rows::<true, true>(w, s.as_ptr() as *const u8, x, r0, out),
            (true, Scales::F32(s)) => rows::<true, false>(w, s.as_ptr() as *const u8, x, r0, out),
            (false, Scales::Bf16(s)) => rows::<false, true>(w, s.as_ptr() as *const u8, x, r0, out),
            (false, Scales::F32(s)) => rows::<false, false>(w, s.as_ptr() as *const u8, x, r0, out),
        }
    }

    #[target_feature(enable = "ssse3,sse4.1")]
    #[inline]
    unsafe fn rows<const Q4: bool, const BF16: bool>(w: &QTensor, sp: *const u8, x: &QAct, r0: usize, out: &mut [f32]) {
        let nb = w.cols / QK;
        let rb = if Q4 { w.cols / 2 } else { w.cols };
        let ones = _mm_set1_epi16(1);
        let m4 = _mm_set1_epi8(0x0F);
        let off = _mm_set1_epi8(8);
        let xq = x.q.as_ptr();
        let xs = x.s.as_ptr();
        for (i, o) in out.iter_mut().enumerate() {
            let r = r0 + i;
            let wr = w.data.as_ptr().add(r * rb);
            let mut acc = _mm_setzero_ps();
            for b in 0..nb {
                let (w0, w1) = if Q4 {
                    let raw = _mm_loadu_si128(wr.add(b * 16) as *const __m128i);
                    (
                        _mm_sub_epi8(_mm_and_si128(raw, m4), off),
                        _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), off),
                    )
                } else {
                    (
                        _mm_loadu_si128(wr.add(b * QK) as *const __m128i),
                        _mm_loadu_si128(wr.add(b * QK + 16) as *const __m128i),
                    )
                };
                let s = block_dot(w0, w1, xq.add(b * QK), ones);
                let ws = if BF16 {
                    f32::from_bits((*(sp as *const u16).add(r * nb + b) as u32) << 16)
                } else {
                    *(sp as *const f32).add(r * nb + b)
                };
                let sc = _mm_set1_ps(ws * *xs.add(b));
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), sc));
            }
            *o = hsum(acc);
        }
    }
}

/// AVX2 (256-bit): the same kernels as `sse`, but one 32-byte QK block fits a single
/// register instead of SSE's two 16-byte halves, so there's no "w0, w1, add the two
/// partial sums" step -- one register, one block, fewer instructions per block.
/// Not bit-identical to the SSE path (summation order differs, like SSE already isn't
/// bit-identical to the scalar fallback); still a plain, correct float computation.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{QAct, QTensor, Scales, QK};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55)))
    }

    /// dot product of one 32-byte weight block against the matching 32-byte activation
    /// block, as 8 x i32 partial sums (same two-step maddubs/madd reduction as SSE's
    /// block_dot, just over the whole block in one register instead of two halves).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn block_dot(w: __m256i, a: *const i8, ones: __m256i) -> __m256i {
        let av = _mm256_loadu_si256(a as *const __m256i);
        let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(av, w));
        _mm256_madd_epi16(p, ones)
    }

    /// Widen 8 bf16 values (as raw u16 bits) to 8 f32, lane-crossing-free: a 128-bit
    /// zero-extend to 32-bit lanes, then shift into the top 16 bits where bf16 lives.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bf16x8_to_f32(p: *const u16) -> __m256 {
        let widened = _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i));
        _mm256_castsi256_ps(_mm256_slli_epi32(widened, 16))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_bf16(q: &[f32], k: &[u16]) -> f32 {
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (qp, kp) = (q.as_ptr(), k.as_ptr());
        let mut i = 0;
        while i + 16 <= q.len() {
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(qp.add(i)), bf16x8_to_f32(kp.add(i))));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(qp.add(i + 8)), bf16x8_to_f32(kp.add(i + 8))));
            i += 16;
        }
        let mut acc = hsum256_ps(_mm256_add_ps(a0, a1));
        while i < q.len() {
            acc += q[i] * f32::from_bits((k[i] as u32) << 16);
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn axpy_bf16(w: f32, v: &[u16], out: &mut [f32]) {
        let wv = _mm256_set1_ps(w);
        let (vp, op) = (v.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i + 8 <= v.len() {
            _mm256_storeu_ps(op.add(i), _mm256_add_ps(_mm256_loadu_ps(op.add(i)), _mm256_mul_ps(wv, bf16x8_to_f32(vp.add(i)))));
            i += 8;
        }
        while i < v.len() {
            out[i] += w * f32::from_bits((v[i] as u32) << 16);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn matvec_f32(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
        let xp = x.as_ptr();
        for (r, o) in out.iter_mut().enumerate() {
            let wp = w.as_ptr().add(r * cols);
            let mut a = [_mm256_setzero_ps(); 4];
            let mut c = 0;
            while c < cols {
                for k in 0..4 {
                    a[k] = _mm256_add_ps(a[k], _mm256_mul_ps(_mm256_loadu_ps(wp.add(c + 8 * k)), _mm256_loadu_ps(xp.add(c + 8 * k))));
                }
                c += 32;
            }
            *o = hsum256_ps(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
        }
    }

    /// Dispatch once per call on the scale format, so the inner loop has no branches
    /// or bounds checks.
    #[target_feature(enable = "avx2")]
    pub unsafe fn matvec(w: &QTensor, x: &QAct, r0: usize, out: &mut [f32]) {
        match (w.q4, w.scales) {
            (true, Scales::Bf16(s)) => rows::<true, true>(w, s.as_ptr() as *const u8, x, r0, out),
            (true, Scales::F32(s)) => rows::<true, false>(w, s.as_ptr() as *const u8, x, r0, out),
            (false, Scales::Bf16(s)) => rows::<false, true>(w, s.as_ptr() as *const u8, x, r0, out),
            (false, Scales::F32(s)) => rows::<false, false>(w, s.as_ptr() as *const u8, x, r0, out),
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rows<const Q4: bool, const BF16: bool>(w: &QTensor, sp: *const u8, x: &QAct, r0: usize, out: &mut [f32]) {
        let nb = w.cols / QK;
        let rb = if Q4 { w.cols / 2 } else { w.cols };
        let ones = _mm256_set1_epi16(1);
        let m4 = _mm_set1_epi8(0x0F);
        let off = _mm_set1_epi8(8);
        let xq = x.q.as_ptr();
        let xs = x.s.as_ptr();
        for (i, o) in out.iter_mut().enumerate() {
            let r = r0 + i;
            let wr = w.data.as_ptr().add(r * rb);
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                // One 32-byte block = all of QK in one register (q8: loaded directly;
                // q4: the 16 packed nibble-bytes unpack into two 16-wide halves that
                // together cover the block, combined into one 256-bit register).
                let wv = if Q4 {
                    let raw = _mm_loadu_si128(wr.add(b * 16) as *const __m128i);
                    let lo = _mm_sub_epi8(_mm_and_si128(raw, m4), off);
                    let hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), off);
                    _mm256_set_m128i(hi, lo)
                } else {
                    _mm256_loadu_si256(wr.add(b * QK) as *const __m256i)
                };
                let s = block_dot(wv, xq.add(b * QK), ones);
                let ws = if BF16 {
                    f32::from_bits((*(sp as *const u16).add(r * nb + b) as u32) << 16)
                } else {
                    *(sp as *const f32).add(r * nb + b)
                };
                let sc = _mm256_set1_ps(ws * *xs.add(b));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(s), sc));
            }
            *o = hsum256_ps(acc);
        }
    }
}
