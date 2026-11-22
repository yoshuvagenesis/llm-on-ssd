"""Fake quantization: round weights / activations the way a low-bit engine would,
then keep computing in float32. Running the model this way and measuring
perplexity shows how much quality each "simpler math" format costs.

Weights: per block of 32 inputs, min/max rounding (like llama.cpp Q4_1-style
formats). This is plain round-to-nearest, a deliberately simple baseline:
better quantizers (k-quants with importance weighting, GPTQ, AWQ) lose less,
especially at 2-3 bits, so treat these numbers as a pessimistic bound.

"tern" = -1/0/+1 per block of 32 with an absmean scale (BitNet b1.58 style).
Converting an existing model this way is expected to break it badly; it is
included to show why ternary needs a model trained for it.
"""
import numpy as np

BLOCK = 32


def _blocks(w):
    n = w.shape[-1]
    b = BLOCK if n % BLOCK == 0 else n
    return w.reshape(-1, b)


def quant_weight(w, bits):
    """bits: 16 (no-op), 8, 6, 5, 4, 3, 2, or 'tern'."""
    if bits in (16, "16", None):
        return w
    x = _blocks(w)
    if bits == "tern":
        s = np.mean(np.abs(x), axis=1, keepdims=True)
        s[s == 0] = 1.0
        return (np.clip(np.round(x / s), -1, 1) * s).reshape(w.shape).astype(np.float32)
    levels = (1 << int(bits)) - 1
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    s = (mx - mn) / levels
    s[s == 0] = 1.0
    q = np.clip(np.round((x - mn) / s), 0, levels)
    return (q * s + mn).reshape(w.shape).astype(np.float32)


def quant_act(x, bits):
    """Symmetric per-32 activation rounding (what the kernels feed in). bits 0 = off."""
    if not bits:
        return x
    qmax = (1 << (int(bits) - 1)) - 1
    v = _blocks(x)
    s = np.max(np.abs(v), axis=1, keepdims=True) / qmax
    s[s == 0] = 1.0
    return (np.clip(np.round(v / s), -qmax, qmax) * s).reshape(x.shape).astype(np.float32)


def parse_bits(s):
    return "tern" if str(s) == "tern" else int(s)
