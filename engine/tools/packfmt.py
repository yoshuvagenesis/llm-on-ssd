"""The .moe pack format shared by convert.py (writer), ref_pack.py (reader) and the
Rust engine (src/pack.rs). Keep the three in sync.

File layout
  [0..8)    magic b"MOEPACK1"
  [8..16)   u64 header length (bytes)
  [16..24)  u64 data start (multiple of 4096)
  [24..)    header: UTF-8 text, one record per line
  [data start..) data; every offset below is relative to data start

Header records
  cfg <key> <value>
  t <name> <fmt> <rows> <cols> <offset> <nbytes>      fmt: f32 | bf16 | q8 | q4 | q4b
  e <layer> <expert> <offset> <nbytes>                one expert, 4096-aligned
  vocab <offset> <nbytes>

Tensor formats (row-major, `cols` divisible by 32)
  f32  : rows*cols float32
  bf16 : rows*cols bfloat16 (raw bits)
  q8   : int8 values [rows*cols], then float32 scales [rows*(cols/32)]
         value = q * scale
  q4   : 4-bit values [rows*cols/2], then float32 scales [rows*(cols/32)]
         per row, per 32-block: byte j (0..16) = (q[j] + 8) | (q[j+16] + 8) << 4
         value = q * scale, q in -8..7
  q4b  : like q4, but the scales are bfloat16 (2 bytes): 4.5 bits per weight
Expert block: gate [F x H], up [F x H], down [H x F] in the format given by
  `cfg expert_fmt` (q4 if absent, or q4b), back to back, padded to a multiple of 4096.
Vocab blob: for each token id: u32 byte length, then the token's raw bytes.
"""
import struct

import numpy as np

MAGIC = b"MOEPACK1"
ALIGN = 4096
QK = 32


def round_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def nbytes(fmt, rows, cols):
    nb = cols // QK
    return {
        "f32": rows * cols * 4,
        "bf16": rows * cols * 2,
        "q8": rows * cols + rows * nb * 4,
        "q4": rows * cols // 2 + rows * nb * 4,
        "q4b": rows * cols // 2 + rows * nb * 2,
    }[fmt]


def expert_nbytes(H, F, fmt="q4"):
    return round_up(2 * nbytes(fmt, F, H) + nbytes(fmt, H, F))


def to_bf16_bits(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- quantizers

def quant_q8(w):
    rows, cols = w.shape
    b = w.reshape(rows, cols // QK, QK).astype(np.float32)
    amax = np.abs(b).max(-1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0).astype(np.float32)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int8)
    return q.reshape(rows, cols), d


def quant_q4(w, bf16_scales=False):
    """Q4_0-style: the value with the largest magnitude maps to -8, so the full range is used.
    With bf16_scales the scale is rounded to bfloat16 first and the values fit to it."""
    rows, cols = w.shape
    b = w.reshape(rows, cols // QK, QK).astype(np.float32)
    idx = np.abs(b).argmax(-1)
    mx = np.take_along_axis(b, idx[..., None], -1)[..., 0]
    d = (mx / -8.0).astype(np.float32)
    if bf16_scales:
        d = bf16_to_f32(to_bf16_bits(d))
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0).astype(np.float32)
    q = np.clip(np.rint(b * inv[..., None]), -8, 7).astype(np.int8)
    return q.reshape(rows, cols), d


def pack_q4(q):
    rows, cols = q.shape
    u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // QK, QK)
    packed = (u[..., :16] & 0x0F) | (u[..., 16:] << 4)
    return packed.reshape(rows, cols // 2)


def unpack_q4(packed, rows, cols):
    p = packed.reshape(rows, cols // QK, 16)
    lo = (p & 0x0F).astype(np.int8) - 8
    hi = (p >> 4).astype(np.int8) - 8
    return np.concatenate([lo, hi], axis=-1).reshape(rows, cols)


def encode(fmt, w):
    """float32 [rows, cols] -> bytes in pack format."""
    if fmt == "f32":
        return np.ascontiguousarray(w, dtype=np.float32).tobytes()
    if fmt == "bf16":
        return to_bf16_bits(w).tobytes()
    if fmt == "q8":
        q, d = quant_q8(w)
        return q.tobytes() + d.astype(np.float32).tobytes()
    if fmt == "q4":
        q, d = quant_q4(w)
        return pack_q4(q).tobytes() + d.astype(np.float32).tobytes()
    if fmt == "q4b":
        q, d = quant_q4(w, bf16_scales=True)
        return pack_q4(q).tobytes() + to_bf16_bits(d).tobytes()
    raise ValueError(fmt)


def encode_large(fmt, w, rows_per=16384):
    """encode() in chunks of rows (same bytes), so huge matrices such as a 248k-row output
    head do not need several full-size temporary copies in RAM."""
    rows = w.shape[0]
    if fmt in ("f32", "bf16") or rows <= rows_per:
        return encode(fmt, w)
    vals, scales = [], []
    for r0 in range(0, rows, rows_per):
        chunk = w[r0:r0 + rows_per]
        part = encode(fmt, chunk)
        nv = chunk.shape[0] * chunk.shape[1] // (1 if fmt == "q8" else 2)  # values come before scales
        vals.append(part[:nv])
        scales.append(part[nv:])
    return b"".join(vals) + b"".join(scales)


def decode(fmt, buf, rows, cols):
    """bytes -> dequantized float32 [rows, cols] (used by the reference)."""
    nb = cols // QK
    if fmt == "f32":
        return np.frombuffer(buf, np.float32, rows * cols).reshape(rows, cols).copy()
    if fmt == "bf16":
        u = np.frombuffer(buf, np.uint16, rows * cols).astype(np.uint32) << 16
        return u.view(np.float32).reshape(rows, cols)
    if fmt == "q8":
        q = np.frombuffer(buf, np.int8, rows * cols).reshape(rows, cols).astype(np.float32)
        d = np.frombuffer(buf, np.float32, rows * nb, offset=rows * cols).reshape(rows, nb)
        return (q.reshape(rows, nb, QK) * d[..., None]).reshape(rows, cols)
    if fmt in ("q4", "q4b"):
        n = rows * cols // 2
        q = unpack_q4(np.frombuffer(buf, np.uint8, n), rows, cols).astype(np.float32)
        if fmt == "q4":
            d = np.frombuffer(buf, np.float32, rows * nb, offset=n).reshape(rows, nb)
        else:
            d = bf16_to_f32(np.frombuffer(buf, np.uint16, rows * nb, offset=n)).reshape(rows, nb)
        return (q.reshape(rows, nb, QK) * d[..., None]).reshape(rows, cols)
    raise ValueError(fmt)


# ---------------------------------------------------------------- reader

class Pack:
    def __init__(self, path):
        self.f = open(path, "rb")
        if self.f.read(8) != MAGIC:
            raise ValueError("not a .moe pack")
        hlen, self.data_start = struct.unpack("<QQ", self.f.read(16))
        self.cfg, self.tensors, self.experts, self.vocab = {}, {}, {}, None
        for line in self.f.read(hlen).decode().splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "cfg":
                self.cfg[p[1]] = p[2]
            elif p[0] == "t":
                self.tensors[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]), int(p[6]))
            elif p[0] == "e":
                self.experts[(int(p[1]), int(p[2]))] = (int(p[3]), int(p[4]))
            elif p[0] == "vocab":
                self.vocab = (int(p[1]), int(p[2]))

    def ci(self, k):
        return int(self.cfg[k])

    def cf(self, k):
        return float(self.cfg[k])

    def raw(self, off, n):
        self.f.seek(self.data_start + off)
        return self.f.read(n)

    def tensor(self, name):
        fmt, rows, cols, off, n = self.tensors[name]
        return decode(fmt, self.raw(off, n), rows, cols)

    def expert(self, l, e):
        H, F = self.ci("hidden_size"), self.ci("moe_intermediate_size")
        fmt = self.cfg.get("expert_fmt", "q4")
        off, _ = self.experts[(l, e)]
        a, b = nbytes(fmt, F, H), nbytes(fmt, H, F)
        buf = self.raw(off, 2 * a + b)
        return decode(fmt, buf[:a], F, H), decode(fmt, buf[a:2 * a], F, H), decode(fmt, buf[2 * a:], H, F)

    def tokens(self):
        off, n = self.vocab
        blob, out, i = self.raw(off, n), [], 0
        while i < len(blob):
            (ln,) = struct.unpack_from("<I", blob, i)
            out.append(blob[i + 4:i + 4 + ln])
            i += 4 + ln
        return out
