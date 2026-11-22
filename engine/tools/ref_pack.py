"""Reference forward pass over a .moe pack, written independently of the engine
(numpy, float64 matmuls) but with the same quantization of activations, so its
logits should match the engine's closely. Used by test_engine.py.

  python ref_pack.py model.moe ids.txt logits.bin   # compare with engine --dump-logits
"""
import sys

import numpy as np

import packfmt as pf

QK = 32


def qact(x):
    """int8 per-32 activation rounding, exactly like the engine's quantize()."""
    b = x.astype(np.float32).reshape(-1, QK)
    amax = np.abs(b).max(1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.clip(np.rint(b * inv[:, None]), -127, 127)
    return (q * d[:, None]).reshape(-1).astype(np.float64)


def bf16(x):
    """round float32 to bfloat16 (nearest even) and back, like the engine's KV cache"""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def rms(x, w, eps):
    x = x.astype(np.float32)
    ms = np.float32((x * x).sum() / np.float32(len(x)))
    return x * (np.float32(1.0) / np.sqrt(ms + np.float32(eps))) * w


def rope(x, pos, theta):
    hd = len(x)
    half = hd // 2
    i = np.arange(half)
    ang = pos * (1.0 / theta ** (2 * i / hd))
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[:half].copy(), x[half:].copy()
    return np.concatenate([a * c - b * s, b * c + a * s]).astype(np.float32)


class Ref:
    def __init__(self, path):
        self.p = p = pf.Pack(path)
        self.H, self.L, self.nh, self.nkv = p.ci("hidden_size"), p.ci("num_hidden_layers"), p.ci("num_attention_heads"), p.ci("num_key_value_heads")
        self.hd, self.E, self.K, self.F = p.ci("head_dim"), p.ci("num_experts"), p.ci("num_experts_per_tok"), p.ci("moe_intermediate_size")
        self.eps, self.theta, self.norm_topk = p.cf("rms_norm_eps"), p.cf("rope_theta"), p.ci("norm_topk_prob")
        self.W = {k: p.tensor(k) for k in p.tensors if k != "embed"}
        self.W = {k: (v.astype(np.float64) if k.split(".")[-1] in ("wq", "wk", "wv", "wo") or k == "lm_head" else v)
                  for k, v in self.W.items()}
        self.experts = {}
        self.kc = [[] for _ in range(self.L)]
        self.vc = [[] for _ in range(self.L)]
        self.pos = 0

    def expert(self, l, e):
        if (l, e) not in self.experts:
            self.experts[(l, e)] = [m.astype(np.float64) for m in self.p.expert(l, e)]
        return self.experts[(l, e)]

    def embed(self, t):
        fmt, rows, cols, off, n = self.p.tensors["embed"]
        return pf.decode("bf16", self.p.raw(off + t * cols * 2, cols * 2), 1, cols)[0].astype(np.float32)

    def forward(self, tok):
        W, hd, eps = self.W, self.hd, self.eps
        h = self.embed(tok)
        for l in range(self.L):
            g = f"L{l}."
            xn = rms(h, W[g + "ln1"][0], eps)
            xq = qact(xn)
            q = (W[g + "wq"] @ xq).astype(np.float32).reshape(self.nh, hd)
            k = (W[g + "wk"] @ xq).astype(np.float32).reshape(self.nkv, hd)
            v = (W[g + "wv"] @ xq).astype(np.float32).reshape(self.nkv, hd)
            q = np.stack([rope(rms(qh, W[g + "qn"][0], eps), self.pos, self.theta) for qh in q])
            k = np.stack([rope(rms(kh, W[g + "kn"][0], eps), self.pos, self.theta) for kh in k])
            self.kc[l].append(bf16(k))
            self.vc[l].append(bf16(v))
            K_, V_ = np.stack(self.kc[l]).astype(np.float64), np.stack(self.vc[l]).astype(np.float64)
            att = np.zeros((self.nh, hd))
            for i in range(self.nh):
                j = i // (self.nh // self.nkv)
                s = K_[:, j] @ q[i] / np.sqrt(hd)
                pr = np.exp(s - s.max())
                att[i] = (pr / pr.sum()) @ V_[:, j]
            h = h + (W[g + "wo"] @ qact(att.reshape(-1))).astype(np.float32)
            xn = rms(h, W[g + "ln2"][0], eps)
            r = W[g + "router"].astype(np.float64) @ xn.astype(np.float64)
            pr = np.exp(r - r.max())
            pr /= pr.sum()
            top = sorted(range(self.E), key=lambda e: (-pr[e], e))[: self.K]
            w = pr[top] / pr[top].sum() if self.norm_topk else pr[top]
            xq = qact(xn)
            moe = np.zeros(self.H)
            for e, we in zip(top, w):
                wg, wu, wd = self.expert(l, e)
                gg, uu = wg @ xq, wu @ xq
                hm = (gg / (1 + np.exp(-gg))) * uu
                moe += we * (wd @ qact(hm))
            h = h + moe.astype(np.float32)
        xn = rms(h, W["norm"][0], eps)
        self.pos += 1
        return W["lm_head"] @ qact(xn)


def main():
    pack, ids_path, out = sys.argv[1:4]
    ids = [int(x) for x in open(ids_path).read().replace(",", " ").split()]
    ref = Ref(pack)
    with open(out, "wb") as f:
        for t in ids[:-1]:
            f.write(ref.forward(t).astype(np.float32).tobytes())


if __name__ == "__main__":
    main()
