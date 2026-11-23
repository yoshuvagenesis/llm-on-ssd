"""Write a .moe pack with REAL model dimensions but random weights, for speed tests.
No model download needed, so it can be generated directly on the machine under test.

  python make_bench_pack.py --out bench.moe                # Qwen3-30B-A3B shape, ~20 GB
  python make_bench_pack.py --out bench.moe --layers 12    # smaller, same per-layer cost
  python make_bench_pack.py --out bench.moe --arch qwen35  # Qwen3.6-35B-A3B shape, ~12 GB

Output text is meaningless and routing is more random than a real model's (so the
cache hit rate is pessimistic), but compute and SSD work per token match the real thing.
"""
import argparse
import struct
import time

import numpy as np

import packfmt as pf

SHAPE = dict(hidden_size=2048, num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
             num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768, rms_norm_eps=1e-6,
             rope_theta=1e6, vocab_size=151936, norm_topk_prob=1, eos="-1")
# Qwen3.6-35B-A3B (text part)
SHAPE35 = dict(hidden_size=2048, num_hidden_layers=40, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
               num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512, rms_norm_eps=1e-6,
               rope_theta=1e7, vocab_size=248320, norm_topk_prob=1, eos="-1", arch="qwen35",
               layer_types="LLLA" * 10, rotary_dim=64, lin_k_heads=16, lin_v_heads=32, lin_k_dim=128, lin_v_dim=128,
               conv_kernel=4, shared_ffn=512, think_open=1)


def rand_q(rng, fmt, rows, cols, std=0.02):
    nb = cols // pf.QK
    if fmt in ("q4", "q4b"):
        data = rng.integers(0, 256, rows * cols // 2, dtype=np.uint8).tobytes()
        sc = np.full(rows * nb, std / 4.6, np.float32)
        if fmt == "q4b":
            return data + pf.to_bf16_bits(sc).tobytes()
    else:
        data = rng.integers(-127, 128, rows * cols, dtype=np.int8).tobytes()
        sc = np.full(rows * nb, std / 73.0, np.float32)
    return data + sc.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--expert-fmt", default="q4b", choices=["q4", "q4b"])
    ap.add_argument("--dense-fmt", default="q8", choices=["q8", "q4b"])
    ap.add_argument("--arch", default="qwen3", choices=["qwen3", "qwen35"])
    args = ap.parse_args()
    if args.arch == "qwen35":
        layers = args.layers if args.layers != 48 else 40
        c = dict(SHAPE35, num_hidden_layers=layers, layer_types=SHAPE35["layer_types"][:layers], expert_fmt=args.expert_fmt)
    else:
        c = dict(SHAPE, num_hidden_layers=args.layers, expert_fmt=args.expert_fmt)
    ef = args.expert_fmt
    H, L, E, F, V = c["hidden_size"], c["num_hidden_layers"], c["num_experts"], c["moe_intermediate_size"], c["vocab_size"]
    nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    df = args.dense_fmt
    items = [("embed", "bf16", V, H), ("norm", "f32", 1, H), ("lm_head", df, V, H)]
    q35 = args.arch == "qwen35"
    for l in range(L):
        items += [(f"L{l}.ln1", "f32", 1, H), (f"L{l}.ln2", "f32", 1, H), (f"L{l}.router", "f32", E, H)]
        if not q35 or c["layer_types"][l] == "A":
            qrows = (2 if q35 else 1) * nh * hd
            items += [(f"L{l}.qn", "f32", 1, hd), (f"L{l}.kn", "f32", 1, hd), (f"L{l}.wq", df, qrows, H),
                      (f"L{l}.wk", df, nkv * hd, H), (f"L{l}.wv", df, nkv * hd, H), (f"L{l}.wo", df, H, nh * hd)]
        else:
            hk, hv, dk, dv, kc = (c["lin_k_heads"], c["lin_v_heads"], c["lin_k_dim"], c["lin_v_dim"], c["conv_kernel"])
            cd = 2 * hk * dk + hv * dv
            items += [(f"L{l}.lqkv", df, cd, H), (f"L{l}.lz", df, hv * dv, H), (f"L{l}.lba", "f32", 2 * hv, H),
                      (f"L{l}.conv", "f32", cd, kc), (f"L{l}.alog", "f32", 1, hv), (f"L{l}.dtb", "f32", 1, hv),
                      (f"L{l}.lnorm", "f32", 1, dv), (f"L{l}.lout", df, H, hv * dv)]
        if q35:
            fs = c["shared_ffn"]
            items += [(f"L{l}.sg", df, fs, H), (f"L{l}.su", df, fs, H), (f"L{l}.sd", df, H, fs),
                      (f"L{l}.sgate", "f32", 1, H)]
    lines = [f"cfg {k} {v}" for k, v in c.items()]
    off, placed = 0, []
    for name, fmt, rows, cols in items:
        n = pf.nbytes(fmt, rows, cols)
        lines.append(f"t {name} {fmt} {rows} {cols} {off} {n}")
        placed.append((off, name, fmt, rows, cols))
        off = pf.round_up(off + n)
    eb = pf.expert_nbytes(H, F, ef)
    e0 = off
    for l in range(L):
        for e in range(E):
            lines.append(f"e {l} {e} {off} {eb}")
            off += eb
    vblob = b"".join(struct.pack("<I", len(t)) + t for t in (f"<{i}>".encode() for i in range(V)))
    lines.append(f"vocab {off} {len(vblob)}")
    header = ("\n".join(lines) + "\n").encode()
    ds = pf.round_up(24 + len(header))
    print(f"writing {(off + len(vblob)) / 1e9:.1f} GB ({L} layers x {E} experts x {eb / 1e6:.2f} MB)")
    rng = np.random.default_rng(0)
    t0 = time.time()
    with open(args.out, "wb") as f:
        f.write(pf.MAGIC + struct.pack("<QQ", len(header), ds) + header)
        for o, name, fmt, rows, cols in placed:
            if fmt == "f32":
                if name.endswith(("ln1", "ln2", "qn", "kn", "norm", "lnorm", "dtb")):
                    w = np.ones((rows, cols), np.float32)
                elif name.endswith("alog"):
                    w = np.log(rng.uniform(1.0, 16.0, (rows, cols))).astype(np.float32)
                elif name.endswith("conv"):
                    w = (rng.standard_normal((rows, cols)) * 0.3).astype(np.float32)
                else:
                    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
                data = w.tobytes()
            elif fmt == "bf16":
                # in chunks of rows: the embedding table is large
                f.seek(ds + o)
                for r0 in range(0, rows, 8192):
                    n = min(8192, rows - r0)
                    f.write(pf.encode("bf16", (rng.standard_normal((n, cols)) * 0.02).astype(np.float32)))
                continue
            else:
                data = rand_q(rng, fmt, rows, cols)
            f.seek(ds + o)
            f.write(data)
        f.seek(ds + e0)
        pad = b"\0" * (eb - (2 * pf.nbytes(ef, F, H) + pf.nbytes(ef, H, F)))
        for l in range(L):
            for e in range(E):
                f.write(rand_q(rng, ef, F, H) + rand_q(rng, ef, F, H) + rand_q(rng, ef, H, F) + pad)
            print(f"\rlayer {l + 1}/{L}", end="", flush=True)
        f.write(vblob)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
