"""Convert a Qwen3-MoE safetensors checkpoint into a .moe pack for the engine.

  python convert.py --model ~/models/Qwen3-30B-A3B --out ~/qwen3-30b-a3b.moe

Experts: 4-bit with bf16 scales (q4b, 4.5 bits/weight), one 4096-aligned block per expert, so the engine reads each
expert with a single SSD request. Attention, output head: 8-bit (q8). Embeddings:
bf16, read one row per token straight from the SSD. Norms and routers: float32.
numpy only; reads the checkpoint one tensor at a time (peak RAM ~3 GB).
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trace"))
from st_reader import SafeTensors  # noqa: E402

import packfmt as pf  # noqa: E402


def bytes_to_unicode():
    """GPT-2 / Qwen byte-level BPE alphabet: byte -> printable unicode char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def build_vocab(model_dir, vocab_size):
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        print("no tokenizer.json: writing placeholder token names")
        return [f"<{i}>".encode() for i in range(vocab_size)]
    tj = json.load(open(path, encoding="utf-8"))
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = [b""] * vocab_size
    for s, i in tj["model"]["vocab"].items():
        if i < vocab_size:
            out[i] = bytes(inv[c] for c in s) if all(c in inv for c in s) else s.encode()
    for t in tj.get("added_tokens", []):
        if t["id"] < vocab_size:
            out[t["id"]] = t["content"].encode()
    return out


def eos_ids(model_dir, cfg):
    ids = None
    gp = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gp):
        ids = json.load(open(gp)).get("eos_token_id")
    if ids is None:
        ids = cfg.get("eos_token_id", [])
    return [ids] if isinstance(ids, int) else list(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert-fmt", default="q4b", choices=["q4", "q4b"],
                    help="q4b = 4-bit with bf16 scales (4.5 bits/weight, default); q4 = f32 scales (5 bits)")
    ap.add_argument("--dense-fmt", default="q8", choices=["q8", "q4b"],
                    help="attention + output head: q8 (default, best quality) or q4b (smaller and faster, "
                         "measured ~+2.7%% perplexity)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    if cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1 or cfg.get("shared_expert_intermediate_size"):
        raise SystemExit("only all-MoE Qwen3 models without shared experts are supported")
    st = SafeTensors(args.model)
    H, L, E = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_experts"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", H // nh)
    F, V = cfg["moe_intermediate_size"], cfg["vocab_size"]
    for n, v in (("hidden_size", H), ("moe_intermediate_size", F), ("q width", nh * hd)):
        if v % 64:
            raise SystemExit(f"{n}={v} must be a multiple of 64")

    lm = "lm_head.weight" if "lm_head.weight" in st else "model.embed_tokens.weight"
    items = [("embed", "bf16", V, H, "model.embed_tokens.weight"),
             ("norm", "f32", 1, H, "model.norm.weight"),
             ("lm_head", args.dense_fmt, V, H, lm)]
    for l in range(L):
        p = f"model.layers.{l}."
        items += [(f"L{l}.ln1", "f32", 1, H, p + "input_layernorm.weight"),
                  (f"L{l}.ln2", "f32", 1, H, p + "post_attention_layernorm.weight"),
                  (f"L{l}.qn", "f32", 1, hd, p + "self_attn.q_norm.weight"),
                  (f"L{l}.kn", "f32", 1, hd, p + "self_attn.k_norm.weight"),
                  (f"L{l}.wq", args.dense_fmt, nh * hd, H, p + "self_attn.q_proj.weight"),
                  (f"L{l}.wk", args.dense_fmt, nkv * hd, H, p + "self_attn.k_proj.weight"),
                  (f"L{l}.wv", args.dense_fmt, nkv * hd, H, p + "self_attn.v_proj.weight"),
                  (f"L{l}.wo", args.dense_fmt, H, nh * hd, p + "self_attn.o_proj.weight"),
                  (f"L{l}.router", "f32", E, H, p + "mlp.gate.weight")]
    vocab = build_vocab(args.model, V)
    vblob = b"".join(struct.pack("<I", len(t)) + t for t in vocab)

    # ---- layout (all sizes are known from shapes)
    off, lines = 0, []
    cfgv = {"hidden_size": H, "num_hidden_layers": L, "num_attention_heads": nh, "num_key_value_heads": nkv,
            "head_dim": hd, "num_experts": E, "num_experts_per_tok": cfg["num_experts_per_tok"],
            "moe_intermediate_size": F, "rms_norm_eps": cfg["rms_norm_eps"],
            "rope_theta": cfg.get("rope_theta", 1e6), "vocab_size": V,
            "norm_topk_prob": int(cfg.get("norm_topk_prob", True)),
            "eos": ",".join(map(str, eos_ids(args.model, cfg))) or "-1", "expert_fmt": args.expert_fmt}
    lines += [f"cfg {k} {v}" for k, v in cfgv.items()]
    placed = []
    for name, fmt, rows, cols, src in items:
        n = pf.nbytes(fmt, rows, cols)
        lines.append(f"t {name} {fmt} {rows} {cols} {off} {n}")
        placed.append((off, name, fmt, rows, cols, src))
        off = pf.round_up(off + n)
    eb = pf.expert_nbytes(H, F, args.expert_fmt)
    expert_off = {}
    for l in range(L):
        for e in range(E):
            lines.append(f"e {l} {e} {off} {eb}")
            expert_off[(l, e)] = off
            off += eb
    lines.append(f"vocab {off} {len(vblob)}")
    total = off + len(vblob)

    header = ("\n".join(lines) + "\n").encode()
    data_start = pf.round_up(24 + len(header))
    print(f"pack: {total / 1e9:.2f} GB data, {L * E} experts x {eb / 1e6:.2f} MB")

    t0 = time.time()
    with open(args.out, "wb") as f:
        f.write(pf.MAGIC + struct.pack("<QQ", len(header), data_start) + header)
        for o, name, fmt, rows, cols, src in placed:
            if fmt == "bf16":
                arr, dt = st.raw(src)
                data = arr.tobytes() if dt == "BF16" else pf.encode("bf16", st.to_f32(arr, dt))
            else:
                data = pf.encode(fmt, st.get(src).reshape(rows, cols))
            f.seek(data_start + o)
            f.write(data)
        for l in range(L):
            for e in range(E):
                q = f"model.layers.{l}.mlp.experts.{e}."
                blob = b"".join(pf.encode(args.expert_fmt, st.get(q + w)) for w in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
                f.seek(data_start + expert_off[(l, e)])
                f.write(blob + b"\0" * (eb - len(blob)))
            el = time.time() - t0
            print(f"layer {l + 1}/{L} experts written  ({el / 60:.1f} min, eta {el / (l + 1) * (L - l - 1) / 60:.1f} min)", flush=True)
        f.seek(data_start + total - len(vblob))
        f.write(vblob)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
