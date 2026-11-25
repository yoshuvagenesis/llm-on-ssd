"""Convert a Qwen MoE safetensors checkpoint into a .moe pack for the engine.

  python convert.py --model ~/models/Qwen3-30B-A3B --out ~/qwen3-30b-a3b.moe
  python convert.py --model ~/models/Qwen3.6-35B-A3B --out ~/qwen3.6-35b-a3b.moe

Supported: Qwen3-MoE (e.g. Qwen3-30B-A3B, Qwen3-Coder-30B-A3B) and Qwen3.5/3.6-MoE
(e.g. Qwen3.6-35B-A3B; text only: the vision tower and the MTP head are not converted).

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


def supports_thinking(model_dir):
    """Qwen3 chat models can toggle a <think>...</think> reasoning block; some fine-tunes
    (e.g. Qwen3-Coder-Instruct) never trained on it at all, so the engine must not insert an
    empty one of its own -- that out-of-distribution sequence can derail generation entirely.
    We ask the model's own chat_template, not its name: if "<think>" never appears there, this
    model's template has no concept of a thinking block."""
    tp = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tp):
        return True  # unknown: keep today's behavior rather than silently changing it
    template = json.load(open(tp)).get("chat_template", "")
    return "<think>" in template


def eos_ids(model_dir, cfg):
    ids = None
    gp = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gp):
        ids = json.load(open(gp)).get("eos_token_id")
    if ids is None:
        ids = cfg.get("eos_token_id", [])
    return [ids] if isinstance(ids, int) else list(ids)


def check_dims(**dims):
    for n, v in dims.items():
        if v % 64:
            raise SystemExit(f"{n}={v} must be a multiple of 64")


def plan_qwen3(cfg, st, args):
    """Qwen3-MoE: every layer is attention + MoE (no shared expert)."""
    if cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1 or cfg.get("shared_expert_intermediate_size"):
        raise SystemExit("only all-MoE Qwen3 models without shared experts are supported")
    H, L, E = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_experts"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", H // nh)
    F, V = cfg["moe_intermediate_size"], cfg["vocab_size"]
    check_dims(hidden_size=H, moe_intermediate_size=F, q_width=nh * hd)
    cfg = dict(cfg, head_dim=hd, rope_theta=cfg.get("rope_theta", 1e6))
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

    def experts(l):
        for e in range(E):
            q = f"model.layers.{l}.mlp.experts.{e}."
            yield e, [st.get(q + w) for w in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")]

    return cfg, items, {}, experts, "model."


def plan_qwen35(full, st, args):
    """Qwen3.5/3.6-MoE (text part): Gated DeltaNet and gated full-attention layers, MoE with
    a shared expert in every layer. Zero-centred RMSNorm weights are stored as (1 + w), so
    the engine's norm stays x_normed * w."""
    cfg = full.get("text_config", full)
    H, L, E = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_experts"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    F, Fs, V = cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"], cfg["vocab_size"]
    Hk, Hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    Dk, Dv, K = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]
    rp = cfg.get("rope_parameters") or {}
    theta = cfg.get("rope_theta") or rp.get("rope_theta") or 1e7
    prf = cfg.get("partial_rotary_factor") or rp.get("partial_rotary_factor") or 1.0
    rot = int(hd * prf)
    if cfg.get("hidden_act", "silu") != "silu":
        raise SystemExit(f"hidden_act {cfg.get('hidden_act')} is not supported (silu only)")
    if not cfg.get("attn_output_gate", True):
        raise SystemExit("attn_output_gate=false is not supported")
    if Hv % Hk:
        raise SystemExit("linear_num_value_heads must be a multiple of linear_num_key_heads")
    types = cfg.get("layer_types") or [
        "full_attention" if (i + 1) % cfg.get("full_attention_interval", 4) == 0 else "linear_attention" for i in range(L)]
    kinds = "".join("A" if t == "full_attention" else "L" for t in types)
    if set(kinds) - {"A", "L"} or len(kinds) != L:
        raise SystemExit(f"unexpected layer_types {types}")
    check_dims(hidden_size=H, moe_intermediate_size=F, shared_expert_intermediate_size=Fs, q_width=nh * hd,
               linear_value_width=Hv * Dv, linear_key_width=Hk * Dk)
    P = "model.language_model." if any(k.startswith("model.language_model.") for k in st.meta) else "model."
    lm = "lm_head.weight" if "lm_head.weight" in st else P + "embed_tokens.weight"
    one_plus = lambda name: (lambda: 1.0 + st.get(name))
    items = [("embed", "bf16", V, H, P + "embed_tokens.weight"),
             ("norm", "f32", 1, H, one_plus(P + "norm.weight")),
             ("lm_head", args.dense_fmt, V, H, lm)]
    cd = 2 * Hk * Dk + Hv * Dv
    for l in range(L):
        p = f"{P}layers.{l}."
        items += [(f"L{l}.ln1", "f32", 1, H, one_plus(p + "input_layernorm.weight")),
                  (f"L{l}.ln2", "f32", 1, H, one_plus(p + "post_attention_layernorm.weight"))]
        if kinds[l] == "A":
            a = p + "self_attn."
            items += [(f"L{l}.qn", "f32", 1, hd, one_plus(a + "q_norm.weight")),
                      (f"L{l}.kn", "f32", 1, hd, one_plus(a + "k_norm.weight")),
                      (f"L{l}.wq", args.dense_fmt, 2 * nh * hd, H, a + "q_proj.weight"),  # [query, gate] per head
                      (f"L{l}.wk", args.dense_fmt, nkv * hd, H, a + "k_proj.weight"),
                      (f"L{l}.wv", args.dense_fmt, nkv * hd, H, a + "v_proj.weight"),
                      (f"L{l}.wo", args.dense_fmt, H, nh * hd, a + "o_proj.weight")]
        else:
            a = p + "linear_attn."
            items += [(f"L{l}.lqkv", args.dense_fmt, cd, H, a + "in_proj_qkv.weight"),
                      (f"L{l}.lz", args.dense_fmt, Hv * Dv, H, a + "in_proj_z.weight"),
                      (f"L{l}.lba", "f32", 2 * Hv, H,
                       (lambda a=a: np.concatenate([st.get(a + "in_proj_b.weight"), st.get(a + "in_proj_a.weight")]))),
                      (f"L{l}.conv", "f32", cd, K, a + "conv1d.weight"),
                      (f"L{l}.alog", "f32", 1, Hv, a + "A_log"),
                      (f"L{l}.dtb", "f32", 1, Hv, a + "dt_bias"),
                      (f"L{l}.lnorm", "f32", 1, Dv, a + "norm.weight"),
                      (f"L{l}.lout", args.dense_fmt, H, Hv * Dv, a + "out_proj.weight")]
        m = p + "mlp."
        items += [(f"L{l}.router", "f32", E, H, m + "gate.weight"),
                  (f"L{l}.sg", args.dense_fmt, Fs, H, m + "shared_expert.gate_proj.weight"),
                  (f"L{l}.su", args.dense_fmt, Fs, H, m + "shared_expert.up_proj.weight"),
                  (f"L{l}.sd", args.dense_fmt, H, Fs, m + "shared_expert.down_proj.weight"),
                  (f"L{l}.sgate", "f32", 1, H, m + "shared_expert_gate.weight")]

    def experts(l):
        gu, gdt = st.raw(f"{P}layers.{l}.mlp.experts.gate_up_proj")  # [E, 2F, H]
        dn, ddt = st.raw(f"{P}layers.{l}.mlp.experts.down_proj")      # [E, H, F]
        if gu.shape != (E, 2 * F, H) or dn.shape != (E, H, F):
            raise SystemExit(f"layer {l}: unexpected expert shapes {gu.shape} {dn.shape}")
        for e in range(E):
            g = st.to_f32(gu[e], gdt)
            yield e, [g[:F], g[F:], st.to_f32(dn[e], ddt)]

    extra = {"arch": "qwen35", "layer_types": kinds, "rotary_dim": rot, "lin_k_heads": Hk, "lin_v_heads": Hv,
             "lin_k_dim": Dk, "lin_v_dim": Dv, "conv_kernel": K, "shared_ffn": Fs, "think_open": 1}
    cfg = dict(cfg, rope_theta=theta, norm_topk_prob=True)
    return cfg, items, extra, experts, P


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

    full = json.load(open(os.path.join(args.model, "config.json")))
    st = SafeTensors(args.model)
    if full.get("model_type", "").startswith("qwen3_5") or "text_config" in full:
        cfg, items, extra, expert_src, P = plan_qwen35(full, st, args)
    else:
        cfg, items, extra, expert_src, P = plan_qwen3(full, st, args)
    H, L, E, F, V = (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_experts"], cfg["moe_intermediate_size"],
                     cfg["vocab_size"])
    vocab = build_vocab(args.model, V)
    vblob = b"".join(struct.pack("<I", len(t)) + t for t in vocab)

    # ---- layout (all sizes are known from shapes)
    off, lines = 0, []
    cfgv = {"hidden_size": H, "num_hidden_layers": L, "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "num_experts": E,
            "num_experts_per_tok": cfg["num_experts_per_tok"], "moe_intermediate_size": F,
            "rms_norm_eps": cfg["rms_norm_eps"], "rope_theta": cfg["rope_theta"], "vocab_size": V,
            "norm_topk_prob": int(cfg.get("norm_topk_prob", True)),
            "eos": ",".join(map(str, eos_ids(args.model, cfg))) or "-1", "expert_fmt": args.expert_fmt,
            "no_think": int(not supports_thinking(args.model)), **extra}
    lines += [f"cfg {k} {v}" for k, v in cfgv.items()]
    placed = []
    for name, fmt, rows, cols, load in items:
        n = pf.nbytes(fmt, rows, cols)
        lines.append(f"t {name} {fmt} {rows} {cols} {off} {n}")
        placed.append((off, name, fmt, rows, cols, load))
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
        for o, name, fmt, rows, cols, load in placed:
            if fmt == "bf16" and isinstance(load, str):
                arr, dt = st.raw(load)  # embeddings: copy the bf16 bits as they are
                data = arr.tobytes() if dt == "BF16" else pf.encode("bf16", st.to_f32(arr, dt))
            else:
                w = st.get(load) if isinstance(load, str) else load()
                data = pf.encode_large(fmt, np.asarray(w, dtype=np.float32).reshape(rows, cols))
                del w
            f.seek(data_start + o)
            f.write(data)
        for l in range(L):
            for e, mats in expert_src(l):
                blob = b"".join(pf.encode(args.expert_fmt, m) for m in mats)
                f.seek(data_start + expert_off[(l, e)])
                f.write(blob + b"\0" * (eb - len(blob)))
            el = time.time() - t0
            print(f"layer {l + 1}/{L} experts written  ({el / 60:.1f} min, eta {el / (l + 1) * (L - l - 1) / 60:.1f} min)", flush=True)
        f.seek(data_start + total - len(vblob))
        f.write(vblob)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
