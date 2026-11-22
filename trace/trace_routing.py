"""Step 0 / expert-routing tracer (run on the Mac).

Runs a Qwen3-MoE model (e.g. Qwen/Qwen3-30B-A3B) over real text, ONE LAYER AT
A TIME: each layer's weights are read from disk once, every token is pushed
through it, then the weights are dropped. Peak RAM stays around 3-4 GB, so a
60 GB bf16 checkpoint works on a 16 GB machine. numpy only (no torch).

It records, for every token and layer:
  * the 8 experts the router actually picked, and their weights
  * a prediction of that layer's experts made one layer EARLY
    (layer L's router applied to the hidden state after layer L-1's attention),
    which is what the engine would use to prefetch from the SSD
  * per-token loss, giving a perplexity sanity check: a correct forward pass on
    normal English text should land roughly in the 3-20 range; a broken one
    lands in the hundreds or thousands.

Usage:
  pip install numpy tokenizers
  python trace_routing.py --model ~/models/Qwen3-30B-A3B --text corpus.txt \
         --seq-len 1024 --num-seqs 8 --out trace.npz
  (or --ids tokens.npy instead of --text)
"""
import argparse
import json
import os
import time

import numpy as np

import sparsity as sp
from quant import parse_bits, quant_act, quant_weight
from st_reader import SafeTensors

PREFETCH_WIDTH = 16  # how many predicted experts to keep (analysis tries 8/12/16)
LOOKAHEAD = 3  # predict each layer's experts 1, 2 and 3 layers early
NEURON_KEEP = (0.1, 0.2, 0.3, 0.5, 0.7)  # fractions for neuron importance statistics


def neuron_scores(hmid, wd):
    """Importance of each expert neuron for each token: |activation| x |its output column|."""
    return np.abs(hmid) * np.linalg.norm(wd, axis=0)[None, :]


def rmsnorm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def topk_desc(x, k):
    idx = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    vals = np.take_along_axis(x, idx, axis=-1)
    order = np.argsort(-vals, axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def rope_tables(T, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    f = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([f, f], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):  # x [T, heads, hd]
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def attention(x, W, cfg, cos, sin, ab=0):
    """Causal GQA self-attention for one sequence. x [T, H] (already normed)."""
    T = x.shape[0]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    x = quant_act(x, ab)
    q = (x @ W["q"].T).reshape(T, nh, hd)
    k = (x @ W["k"].T).reshape(T, nkv, hd)
    v = (x @ W["v"].T).reshape(T, nkv, hd)
    q = rmsnorm(q, W["q_norm"], eps)  # Qwen3 QK-norm (per head, before RoPE)
    k = rmsnorm(k, W["k_norm"], eps)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    rep = nh // nkv
    q = q.transpose(1, 0, 2).reshape(nkv, rep, T, hd)
    k = k.transpose(1, 0, 2)[:, None]  # [nkv,1,T,hd]
    v = v.transpose(1, 0, 2)[:, None]
    s = np.matmul(q, k.swapaxes(-1, -2)) * (1.0 / np.sqrt(hd))
    s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, s)
    p = softmax(s, axis=-1)
    o = np.matmul(p, v).reshape(nh, T, hd).transpose(1, 0, 2).reshape(T, nh * hd)
    return quant_act(o, ab) @ W["o"].T


def load_config(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    need = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "num_experts", "num_experts_per_tok", "moe_intermediate_size", "rms_norm_eps",
            "vocab_size"]
    for n in need:
        if n not in cfg:
            raise SystemExit(f"config.json has no '{n}': this tracer supports Qwen3-MoE models only")
    cfg.setdefault("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    cfg.setdefault("rope_theta", 1e6)
    cfg.setdefault("norm_topk_prob", True)
    if cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1:
        raise SystemExit("dense (non-MoE) layers present: not supported yet")
    if cfg.get("shared_expert_intermediate_size"):
        raise SystemExit("shared experts present: not supported yet")
    if cfg["num_experts"] > 255:
        raise SystemExit("more than 255 experts: trace format uses uint8 ids")
    return cfg


def get_tokens(args, model_dir):
    if args.ids:
        ids = np.load(args.ids).astype(np.int64).ravel()
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        ids = np.array(tok.encode(open(args.text, encoding="utf-8").read()).ids, dtype=np.int64)
    need = args.seq_len * args.num_seqs
    if len(ids) < need:
        n = len(ids) // args.seq_len
        if n == 0:
            raise SystemExit(f"only {len(ids)} tokens, need at least seq-len={args.seq_len}")
        print(f"note: only {len(ids)} tokens, using {n} sequences instead of {args.num_seqs}")
        args.num_seqs = n
        need = args.seq_len * n
    return ids[:need].reshape(args.num_seqs, args.seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder with config.json + *.safetensors")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="UTF-8 text file (needs `pip install tokenizers`)")
    src.add_argument("--ids", help=".npy file of token ids")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-seqs", type=int, default=8)
    ap.add_argument("--layers", type=int, default=0, help="only first N layers (debug; no perplexity)")
    ap.add_argument("--out", default="trace.npz")
    ap.add_argument("--expert-bits", default="16", help="16 (off), 8, 6, 5, 4, 3, 2 or tern")
    ap.add_argument("--dense-bits", default="16", help="attention + output-head weights: 16 (off), 8, 6, 5, 4, 3, 2")
    ap.add_argument("--act-bits", type=int, default=0, help="round matmul inputs to N bits per 32 (0 = off; 8, or 6 like the LUT kernels)")
    ap.add_argument("--neuron-skip", type=float, default=0.0,
                    help="oracle: in every expert, skip this fraction of neurons with the smallest contribution (0-0.95)")
    ap.add_argument("--skip-mode", default="oracle", choices=["oracle", "gate"],
                    help="oracle = perfect choice (upper bound); gate = real rule using only the gate projection")
    ap.add_argument("--neuron-group", type=int, default=1, help="skip neurons in groups of this size (SSD read unit)")
    ap.add_argument("--group-order", default="plain", choices=["plain", "cluster"],
                    help="plain = original neuron order; cluster = group neurons that matter together "
                         "(learned from the first half of the sequences)")
    ap.add_argument("--neuron-stats", action="store_true",
                    help="record how much of each expert's output its most important neurons carry")
    args = ap.parse_args()

    eb, db, ab = parse_bits(args.expert_bits), parse_bits(args.dense_bits), args.act_bits
    if db == "tern":
        raise SystemExit("--dense-bits tern is not supported")
    cfg = load_config(args.model)
    st = SafeTensors(args.model)
    ids = get_tokens(args, args.model)
    S, T = ids.shape
    H, E, K = cfg["hidden_size"], cfg["num_experts"], cfg["num_experts_per_tok"]
    L = args.layers or cfg["num_hidden_layers"]
    eps = cfg["rms_norm_eps"]
    N = S * T
    n_cal = (S // 2) * T if S >= 2 else N // 2  # tokens used to learn neuron groups
    print(f"{S} sequences x {T} tokens = {N} tokens, {L} layers, {E} experts, top-{K}")
    print(f"quantization: experts={eb} dense={db} activations={ab or 'off'} (router kept at full precision)")
    skip = float(args.neuron_skip)
    if not 0.0 <= skip <= 0.95:
        raise SystemExit("--neuron-skip must be between 0 and 0.95")
    F = cfg["moe_intermediate_size"]
    G = args.neuron_group
    if F % G:
        raise SystemExit(f"--neuron-group must divide {F}")
    if skip:
        print(f"neuron skipping: {skip * 100:.0f}% per expert per token, mode={args.skip_mode}, "
              f"group={G}" + (f" ({args.group_order} order)" if G > 1 else ""))

    cos, sin = rope_tables(T, cfg["head_dim"], cfg["rope_theta"])
    h = st.rows("model.embed_tokens.weight", ids.ravel()).reshape(S, T, H)

    actual = np.zeros((L, N, K), dtype=np.uint8)
    weights = np.zeros((L, N, K), dtype=np.float16)
    preds = [np.full((L, N, PREFETCH_WIDTH), 255, dtype=np.uint8) for _ in range(LOOKAHEAD)]
    mids = []  # residual streams after the last LOOKAHEAD layers' attention, [S,T,H] each
    share_sum = np.zeros((L, len(NEURON_KEEP)), dtype=np.float64)
    share_cnt = np.zeros(L, dtype=np.float64)

    t_start = time.time()
    for l in range(L):
        t0 = time.time()
        p = f"model.layers.{l}."
        W = {
            "q": quant_weight(st.get(p + "self_attn.q_proj.weight"), db),
            "k": quant_weight(st.get(p + "self_attn.k_proj.weight"), db),
            "v": quant_weight(st.get(p + "self_attn.v_proj.weight"), db),
            "o": quant_weight(st.get(p + "self_attn.o_proj.weight"), db),
            "q_norm": st.get(p + "self_attn.q_norm.weight"),
            "k_norm": st.get(p + "self_attn.k_norm.weight"),
        }
        ln1 = st.get(p + "input_layernorm.weight")
        ln2 = st.get(p + "post_attention_layernorm.weight")
        router = st.get(p + "mlp.gate.weight")  # [E, H]

        # early predictions for this layer: its router applied to the hidden state
        # after the attention of layer l-1, l-2, l-3
        for d in range(1, LOOKAHEAD + 1):
            if len(mids) >= d:
                r = rmsnorm(mids[-d].reshape(N, H), ln2, eps) @ router.T
                preds[d - 1][l] = topk_desc(r, PREFETCH_WIDTH).astype(np.uint8)

        for s in range(S):
            h[s] = h[s] + attention(rmsnorm(h[s], ln1, eps), W, cfg, cos, sin, ab)
        mids.append(h.copy())
        if len(mids) > LOOKAHEAD:
            mids.pop(0)

        x = rmsnorm(h.reshape(N, H), ln2, eps)
        probs = softmax(x @ router.T, axis=-1)
        top = topk_desc(probs, K)
        tw = np.take_along_axis(probs, top, axis=-1)
        if cfg["norm_topk_prob"]:
            tw = tw / tw.sum(-1, keepdims=True)
        actual[l] = top.astype(np.uint8)
        weights[l] = tw.astype(np.float16)

        out = np.zeros((N, H), dtype=np.float32)
        flat_e = top.ravel()
        flat_t = np.repeat(np.arange(N), K)
        flat_w = tw.ravel()
        order = np.argsort(flat_e, kind="stable")
        bounds = np.searchsorted(flat_e[order], np.arange(E + 1))
        for e in range(E):
            sel = order[bounds[e]:bounds[e + 1]]
            if len(sel) == 0:
                continue
            tok = flat_t[sel]
            q = p + f"mlp.experts.{e}."
            xe = quant_act(x[tok], ab)
            wu = quant_weight(st.get(q + "up_proj.weight"), eb)
            ga = silu(xe @ quant_weight(st.get(q + "gate_proj.weight"), eb).T)
            hm = ga * (xe @ wu.T)
            wd = quant_weight(st.get(q + "down_proj.weight"), eb)
            if args.neuron_stats:
                sc = neuron_scores(hm, wd)
                srt = -np.sort(-sc, axis=1)
                tot = srt.sum(1, keepdims=True) + 1e-30
                cs = np.cumsum(srt, axis=1) / tot
                for i, f in enumerate(NEURON_KEEP):
                    share_sum[l, i] += cs[:, max(1, int(round(F * f))) - 1].sum()
                share_cnt[l] += len(tok)
            if skip:
                perm = None
                if G > 1 and args.group_order == "cluster":
                    cal = tok < n_cal
                    oc = sp.scores("oracle", ga[cal], hm[cal], wu, wd)
                    imp = oc >= np.median(oc, axis=1, keepdims=True) if len(oc) else np.zeros((0, F), bool)
                    perm = sp.cluster_order(imp, G)
                hm = hm * sp.skip_mask(sp.scores(args.skip_mode, ga, hm, wu, wd), skip, G, perm)
            y = quant_act(hm, ab) @ wd.T
            out[tok] += y * flat_w[sel, None]  # a token picks each expert at most once
        h = h + out.reshape(S, T, H)
        dt = time.time() - t0
        eta = (time.time() - t_start) / (l + 1) * (L - l - 1)
        print(f"layer {l + 1}/{L}  {dt:5.1f}s  (eta {eta / 60:4.1f} min)", flush=True)

    loss = np.full(N, np.nan, dtype=np.float32)
    if L == cfg["num_hidden_layers"]:
        hn = quant_act(rmsnorm(h.reshape(N, H), st.get("model.norm.weight"), eps), ab)
        lm_name = "lm_head.weight" if "lm_head.weight" in st else "model.embed_tokens.weight"
        lm = quant_weight(st.get(lm_name), db)
        tgt = ids.ravel()
        for a in range(0, N, 256):
            b = min(a + 256, N)
            lg = hn[a:b] @ lm.T
            m = lg.max(-1)
            lse = m + np.log(np.exp(lg - m[:, None]).sum(-1))
            pos = np.arange(a, b)
            valid = (pos % T) != T - 1  # last token of each sequence has no target
            nxt = np.where(valid, pos + 1, pos)
            tok_loss = lse - lg[np.arange(b - a), tgt[nxt]]
            loss[pos[valid]] = tok_loss[valid]
        ppl = float(np.exp(np.nanmean(loss)))
        print(f"\nperplexity on this text: {ppl:.2f}  (normal text, correct model: roughly 3-20)")
        print(f"PPL={ppl:.4f}")
    np.savez_compressed(
        args.out, actual=actual, weights=weights, pred=preds[0], pred2=preds[1], pred3=preds[2],
        loss=loss, ids=ids, neuron_skip=skip, neuron_keep=np.array(NEURON_KEEP),
        skip_mode=args.skip_mode, neuron_group=G, group_order=args.group_order,
        neuron_share=(share_sum / np.maximum(share_cnt, 1)[:, None]) if args.neuron_stats else np.zeros(0),
        seq_len=T, num_experts=E, top_k=K, num_layers=L, hidden=H,
        moe_intermediate=cfg["moe_intermediate_size"], full_layers=cfg["num_hidden_layers"],
        num_heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], vocab=cfg["vocab_size"],
        expert_bits=str(eb), dense_bits=str(db), act_bits=ab,
    )
    print(f"saved {args.out}  ({(time.time() - t_start) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
