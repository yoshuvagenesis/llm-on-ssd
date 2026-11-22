"""Self-test for trace_routing.py, no real model needed.

Builds a tiny random Qwen3-MoE checkpoint, runs the tracer on it, and compares
the results with an independent reference written differently on purpose:
token-by-token decoding with a KV cache, and every expert evaluated densely.
If both agree on routing, early predictions and loss, the tracer's batching,
causal mask, RoPE positions and expert scatter are correct.

Run:  python selftest.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import sparsity as sp
from quant import quant_act, quant_weight
from st_reader import SafeTensors, write_safetensors

CFG = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
           head_dim=32, num_experts=32, num_experts_per_tok=4, moe_intermediate_size=24,
           rms_norm_eps=1e-6, rope_theta=10000.0, vocab_size=97, norm_topk_prob=True,
           tie_word_embeddings=False, model_type="qwen3_moe")


def build(d, rng):
    c = CFG
    H, nh, nkv, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, F, V = c["num_experts"], c["moe_intermediate_size"], c["vocab_size"]
    n = lambda *s: (rng.standard_normal(s) * 0.15).astype(np.float32)
    one = lambda k: (1.0 + 0.1 * rng.standard_normal(k)).astype(np.float32)
    t = {"model.embed_tokens.weight": n(V, H), "model.norm.weight": one(H), "lm_head.weight": n(V, H)}
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        t.update({p + "self_attn.q_proj.weight": n(nh * hd, H), p + "self_attn.k_proj.weight": n(nkv * hd, H),
                  p + "self_attn.v_proj.weight": n(nkv * hd, H), p + "self_attn.o_proj.weight": n(H, nh * hd),
                  p + "self_attn.q_norm.weight": one(hd), p + "self_attn.k_norm.weight": one(hd),
                  p + "input_layernorm.weight": one(H), p + "post_attention_layernorm.weight": one(H),
                  p + "mlp.gate.weight": n(E, H) * 4})
        for e in range(E):
            q = p + f"mlp.experts.{e}."
            t.update({q + "gate_proj.weight": n(F, H), q + "up_proj.weight": n(F, H), q + "down_proj.weight": n(H, F)})
    # split into two shards to exercise the index file
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wm = {}
    for fn, ns in shards.items():
        write_safetensors(os.path.join(d, fn), {k: t[k] for k in ns})
        wm.update({k: fn for k in ns})
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump(c, open(os.path.join(d, "config.json"), "w"))


def reference(d, ids, skip=0.0, mode="oracle"):
    """Token-by-token decode with KV cache; dense experts. Returns actual, weights, [pred8 at
    lookahead 1,2,3], loss. skip = oracle fraction of neurons dropped per expert."""
    c, st = CFG, SafeTensors(d)
    H, nh, nkv, hd, E, K = (c[k] for k in ["hidden_size", "num_attention_heads", "num_key_value_heads",
                                            "head_dim", "num_experts", "num_experts_per_tok"])
    L, eps = c["num_hidden_layers"], c["rms_norm_eps"]
    g = st.get
    norm = lambda x, w: x / np.sqrt(np.mean(x * x) + eps) * w

    def rope(v, pos):  # v [hd], rotate-half convention, written element-wise
        out = np.empty_like(v)
        half = hd // 2
        for i in range(half):
            ang = pos / (c["rope_theta"] ** (2 * i / hd))
            co, si = np.cos(ang), np.sin(ang)
            a, b = v[i], v[i + half]
            out[i] = a * co - b * si
            out[i + half] = b * co + a * si
        return out

    S, T = ids.shape
    act, wts, pred, loss = [], [], [], []
    for s in range(S):
        cache = [([], []) for _ in range(L)]
        for t in range(T):
            h = g("model.embed_tokens.weight")[ids[s, t]].copy()
            mids_t = []
            a_t, w_t = [], []
            p_t = [[], [], []]
            for l in range(L):
                p = f"model.layers.{l}."
                for dd in (1, 2, 3):
                    if l - dd >= 0:
                        r = g(p + "mlp.gate.weight") @ norm(mids_t[l - dd], g(p + "post_attention_layernorm.weight"))
                        p_t[dd - 1].append(list(np.argsort(-r)[:8]))
                    else:
                        p_t[dd - 1].append([255] * 8)
                x = norm(h, g(p + "input_layernorm.weight"))
                q = (g(p + "self_attn.q_proj.weight") @ x).reshape(nh, hd)
                k = (g(p + "self_attn.k_proj.weight") @ x).reshape(nkv, hd)
                v = (g(p + "self_attn.v_proj.weight") @ x).reshape(nkv, hd)
                q = np.stack([rope(norm(q[i], g(p + "self_attn.q_norm.weight")), t) for i in range(nh)])
                k = np.stack([rope(norm(k[i], g(p + "self_attn.k_norm.weight")), t) for i in range(nkv)])
                cache[l][0].append(k)
                cache[l][1].append(v)
                Ks, Vs = np.stack(cache[l][0]), np.stack(cache[l][1])  # [t+1, nkv, hd]
                o = []
                for i in range(nh):
                    j = i // (nh // nkv)
                    sc = Ks[:, j] @ q[i] / np.sqrt(hd)
                    pr = np.exp(sc - sc.max())
                    pr /= pr.sum()
                    o.append(pr @ Vs[:, j])
                h = h + g(p + "self_attn.o_proj.weight") @ np.concatenate(o)
                mids_t.append(h.copy())
                x2 = norm(h, g(p + "post_attention_layernorm.weight"))
                r = g(p + "mlp.gate.weight") @ x2
                pr = np.exp(r - r.max())
                pr /= pr.sum()
                top = np.argsort(-pr)[:K]
                w = pr[top] / pr[top].sum()
                a_t.append(top)
                w_t.append(w)
                mix = np.zeros(E, dtype=np.float32)
                mix[top] = w
                y = np.zeros(H, dtype=np.float32)
                for e in range(E):  # dense: every expert computed, most weighted by 0
                    q_ = p + f"mlp.experts.{e}."
                    gg = g(q_ + "gate_proj.weight") @ x2
                    uu = g(q_ + "up_proj.weight") @ x2
                    ga = gg / (1 + np.exp(-gg))
                    hm = ga * uu
                    wd = g(q_ + "down_proj.weight")
                    wu = g(q_ + "up_proj.weight")
                    nd = int(round(len(hm) * skip))
                    if nd:  # drop the nd neurons with the smallest score
                        dn = np.sqrt((wd * wd).sum(0))
                        if mode == "oracle":
                            score = np.abs(hm) * dn
                        else:
                            score = np.abs(ga) * np.sqrt((wu * wu).sum(1)) * dn
                        hm = hm.copy()
                        hm[np.argsort(score, kind="stable")[:nd]] = 0.0
                    y += mix[e] * (wd @ hm)
                h = h + y
            logits = g("lm_head.weight") @ norm(h, g("model.norm.weight"))
            lse = logits.max() + np.log(np.exp(logits - logits.max()).sum())
            loss.append(lse - logits[ids[s, t + 1]] if t + 1 < T else np.nan)
            act.append(a_t)
            wts.append(w_t)
            pred.append(p_t)
    tr = lambda a: np.array(a).transpose(1, 0, 2)  # -> [L, N, k]
    preds = [tr([pt[dd] for pt in pred]) for dd in range(3)]
    return tr(act), tr(wts), preds, np.array(loss, dtype=np.float32)


def test_quant(rng):
    """Fake-quantizer sanity: level counts, error bounds, monotonic error."""
    ok = True
    w = rng.standard_normal((64, 128)).astype(np.float32)
    errs = []
    for bits in (8, 4, 3, 2):
        q = quant_weight(w, bits)
        blocks, qb = w.reshape(-1, 32), q.reshape(-1, 32)
        levels = max(len(np.unique(r)) for r in qb)
        step = (blocks.max(1) - blocks.min(1)) / ((1 << bits) - 1)
        bound = np.all(np.abs(blocks - qb).max(1) <= step / 2 + 1e-5)
        ok &= levels <= (1 << bits) and bool(bound)
        errs.append(np.abs(w - q).mean())
    ok &= all(a < b for a, b in zip(errs, errs[1:]))
    t = quant_weight(w, "tern").reshape(-1, 32)
    ok &= all(len(np.unique(np.abs(r))) <= 2 for r in t)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    xa = quant_act(x, 8).reshape(-1, 32)
    bound = np.abs(x.reshape(-1, 32)).max(1) / 127 / 2 + 1e-6
    ok &= bool(np.all(np.abs(x.reshape(-1, 32) - xa).max(1) <= bound))
    print(f"quantizer checks:              {'OK' if ok else 'FAIL'}  (mean error 8/4/3/2 bits: "
          + ", ".join(f"{e:.4f}" for e in errs) + ")")
    return ok


def test_sparsity(rng):
    ok = True
    sc = rng.random((50, 64)).astype(np.float32)
    m = sp.skip_mask(sc, 0.5)
    ok &= bool(np.all((m == 0).sum(1) == 32))
    ok &= bool(np.all(np.where(m == 0, sc, -1).max(1) <= np.where(m == 1, sc, 2).min(1)))
    perm = rng.permutation(64)
    mg = sp.skip_mask(sc, 0.25, group=8, perm=perm)
    gm = mg[:, perm].reshape(50, 8, 8)
    whole = bool(np.all((gm.min(-1) == gm.max(-1))))  # groups are all-kept or all-dropped
    ok &= whole and bool(np.all((gm[:, :, 0] == 0).sum(1) == 2))
    gs = sc[:, perm].reshape(50, 8, 8).sum(-1)
    ok &= bool(np.all(np.where(gm[:, :, 0] == 0, gs, -1).max(1) <= np.where(gm[:, :, 0] == 1, gs, 1e9).min(1)))
    # planted structure: 8 hidden groups of 8 neurons that switch on together
    planted = rng.permutation(64).reshape(8, 8)
    on = rng.random((400, 8)) < 0.4
    imp = np.zeros((400, 64), dtype=bool)
    for k in range(8):
        imp[:, planted[k]] = on[:, k:k + 1]
    imp ^= rng.random((400, 64)) < 0.03  # a little noise
    order = sp.cluster_order(imp, 8)
    ok &= sorted(order.tolist()) == list(range(64))
    found = {frozenset(order[i * 8:(i + 1) * 8].tolist()) for i in range(8)}
    rec = len(found & {frozenset(p.tolist()) for p in planted})
    ok &= rec == 8
    print(f"sparsity checks:               {'OK' if ok else 'FAIL'}  (clustering recovered {rec}/8 planted groups)")
    return ok


def main():
    rng = np.random.default_rng(7)
    qok = test_quant(rng) & test_sparsity(np.random.default_rng(3))
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as d:
        build(d, rng)
        S, T = 2, 12
        ids = rng.integers(0, CFG["vocab_size"], size=(S, T))
        np.save(os.path.join(d, "ids.npy"), ids)
        out = os.path.join(d, "trace.npz")
        subprocess.run([sys.executable, os.path.join(here, "trace_routing.py"), "--model", d,
                        "--ids", os.path.join(d, "ids.npy"), "--seq-len", str(T), "--num-seqs", str(S),
                        "--out", out], check=True)
        z = np.load(out)
        ra, rw, rp, rl = reference(d, ids)
        # quantized run must work and stay close to the unquantized one
        outq = os.path.join(d, "trace_q.npz")
        subprocess.run([sys.executable, os.path.join(here, "trace_routing.py"), "--model", d,
                        "--ids", os.path.join(d, "ids.npy"), "--seq-len", str(T), "--num-seqs", str(S),
                        "--expert-bits", "4", "--dense-bits", "8", "--act-bits", "6", "--out", outq],
                       check=True, stdout=subprocess.DEVNULL)
        zq = np.load(outq)
        outs = os.path.join(d, "trace_s.npz")
        subprocess.run([sys.executable, os.path.join(here, "trace_routing.py"), "--model", d,
                        "--ids", os.path.join(d, "ids.npy"), "--seq-len", str(T), "--num-seqs", str(S),
                        "--neuron-skip", "0.5", "--neuron-stats", "--out", outs],
                       check=True, stdout=subprocess.DEVNULL)
        zs = np.load(outs)
        _, _, _, rl_s = reference(d, ids, skip=0.5)
        outg = os.path.join(d, "trace_g.npz")
        subprocess.run([sys.executable, os.path.join(here, "trace_routing.py"), "--model", d,
                        "--ids", os.path.join(d, "ids.npy"), "--seq-len", str(T), "--num-seqs", str(S),
                        "--neuron-skip", "0.5", "--skip-mode", "gate", "--out", outg],
                       check=True, stdout=subprocess.DEVNULL)
        zg = np.load(outg)
        _, _, _, rl_g = reference(d, ids, skip=0.5, mode="gate")
        outc = os.path.join(d, "trace_c.npz")
        subprocess.run([sys.executable, os.path.join(here, "trace_routing.py"), "--model", d,
                        "--ids", os.path.join(d, "ids.npy"), "--seq-len", str(T), "--num-seqs", str(S),
                        "--neuron-skip", "0.5", "--skip-mode", "gate", "--neuron-group", "8",
                        "--group-order", "cluster", "--out", outc],
                       check=True, stdout=subprocess.DEVNULL)
        zc = np.load(outc)

    ok = True
    same_sets = np.array([[set(a) == set(b) for a, b in zip(x, y)] for x, y in zip(z["actual"], ra)])
    ok &= bool(same_sets.all())
    print(f"routing sets identical:        {same_sets.mean() * 100:.1f}%")
    werr = np.abs(np.sort(z["weights"].astype(np.float32), -1) - np.sort(rw, -1)).max()
    ok &= bool(werr < 2e-3)  # weights stored as float16
    print(f"routing weight max diff:       {werr:.2e}")
    for dd, key in enumerate(("pred", "pred2", "pred3")):
        pm = (z[key][dd + 1:, :, :8] == rp[dd][dd + 1:]).mean()
        ok &= bool(pm == 1.0)
        print(f"{dd + 1}-layer-early prediction match: {pm * 100:.1f}%")
    m = ~np.isnan(rl)
    lerr = np.abs(z["loss"][m] - rl[m]).max()
    ok &= bool(lerr < 1e-3) and bool(np.array_equal(np.isnan(z["loss"]), ~m))
    print(f"per-token loss max diff:       {lerr:.2e}")
    ok &= qok
    ms = ~np.isnan(rl_s)
    serr = np.abs(zs["loss"][ms] - rl_s[ms]).max()
    share = zs["neuron_share"]
    share_ok = share.shape == (CFG["num_hidden_layers"], 5) and bool(np.all(np.diff(share, axis=1) >= -1e-9)) \
        and bool(np.all((share > 0) & (share <= 1 + 1e-6)))
    ok &= bool(serr < 1e-3) and share_ok
    print(f"oracle neuron skip 50% loss diff: {serr:.2e} | neuron share stats {'OK' if share_ok else 'FAIL'}")
    gerr = np.abs(zg["loss"][ms] - rl_g[ms]).max()
    cl_ok = bool(np.isfinite(np.nanmean(zc["loss"])))
    ok &= bool(gerr < 1e-3) and cl_ok
    print(f"gate-first skip 50% loss diff:    {gerr:.2e} | clustered groups of 8 run {'OK' if cl_ok else 'FAIL'}"
          f" (loss {np.nanmean(zc['loss']):.3f})")
    same = np.mean([[len(set(a) & set(b)) / len(a) for a, b in zip(x, y)] for x, y in zip(zq["actual"], z["actual"])])
    qloss_ok = bool(np.isfinite(np.nanmean(zq["loss"])))
    ok &= qloss_ok and same > 0.8
    print(f"quantized run (4b experts, 8b dense, 6b acts): routing overlap {same * 100:.1f}%, "
          f"loss {np.nanmean(zq['loss']):.3f} vs {np.nanmean(z['loss']):.3f}")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
