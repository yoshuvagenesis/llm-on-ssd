"""Step 0 / trace analyzer and speed estimator.

Reads trace.npz from trace_routing.py and answers the feasibility questions:
  1. How skewed is expert popularity?            (can a RAM cache help?)
  2. How often does a token reuse the previous token's experts?
  3. How well does the one-layer-early prediction work?   (can prefetch hide SSD latency?)
  4. Cache hit rates for a given RAM budget (LRU vs pinned-hot experts)
  5. Speculative decoding: how many distinct experts does a verify batch touch?
  6. Estimated tokens/s for strict and relaxed modes, using YOUR measured
     SSD GB/s (bench_ssd.sh) and compute GMAC/s (int8bench).

Usage:
  python analyze_trace.py trace.npz --ssd-gbs 2.9 --gmacs 1:6.1,2:8.7,4:12.0
"""
import argparse
import collections
import sys

import numpy as np


def parse_gmacs(s):
    d = {}
    for part in s.split(","):
        b, v = part.split(":")
        d[int(b)] = float(v)
    return dict(sorted(d.items()))


def gmacs_at(table, b):
    ks = list(table)
    if b <= ks[0]:
        return table[ks[0]]
    if b >= ks[-1]:
        return table[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= b <= hi:
            f = (b - lo) / (hi - lo)
            return table[lo] * (1 - f) + table[hi] * f


def simulate_cache(A, P, T, E, slots, width, policy, calib_frac=0.5):
    """Walk tokens in decode order. Returns per-(token,layer) averages:
    hit, prefetched-and-used, stall (unpredicted miss), prefetched-but-unused.
    For 'pinned', the hottest `slots` experts are chosen on the first part of the
    trace and evaluated on the rest (no peeking)."""
    L, N, K = A.shape
    start = 0
    if policy == "pinned":
        start = int(N * calib_frac)
        cnt = np.zeros(L * E, dtype=np.int64)
        for l in range(L):
            cnt[l * E:(l + 1) * E] = np.bincount(A[l, :start].ravel(), minlength=E)
        pinned = set(np.argsort(-cnt)[:slots].tolist()) if slots > 0 else set()
    lru = collections.OrderedDict()
    hit = pre = stall = waste = 0
    for n in range(start, N):
        for l in range(L):
            base = l * E
            used = [base + int(e) for e in A[l, n]]
            pred = set() if (l == 0 or width == 0) else {base + int(e) for e in P[l, n, :width]}
            if policy == "pinned":
                cached = pinned
                for k in used:
                    if k in cached:
                        hit += 1
                    elif k in pred:
                        pre += 1
                    else:
                        stall += 1
                waste += sum(1 for k in pred if k not in cached and k not in used)
            else:
                waste += sum(1 for k in pred if k not in lru and k not in used)
                for k in used:
                    if k in lru:
                        hit += 1
                        lru.move_to_end(k)
                    else:
                        if k in pred:
                            pre += 1
                        else:
                            stall += 1
                        if slots > 0:
                            lru[k] = True
                            if len(lru) > slots:
                                lru.popitem(last=False)
    steps = (N - start) * L
    return {"hit": hit / (steps * K), "pre": pre / steps, "stall": stall / steps, "waste": waste / steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--ssd-gbs", type=float, default=None, help="measured random-read GB/s (bench_ssd.sh, 2M QD32)")
    ap.add_argument("--gmacs", default=None, help="int8bench GMAC/s per batch size, all threads, e.g. 1:6,2:8.5,4:12")
    ap.add_argument("--bits-expert", type=float, default=3.5, help="bits/weight incl. scales for experts")
    ap.add_argument("--bits-dense", type=float, default=4.5, help="bits/weight for attention, router, lm_head")
    ap.add_argument("--kv-bits", type=float, default=8)
    ap.add_argument("--ctx", default="1024,4096,8192", help="context positions to estimate at")
    ap.add_argument("--gamma", default="0,1,2,3,4,5", help="draft lengths to try (0 = no speculation)")
    ap.add_argument("--accept", type=float, default=0.7, help="draft token acceptance rate (measure later)")
    ap.add_argument("--draft-gmac", type=float, default=0.6, help="draft model GMAC per token (Qwen3-0.6B ~0.6)")
    ap.add_argument("--draft-gb", type=float, default=0.65, help="draft model size in RAM")
    ap.add_argument("--ram-cap-gb", type=float, default=6.0)
    ap.add_argument("--prefetch-width", type=int, default=12)
    ap.add_argument("--cache-gb", default="0.5,1,2,3,4", help="cache sizes for the hit-rate table")
    args = ap.parse_args()

    placeholder = []
    if args.ssd_gbs is None:
        args.ssd_gbs = 2.8
        placeholder.append("--ssd-gbs")
    if args.gmacs is None:
        args.gmacs = "1:6,2:9,4:13"
        placeholder.append("--gmacs")
    gm = parse_gmacs(args.gmacs)

    z = np.load(args.trace)
    A = z["actual"].astype(np.int64)
    P = z["pred"].astype(np.int64)
    Lt, N, K = A.shape
    T, E = int(z["seq_len"]), int(z["num_experts"])
    Lf = int(z["full_layers"])
    H, F = int(z["hidden"]), int(z["moe_intermediate"])
    nh, nkv, hd, V = int(z["num_heads"]), int(z["kv_heads"]), int(z["head_dim"]), int(z["vocab"])

    expert_params = 3 * H * F
    eb = expert_params * args.bits_expert / 8
    dense_layer = H * nh * hd * 2 + H * nkv * hd * 2 + E * H
    dense_bytes = (dense_layer * Lf + V * H) * args.bits_dense / 8
    mac_tok = Lf * (K * expert_params + dense_layer) + V * H
    kv_tok = Lf * 2 * nkv * hd * (args.kv_bits / 8 + 4 / 32)
    attn_mac_per_pos = 2 * Lf * nh * hd

    out = []
    pr = lambda s="": (print(s), out.append(s))
    pr(f"trace: {N} tokens ({N // T} seqs x {T}), {Lt}/{Lf} layers traced, {E} experts, top-{K}")
    if Lt < Lf:
        pr("WARNING: partial trace (debug run); estimates assume the untraced layers behave alike.")
    ppl = np.exp(np.nanmean(z["loss"])) if np.isfinite(z["loss"]).any() else float("nan")
    pr(f"perplexity of traced run: {ppl:.2f}")
    pr(f"expert block: {eb / 1e6:.2f} MB at {args.bits_expert} bits | dense weights: {dense_bytes / 1e9:.2f} GB "
       f"| active per token: {mac_tok / 1e9:.2f} GMAC | KV: {kv_tok / 1e3:.0f} KB per context token")

    # masks [L, N, E]
    M = np.zeros((Lt, N, E), dtype=bool)
    np.put_along_axis(M, A, True, axis=2)

    pr("\n== 1. Expert popularity (share of all activations taken by the hottest experts)")
    cnt = M.sum(1)  # [L, E]
    flat = np.sort(cnt.ravel())[::-1]
    tot = flat.sum()
    for pct in (5, 10, 25, 50):
        n_ = max(1, int(len(flat) * pct / 100))
        pr(f"   hottest {pct:>2}% of (layer,expert) blocks -> {flat[:n_].sum() / tot * 100:5.1f}% of activations")
    pr("   (uniform routing would give the same % on both sides)")

    pr("\n== 2. Temporal reuse (experts shared with the previous token, same layer)")
    cont = (np.arange(1, N) % T) != 0
    reuse = ((M[:, 1:] & M[:, :-1]).sum(-1) / K)[:, cont]
    pr(f"   average {reuse.mean() * 100:.1f}% of a token's experts were used by the previous token "
       f"(random would be {K / E * 100:.1f}%)")

    pr("\n== 3. Early prediction recall (how many of the real experts a prefetch would already have)")
    rec = {}
    for d, key in enumerate(("pred", "pred2", "pred3"), 1):
        if key not in z or Lt <= d:
            continue
        Pd = z[key].astype(np.int64)
        parts = []
        for w in (8, 12, 16):
            PM = np.zeros((Lt - d, N, E), dtype=bool)
            np.put_along_axis(PM, Pd[d:, :, :w], True, axis=2)
            r = (PM & M[d:]).sum(-1).mean() / K
            if d == 1:
                rec[w] = r
            parts.append(f"{w:>2} -> {r * 100:5.1f}%")
        pr(f"   {d} layer{'s' if d > 1 else ' '} early (first {d} excluded): " + "   ".join(parts))
    pr("   (reading 12 for 8 used = 1.5x the bytes if nothing is cached)")

    if "neuron_share" in z and z["neuron_share"].size:
        pr("\n== 3b. Neurons inside experts: share of an expert's output carried by its top neurons")
        share, keep = z["neuron_share"], z["neuron_keep"]
        thirds = np.array_split(np.arange(Lt), 3)
        pr(f"   {'top neurons':>12} {'all':>6} {'early':>6} {'middle':>6} {'late':>6}  (layers)")
        for i, f in enumerate(keep):
            row = [share[:, i].mean()] + [share[t, i].mean() for t in thirds]
            pr(f"   {f * 100:>11.0f}% " + " ".join(f"{v * 100:6.1f}" for v in row))
        pr("   share = sum of |activation| x |output column| for the top neurons, over all neurons")

    pr("\n== 4. RAM cache hit rates (prefetch width %d)" % args.prefetch_width)
    pr(f"   {'cache GB':>8} {'slots':>6} {'policy':>7} {'hit%':>6} {'prefetched':>10} {'stall':>6} {'wasted':>7}  (per token-layer, in experts)")
    ram_fixed = args.draft_gb + dense_bytes / 1e9
    sizes = [float(x) for x in args.cache_gb.split(",")]
    fit = max(0.0, args.ram_cap_gb - ram_fixed - 0.35)  # 0.35 GB: staging buffers, KV write buffer, runtime
    sizes = sorted(set(sizes + [round(fit, 2), 0.0]))
    sims = {}
    for gb in sizes:
        slots = int(gb * 1e9 / eb * Lt / Lf)
        best = None
        for pol in ("lru", "pinned"):
            r = simulate_cache(A, P, T, E, slots, args.prefetch_width, pol)
            pr(f"   {gb:>8.2f} {slots:>6} {pol:>7} {r['hit'] * 100:>6.1f} {r['pre']:>10.2f} {r['stall']:>6.2f} {r['waste']:>7.2f}")
            if best is None or r["hit"] > best["hit"]:
                best = r
        sims[gb] = best
    pr(f"   relaxed-mode cache that fits in {args.ram_cap_gb} GB: {fit:.2f} GB "
       f"(draft {args.draft_gb} + dense {dense_bytes / 1e9:.2f} + 0.35 buffers)")

    pr("\n== 5. Speculative verify batches: distinct experts per layer")
    gammas = [int(g) for g in args.gamma.split(",")]
    union = {}
    for B in sorted({g + 1 for g in gammas}):
        U = M[:, : N - B + 1].copy()
        for j in range(1, B):
            U |= M[:, j: N - B + 1 + j]
        starts = np.arange(N - B + 1)
        ok = (starts % T) + B <= T
        union[B] = U.sum(-1)[:, ok].mean()
        pr(f"   batch {B}: {union[B]:5.1f} distinct experts (vs {B * K} if never shared) -> {union[B] / B:4.1f} per token")

    pr("\n== 6. Estimated speed")
    if placeholder:
        pr("   !! PLACEHOLDER values for " + ", ".join(placeholder) + ": rerun with numbers measured on the N5105 !!")
    pr(f"   SSD {args.ssd_gbs} GB/s, compute {gm} GMAC/s, acceptance {args.accept}, "
       f"experts {args.bits_expert}b, dense {args.bits_dense}b, KV {args.kv_bits}b")
    a = args.accept
    for mode in ("strict", "relaxed"):
        sim = sims[0.0] if mode == "strict" else sims[round(fit, 2)]
        miss = 1 - sim["hit"]
        used_miss = sim["pre"] + sim["stall"]
        stall_frac = sim["stall"] / used_miss if used_miss > 0 else 0.0
        pr(f"\n   [{mode}] {'RAM = draft model + buffers only' if mode == 'strict' else f'RAM = draft + dense weights + {fit:.2f} GB expert cache'}"
           f" | hit {sim['hit'] * 100:.1f}%, stall {stall_frac * 100:.0f}% of misses")
        pr(f"   {'ctx':>6} {'gamma':>5} {'tok/pass':>8} {'SSD ms':>7} {'CPU ms':>7} {'stall ms':>8} {'draft ms':>8} {'tok/s':>6}  limit")
        for ctx in [int(c) for c in args.ctx.split(",")]:
            best = None
            for g in gammas:
                B = g + 1
                ex = union[B] * miss * Lf  # expert reads per pass
                io_pre = ex * (1 - stall_frac) * eb + sim["waste"] * B * Lf * eb + ctx * kv_tok
                if mode == "strict":
                    io_pre += dense_bytes
                io_stall = ex * stall_frac * eb
                cpu = B * (mac_tok + ctx * attn_mac_per_pos) / (gmacs_at(gm, B) * 1e9)
                draft = g * args.draft_gmac / gmacs_at(gm, 1)
                t_io = io_pre / (args.ssd_gbs * 1e9)
                t = max(t_io, cpu) + io_stall / (args.ssd_gbs * 1e9) + draft
                toks = (1 - a ** (g + 1)) / (1 - a) if a < 1 else g + 1
                lim = "SSD" if t_io > cpu else "CPU"
                row = (ctx, g, toks, t_io * 1e3, cpu * 1e3, io_stall / (args.ssd_gbs * 1e9) * 1e3, draft * 1e3, toks / t, lim)
                pr("   {:>6} {:>5} {:>8.2f} {:>7.0f} {:>7.0f} {:>8.0f} {:>8.0f} {:>6.2f}  {}".format(*row))
                if best is None or row[7] > best[7]:
                    best = row
            pr(f"   -> best at ctx {ctx}: gamma {best[1]}, {best[7]:.2f} tok/s ({best[8]}-bound)")
    pr("\nModel notes: SSD prefetch overlaps with compute; unpredicted misses (stalls) do not; the draft model"
       "\nruns before each verify pass; draft tokens are assumed to route like the real continuation.")

    with open(args.trace.rsplit(".", 1)[0] + "_report.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n(report also saved next to the trace)")


if __name__ == "__main__":
    main()
