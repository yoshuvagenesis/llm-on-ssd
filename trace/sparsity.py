"""Neuron skipping inside experts.

Each expert has F neurons (768 for Qwen3-30B-A3B). Neuron i needs row i of the gate
and up matrices and column i of the down matrix. Skipping it saves those reads and
that compute.

Two ways to choose what to skip:
  oracle : uses the true contribution |silu(gate)*up| x |down column|.
           Peeks at values you only have after computing everything: an upper bound.
  gate   : uses |silu(gate)| x |up row| x |down column|. Needs only the gate
           projection (1/3 of the expert's work), so it is a real, implementable rule.

Skipping can work on single neurons or on groups of G neurons stored together on
the SSD (so one read fetches a whole group). Groups can follow the original neuron
order or be clustered so that neurons which tend to matter together share a group.
"""
import numpy as np


def scores(mode, g_act, h, wu, wd):
    """g_act = silu(gate x), h = g_act * (up x); both [n, F]."""
    dcol = np.linalg.norm(wd, axis=0)[None, :]
    if mode == "oracle":
        return np.abs(h) * dcol
    if mode == "gate":
        urow = np.linalg.norm(wu, axis=1)[None, :]
        return np.abs(g_act) * urow * dcol
    raise ValueError(mode)


def skip_mask(sc, skip, group=1, perm=None):
    """Return a 0/1 mask [n, F] keeping the most important neurons (or groups).
    Exactly round(n_groups * skip) groups are dropped for every token."""
    n, F = sc.shape
    if skip <= 0:
        return np.ones_like(sc)
    if group == 1:
        drop = int(round(F * skip))
        mask = np.ones_like(sc)
        if drop:
            idx = np.argpartition(sc, drop - 1, axis=1)[:, :drop]
            np.put_along_axis(mask, idx, 0.0, axis=1)
        return mask
    if F % group:
        raise ValueError(f"{F} neurons are not divisible into groups of {group}")
    perm = np.arange(F) if perm is None else perm
    ng = F // group
    gs = sc[:, perm].reshape(n, ng, group).sum(-1)
    drop = int(round(ng * skip))
    gmask = np.ones((n, ng), dtype=sc.dtype)
    if drop:
        idx = np.argpartition(gs, drop - 1, axis=1)[:, :drop]
        np.put_along_axis(gmask, idx, 0.0, axis=1)
    mask = np.empty_like(sc)
    mask[:, perm] = np.repeat(gmask, group, axis=1)
    return mask


def cluster_order(important, group):
    """Order neurons so that each run of `group` neurons tends to be important together.
    important: bool [n_tokens, F] from calibration tokens. Greedy: seed with the most
    often important unassigned neuron, then add the neurons most co-active with the group."""
    n, F = important.shape
    if n < 8 or group <= 1:
        return np.arange(F)
    A = important.astype(np.float32)
    freq = A.sum(0)
    nrm = np.sqrt(freq) + 1e-6
    sim = (A.T @ A) / (nrm[:, None] * nrm[None, :])
    free = np.ones(F, dtype=bool)
    order = []
    while free.any():
        cand = np.flatnonzero(free)
        seed = cand[np.argmax(freq[cand])]
        members = [seed]
        free[seed] = False
        acc = sim[seed].copy()
        for _ in range(group - 1):
            masked = np.where(free, acc, -np.inf)
            nxt = int(np.argmax(masked))
            if not np.isfinite(masked[nxt]):
                break
            members.append(nxt)
            free[nxt] = False
            acc += sim[nxt]
        order.extend(members)
    return np.array(order, dtype=np.int64)
