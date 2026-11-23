"""Qwen3.5 / Qwen3.6 MoE (text part): a tiny random checkpoint in Hugging Face's own
layout, and an independent float64 reference forward pass that reads it directly.

The reference follows the transformers implementation (Qwen3_5MoeForConditionalGeneration):
  - 3 of every 4 layers are Gated DeltaNet ("linear_attention"), every 4th is gated
    full attention; every layer ends in a MoE block with a shared expert
  - RMSNorm computes x_normed * (1 + weight) (weights stored zero-centred), except the
    DeltaNet output norm, which is x_normed * weight followed by * silu(z)
  - full attention: q_proj gives [query, gate] per head; q/k RMSNorm per head; RoPE on the
    first head_dim * partial_rotary_factor dims (rotate-half); output * sigmoid(gate)
  - DeltaNet: in_proj_qkv -> causal depthwise conv (kernel 4) -> silu -> split q|k|v;
    l2-normalized q,k (q also / sqrt(Dk)); key heads repeat_interleave'd to value heads;
    beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias); recurrent delta rule
  - MoE: softmax over all experts, top-k renormalized; experts stored fused
    (gate_up_proj [E, 2F, H], down_proj [E, H, F]); shared expert * sigmoid(gate)

It shares no code with the converter or the engine. With `fake_quant` it rounds the
weights exactly like the converter would (but keeps activations in float64), so the
engine's result should differ only by its int8 activations and bf16 KV cache.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_reader import SafeTensors, write_safetensors  # noqa: E402

TEXT = dict(hidden_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            attn_output_gate=True, full_attention_interval=4,
            linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4,
            num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            hidden_act="silu", rms_norm_eps=1e-6, partial_rotary_factor=0.25,
            rope_parameters={"rope_theta": 10000000, "rope_type": "default", "mrope_interleaved": True,
                             "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25},
            vocab_size=512, bos_token_id=248044, eos_token_id=500, tie_word_embeddings=False,
            model_type="qwen3_5_moe_text", mtp_num_hidden_layers=1)


def layer_types(t):
    n = t["full_attention_interval"]
    return ["full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(t["num_hidden_layers"])]


def build(d, rng, text=None, scale=0.15):
    """Write a tiny random Qwen3.5-MoE checkpoint (HF layout, two shards, vision and MTP
    tensors included so the converter must skip them)."""
    t = dict(TEXT if text is None else text)
    t["layer_types"] = layer_types(t)
    H, L, E = t["hidden_size"], t["num_hidden_layers"], t["num_experts"]
    nh, nkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    Hk, Hv, Dk, Dv, K = (t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                         t["linear_value_head_dim"], t["linear_conv_kernel_dim"])
    F, Fs, V = t["moe_intermediate_size"], t["shared_expert_intermediate_size"], t["vocab_size"]
    n = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)
    zc = lambda k: (0.1 * rng.standard_normal(k)).astype(np.float32)  # zero-centred norm weights
    P = "model.language_model."
    w = {P + "embed_tokens.weight": n(V, H), P + "norm.weight": zc(H), "lm_head.weight": n(V, H),
         "mtp.fc.weight": n(H, 2 * H), "mtp.norm.weight": zc(H), "model.visual.patch_embed.proj.weight": n(8, 3, 2, 2)}
    f32 = set()
    for l in range(L):
        p = f"{P}layers.{l}."
        w[p + "input_layernorm.weight"] = zc(H)
        w[p + "post_attention_layernorm.weight"] = zc(H)
        if t["layer_types"][l] == "full_attention":
            w.update({p + "self_attn.q_proj.weight": n(nh * hd * 2, H), p + "self_attn.k_proj.weight": n(nkv * hd, H),
                      p + "self_attn.v_proj.weight": n(nkv * hd, H), p + "self_attn.o_proj.weight": n(H, nh * hd),
                      p + "self_attn.q_norm.weight": zc(hd), p + "self_attn.k_norm.weight": zc(hd)})
        else:
            cd = 2 * Hk * Dk + Hv * Dv
            w.update({p + "linear_attn.in_proj_qkv.weight": n(cd, H), p + "linear_attn.in_proj_z.weight": n(Hv * Dv, H),
                      p + "linear_attn.in_proj_b.weight": n(Hv, H), p + "linear_attn.in_proj_a.weight": n(Hv, H),
                      p + "linear_attn.conv1d.weight": (rng.standard_normal((cd, 1, K)) * 0.4).astype(np.float32),
                      p + "linear_attn.dt_bias": (1.0 + 0.2 * rng.standard_normal(Hv)).astype(np.float32),
                      p + "linear_attn.A_log": np.log(rng.uniform(0.5, 8.0, Hv)).astype(np.float32),
                      p + "linear_attn.norm.weight": (1.0 + 0.1 * rng.standard_normal(Dv)).astype(np.float32),
                      p + "linear_attn.out_proj.weight": n(H, Hv * Dv)})
            f32 |= {p + "linear_attn.dt_bias", p + "linear_attn.A_log"}  # stored in float32, as in Qwen3.6
        w.update({p + "mlp.gate.weight": n(E, H) * 4,
                  p + "mlp.experts.gate_up_proj": n(E, 2 * F, H), p + "mlp.experts.down_proj": n(E, H, F),
                  p + "mlp.shared_expert.gate_proj.weight": n(Fs, H), p + "mlp.shared_expert.up_proj.weight": n(Fs, H),
                  p + "mlp.shared_expert.down_proj.weight": n(H, Fs), p + "mlp.shared_expert_gate.weight": n(1, H)})
    names = sorted(w)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    wm = {}
    for fn, ns in shards.items():
        write_safetensors(os.path.join(d, fn), {k: w[k] for k in ns}, f32=f32)
        wm.update({k: fn for k in ns})
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
           "tie_word_embeddings": False, "text_config": t, "vision_config": {"depth": 1}}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    json.dump({"eos_token_id": [t["eos_token_id"], t["eos_token_id"] - 2]}, open(os.path.join(d, "generation_config.json"), "w"))
    return t


def _fake_quant(fmt):
    sys.path.insert(0, os.path.join(HERE, "..", "engine", "tools"))
    import packfmt as pf

    def q(m):
        rows, cols = m.shape
        return pf.decode(fmt, pf.encode(fmt, m), rows, cols)

    return q


class HFRef:
    """Float64 forward pass, one token at a time, straight from the HF checkpoint."""

    def __init__(self, d, dense_fmt=None, expert_fmt=None):
        self.st = SafeTensors(d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.t = t = cfg.get("text_config", cfg)
        self.types = t.get("layer_types") or layer_types(t)
        self.P = "model.language_model." if any(k.startswith("model.language_model.") for k in self.st.meta) else "model."
        self.dq = _fake_quant(dense_fmt) if dense_fmt else (lambda m: m)
        self.eq = _fake_quant(expert_fmt) if expert_fmt else (lambda m: m)
        self.cache = {}
        L = t["num_hidden_layers"]
        self.kc, self.vc = [[] for _ in range(L)], [[] for _ in range(L)]
        cd = 2 * t["linear_num_key_heads"] * t["linear_key_head_dim"] + t["linear_num_value_heads"] * t["linear_value_head_dim"]
        self.conv = [np.zeros((t["linear_conv_kernel_dim"] - 1, cd)) for _ in range(L)]
        self.S = [np.zeros((t["linear_num_value_heads"], t["linear_key_head_dim"], t["linear_value_head_dim"]))
                  for _ in range(L)]
        self.pos = 0

    def g(self, name, kind="plain"):
        key = (name, kind)
        if key not in self.cache:
            m = self.st.get(name).astype(np.float32)
            if kind == "dense":
                m = self.dq(m)
            self.cache[key] = m.astype(np.float64)
        return self.cache[key]

    def norm1p(self, x, w):  # Qwen3_5MoeRMSNorm: x_normed * (1 + weight)
        return x / np.sqrt(np.mean(x * x) + self.t["rms_norm_eps"]) * (1.0 + w)

    def forward(self, tok):
        t, P = self.t, self.P
        eps = t["rms_norm_eps"]
        h = self.g(P + "embed_tokens.weight")[tok].copy()
        for l, kind in enumerate(self.types):
            p = f"{P}layers.{l}."
            x = self.norm1p(h, self.g(p + "input_layernorm.weight"))
            h = h + (self.attention(l, p, x) if kind == "full_attention" else self.deltanet(l, p, x))
            x = self.norm1p(h, self.g(p + "post_attention_layernorm.weight"))
            h = h + self.moe(l, p, x)
        self.pos += 1
        x = self.norm1p(h, self.g(P + "norm.weight"))
        return self.g("lm_head.weight", "dense") @ x

    def attention(self, l, p, x):
        t = self.t
        nh, nkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        qg = (self.g(p + "self_attn.q_proj.weight", "dense") @ x).reshape(nh, 2 * hd)
        q, gate = qg[:, :hd], qg[:, hd:].reshape(-1)
        k = (self.g(p + "self_attn.k_proj.weight", "dense") @ x).reshape(nkv, hd)
        v = (self.g(p + "self_attn.v_proj.weight", "dense") @ x).reshape(nkv, hd)
        q = np.stack([self.norm1p(qh, self.g(p + "self_attn.q_norm.weight")) for qh in q])
        k = np.stack([self.norm1p(kh, self.g(p + "self_attn.k_norm.weight")) for kh in k])
        rot = int(hd * t.get("partial_rotary_factor", t.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)))
        theta = t.get("rope_theta") or t["rope_parameters"]["rope_theta"]
        inv = 1.0 / theta ** (np.arange(0, rot, 2) / rot)
        ang = self.pos * inv
        cos, sin = np.cos(np.concatenate([ang, ang])), np.sin(np.concatenate([ang, ang]))

        def rope(v):
            r, rest = v[:rot], v[rot:]
            rh = np.concatenate([-r[rot // 2:], r[: rot // 2]])  # rotate_half
            return np.concatenate([r * cos + rh * sin, rest])

        q = np.stack([rope(qh) for qh in q])
        k = np.stack([rope(kh) for kh in k])
        self.kc[l].append(k)
        self.vc[l].append(v)
        K_, V_ = np.stack(self.kc[l]), np.stack(self.vc[l])
        out = np.zeros((nh, hd))
        for i in range(nh):
            j = i // (nh // nkv)
            s = K_[:, j] @ q[i] / np.sqrt(hd)
            pr = np.exp(s - s.max())
            out[i] = (pr / pr.sum()) @ V_[:, j]
        out = out.reshape(-1) * (1.0 / (1.0 + np.exp(-gate)))
        return self.g(p + "self_attn.o_proj.weight", "dense") @ out

    def deltanet(self, l, p, x):
        t = self.t
        Hk, Hv, Dk, Dv = (t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                          t["linear_value_head_dim"])
        a_ = p + "linear_attn."
        mixed = self.g(a_ + "in_proj_qkv.weight", "dense") @ x
        z = (self.g(a_ + "in_proj_z.weight", "dense") @ x).reshape(Hv, Dv)
        b = self.g(a_ + "in_proj_b.weight") @ x
        a = self.g(a_ + "in_proj_a.weight") @ x
        w = self.g(a_ + "conv1d.weight")[:, 0, :]  # [conv_dim, K]
        seq = np.vstack([self.conv[l], mixed[None]])  # last K inputs, oldest first
        self.conv[l] = seq[1:]
        y = (w.T * seq).sum(0)
        y = y / (1.0 + np.exp(-y))  # silu
        kd = Hk * Dk
        q, k, v = y[:kd].reshape(Hk, Dk), y[kd:2 * kd].reshape(Hk, Dk), y[2 * kd:].reshape(Hv, Dv)
        l2 = lambda m: m / np.sqrt((m * m).sum(-1, keepdims=True) + 1e-6)
        q, k = l2(q) / np.sqrt(Dk), l2(k)
        q, k = np.repeat(q, Hv // Hk, axis=0), np.repeat(k, Hv // Hk, axis=0)  # repeat_interleave
        beta = 1.0 / (1.0 + np.exp(-b))
        sp = np.where(a + self.g(a_ + "dt_bias") > 20, a + self.g(a_ + "dt_bias"),
                      np.log1p(np.exp(a + self.g(a_ + "dt_bias"))))
        gdec = -np.exp(self.g(a_ + "A_log")) * sp
        S = self.S[l]
        o = np.zeros((Hv, Dv))
        for h in range(Hv):
            S[h] *= np.exp(gdec[h])
            kv_mem = S[h].T @ k[h]
            delta = (v[h] - kv_mem) * beta[h]
            S[h] += np.outer(k[h], delta)
            o[h] = S[h].T @ q[h]
        nw = self.g(a_ + "norm.weight")
        o = np.stack([oh / np.sqrt(np.mean(oh * oh) + t["rms_norm_eps"]) * nw for oh in o])
        o = o * (z / (1.0 + np.exp(-z)))
        return self.g(a_ + "out_proj.weight", "dense") @ o.reshape(-1)

    def moe(self, l, p, x):
        t = self.t
        E, K, F = t["num_experts"], t["num_experts_per_tok"], t["moe_intermediate_size"]
        r = self.g(p + "mlp.gate.weight") @ x
        pr = np.exp(r - r.max())
        pr /= pr.sum()
        top = sorted(range(E), key=lambda e: (-pr[e], e))[:K]
        wts = pr[top] / pr[top].sum()
        key = ("experts", l)
        if key not in self.cache:
            gu = self.st.get(p + "mlp.experts.gate_up_proj").astype(np.float32)
            dn = self.st.get(p + "mlp.experts.down_proj").astype(np.float32)
            self.cache[key] = [(self.eq(gu[e, :F]).astype(np.float64), self.eq(gu[e, F:]).astype(np.float64),
                                self.eq(dn[e]).astype(np.float64)) for e in range(E)]
        out = np.zeros_like(x)
        for e, we in zip(top, wts):
            wg, wu, wd = self.cache[key][e]
            gg, uu = wg @ x, wu @ x
            out += we * (wd @ (gg / (1 + np.exp(-gg)) * uu))
        sg = self.g(p + "mlp.shared_expert.gate_proj.weight", "dense") @ x
        su = self.g(p + "mlp.shared_expert.up_proj.weight", "dense") @ x
        ys = self.g(p + "mlp.shared_expert.down_proj.weight", "dense") @ (sg / (1 + np.exp(-sg)) * su)
        gate = 1.0 / (1.0 + np.exp(-(self.g(p + "mlp.shared_expert_gate.weight") @ x)[0]))
        return out + gate * ys
