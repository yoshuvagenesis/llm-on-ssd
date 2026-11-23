# The `.moe` pack format

A pack is one file that holds a whole converted model. It is written by
`engine/tools/convert.py`, read by the engine (`engine/src/pack.rs`) and by the
independent reference (`engine/tools/ref_pack.py`). The shared definitions live in
`engine/tools/packfmt.py`; keep all three in sync when changing the format.

All integers are little-endian.

## File layout

| Bytes | Content |
|---|---|
| `0..8` | magic `MOEPACK1` |
| `8..16` | `u64` header length in bytes |
| `16..24` | `u64` data start (a multiple of 4096) |
| `24..` | header: UTF-8 text, one record per line |
| data start.. | data. Every offset in the header is relative to the data start. |

## Header records

```
cfg <key> <value>
t <name> <fmt> <rows> <cols> <offset> <nbytes>     a tensor
e <layer> <expert> <offset> <nbytes>               one expert block, 4096-aligned
vocab <offset> <nbytes>                            the vocabulary blob
```

### `cfg` keys

| Key | Meaning |
|---|---|
| `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` | Model shape |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | MoE shape (Qwen3-30B-A3B: 128, 8, 768) |
| `rms_norm_eps`, `rope_theta`, `vocab_size` | As in the model's `config.json` |
| `norm_topk_prob` | `1` if the top-k router probabilities are renormalized to sum to 1 |
| `eos` | Comma-separated end-of-sequence token ids (`-1` if none) |
| `expert_fmt` | `q4b` or `q4` (format of the expert blocks; `q4` if absent) |

Qwen3.5/3.6 packs add:

| Key | Meaning |
|---|---|
| `arch` | `qwen35` (absent or `qwen3` for Qwen3-MoE) |
| `layer_types` | One letter per layer: `L` Gated DeltaNet, `A` full attention (Qwen3.6-35B-A3B: `LLLA` × 10) |
| `rotary_dim` | Dims at the start of each head that get RoPE (head_dim × partial_rotary_factor; 64 of 256) |
| `lin_k_heads`, `lin_v_heads`, `lin_k_dim`, `lin_v_dim`, `conv_kernel` | DeltaNet sizes (16, 32, 128, 128, 4) |
| `shared_ffn` | Width of the shared expert (512) |
| `think_open` | `1`: the chat format opens `<think>\n` itself when thinking is on |

### Tensor names

| Name | Shape | Format |
|---|---|---|
| `embed` | vocab × hidden | bf16 (rows are read from the SSD per token, not kept in RAM) |
| `norm` | 1 × hidden | f32 |
| `lm_head` | vocab × hidden | q8 or q4b (`--dense-fmt`) |
| `L<n>.ln1`, `L<n>.ln2` | 1 × hidden | f32 (RMSNorm weights before attention and before the MoE) |
| `L<n>.qn`, `L<n>.kn` | 1 × head_dim | f32 (Qwen3 per-head q/k norms) |
| `L<n>.wq`, `L<n>.wk`, `L<n>.wv`, `L<n>.wo` | as in the model | q8 or q4b (`--dense-fmt`) |
| `L<n>.router` | experts × hidden | f32 |

Qwen3.5/3.6 packs store the zero-centred RMSNorm weights (`ln1`, `ln2`, `qn`, `kn`,
`norm`) as `1 + w`, so every norm is `x_normed * weight` in the engine. Attention layers
(`A`) have `qn`, `kn`, `wk`, `wv`, `wo` as above, and `wq` with 2 × heads × head_dim rows:
for each head, its query rows followed by its output-gate rows. DeltaNet layers (`L`) have
instead:

| Name | Shape | Format |
|---|---|---|
| `L<n>.lqkv` | (2 × k_heads × k_dim + v_heads × v_dim) × hidden | q8 or q4b (`in_proj_qkv`: q, k, v) |
| `L<n>.lz` | (v_heads × v_dim) × hidden | q8 or q4b (`in_proj_z`, output gate) |
| `L<n>.lba` | (2 × v_heads) × hidden | f32 (`in_proj_b` rows, then `in_proj_a` rows) |
| `L<n>.conv` | conv_dim × conv_kernel | f32 (depthwise causal convolution) |
| `L<n>.alog`, `L<n>.dtb` | 1 × v_heads | f32 (`A_log`, `dt_bias`) |
| `L<n>.lnorm` | 1 × v_dim | f32 (gated RMSNorm weight, used as is) |
| `L<n>.lout` | hidden × (v_heads × v_dim) | q8 or q4b (`out_proj`) |

Every Qwen3.5/3.6 layer also has the shared expert: `L<n>.sg`, `L<n>.su` (shared × hidden),
`L<n>.sd` (hidden × shared) in the dense format, and `L<n>.sgate` (1 × hidden, f32). Its
routed experts are stored like Qwen3's: the checkpoint's fused `gate_up_proj` is split
into gate (first half of the rows) and up (second half).

## Tensor formats

Matrices are row-major, and `cols` is divisible by 32. Quantized formats store all
values first, then all scales, one scale per 32 values of a row.

| Format | Values | Scales | Bits per weight |
|---|---|---|---|
| `f32` | float32 | none | 32 |
| `bf16` | bfloat16 (raw bits) | none | 16 |
| `q8` | int8, `value = q × scale` | float32 | 9 |
| `q4` | 4-bit, `value = q × scale`, q in −8..7 | float32 | 5 |
| `q4b` | like `q4` | bfloat16 | 4.5 |

4-bit packing, per row and per block of 32: byte `j` (0..15) holds
`(q[j] + 8) | (q[j + 16] + 8) << 4`. The converter picks the scale so that the value with
the largest magnitude maps to −8, which uses the full range (the Q4_0 idea from ggml).

## Expert blocks

Each expert is one block: `gate` [F × H], `up` [F × H] and `down` [H × F] in the
`expert_fmt` format, back to back, padded with zeros to a multiple of 4096 bytes. One
block arrives in one SSD read. For Qwen3-30B-A3B with `q4b`, a block is 2.65 MB and the
6,144 blocks take 16.3 GB.

## Vocabulary blob

For each token id in order: a `u32` byte length, then the token's raw bytes (the
byte-level BPE characters already mapped back to bytes).
