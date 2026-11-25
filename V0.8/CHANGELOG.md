# V0.8 — changes beyond the v0.7 release

Tracked on git branch `v0.8` (commit `51b9296`). The actual code lives in its normal place
(`engine/`, `docs/`) — this file is the record of what changed and why, and what was
verified on this server.

## Fix: engine broke on models with no thinking mode (Qwen3-Coder-Instruct)

**Bug found while testing `Qwen3-Coder-30B-A3B-Instruct` on this server.** The engine
unconditionally inserted an empty `<think>\n\n</think>\n\n` block after the assistant tag
for every `--prompt`/`--serve` request that didn't pass `--think`. That's correct for base
Qwen3 (which toggles thinking on/off with that exact pattern), but Qwen3-Coder-Instruct's
own chat template has **no concept of a thinking block at all** — the sequence is
out-of-distribution for it. Result: the model emitted `<|endoftext|>` as its very first
generated token instead of answering.

**Diagnosis, in order:**
1. `--perplexity` self-check on a known text: **9.34**, a sane number — proved the pack
   conversion and the engine's core math were correct. The bug was isolated to prompt
   construction, not computation.
2. Checked the model's own `tokenizer_config.json` `chat_template` directly: zero
   references to `think`, confirming this model never uses the tag at all.
3. Manually built the prompt via `--tokenize` + `--prompt-ids`, skipping the forced
   think-block insertion: generation worked immediately and produced a coherent answer.

**Fix — detect per-model, not per-architecture:**
- `engine/tools/convert.py`: new `supports_thinking(model_dir)` reads the model's actual
  `chat_template` and checks for `<think>`. Writes a new pack cfg key `no_think`
  (0 = default/absent, so every existing pack keeps today's behavior unchanged).
- `engine/src/model.rs`: `Config.no_think: bool`, parsed from the pack.
- `engine/src/main.rs`: skip the empty think-block insertion when `no_think`; `--think`
  is refused outright for a `no_think` pack with a clear error, instead of silently
  producing broken output.
- `engine/src/server.rs` (`--serve`, the primary documented usage): identical fix in
  `render()`, for both the live turn and replayed assistant history.
- `docs/PACK_FORMAT.md`: documented the new key.

**Verified on this server (2026-11-25):**
- Rebuilt pack for `Qwen3-Coder-30B-A3B-Instruct` with `no_think=1` now set correctly.
- Plain `--prompt "Write a one-paragraph explanation of what a mixture-of-experts model is."`
  now generates the expected coherent answer automatically — no workaround needed.
- Same correctness as the earlier manual `--prompt-ids` workaround, confirming the fix
  doesn't change behavior for the thinking path, only skips the broken insertion.

## Hardware-tuning findings from this server's benchmarks (not yet code changes)

Measured on this box (Intel i5-1235U, hybrid 2 P-core+HT / 8 E-core, NVMe PCIe3 x4,
NVIDIA MX550 unused by this engine):

- **`--threads 4` beats the default (`--threads 12`, all logical cores) by ~12%**
  (37.3 vs 33.4 GMAC/s, measured via `--bench-kernels`). E-cores/HT add contention, not
  throughput, for this tight SIMD workload. Worth benchmarking per-machine rather than
  defaulting to `available_parallelism()`.
- **SSD thermal throttling is real on this hardware**: sustained 2MB/QD32 reads climbed
  the NVMe from 57°C to 80°C over six 20s runs, with throughput falling ~10% by the end.
  The default `--max-ssd-temp 70` guard is already crossed well before the drive
  stabilizes — don't push `--prefetch-width`/`--lookahead` higher on this box without
  re-testing sustained thermals, despite the ~2.5GB/s peak bandwidth available.
- **Real run observed CPU hit 93°C and the heat guard correctly paused 11.2s** — confirms
  `--max-cpu-temp` works as intended on real hardware, not just in the design doc.

## AVX2 kernels (added after this changelog's first version)

Added an `avx2` module in `engine/src/kernels.rs`, parallel to the existing `sse` one:
`matvec` (the main int8/q4b expert kernel), `dot_bf16`, `axpy_bf16`, `matvec_f32` (routers),
all at 256-bit width. Runtime-dispatched via `is_x86_feature_detected!("avx2")`, preferred
over SSE, SSE kept as the fallback for CPUs without it. One real simplification AVX2 gives
for free: a whole 32-byte QK block fits one register, so there's no more "two 16-byte
halves, add the partial sums" step SSE needed.

**Verified correct:** perplexity 9.328 (AVX2) vs 9.341 (SSE) on the same text — effectively
the same answer, the tiny difference being float summation order, same category of
non-determinism SSE already has versus the scalar fallback. Real `--prompt` generation
with `kernels: AVX2` confirmed in the log produced a coherent, correct answer end to end.

**Honest result — the speedup is small to negligible on this hardware, not the ~2x hoped
for**, and here's why, confirmed by hardware inspection, not guesswork: **`dmidecode`
shows only one of this laptop's two DIMM slots is populated — it runs single-channel
DDR4-3200, not dual-channel.** `--bench-kernels`'s working set (~170MB across 64 synthetic
experts) is far past L3 cache, so it's memory-bandwidth-bound: the CPU's ALUs can already
consume quantized weight data faster than single-channel RAM can supply it. Doubling SIMD
width doesn't help when the bottleneck is RAM bandwidth, not instruction throughput.
Measured: 36.1-36.9 GMAC/s (AVX2, threads 4-8) vs 33.4-37.3 GMAC/s (SSE, same range) —
within noise of each other.

**Takeaway for this machine specifically:** the single-channel RAM is a more fundamental
constraint on this workload than CPU SIMD width. On a dual-channel machine, these same
AVX2 kernels should show a real gap over SSE, since the bottleneck would shift back to
compute. Code correctness and the runtime-dispatch fallback chain (AVX2 → SSE → scalar)
are solid either way.

## Open, not yet done
- No GPU path exists in the engine at all. Investigated adding one and decided against it
  for now: the engine's own stats show the per-token bottleneck is SSD read *latency*
  (experts: 145ms/token, 91ms of that waiting on the SSD), not compute throughput, and the
  MX550's 2GB VRAM is smaller than the 3.28GB expert cache already held in RAM — a GPU
  can't reduce SSD reads if it can't even hold as much as RAM already does. Revisit only if
  the SSD-latency bottleneck itself is addressed first (faster drive, deeper cache).
