# How it works, and how we got here

The goal: **considerable speed for a ~30B MoE model on the cheapest hardware people
already own, without overheating it.** The concrete targets were an 8 GB RAM machine with
one SSD, speed well above existing methods (llama.cpp), and minimal quality loss. The
rule: get the speed from software first, and reach for hardware only when nothing else
works.

Every design choice below came from a measurement. The tools for those measurements are
in this repository, so they can be repeated on other hardware.

## 1. Measure the limits first

| Measured on the N5105 | Result | Consequence |
|---|---|---|
| SSD reads (`ssd/bench_ssd.sh`) | 0.89 GB/s per drive: the NAS gives each NVMe slot one PCIe 3.0 lane | Bytes read per token decide the speed. Read as little as possible, in large blocks. |
| CPU math formats (`kernel/`) | No AVX2, but SSSE3 multiplies int8 pairs quickly (`pmaddubsw`) | 4-bit weights × int8 activations with SSE. Table lookup and ternary formats were tested too: ternary needs a model trained for it, and 2-bit lookup costs too much quality. |
| Quality of formats (`trace/quant_sweep.py`, on a Mac) | 4-bit experts +3.2% perplexity, 3-bit +5.4%, 2-bit +36.8%; ternary unusable without retraining | 4.5-bit experts (q4b); 8-bit attention by default, 4-bit as the fast option (about +2.7% more). |
| Which experts tokens use (`trace/trace_routing.py`, `trace/analyze_trace.py`) | Expert use is uneven and repeats across nearby tokens; the next layers' choices can be guessed from the current hidden state | An LRU cache works, and prefetching is worth it. |

Per generated token, Qwen3-30B-A3B uses 8 of 128 experts in each of its 48 layers:
384 blocks of 2.65 MB, about 1 GB. At 0.89 GB/s that would be over a second per token if
nothing were cached. With the cache, v0.5 reads 180 MB per token.

## 2. The pack: one read per expert

`convert.py` writes the model into one file (format: [PACK_FORMAT.md](PACK_FORMAT.md)).
Each expert's three matrices are stored together in one 4 KB-aligned block, so an expert
arrives in a single SSD read. Reads use direct I/O (`O_DIRECT` on Linux, `F_NOCACHE` on
macOS). The OS file cache would otherwise hold a second copy of everything and break the
RAM cap. The embedding table stays on the SSD; one row (4 KB) is read per token.

## 3. RAM: one hard cap

`--ram-gb` (default 7) covers everything the engine allocates:

```
attention + routers + output head   (0.7–1.4 GB, always in RAM)
+ KV cache reserved for --max-context tokens (bf16, ~98 KB per token for Qwen3-30B-A3B)
+ ~0.3 GB buffers, vocabulary, threads
+ expert cache = whatever is left
```

The engine prints this plan and refuses to start if it does not fit. The expert cache is
one fixed region of slots, each exactly one expert block, backed by 2 MB huge pages
when the system allows it. Nothing is allocated per read, so memory use cannot creep up.

The cache drops the least recently used block, except that wrong guesses go first and
guesses for a layer that hasn't run yet are protected. After a prompt is read, the
experts used by its last tokens are marked most recent, because generation starts
with them.

## 4. Guessing the next experts

After layer *l*'s attention, the engine runs the routers of layers *l+1* and *l+2* on the
current hidden state. They usually pick almost the same experts as they will once those
layers actually run. The top 8 guesses for the next layer and the top 4 for the layer
after are queued for reading while layer *l* computes.

- **Two queues.** Blocks the current layer needs are always read first. Guesses use at
  most 3 of the 4 reader threads, so a needed block never waits behind many guesses.
- **Late guesses are dropped.** A guess whose layer has already passed is never read.
- **Experts are computed in arrival order**: those already in RAM first.

**Why not 100% prediction?** Which experts layer *l+1* uses depends on layer *l*'s
output, which does not exist until layer *l* is done. Any guess made early enough to hide
the read can be wrong. What can be done is to make misses rare and cheap. Measured in
v0.5: 3.2% of needed experts were not predicted at all.

We also tried a **second, later guess** (`--refine`): after layer *l* finishes, re-run
the next router on the final hidden state, add new guesses and withdraw queued ones it
no longer supports. Unpredicted experts fell from 3.2% to 2.1%, but extra reads rose
from 180 to 195 MB per token, and generation got slower (2.08 vs 2.75 tok/s). So it is
off by default.

We tried guessing 3 layers ahead. On one drive, it was slightly worse than 2: more
wrong guesses compete for the same narrow link.

## 5. Reading the prompt

A prompt is processed layer by layer for all its tokens at once (`--prefill-batch`,
default 512). Each expert is then read at most once per layer for the whole prompt
instead of once per token. That took prompt reading on the N5105 to 4.1 tok/s, faster
than llama.cpp on the same box (2.69 tok/s). The result is bit-identical to processing
the prompt token by token.

## 6. Compute

- **Kernels**: int8 activations (one scale per 32 values) times 4-bit or 8-bit weights,
  using SSSE3/SSE4.1. On one thread inside the engine they run within 1% of the
  standalone benchmark (`kernel/`), thanks to the huge-page slot arena. On four threads
  they reach about 78% of it.
- **Threads**: a persistent pool. Experts are summed in a fixed order, so the output is
  bit-identical for any number of threads.
- **Attention**: KV cache in bfloat16, Qwen3's native precision, with SIMD dot products.
  It uses half the RAM of float32.

## 7. Two SSDs

`--mirror` spreads reads across identical copies of the pack on several drives. On the
N5105, two drives raised generation with the 8-bit attention pack from 2.14 to 2.38
tok/s. The gain is smaller than the doubled bandwidth suggests, because on this CPU,
handling more I/O takes compute time away from attention.

## 8. What did not help (on this hardware)

| Idea | Finding |
|---|---|
| Speculative decoding (draft model) | Checking 2 tokens at once makes the expert kernels only ~25% more efficient on this CPU, and more tokens need more distinct experts. Not worth it here. |
| KV cache on the SSD | On a PCIe x1 link it would compete with expert reads. It stays in RAM, sized by `--max-context`. |
| Skipping unimportant neurons inside experts | Saves part of the compute, but the SSD is the limit, and the whole block is read anyway. It would need a different storage layout. |
| Guessing 3 layers ahead, `--refine` | Slightly or clearly slower on one drive (see section 4). |

## 9. Correctness

`engine/tools/test_engine.py` builds a tiny random model, converts it in three formats and
checks the following:

1. The engine matches an independent numpy implementation (`engine/tools/ref_pack.py`).
   It is exact until an activation lands on an int8 rounding boundary, where float32 and
   float64 round differently. After that, tiny differences carry forward, and the
   predicted tokens still agree.
2. Thirteen settings give **bit-identical** output: cache sizes from none to large; 1–4
   compute threads; 1–4 reader threads; prefetching off, 1 or 3 layers ahead; direct or
   buffered reads; a mirror copy; prompt in one batch, batches of 4, or token by token;
   `--refine`; and a tight RAM cap.

## 10. Heat

On the N5105, CPU temperature under load is reported by `coretemp`. The board sensor
(`acpitz`) barely moves, and an early version mistakenly read it. The engine reads the
CPU's own sensor and the NVMe sensors every half second. Between tokens, it pauses while
the CPU is above `--max-cpu-temp` (default 90 °C) or an SSD is above `--max-ssd-temp`
(70 °C), until they are 5 °C lower again. Under full load the N5105 reached 85–93 °C,
within its specification. The SSDs stayed below 41 °C because they are only read.
