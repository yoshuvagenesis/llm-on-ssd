# Results

All numbers below were measured by us. The raw output of the release checks is in
[results/nas_n5105_v0.5_release_check.txt](results/nas_n5105_v0.5_release_check.txt) and
[results/nas_n5105_v0.7_release_check.txt](results/nas_n5105_v0.7_release_check.txt).

## Test machine

| Part | Details |
|---|---|
| CPU | Intel Celeron N5105: 4 cores and 4 threads, up to 2.9 GHz, 10 W. Has SSE4.2 but **no AVX, AVX2, FMA or F16C**. |
| RAM | 12 GB installed. The engine was held to **7 GB** by the container (`--memory 7g --memory-swap 7g`) and by `--ram-gb 7`. |
| Storage | Two NVMe SSDs (ADATA LEGEND 900), **each on a PCIe 3.0 x1 link**. Measured with large reads: **0.89 GB/s per drive** (the link, not the drive, is the limit). |
| System | A NAS running its vendor's Linux, with Docker. The engine ran in the container built from `docker/Dockerfile`. |

## Model and packs

Model: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), converted with
`engine/tools/convert.py`.

| Pack | Experts | Attention and output head | Kept in RAM |
|---|---|---|---|
| `qwen3-30b-a3b.moe` (best quality) | q4b, 4.5 bits/weight | q8, 8-bit | 1.42 GB |
| `qwen3-q4b-d4.moe` (fastest) | q4b, 4.5 bits/weight | q4b, 4.5 bits/weight | 0.73 GB |

Both packs have identical experts: 6,144 blocks of 2.65 MB.

## Benchmark method

- Prompt: 512 tokens. `engine/tools/make_prompt.py` makes an equivalent prompt from
  `trace/corpus.txt`.
- 128 generated tokens, greedy decoding.
- The OS file cache was dropped before every run. The engine reads with direct I/O, so
  every run starts with an empty cache.
- Run with `docker/release_check.sh`, one run per row.

How much is read per token depends a little on the text being generated (about ±10%
between prompts), so expect small differences with your own prompt.

## v0.5 on the N5105

| Setup | Prompt reading | Generation | SSD read per token | Experts from RAM cache |
|---|---|---|---|---|
| **4-bit attention pack, 1 SSD** | **4.11 tok/s** | **2.75 tok/s** | 180 MB | 79.0% |
| 4-bit attention pack, 1 SSD, `--refine` | 3.86 tok/s | 2.08 tok/s | 195 MB | 79.0% |
| 8-bit attention pack, 1 SSD | 3.33 tok/s | 2.14 tok/s | 283 MB | 73.5% |
| 8-bit attention pack, 2 SSDs (`--mirror`) | 3.35 tok/s | 2.38 tok/s | 283 MB | 73.5% |

RAM plan printed by the engine for the 4-bit attention pack (all under the 7.0 GB cap):
attention/output 0.73 GB + expert cache 5.56 GB + KV cache for 4,096 tokens 0.40 GB +
other 0.30 GB. Measured peak memory stays below the printed plan, because the KV
reservation fills only as the conversation grows. In a test with a 4.5 GB plan, the peak
resident memory was 3.79 GB.

Where the time goes per generated token (4-bit attention pack, 1 SSD): attention 113 ms,
experts 197 ms (of which about 96 ms waiting for the SSD), routers 23 ms, output head
20 ms. Of the experts needed, 79.0% were already in RAM, 14.6% had been guessed and had
arrived in time, 6.4% had been guessed but were still being read, and 3.2% had not been
predicted at all.

Heat: CPU peaks of 85–93 °C, SSDs at most 40 °C. The heat guard paused for 1 s in the
fastest run and 20 s in the `--refine` run.

## v0.7 on the N5105: Qwen3.6-35B-A3B

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), converted with
`engine/tools/convert.py` (experts q4b, everything else 8-bit; 10,240 expert blocks of
1.77 MB, 21.4 GB in total). Same machine, same 7 GB cap, same method; the 512-token
prompt is built by the engine (`--bench-prompt 512 --bench-text trace/corpus.txt`).
Qwen3 rows are from the same release check, for comparison.

| Model and pack | Prompt reading | Generation | SSD read per token | Experts from RAM cache | Peak CPU |
|---|---|---|---|---|---|
| Qwen3-30B-A3B, 4-bit attention | 2.96 tok/s | **2.69 tok/s** | 180 MB | 79.0% | 88 °C |
| Qwen3-30B-A3B, 8-bit attention | 3.35 tok/s | 2.17 tok/s | 283 MB | 73.5% | 88 °C |
| **Qwen3.6-35B-A3B, 8-bit dense** | **3.09 tok/s** | **2.39 tok/s** | **136 MB** | 70.6% | **85 °C** |

RAM plan for Qwen3.6 (all under the 7.0 GB cap): dense weights 2.26 GB + expert cache
3.88 GB + conversation memory for 16,384 tokens 0.47 GB + other 0.40 GB.

Where the time goes per generated token for Qwen3.6: attention and DeltaNet layers
191 ms, experts 136 ms (of which about 66 ms waiting for the SSD), routers 33 ms, output
head 57 ms. Unlike Qwen3, the SSD is no longer the main cost: most of the time is spent
reading the always-in-RAM weights (about 1.3 GB per token for the layers and 0.5 GB for
the 248k-word output head), so a 4-bit dense pack is the obvious next measurement.

Quality: perplexity on the first 1,024 tokens of `trace/corpus.txt` (lower is better;
each model uses its own tokenizer, so the numbers compare packs of the same model, not
the two models):

| Pack | Perplexity |
|---|---|
| Qwen3-30B-A3B, 8-bit attention | 7.372 |
| Qwen3-30B-A3B, 4-bit attention | 7.878 (+6.9%) |
| Qwen3.6-35B-A3B, 8-bit dense | 3.246 (first 512 tokens, measured on the Mac) |

## Real chat use (v0.6 chat server)

The numbers above are from a controlled benchmark that starts with an empty cache. These
come from the chat server (`--serve`, 4-bit attention pack, 1 SSD, 7 GB cap) during the
first real conversations on the N5105, read from `/api/stats`. They are a few answers, not
a controlled benchmark: speed depends on the topic, the answer length and how long the
conversation already is.

| Answer | Conversation before it | New tokens read | Answer length | Writing | Experts from RAM | SSD read per token |
|---|---|---|---|---|---|---|
| Short chat | short | – | – | **3.49 tok/s** | 91.1% | 122 MB |
| Follow-up, long answer (measured while writing) | 190 tokens, reused | 44 | 668 tokens so far | 2.52 tok/s | – | – |
| Follow-up in a long chat | 1,257 tokens, reused | 20 | 512 tokens | 2.10 tok/s | 86.3% | 181 MB |

Over the first four answers, **90.5%** of the experts needed came from the RAM cache.
New messages were read at 3.3–5.2 tok/s. Continuing a conversation read only the new
message: 1,257 earlier tokens were reused instead of read again. The CPU was at 65–79 °C
when checked, with no heat pause in progress; the SSDs stayed at about 35 °C.

Why the range: in a real chat the same experts keep coming back, so the cache serves more
of them than in the cold-start benchmark, which makes short answers faster (3.49 vs 2.75
tok/s). Long answers in long conversations are slower, because every new token attends
to everything before it, so attention takes longer as the conversation grows.

## Baseline: llama.cpp on the same machine

| Run | Memory | Prompt reading | Generation |
|---|---|---|---|
| **llm-on-ssd v0.5**, 4-bit attention pack | 7 GB limit | **4.11 tok/s** | **2.75 tok/s** |
| **llama.cpp**, Q4_K_M | **7 GB limit (same)** | 2.69 tok/s | 0.57 tok/s |
| llama.cpp, Q4_K_M | 6 GB limit | 2.69 tok/s | 0.43 tok/s |
| llama.cpp, Q4_K_M | no limit (all 12 GB) | 2.62 tok/s | 0.94 tok/s |

Under the same 7 GB limit, llm-on-ssd reads prompts **1.5×** and generates **4.8×**
faster. Even when llama.cpp may use all 12 GB of the machine, llm-on-ssd with 7 GB
generates **2.9×** faster. Q4_K_M keeps somewhat more precision than our fastest pack;
our best-quality pack (8-bit attention) is closer to it and generates at 2.14 tok/s,
**3.8×** llama.cpp under the same limit.

Conditions: same NAS, same model. llama.cpp build `8d60873`, built in Docker with
`-DGGML_NATIVE=ON`. Model file `Qwen3-30B-A3B-Q4_K_M.gguf` (17.28 GiB, about 4.9 bits per
weight, slightly more than our 4.5). Command:
`llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -p 512 -n 128 -t 4 -r 1`, each run in a
container with the memory limit shown. Raw output:
[results/llama_cpp_8d60873_n5105.txt](results/llama_cpp_8d60873_n5105.txt).

One difference in method: `llama-bench` measures generation (`tg128`) without a prompt
before it, while our numbers are for generation after the 512-token prompt. The longer
context makes our generation slightly harder, not easier.

By default llama.cpp memory-maps the weights and relies on the OS file cache. When the
model does not fit in RAM, the OS has to keep dropping and re-reading pages, which is
the likely reason its generation speed stays far below what the SSD could deliver.

## Quality

- **Correctness of the engine.** The test suite (`engine/tools/test_engine.py`) compares
  the engine with an independent numpy implementation of the same model. It then checks
  that 13 different cache, thread, drive, read-mode and prediction settings give
  **bit-identical** output, for three pack formats. It passes on the N5105.
- **Real text.** The converted Qwen3-30B-A3B answers normally, e.g. "The capital of
  France is Paris." to the prompt in the quick start.
- **Tokenizer.** The engine's built-in tokenizer gives exactly the same tokens as Hugging
  Face's on Qwen3-30B-A3B's own `tokenizer.json`: 24 of 24 test texts, 11,651 tokens, in
  English, code, chat markup, 10 Indian scripts, Chinese, Japanese, Korean, Arabic, Hebrew,
  Thai, Greek, Cyrillic, emoji and unusual spacing (`engine/tools/verify_tokenizer.py`).
- **Cost of compression**, measured as perplexity on `trace/corpus.txt` (lower is
  better; full precision is 6.011):

| Experts stored as | Perplexity | vs full precision |
|---|---|---|
| bf16 (reference) | 6.011 | +0.0% |
| 8-bit | 6.042 | +0.5% |
| **4-bit (used by the packs)** | **6.205** | **+3.2%** |
| 4-bit, lookup kernel with 6-bit activations | 6.259 | +4.1% |
| 3-bit | 6.337 | +5.4% |
| 2-bit | 8.223 | +36.8% |
| ternary (−1/0/+1), no retraining | 6661 | unusable |

Measured with `trace/quant_sweep.py`, with attention at full precision. The packs use
the 4-bit format with bfloat16 scales (q4b), which rounds the same way. Switching
attention and the output head from 8-bit to 4-bit adds about **+2.7%** more. The rounding
here is the simple kind; better quantizers lose less, which is a planned improvement.

## Development history

On the N5105, generation speed with the 4-bit attention pack and one SSD:

| Version | Main change | Generation |
|---|---|---|
| v0.3 | First full test with the real model on the N5105 | 2.19 tok/s |
| v0.4 | Batched prompt reading, bf16 KV cache (6 GB limit) | 2.25–2.37 tok/s |
| v0.5 | 7 GB hard cap with automatic cache sizing, lookahead 2, cache warm-up | 2.75 tok/s |

Earlier versions were tuned on a 2-core cloud VM with a random-weight pack of the real
shape (`engine/tools/make_bench_pack.py`). Those numbers are in
[engine/README.md](../engine/README.md) and are not comparable with the N5105.
