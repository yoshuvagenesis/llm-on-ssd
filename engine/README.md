# moe-engine v0.6: storage-first MoE inference

Runs Qwen3-MoE models (reference: Qwen3-30B-A3B) with the experts streamed from the
SSD. RAM holds only the attention weights, routers and output head (~1.4 GB), the KV
cache, and an LRU cache of experts. All of it stays under one hard cap, `--ram-gb`
(default 7): the expert cache gets what is left after the rest is reserved, and the
engine refuses to start if the plan does not fit.

## New in v0.6: chat server

- `--serve --tokenizer tokenizer.json`: one server with a **chat page** at `/` and an
  **OpenAI-compatible API** at `/v1/chat/completions` (streaming and not), `/v1/models`,
  plus `/api/stats`. Options: `--port` (8080), `--host` (0.0.0.0), `--password`,
  `--max-answer` (1024 tokens). `docker/serve.sh` starts it in the background.
- **Tokenizer inside the engine**, read from the model's own `tokenizer.json`: text in,
  text out, no Python step. `--prompt "text"` gives one answer in the terminal.
  `tools/test_tokenizer.py` checks it against Python's regex engine;
  `tools/verify_tokenizer.py` checks it against Hugging Face on the real file.
- **Conversations reuse the KV cache**: each request is compared with the tokens already
  in memory, and only the new part is read. Answers are remembered token by token, so
  continuing a chat does not re-read the previous answer.
- **Queue**: one answer at a time; waiting requests get their place in line and an
  estimated wait (as SSE comments `: queue <place> <seconds>`, which other clients ignore).
- **Sampling** with Qwen3's recommended settings (temperature 0.7, top-p 0.8, top-k 20;
  thinking 0.6 / 0.95 / 20), or greedy with `"temperature": 0`.
- **Thinking switch**: `"think": true` or `chat_template_kwargs.enable_thinking`; the
  reasoning arrives as `reasoning_content`, the page shows it folded.
- **Long chats**: when a conversation would pass `--max-context`, its oldest messages are
  left out (system messages are kept) and the client is told.
- `tools/test_server.py` tests all of this end to end on a tiny model.

## New in v0.5

- `--ram-gb G` hard cap (default 7). It reserves the KV cache for `--max-context N`
  (default 4096 tokens, ~98 KB per token for Qwen3-30B-A3B) and ~0.3 GB for buffers.
  Generation stops at the context limit; a prompt longer than it is refused.
  `--cache-gb` still sets the expert cache directly, but it is checked against the cap.
  Measured peak memory stays below the printed plan.
- `--refine` (opt-in): after each layer, re-predict the next layer from the newer hidden
  state, add what it now expects and withdraw queued guesses it no longer supports.
  Output is bit-identical with or without it (it changes only what is read). On random
  weights it was neutral; the release check measures it on the real model.
- Heat guard reads the CPU's own sensor (coretemp) instead of the board sensor (acpitz).
- `--lookahead` default back to 2 (3 was slightly worse on one drive).
- After reading a prompt, the experts of its last tokens are kept in the cache first.
- Exact least-recently-used order in the cache; new statistic "not predicted at all".

## How it works

- **Pack format** (`tools/packfmt.py`): every expert is one 4 KB-aligned block of
  2.65 MB (gate, up, down at 4 bits with bf16 scales, "q4b"), so it arrives in one SSD
  read. Older packs with 2.95 MB "q4" blocks still work.
  Attention and output head are 8-bit. Embedding rows are read per token from the SSD.
- **Reads** bypass the OS cache (O_DIRECT on Linux, F_NOCACHE on macOS), so RAM use stays
  within the cap, using 4 parallel reader threads. At most 3 of them read guesses, so a
  block needed now never waits behind many guesses on a slow link.
- **Cache = one fixed region of slots** (one expert block each), backed by 2 MB huge pages
  on Linux when available. Nothing is allocated per read, and RAM use is fixed: the
  `RAM:` line shows it. Even `--cache-gb 0` keeps ~40 slots (~0.1 GB) for this layer's
  experts and the guesses in flight.
- **Prefetch**: after each layer's attention, the next layer's router guesses 8 experts
  and the two after guess 4 each. All routers run in parallel (SIMD).
- **Prompt reading in batches** (`--prefill-batch`, default 512): each layer runs for all
  prompt tokens before the next, so every expert is read once per layer for the whole
  prompt, and the output layer runs only for the last token. Bit-identical to token by token.
- **Experts are computed as they arrive**, ready ones first.
- **KV cache in bfloat16** (Qwen3's native precision), SIMD attention: half the RAM.
- **Heat guard**: pauses between tokens if the CPU passes 90°C or an NVMe 70°C
  (`--max-cpu-temp`, `--max-ssd-temp`, 0 = off). The SSD is only ever read, never written.
- **Two read queues**: blocks the current layer needs are always read first. Guesses wait
  behind them and are dropped unread if their layer has already passed.
- **Cache eviction**: wrong guesses go first, then least recently used blocks. Guesses are
  protected until their layer runs, so prefetching works even with no cache.
- **Copies on other drives** (`--mirror`): reads are spread over identical pack copies.
- **Persistent worker threads**; experts are summed in a fixed order, so output is
  bit-identical for any number of threads.
- **Kernels**: SSE (SSSE3/SSE4.1) on x86-64, fine on the N5105. Plain Rust elsewhere,
  including Apple Silicon (correct, but slow until NEON kernels are added).

## Verified

`python tools/test_engine.py` builds a tiny random model, converts it, and checks:
1. The engine matches an independent numpy reference (`tools/ref_pack.py`). Results are
   exact until the first time an activation sits on an int8 rounding boundary, where
   float32 and float64 round differently. After that, tiny differences carry forward
   (max ~2%, same next token 95%).
2. Cache size, strict mode, prefetch on/off, I/O threads, compute threads, mirror copies
   and buffered/direct reads all give **bit-identical** output.
   Both formats (q4b, q4) are tested.

Full-size runs with random weights (`tools/make_bench_pack.py`) on a 2-core cloud VM
(not the N5105), 24 decoded tokens:

| version, cache | tok/s | SSD read per token |
|---|---|---|
| v0.1, q4, 3 GB | 1.73 | n/a |
| v0.2, q4, 3 GB | 2.33 | 315 MB |
| v0.2, q4b, 3 GB | 2.32 | 263 MB |
| v0.2, q4b, no cache | 1.76 | 1378 MB |
| v0.3, q4b, 3 GB | 2.74 | 103 MB |
| v0.3, q4b, no cache | 2.04 | 1156 MB |
| v0.4, q4b, 3 GB | 2.79 | 122 MB; prompt reading 5.9 tok/s (v0.3: ~1.4-1.9) |

Random weights route more randomly than a real model, so real reuse is better.

## Run it on the Mac (real model)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh      # once
cd llm-on-ssd/engine
cargo build --release
python tools/test_engine.py                                        # must end with ENGINE TEST PASSED

# convert (needs ~20 GB free disk, takes roughly 15-30 min)
python tools/convert.py --model ~/models/Qwen3-30B-A3B --out ~/models/qwen3-30b-a3b.moe

python tools/encode.py --model ~/models/Qwen3-30B-A3B \
    "What is the capital of France? Answer in one sentence." > prompt.txt
./target/release/moe-engine --pack ~/models/qwen3-30b-a3b.moe --prompt-file prompt.txt \
    --max-new 40
```
On the Mac this is a **quality check**: real text should come out. It will be slow,
because there are no NEON kernels yet.

## Run it on the NAS (container, speed test)

```bash
docker build -t moe-bench -f docker/Dockerfile .
docker run --rm --security-opt seccomp=unconfined -e MODE=engine \
    -v /path/on/the/nvme/moe-bench:/bench moe-bench
```
Writes a 20 GB random-weight pack with the real model's shape (one time), then runs the
engine with a 4 GB cache and with none. Output text is meaningless, but compute and SSD
work per token match the real model. Routing is more random than real, so cache numbers
are pessimistic. To run the real model, copy the converted `.moe` file to the NAS and
pass `--pack` instead.

## Options

`--ram-gb G` (default 7, hard cap) · `--max-context N` (4096) · `--cache-gb G` (default: what
the cap leaves; 0 = strict) · `--refine` · `--threads N` · `--io-threads N` (4) ·
`--max-guess-reads N` (io-threads - 1) · `--lookahead N` (2) · `--prefetch-width N` (8, 0 = off) ·
`--prefetch-width2 N` (4) · `--mirror path1,path2` · `--max-new N` · `--buffered` ·
`--dump-logits f` / `--out-ids f` (verification) · `--quiet` ·
`--bench-kernels [--threads N]` (time the expert kernels alone, no model needed)

Converter: `--expert-fmt q4b|q4` (default q4b) · `--dense-fmt q8|q4b` (default q8; q4b makes
attention and the output head half the size and faster, measured ~+2.7% perplexity).

## Not done yet (planned, in rough order)

1. Save a conversation's KV cache to the SSD, so switching between people is quick
2. NFC text normalization in the tokenizer (only matters for decomposed Unicode input)
3. Cache policy tuned on real routing traces (segmented LRU, per-layer budgets)
4. Faster multi-threaded kernels (4 threads reach ~78% of the standalone benchmark)
5. NEON kernels for ARM / Apple Silicon; io_uring reads
6. More model families (other MoE architectures)
