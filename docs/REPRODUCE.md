# Reproduce the results

This guide repeats the benchmark in [RESULTS.md](RESULTS.md) on your own machine. Any
x86-64 Linux box with an NVMe SSD works; the closer it is to the N5105 setup, the more
directly the numbers compare.

## What you need

- **Test machine**: x86-64 Linux with SSSE3 and SSE4.1 (any x86-64 CPU from roughly the
  last 10 years), 8 GB of RAM or more, an NVMe SSD with about 40 GB free for both packs,
  Docker (or Rust 1.78+ and Python 3 with numpy), and a root shell (only needed to drop the
  file cache between runs).
- **Conversion machine** (can be the same one, any OS): Python 3 with `numpy`,
  `tokenizers` and `huggingface_hub`, and about 100 GB of free disk (61 GB for the
  original model and about 18 GB per pack). A 16 GB RAM laptop is enough, because the
  converter works one tensor at a time.

```bash
git clone https://github.com/yoshuvagenesis/llm-on-ssd && cd llm-on-ssd
python3 -m pip install numpy tokenizers huggingface_hub
```

## 1. Download the model

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ~/models/Qwen3-30B-A3B
```

The model is Apache-2.0. Downloads resume if they are interrupted; run the command again.

## 2. Convert it into two packs

```bash
# best quality: 4.5-bit experts, 8-bit attention and output head
python3 engine/tools/convert.py --model ~/models/Qwen3-30B-A3B --out ~/models/qwen3-30b-a3b.moe

# fastest: 4.5-bit experts, 4.5-bit attention and output head
python3 engine/tools/convert.py --model ~/models/Qwen3-30B-A3B --out ~/models/qwen3-q4b-d4.moe \
    --dense-fmt q4b
```

Each takes roughly 15–30 minutes, depending on the disk.

## 3. Make the benchmark prompt

```bash
python3 engine/tools/make_prompt.py --tokenizer ~/models/Qwen3-30B-A3B/tokenizer.json \
    --text trace/corpus.txt --tokens 512 > ~/models/prompt512.txt
```

For your own questions, use `engine/tools/encode.py --model ~/models/Qwen3-30B-A3B "your question"`.

## 4. Copy everything to the test machine's SSD

Put the two packs and `prompt512.txt` in one folder **on the NVMe SSD**, not on a hard
disk or a network share, for example `/ssd/moe`:

```bash
rsync -av --progress ~/models/qwen3-30b-a3b.moe ~/models/qwen3-q4b-d4.moe ~/models/prompt512.txt \
    user@test-machine:/ssd/moe/
```

Optional, to test two drives: put a second copy of `qwen3-30b-a3b.moe` on another SSD,
for example `/ssd2/moe`.

## 5. Build and check

On the test machine, in the repository folder:

```bash
docker build -t moe-bench -f docker/Dockerfile .
docker run --rm --entrypoint sh moe-bench -c "cd /kit/engine && python3 tools/test_engine.py /usr/local/bin/moe-engine"
```

The test must end with `ENGINE TEST PASSED`. It checks the engine against an independent
implementation and confirms that every setting gives bit-identical output on your CPU.

## 6. Run the benchmark

```bash
MODELS=/ssd/moe sh docker/release_check.sh
# with a second drive:
MODELS=/ssd/moe MIRROR=/ssd2/moe sh docker/release_check.sh
```

It runs the self-test, then each setup with the 512-token prompt and 128 generated
tokens, in a container limited to 7 GB. The results are saved to
`/ssd/moe/release_check_results.txt`.

How to read the output lines:

| Line | Meaning |
|---|---|
| `RAM (cap 7.0 GB): ...` | The memory plan. The expert cache gets what the cap leaves. |
| `prefill: ... decode: ...` | Speed of prompt reading and of generation |
| `decode per token: SSD read ...` | Data read per generated token. Lower is better. |
| `decode time per token: ...` | Where the time goes, including time spent waiting for the SSD |
| `expert blocks used: ...` | Share of experts found in the cache, guessed in time, still being read, or not predicted |
| `heat: ...` | Peak CPU and SSD temperatures and time paused by the heat guard |
| `SSD: ...` | Reads, data volume, and how many guesses were used |

## 7. Without Docker

```bash
cd engine
cargo build --release
python3 tools/test_engine.py              # must end with ENGINE TEST PASSED
./target/release/moe-engine --pack /ssd/moe/qwen3-q4b-d4.moe --prompt-file /ssd/moe/prompt512.txt \
    --max-new 128 --ram-gb 7
```

`--ram-gb` limits the engine's own allocations. A container limit (or a cgroup) also
makes sure nothing else on the machine counts against it.

## 8. Optional: the research measurements

These were run on a Mac (Apple M2, 16 GB). They read the original model, not a pack.

```bash
cd trace
python3 selftest.py                                                    # must print SELFTEST PASSED
python3 trace_routing.py --model ~/models/Qwen3-30B-A3B --text corpus.txt   # routing trace + perplexity
python3 quant_sweep.py --model ~/models/Qwen3-30B-A3B --text corpus.txt     # quality of each number format
python3 analyze_trace.py trace.npz                                     # cache and prediction analysis
```

On the test machine, `docker run --rm -e MODE=cpu moe-bench` times the kernel formats.
`docker run --rm --security-opt seccomp=unconfined -e MODE=ssd -v /ssd/bench:/bench moe-bench`
measures the SSD; it writes a 16 GB test file and deletes it afterwards.

## Sharing your results

Please include: the CPU model, the RAM size and the cap you used, the SSD and its link
(PCIe generation and lanes), and the complete `release_check_results.txt`.
