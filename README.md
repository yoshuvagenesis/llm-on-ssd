# llm-on-ssd

**Run a 30B mixture-of-experts LLM from an SSD, on a cheap CPU, in 7 GB of RAM. No GPU.**

`llm-on-ssd` is a small inference engine that keeps a mixture-of-experts (MoE) model on
an NVMe SSD and streams each expert into RAM only when it is needed. It is built for the
hardware ordinary people already have: a low-power quad-core CPU without AVX2, one SSD,
and 8 GB of RAM.

Supported models: **Qwen3.6-35B-A3B** (and Qwen3.5-35B-A3B), **Qwen3-30B-A3B** and
**Qwen3-Coder-30B-A3B**. Reference setup: Qwen3-30B-A3B (30.5B parameters, 3.3B active
per token) on an **Intel Celeron N5105** NAS (4 cores, no AVX/AVX2/FMA, NVMe drives on
PCIe 3.0 x1).

| On the same N5105 box, 512-token prompt, 128 generated tokens | RAM | Prompt reading | Generation |
|---|---|---|---|
| **llm-on-ssd v0.5**, 4-bit attention pack, 1 SSD | 7 GB cap | **4.11 tok/s** | **2.75 tok/s** |
| llm-on-ssd v0.5, 8-bit attention pack (best quality), 1 SSD | 7 GB cap | 3.33 tok/s | 2.14 tok/s |
| llm-on-ssd v0.5, 8-bit attention pack, 2 SSDs (`--mirror`) | 7 GB cap | 3.35 tok/s | 2.38 tok/s |
| llama.cpp, Q4_K_M | 7 GB limit | 2.69 tok/s | 0.57 tok/s |
| llama.cpp, Q4_K_M | all 12 GB | 2.62 tok/s | 0.94 tok/s |

Under the same 7 GB limit, generation is **4.8× faster** than llama.cpp, and still
**2.9× faster** than llama.cpp using all 12 GB of the machine. In real chats on the chat
server, answers were written at **2.1–3.5 tok/s**, with about 90% of the experts served
from RAM. Full numbers, conditions and the raw log are in [docs/RESULTS.md](docs/RESULTS.md).

It comes with a **chat page and an OpenAI-compatible API**: start it once on the
machine with the SSD, then chat from any phone or computer on your home network.

> **Status: research prototype.** It works and it is tested, but it supports two model
> families (Qwen3-MoE and Qwen3.5/3.6-MoE) and has fast kernels only for x86-64. See
> [Limitations](#limitations).

## Why this works

In an MoE model, each token uses only 8 of the 128 experts in each layer. So per token,
the engine needs about 1 GB of the 16 GB of expert weights, and most of that is already
in RAM from earlier tokens. The engine:

- **Keeps only the small, always-used parts in RAM**: attention, routers, output head
  (0.7–1.4 GB) and the conversation's KV cache.
- **Caches experts in a fixed RAM region** under one hard cap (`--ram-gb`, default 7).
  It works out the cache size itself and refuses to start if the plan does not fit.
- **Guesses the next layers' experts early** by running their routers on the current
  hidden state, and reads them from the SSD while it computes. Experts needed right now
  always jump ahead of guesses.
- **Reads whole experts in one go**: every expert is one 4 KB-aligned 2.65 MB block,
  read with direct I/O that bypasses the OS cache, so RAM use stays fixed.
- **Reads the prompt layer by layer** for all tokens at once, so each expert is read
  once per layer for the whole prompt.
- **Computes with int8 SSE kernels** that need only SSSE3/SSE4.1, which every x86-64
  CPU from the last decade has.
- **Protects the hardware**: it pauses between tokens if the CPU passes 90 °C or an
  SSD passes 70 °C. While running a model it only reads from the SSD, so it causes no
  write wear.

Output is **bit-identical** for any cache size, thread count, number of drives, read mode
or prediction setting. These only change speed, never the answer. The test suite checks
this. See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the design and the
measurements behind every choice.

## Quick start

You need: an x86-64 Linux machine with an NVMe SSD (about 20 GB free per model pack),
8 GB of RAM or more, and Docker (or Rust 1.78+). Converting the model needs Python 3 with
numpy and about 80 GB of free disk once; any OS works for that step.

```bash
git clone https://github.com/yoshuvagenesis/llm-on-ssd && cd llm-on-ssd

# 1. Get the model (Apache-2.0, about 61 GB) and convert it to a pack (about 18 GB)
pip install numpy huggingface_hub
hf download Qwen/Qwen3-30B-A3B --local-dir ~/models/Qwen3-30B-A3B
python engine/tools/convert.py --model ~/models/Qwen3-30B-A3B --out /ssd/qwen3-q4b-d4.moe --dense-fmt q4b
cp ~/models/Qwen3-30B-A3B/tokenizer.json /ssd/

# 2. Build and start the chat server (runs in the background, restarts after a reboot)
docker build -t moe-bench -f docker/Dockerfile .
MODELS=/ssd sh docker/serve.sh
```

For **Qwen3.6-35B-A3B** (newer and stronger; its download is about 70 GB and its pack about
21 GB), download `Qwen/Qwen3.6-35B-A3B` the same way, convert it to
`/ssd/qwen3.6-35b-a3b.moe`, copy its `tokenizer.json` to `/ssd/tokenizer-qwen3.6.json`
(it differs from Qwen3's), and start the server with
`MODELS=/ssd PACK=qwen3.6-35b-a3b.moe TOKENIZER=tokenizer-qwen3.6.json sh docker/serve.sh`.

Then open `http://<the machine's IP address>:8080` in a browser on any device in your
home network. Answers appear word by word. A panel shows the queue, speed, temperatures,
memory and cache use. Use `--dense-fmt q8` and `PACK=qwen3-30b-a3b.moe` for the
best-quality pack.

**API.** The same port serves an OpenAI-compatible API, so existing chat apps can use it:
base URL `http://<address>:8080/v1`, any API key (or the password, if set), model name as
shown at `/v1/models`. `"think": true` (or `chat_template_kwargs.enable_thinking`) lets
the model reason first; the reasoning arrives as `reasoning_content`.

**One answer in the terminal:**
```bash
docker run --rm --memory 7g -v /ssd:/models --entrypoint moe-engine moe-bench \
    --pack /models/qwen3-q4b-d4.moe --tokenizer /models/tokenizer.json --prompt "What is the capital of France?"
```

Without Docker: `cd engine && cargo build --release`, then run `./target/release/moe-engine`
with the same options (`--serve` for the server). Step-by-step instructions, including the
exact benchmark, are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository layout

| Folder | What it is |
|---|---|
| `engine/` | The inference engine with its chat server and tokenizer (Rust, no third-party crates), and its Python tools: converter, independent reference implementation, test suites |
| `docker/` | Container build, chat server start script, benchmark runner, release check script |
| `docs/` | How it works, results, how to reproduce, pack format |
| `kernel/` | Standalone CPU benchmark for the int8/int4 kernel formats we evaluated |
| `ssd/` | SSD read benchmark (fio) |
| `trace/` | Research tools: routing traces, quantization and neuron-skipping quality sweeps |

## Limitations

- **Two model families**: Qwen3-MoE and Qwen3.5/3.6-MoE (text only; the vision part of
  Qwen3.5/3.6 is not converted). Other MoE models need converter and engine work.
- **One answer at a time.** Requests wait in line; the page shows the place in line and
  an estimated wait. When a different conversation takes its turn, the engine reads that
  conversation again (about 2 minutes per 500 tokens on the N5105); continuing the same
  conversation reuses what is already in memory, and so does asking the same message again.
- **Fast kernels are x86-64 SSE only.** It runs correctly on ARM, including Apple
  Silicon, but slowly.
- **Context** up to `--max-context` tokens (default 4,096 for Qwen3 and 16,384 for
  Qwen3.6, whose memory per token is 5× smaller). When a conversation gets longer, its
  oldest messages are left out, and the page says so.
- **Quality**: experts are stored at 4.5 bits per weight, which costs about 3% in
  perplexity (6.21 vs 6.01). See [docs/RESULTS.md](docs/RESULTS.md#quality).
- **Heat**: generation keeps the CPU fully busy. On our NAS the CPU reached
  85–93 °C, within Intel's specification. The heat guard pauses at 90 °C by
  default; lower it with `--max-cpu-temp`.

## Related work

The idea of serving model weights from flash storage is not new, and several projects
explore SSD streaming for MoE models. `llm-on-ssd` focuses on the low end: CPU only, a
budget CPU without AVX2, a hard RAM cap, and a slow PCIe x1 SSD link. Related work we
know of:

- Alizadeh et al., *LLM in a Flash: Efficient Large Language Model Inference with
  Limited Memory* (Apple, 2023), [arXiv:2312.11514](https://arxiv.org/abs/2312.11514)
- Eliseev & Mazur, *Fast Inference of Mixture-of-Experts Language Models with
  Offloading* (2023), [arXiv:2312.17238](https://arxiv.org/abs/2312.17238)
- Song et al., *PowerInfer* (2023), [arXiv:2312.12456](https://arxiv.org/abs/2312.12456)
- Kamahori et al., *Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models*
  (ICLR 2025), [arXiv:2402.07033](https://arxiv.org/abs/2402.07033)
- Kim et al., *FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for
  MoE Inference on Edge Devices* (2026), [arXiv:2601.17063](https://arxiv.org/abs/2601.17063)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (memory-mapped weights; the 4-bit
  block layout here follows its Q4_0 idea) and its discussion on
  [MoE offload to disk](https://github.com/ggml-org/llama.cpp/discussions/23324)
- Apple Silicon SSD-streaming engines such as Flash-MoE and
  [TurboFieldfare](https://github.com/drumih/turbo-fieldfare); an overview of these
  engines: [InsiderLLM](https://insiderllm.com/guides/ssd-streaming-moe-engines/)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party software,
the model and their licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md). No model
weights are included in this repository.

This software is provided "as is", without warranty of any kind (see sections 7 and 8 of
the license). You run it on your own hardware at your own risk.

Qwen is a trademark of Alibaba. Intel and Celeron are trademarks of Intel Corporation.
Other names are trademarks of their owners. This project is not affiliated with or
endorsed by any of them.
