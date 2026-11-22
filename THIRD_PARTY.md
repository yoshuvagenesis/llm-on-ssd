# Third-party software, models and data

Everything in this repository was written for this project and is licensed under
Apache-2.0 (see [LICENSE](LICENSE)). The repository bundles **no** third-party code, no
model weights and no third-party data. The items below are things you install or download
yourself to use it, or that we used for comparison.

## Needed to build and run

| Item | Used for | License | Bundled here? |
|---|---|---|---|
| [Rust](https://www.rust-lang.org/) toolchain and standard library | Building the engine (`engine/`) and the kernel benchmark (`kernel/`). Neither uses any third-party crate. | MIT OR Apache-2.0 | No |
| [Python 3](https://www.python.org/) | Converter, test suite, research tools | PSF License | No |
| [NumPy](https://numpy.org/) | Converter, reference implementation, research tools | BSD-3-Clause | No |
| [Hugging Face `tokenizers`](https://github.com/huggingface/tokenizers) | Turning text into token ids (`encode.py`, `make_prompt.py`, `trace_routing.py --text`) | Apache-2.0 | No |
| [Hugging Face Hub CLI](https://github.com/huggingface/huggingface_hub) (optional) | Downloading the model | Apache-2.0 | No |

## Container (`docker/Dockerfile`)

The Dockerfile builds an image on your own machine. Nothing from it is distributed
by this repository.

| Item | Used for | License |
|---|---|---|
| [`rust:1-slim-bookworm`](https://hub.docker.com/_/rust) base image (Debian 12 + Rust) | Build and run environment | Debian packages under their own licenses (mostly free software licenses, some GPL); Rust MIT OR Apache-2.0 |
| [fio](https://github.com/axboe/fio), installed with `apt` | SSD benchmark (`ssd/bench_ssd.sh`) | GPL-2.0. Used unmodified as a separate program; not linked with or bundled in this project |
| `python3`, `python3-numpy` (Debian packages) | Self-test and benchmark pack generation inside the container | PSF License, BSD-3-Clause |

## Model (not included)

| Model | License | Notes |
|---|---|---|
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) by the Qwen team, Alibaba | Apache-2.0 | You download it yourself. A `.moe` pack you make with `convert.py` is a derivative of the model. If you share a pack, include the model's license and say that it was converted (Apache-2.0, section 4). |

## Comparison baseline (not included)

| Item | Used for | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Speed baseline in `docs/RESULTS.md` | MIT |

## Ideas and formats we followed (no code copied)

- The 4-bit block layout ("q4": 32 weights per block, one scale, values −8..7) follows
  the Q4_0 format idea from ggml/llama.cpp (MIT). It was implemented from scratch here.
- The byte-to-character table used to read Qwen3's vocabulary is the standard byte-level
  BPE mapping introduced with GPT-2. It was implemented from scratch here.
- The `.safetensors` reader (`trace/st_reader.py`) follows the public safetensors file
  format specification. It does not use the safetensors library.

## Data

- `trace/corpus.txt` is original text written for this project (chats, code, short
  articles and stories), Apache-2.0 like the rest of the repository. It is used to measure
  perplexity and to make the benchmark prompt.
