"""Turn a chat message into prompt token ids for the engine (Qwen3 chat format).

  python encode.py --model ~/models/Qwen3-30B-A3B "What is the capital of France?" > prompt.txt
  moe-engine --pack model.moe --prompt-file prompt.txt

Needs `pip install tokenizers` and the model folder's tokenizer.json (only that file
is needed; copy it anywhere and pass --tokenizer instead of --model).
Thinking is switched off by default (empty <think> block), like Qwen3's
enable_thinking=False; use --think to let the model reason first.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="model folder containing tokenizer.json")
    g.add_argument("--tokenizer", help="path to tokenizer.json")
    ap.add_argument("--system", default="")
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer or os.path.join(args.model, "tokenizer.json"))
    text = ""
    if args.system:
        text += f"<|im_start|>system\n{args.system}<|im_end|>\n"
    text += f"<|im_start|>user\n{args.message}<|im_end|>\n<|im_start|>assistant\n"
    if not args.think:
        text += "<think>\n\n</think>\n\n"
    ids = tok.encode(text, add_special_tokens=False).ids
    sys.stdout.write(",".join(map(str, ids)) + "\n")


if __name__ == "__main__":
    main()
