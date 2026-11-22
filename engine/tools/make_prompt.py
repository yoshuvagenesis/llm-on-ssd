"""Make a benchmark prompt of exactly N tokens (default 512) from a text file, in the
Qwen3 chat format, for repeatable speed tests:

  python make_prompt.py --tokenizer tokenizer.json --text ../../trace/corpus.txt > prompt512.txt

The user message is the start of the text, cut so that the whole prompt (template
included) has exactly N tokens. Needs `pip install tokenizers`.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to the model's tokenizer.json")
    ap.add_argument("--text", required=True, help="UTF-8 text file")
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    head = enc("<|im_start|>user\nSummarize this text:\n\n")
    tail = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body = enc(open(args.text, encoding="utf-8").read())
    n = args.tokens - len(head) - len(tail)
    if n <= 0 or len(body) < n:
        sys.exit(f"text too short or --tokens too small (need {n} body tokens, have {len(body)})")
    ids = head + body[:n] + tail
    assert len(ids) == args.tokens
    sys.stdout.write(",".join(map(str, ids)) + "\n")


if __name__ == "__main__":
    main()
