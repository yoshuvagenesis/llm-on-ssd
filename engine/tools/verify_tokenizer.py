"""Compare the engine's tokenizer with Hugging Face's on the model's real tokenizer.json.

  pip install tokenizers
  python3 tools/verify_tokenizer.py --tokenizer ~/models/Qwen3-30B-A3B/tokenizer.json [--engine path]

Encodes many texts (English, code, Indian and other scripts, emoji, odd spacing, chat
markup and trace/corpus.txt) both ways and reports any difference.
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_tokenizer import TEXTS  # noqa: E402

EXTRA = [
    "বাংলা ভাষা, ગુજરાતી, ਪੰਜਾਬੀ, తెలుగు, ಕನ್ನಡ, മലയാളം, ଓଡ଼ିଆ, اردو",
    "Ελληνικά, Русский язык, Українська, עברית, ไทย, Tiếng Việt, ქართული",
    "JSON {\"a\": [1, 2.5e-3, null], \"b\": \"x\\ny\"} and URLs https://example.com/a?b=c&d=e#f",
    "Math: ∑_{i=1}^{n} i² = n(n+1)(2n+1)/6, ∫ e^x dx, α β γ, x ≤ y ≥ z ≠ w",
    "Mixed: abc123def 3D 2nd 1st 100% $5.99 €10 ₹500 #hashtag @user",
    "Repeated!!!!!!! ??? ... --- ___ *** ===",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default=os.path.join(HERE, "..", "target", "release", "moe-engine"))
    a = ap.parse_args()
    from tokenizers import Tokenizer

    hf = Tokenizer.from_file(os.path.expanduser(a.tokenizer))
    corpus = open(os.path.join(HERE, "..", "..", "trace", "corpus.txt"), encoding="utf-8").read()
    texts = TEXTS + EXTRA + [corpus]
    bad = 0
    for t in texts:
        want = hf.encode(t, add_special_tokens=False).ids
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write(t)
            name = f.name
        out = subprocess.run([a.engine, "--tokenizer", os.path.expanduser(a.tokenizer), "--tokenize", name],
                             capture_output=True, text=True, check=True).stdout
        os.unlink(name)
        got = [int(x) for x in out.strip().split(",") if x]
        if want != got:
            bad += 1
            k = next((i for i, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
            print(f"DIFFERENT for {t[:50]!r}: first at token {k}: HF {want[k:k+4]} engine {got[k:k+4]}")
    print(f"{len(texts) - bad}/{len(texts)} texts tokenized identically "
          f"({sum(len(hf.encode(t, add_special_tokens=False).ids) for t in texts)} tokens)")
    print("TOKENIZER MATCHES HUGGING FACE" if not bad else "TOKENIZER DIFFERS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
