"""Check the engine's tokenizer without Hugging Face libraries.

For both split rules (Qwen2/Qwen3's, and Qwen3.5/3.6's with combining marks):
1. Pre-tokenizer: compares the engine's pieces with Python's `re` running the split
   pattern, with \\p{L}, \\p{M}, \\p{N} and \\s expanded into explicit Unicode classes.
2. NFC: compares the engine's normalization with Python's unicodedata.normalize.
3. BPE: builds a small byte-level BPE tokenizer.json trained on the test text, and
   compares the engine's token ids with a straightforward Python BPE (after NFC).

  python3 tools/test_tokenizer.py [path/to/moe-engine]

For the real Qwen3 tokenizer, compared with Hugging Face's, use tools/verify_tokenizer.py.
"""
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "target", "release", "moe-engine")
QWEN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
          r"|\s*[\r\n]+|\s+(?!\S)|\s+")

# Unicode White_Space (what Oniguruma and Rust use for \s)
WHITE_SPACE = [0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B)) + \
              [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]


def cls(pred):
    out, start = [], None
    for cp in range(0x110001):
        ok = cp < 0x110000 and not (0xD800 <= cp < 0xE000) and pred(cp)
        if ok and start is None:
            start = cp
        if not ok and start is not None:
            a, b = start, cp - 1
            out.append(re.escape(chr(a)) if a == b else re.escape(chr(a)) + "-" + re.escape(chr(b)))
            start = None
    return "".join(out)


L = cls(lambda c: unicodedata.category(chr(c)).startswith("L"))
M = cls(lambda c: unicodedata.category(chr(c)).startswith("M"))
N = cls(lambda c: unicodedata.category(chr(c)).startswith("N"))
S = "".join(re.escape(chr(c)) for c in WHITE_SPACE)


def compile_split(pattern):
    """Python `re` version of a Qwen split pattern, Unicode classes spelled out."""
    p = pattern
    p = p.replace(r"[^\r\n\p{L}\p{N}]", f"[^\\r\\n{L}{N}]")
    p = p.replace(r"[^\s\p{L}\p{M}\p{N}]", f"[^{S}{L}{M}{N}]").replace(r"[^\s\p{L}\p{N}]", f"[^{S}{L}{N}]")
    p = p.replace(r"[\p{L}\p{M}]", f"[{L}{M}]")
    p = p.replace(r"\p{L}", f"[{L}]").replace(r"\p{N}", f"[{N}]").replace(r"\s", f"[{S}]").replace(r"\S", f"[^{S}]")
    return re.compile(p)


SPLIT = compile_split(QWEN)

TEXTS = [
    "Hello world! Don't stop, I'M sure they'll come. We've got it, you'd see. It'S fine.",
    "Numbers 1234567 and 3.14159, ½ ⅓ ² ³ ٣٤٥ ௧௨௩ and phone +91-98765 43210.",
    "  leading spaces,   three spaces,\ttab\t\there\n\nnewlines\n\n\nand trailing   ",
    "Windows\r\nline\r\n\r\nendings \r\n mixed \n\r",
    "தமிழ் ஒரு செம்மொழி. இந்தியாவில் பேசப்படுகிறது!",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "中文测试：你好，世界！日本語のテスト、カタカナとひらがな。한국어 테스트입니다.",
    "Emoji 😀👍🏽 and flags 🇮🇳 mixed with text😀text",
    "العربية من اليمين إلى اليسار، مرحبا بالعالم.",
    "Ünïcödé àccents: café, naïve, résumé, Ærø, ß, İstanbul, ﬁ ligature",
    "def f(x):\n    return x**2  # comment\n\nprint(f(3))\n    \n",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>",
    "'s 'S 'tis 'RE 'Ve 'll 'LL 'd 'x '' ''' quotes 'quoted'",
    "a b c　d ​ e   f\u0085g",
    "    ",
    "end with space ",
    "x",
    # decomposed forms (NFC changes these) and scripts with many combining marks
    "Cafe\u0301 nai\u0308ve, A\u030angstro\u0308m, Vie\u0323\u0302t Nam, \u1100\u1161\u11a8 \u1112\u1161\u11ab",
    "q\u0307\u0323 x\u0323\u0307 e\u0301\u0327 \u0301start, \u0915\u093c\u093f \u0b95\u0bc6\u0bbe \u0b95\u0bca",
    "ไทย ภาษาไทย ที่ ผู้ใหญ่, ລາວ, ខ្មែរ, မြန်မာ, སྐད, ꦗꦮ, 𑀩𑁆𑀭𑀸𑀳𑁆𑀫𑀻",
]


def engine(args, text):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(text)
        name = f.name
    try:
        return subprocess.run([ENGINE] + args + ["--tokenize", name], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(name)


# ---- byte-level alphabet and a tiny BPE trained on the texts
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()


def train(texts, n_merges):
    words = collections.Counter()
    for t in texts:
        for p in SPLIT.findall(t):
            words[tuple(B2U[b] for b in p.encode())] += 1
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append((a, b))
        nw = collections.Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            nw[tuple(out)] += c
        words = nw
    return merges


def py_bpe(piece, ranks):
    word = [B2U[b] for b in piece.encode()]
    while len(word) > 1:
        pairs = [(ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(word, word[1:]))]
        r, _ = min(pairs)
        if r == 1 << 30:
            break
        i = min(i for rr, i in pairs if rr == r)
        a, b = word[i], word[i + 1]
        out, j = [], 0
        while j < len(word):
            if j + 1 < len(word) and word[j] == a and word[j + 1] == b:
                out.append(a + b)
                j += 2
            else:
                out.append(word[j])
                j += 1
        word = out
    return word


def main(pattern, name):
    global SPLIT
    SPLIT = compile_split(pattern)
    tok = make_tokenizer(pattern)
    corpus = open(os.path.join(HERE, "..", "..", "trace", "corpus.txt"), encoding="utf-8").read()
    texts = TEXTS + [corpus]
    fails = 0
    print(f"== {name} split rule")

    # 1. pre-tokenizer
    for t in texts:
        want = SPLIT.findall(t)  # pieces are compared on the text as given (before NFC)
        got = [json.loads(l) for l in engine(["--tokenizer", tok, "--pieces"], t).split("\n") if l]
        if want != got:
            fails += 1
            k = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
            print(f"FAIL pieces for {t[:40]!r}: first difference at piece {k}: want {want[k:k+3]!r} got {got[k:k+3]!r}")
    print(f"pre-tokenizer: {len(texts) - fails}/{len(texts)} texts identical to Python re")

    # 2. NFC
    nbad = 0
    for t in texts + NFC_EXTRA:
        got = json.loads(engine(["--tokenizer", tok, "--nfc"], t).strip())
        if got != unicodedata.normalize("NFC", t):
            nbad += 1
            print(f"FAIL NFC for {t[:40]!r}")
    print(f"NFC: {len(texts) + len(NFC_EXTRA) - nbad}/{len(texts) + len(NFC_EXTRA)} texts identical to Python")
    fails += nbad

    # 2. BPE ids, with and without special tokens
    bad = 0
    for t in texts:
        for special in (True, False):
            want = []
            rest = t
            if special:
                parts = re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", t)
            else:
                parts = [t]
            for part in parts:
                if special and part in SPECIAL:
                    want.append(SPECIAL[part])
                    continue
                for p in SPLIT.findall(unicodedata.normalize("NFC", part)):
                    want += [VOCAB[s] for s in py_bpe(p, RANKS)]
            args = ["--tokenizer", tok] + ([] if special else ["--no-special"])
            got = [int(x) for x in engine(args, t).strip().split(",") if x]
            if want != got:
                bad += 1
                k = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
                print(f"FAIL ids for {t[:40]!r} (special={special}): first difference at {k}")
    print(f"BPE: {2 * len(texts) - bad}/{2 * len(texts)} encodings identical")
    return fails + bad


# extra NFC cases: random sequences of letters and combining marks
def nfc_extra():
    import random
    rng = random.Random(7)
    bases = "aeiouAEIOUcnsyzkgqwxCNSYZ\u0915\u0b95\u0e01\u1100\u1161\u11a8\uac00\u0391\u0410 "
    marks = ["\u0300", "\u0301", "\u0302", "\u0303", "\u0308", "\u030a", "\u0323", "\u0327", "\u0328",
             "\u031b", "\u0345", "\u093c", "\u094d", "\u0bcd", "\u0bbe", "\u0bc6", "\u0e48", "\u05b4",
             "\u0651", "\u0654", "\u1161", "\u11a8", "\u0342", "\u0313"]
    out = []
    for _ in range(300):
        s = "".join(rng.choice(bases) + "".join(rng.choice(marks) for _ in range(rng.randint(0, 3)))
                    for _ in range(rng.randint(1, 12)))
        out.append(s)
    return out


def make_tokenizer(pattern):
    tj = {
        "added_tokens": [{"id": i, "content": s, "special": True} for s, i in SPECIAL.items()],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "model": {"type": "BPE", "vocab": VOCAB, "merges": [[a, b] for a, b in MERGES]},
    }
    path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    json.dump(tj, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return path


if __name__ == "__main__":
    corpus = open(os.path.join(HERE, "..", "..", "trace", "corpus.txt"), encoding="utf-8").read()
    MERGES = train(TEXTS + [corpus[:20000]], 3000)
    VOCAB = {}
    for b in range(256):
        VOCAB[B2U[b]] = len(VOCAB)
    for a, b in MERGES:
        if a + b not in VOCAB:
            VOCAB[a + b] = len(VOCAB)
    RANKS = {m: i for i, m in enumerate(MERGES)}
    SPECIAL = {s: len(VOCAB) + i for i, s in enumerate(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"])}
    NFC_EXTRA = nfc_extra()
    bad = main(QWEN, "Qwen2/Qwen3") + main(QWEN35, "Qwen3.5/3.6")
    print("TOKENIZER TEST PASSED" if not bad else "TOKENIZER TEST FAILED")
    sys.exit(1 if bad else 0)
