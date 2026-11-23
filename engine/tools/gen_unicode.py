"""Regenerate src/unicode_data.rs from Python's unicodedata (the Unicode Character Database).
Run from the engine folder:  python3 tools/gen_unicode.py

Tables:
  LETTERS   ranges of general category L   (tokenizer \\p{L})
  MARKS     ranges of general category M   (tokenizer \\p{M}, Qwen3.5 / 3.6 split rule)
  CCC       ranges with a non-zero canonical combining class (NFC ordering)
  DECOMP    canonical decompositions, one step each (NFC decomposition, applied recursively)
  COMPOSE   primary composites (a, b) -> c, sorted by (a, b) (NFC composition)
Hangul syllables are handled by formula in the engine, not by these tables.
"""
import unicodedata

N = 0x110000


def chars():
    for cp in range(N):
        if 0xD800 <= cp < 0xE000:
            continue
        yield cp


def ranges(pred):
    out, start, prev = [], None, None
    for cp in range(N + 1):
        ok = cp < N and not (0xD800 <= cp < 0xE000) and pred(cp)
        if ok and start is None:
            start = cp
        if not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    return out


def is_hangul_syllable(cp):
    return 0xAC00 <= cp <= 0xD7A3


def main():
    cat = lambda cp: unicodedata.category(chr(cp))
    letters = ranges(lambda cp: cat(cp).startswith("L"))
    marks = ranges(lambda cp: cat(cp).startswith("M"))

    ccc = []  # (start, end, class) runs with the same non-zero class
    cur = None
    for cp in range(N + 1):
        c = unicodedata.combining(chr(cp)) if cp < N and not (0xD800 <= cp < 0xE000) else 0
        if cur and (c != cur[2] or cp != cur[1] + 1):
            ccc.append(tuple(cur))
            cur = None
        if c:
            cur = [cp, cp, c] if cur is None else [cur[0], cp, c]
    decomp, flat = [], []
    compose = []
    for cp in chars():
        if is_hangul_syllable(cp):
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = [int(x, 16) for x in d.split()]
        decomp.append((cp, len(flat), len(parts)))
        flat += parts
        if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == chr(cp):
            compose.append((parts[0], parts[1], cp))
    compose.sort()

    def table(name, typ, rows, fmt):
        lines = [f"pub const {name}: [{typ}; {len(rows)}] = ["]
        line = "   "
        for r in rows:
            item = " " + fmt(r) + ","
            if len(line) + len(item) > 96:
                lines.append(line)
                line = "   "
            line += item
        lines += [line, "];", ""]
        return lines

    out = [
        "//! Unicode tables generated from Python's unicodedata "
        f"(Unicode {unicodedata.unidata_version})",
        "//! by tools/gen_unicode.py. Unicode data: Copyright (c) Unicode, Inc., Unicode License v3.",
        "//! Used by the tokenizer: letters and marks for the split rules, and NFC normalization.",
        "",
    ]
    h = lambda r: f"(0x{r[0]:X}, 0x{r[1]:X})"
    out += table("LETTERS", "(u32, u32)", letters, h)
    out += table("MARKS", "(u32, u32)", marks, h)
    out += table("CCC", "(u32, u32, u8)", ccc, lambda r: f"(0x{r[0]:X}, 0x{r[1]:X}, {r[2]})")
    out += table("DECOMP", "(u32, u16, u8)", decomp, lambda r: f"(0x{r[0]:X}, {r[1]}, {r[2]})")
    out += table("DECOMP_CHARS", "u32", flat, lambda r: f"0x{r:X}")
    out += table("COMPOSE", "(u32, u32, u32)", compose, lambda r: f"(0x{r[0]:X}, 0x{r[1]:X}, 0x{r[2]:X})")
    open("src/unicode_data.rs", "w").write("\n".join(out))
    print(f"letters {len(letters)}, marks {len(marks)}, ccc runs {len(ccc)}, decompositions {len(decomp)}, "
          f"compositions {len(compose)}")


if __name__ == "__main__":
    main()
