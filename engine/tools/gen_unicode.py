"""Regenerate src/unicode_letters.rs (Unicode category L ranges) from Python's unicodedata.
Run from the engine folder:  python3 tools/gen_unicode.py"""
import unicodedata

rs, start = [], None
for cp in range(0x110001):
    is_l = cp < 0x110000 and unicodedata.category(chr(cp)).startswith("L")
    if is_l and start is None:
        start = cp
    if not is_l and start is not None:
        rs.append((start, cp - 1))
        start = None
out = ["//! Unicode general category L (letters), generated from Python's unicodedata",
       f"//! (Unicode {unicodedata.unidata_version}) by tools/gen_unicode.py. Used by the tokenizer's \\p{{L}}.",
       "", f"pub const LETTERS: [(u32, u32); {len(rs)}] = ["]
line = "   "
for a, b in rs:
    item = f" (0x{a:X}, 0x{b:X}),"
    if len(line) + len(item) > 96:
        out.append(line)
        line = "   "
    line += item
out += [line, "];"]
open("src/unicode_letters.rs", "w").write("\n".join(out) + "\n")
print(len(rs), "ranges")
