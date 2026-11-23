//! Qwen tokenizer: byte-level BPE read from the model's own `tokenizer.json`.
//!
//! Steps, as in Hugging Face `tokenizers` for these models:
//! 1. split off added tokens (`<|im_start|>`, `<think>`, ...) when allowed,
//! 2. NFC-normalize the text (if tokenizer.json asks for it, as Qwen's do),
//! 3. pre-tokenize with the model's split pattern (hand-written matchers below):
//!    Qwen2/Qwen3's, or Qwen3.5/3.6's, which keeps combining marks inside letter runs,
//! 4. map each piece's UTF-8 bytes to the byte-level alphabet,
//! 5. apply BPE merges by rank.
//!
//! `tools/verify_tokenizer.py` compares the result with Hugging Face's on real text.

use crate::json::{self, J};
use crate::unicode_data::{CCC, COMPOSE, DECOMP, DECOMP_CHARS, LETTERS, MARKS};
use std::collections::HashMap;

/// The split pattern used by Qwen2/Qwen3.
pub const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The split pattern used by Qwen3.5/Qwen3.6: combining marks (\p{M}) count with letters.
pub const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    byte_id: [u32; 256],              // id of each single byte symbol
    merges: HashMap<(u32, u32), (u32, u32)>, // (left, right) -> (rank, merged id)
    added: Vec<(String, u32)>,        // added tokens, longest first
    pub marks: bool,                  // Qwen3.5 split rule
    pub nfc: bool,                    // NFC-normalize before splitting
    pub warning: Option<String>,      // set if tokenizer.json is not the expected kind
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut m = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        m[*b as usize] = char::from_u32(c).unwrap();
    }
    m
}

fn in_ranges(t: &[(u32, u32)], c: u32) -> bool {
    t.binary_search_by(|&(a, b)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

fn is_letter(c: char) -> bool {
    let c = c as u32;
    if c < 0x80 {
        return (c | 0x20).wrapping_sub(b'a' as u32) < 26;
    }
    in_ranges(&LETTERS, c)
}
fn is_mark(c: char) -> bool {
    (c as u32) >= 0x300 && in_ranges(&MARKS, c as u32)
}

// ------------------------------------------------------------------ NFC normalization

fn ccc(c: u32) -> u8 {
    if c < 0x300 {
        return 0;
    }
    match CCC.binary_search_by(|&(a, b, _)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => CCC[i].2,
        Err(_) => 0,
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decompose(c: u32, out: &mut Vec<u32>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&c) {
        let s = c - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if s % T_COUNT != 0 {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    if c >= 0xC0 {
        if let Ok(i) = DECOMP.binary_search_by_key(&c, |e| e.0) {
            let (_, start, len) = DECOMP[i];
            for &d in &DECOMP_CHARS[start as usize..start as usize + len as usize] {
                decompose(d, out);
            }
            return;
        }
    }
    out.push(c);
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a) && (a - S_BASE) % T_COUNT == 0 && (T_BASE + 1..T_BASE + T_COUNT).contains(&b) {
        return Some(a + (b - T_BASE));
    }
    COMPOSE.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok().map(|i| COMPOSE[i].2)
}

/// Unicode Normalization Form C (canonical decomposition, ordering, composition).
pub fn nfc(text: &str) -> String {
    if text.bytes().all(|b| b < 0xCC) {
        // only characters below U+0300: none of them changes under NFC
        return text.to_string();
    }
    let mut d: Vec<u32> = Vec::with_capacity(text.len());
    for c in text.chars() {
        decompose(c as u32, &mut d);
    }
    // canonical ordering: stable sort of each run of non-starters by combining class
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let j = (i..d.len()).find(|&k| ccc(d[k]) == 0).unwrap_or(d.len());
        d[i..j].sort_by_key(|&c| ccc(c));
        i = j;
    }
    // canonical composition
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    let mut last_cc: u8 = 0;
    for &c in &d {
        let cc = ccc(c);
        if let Some(si) = starter {
            let adjacent = out.len() == si + 1;
            let blocked = !adjacent && (last_cc == 0 || last_cc >= cc);
            if !blocked {
                if let Some(comp) = compose_pair(out[si], c) {
                    out[si] = comp;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
        }
        last_cc = cc;
        out.push(c);
    }
    out.into_iter().filter_map(char::from_u32).collect()
}
fn is_number(c: char) -> bool {
    c.is_numeric() // general categories Nd, Nl, No = \p{N}
}
fn is_space(c: char) -> bool {
    c.is_whitespace() // Unicode White_Space = \s
}
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Split text into pieces exactly like QWEN_PATTERN, or QWEN35_PATTERN with `marks`
/// (leftmost alternative first, greedy).
pub fn pretokenize(text: &str, marks: bool) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| cs.get(i).map(|x| x.1);
    let byte = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = match_at(i, n, &at, marks);
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}

fn match_at(i: usize, n: usize, at: &dyn Fn(usize) -> Option<char>, marks: bool) -> usize {
    let c = at(i).unwrap();
    // the letter-run class: \p{L}, or [\p{L}\p{M}] for Qwen3.5
    let word = |x: char| is_letter(x) || (marks && is_mark(x));
    // 1. contractions, case-insensitive: 's 't 're 've 'm 'll 'd
    if c == '\'' {
        let l = |k: usize| at(i + k).map(|x| x.to_ascii_lowercase());
        match l(1) {
            Some('s') | Some('t') | Some('m') | Some('d') => return i + 2,
            Some('r') if l(2) == Some('e') => return i + 3,
            Some('v') if l(2) == Some('e') => return i + 3,
            Some('l') if l(2) == Some('l') => return i + 3,
            _ => {}
        }
    }
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+   (Qwen3.5: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+)
    {
        let mut j = i;
        if !is_nl(c) && !is_letter(c) && !is_number(c) && at(i + 1).is_some_and(word) {
            j = i + 1;
        }
        if at(j).is_some_and(word) {
            while at(j).is_some_and(word) {
                j += 1;
            }
            return j;
        }
    }
    // 3. \p{N}
    if is_number(c) {
        return i + 1;
    }
    // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'   (Qwen3.5 also excludes \p{M})
    {
        let punct = |x: char| !is_space(x) && !word(x) && !is_number(x);
        let mut j = i;
        if c == ' ' && at(i + 1).is_some_and(punct) {
            j = i + 1;
        }
        if at(j).is_some_and(punct) {
            while at(j).is_some_and(punct) {
                j += 1;
            }
            while at(j).is_some_and(is_nl) {
                j += 1;
            }
            return j;
        }
    }
    if is_space(c) {
        let mut j = i;
        while j < n && at(j).is_some_and(is_space) {
            j += 1;
        }
        // 5. \s*[\r\n]+ : up to and including the last newline in the whitespace run
        if let Some(last_nl) = (i..j).rev().find(|&k| at(k).is_some_and(is_nl)) {
            return last_nl + 1;
        }
        // 6. \s+(?!\S) : the whole run at the end of text, else all but its last char
        if j == n {
            return j;
        }
        if j - i >= 2 {
            return j - 1;
        }
        // 7. \s+
        return j;
    }
    // nothing matched (cannot happen: every char falls in one class); take one char
    i + 1
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let root = json::parse(&text).map_err(|e| format!("{path}: {e}"))?;
        drop(text);
        let model = root.get("model").ok_or("tokenizer.json: no model")?;
        if model.get("type").and_then(J::as_str) != Some("BPE") {
            return Err("tokenizer.json: only BPE models are supported".into());
        }
        let mut warning = None;
        let pat = root
            .get("pre_tokenizer")
            .and_then(|p| p.get("pretokenizers"))
            .and_then(J::as_arr)
            .and_then(|a| a.iter().find_map(|x| x.get("pattern").and_then(|p| p.get("Regex")).and_then(J::as_str)));
        let marks = match pat {
            Some(QWEN_PATTERN) => false,
            Some(QWEN35_PATTERN) => true,
            _ => {
                warning = Some("tokenizer.json split pattern is neither Qwen3's nor Qwen3.5's; tokens may not match exactly".into());
                pat.is_some_and(|p| p.contains("\\p{M}"))
            }
        };
        let norm_type = |n: &J| n.get("type").and_then(J::as_str).map(|s| s.to_string());
        let nfc = match root.get("normalizer") {
            None | Some(J::Null) => false,
            Some(n) => match norm_type(n).as_deref() {
                Some("NFC") => true,
                Some("Sequence") => n
                    .get("normalizers")
                    .and_then(J::as_arr)
                    .is_some_and(|a| a.iter().any(|x| norm_type(x).as_deref() == Some("NFC"))),
                _ => {
                    warning = Some("tokenizer.json uses a normalizer other than NFC; tokens may not match exactly".into());
                    false
                }
            },
        };
        let vocab_j = model.get("vocab").and_then(J::as_obj).ok_or("tokenizer.json: no vocab")?;
        let mut vocab: HashMap<&str, u32> = HashMap::with_capacity(vocab_j.len());
        for (k, v) in vocab_j {
            vocab.insert(k.as_str(), v.as_f64().ok_or("bad vocab id")? as u32);
        }
        let b2u = bytes_to_unicode();
        let mut byte_id = [0u32; 256];
        for b in 0..256 {
            let s = b2u[b].to_string();
            byte_id[b] = *vocab.get(s.as_str()).ok_or("tokenizer.json: vocab lacks a byte symbol")?;
        }
        let mut merges = HashMap::new();
        let ml = model.get("merges").and_then(J::as_arr).ok_or("tokenizer.json: no merges")?;
        for (rank, m) in ml.iter().enumerate() {
            let (a, b) = match m {
                J::Str(s) => match s.split_once(' ') {
                    Some(p) => p,
                    None => continue,
                },
                J::Arr(p) if p.len() == 2 => match (p[0].as_str(), p[1].as_str()) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                },
                _ => continue,
            };
            let ab = format!("{a}{b}");
            if let (Some(&ia), Some(&ib), Some(&iab)) = (vocab.get(a), vocab.get(b), vocab.get(ab.as_str())) {
                merges.entry((ia, ib)).or_insert((rank as u32, iab));
            }
        }
        let mut added: Vec<(String, u32)> = root
            .get("added_tokens")
            .and_then(J::as_arr)
            .unwrap_or(&[])
            .iter()
            .filter_map(|t| Some((t.get("content")?.as_str()?.to_string(), t.get("id")?.as_f64()? as u32)))
            .collect();
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Tokenizer { byte_id, merges, added, marks, nfc, warning })
    }

    pub fn added_id(&self, s: &str) -> Option<u32> {
        self.added.iter().find(|(t, _)| t == s).map(|x| x.1)
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut sym: Vec<u32> = piece.bytes().map(|b| self.byte_id[b as usize]).collect();
        while sym.len() > 1 {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..sym.len() - 1 {
                if let Some(&(r, id)) = self.merges.get(&(sym[i], sym[i + 1])) {
                    if best.is_none_or(|b| r < b.0) {
                        best = Some((r, i, id));
                    }
                }
            }
            let Some((_, i, id)) = best else { break };
            // merge every occurrence of this pair, left to right (same as HF for one rank)
            let (l, rr) = (sym[i], sym[i + 1]);
            let mut j = 0;
            let mut next = Vec::with_capacity(sym.len());
            while j < sym.len() {
                if j + 1 < sym.len() && sym[j] == l && sym[j + 1] == rr {
                    next.push(id);
                    j += 2;
                } else {
                    next.push(sym[j]);
                    j += 1;
                }
            }
            sym = next;
        }
        out.extend_from_slice(&sym);
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let normed;
        let text = if self.nfc {
            normed = nfc(text);
            normed.as_str()
        } else {
            text
        };
        for piece in pretokenize(text, self.marks) {
            self.bpe(piece, out);
        }
    }

    /// Encode text. With `allow_special`, added tokens written in the text (such as
    /// `<|im_start|>`) become their single ids, as Hugging Face does; without it they
    /// are treated as ordinary text (use this for user input).
    pub fn encode(&self, text: &str, allow_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if !allow_special {
            self.encode_plain(text, &mut out);
            return out;
        }
        let mut start = 0;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if let Some((t, id)) = self.added.iter().find(|(t, _)| text[i..].starts_with(t.as_str())) {
                self.encode_plain(&text[start..i], &mut out);
                out.push(*id);
                i += t.len();
                start = i;
            } else {
                i += 1;
                while i < bytes.len() && !text.is_char_boundary(i) {
                    i += 1;
                }
            }
        }
        self.encode_plain(&text[start..], &mut out);
        out
    }
}
