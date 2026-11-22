//! Reader for the .moe pack format (see tools/packfmt.py for the layout).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::FileExt;

pub const MAGIC: &[u8; 8] = b"MOEPACK1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fmt {
    F32,
    Bf16,
    Q8,
    Q4,
    Q4b,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub fmt: Fmt,
    pub rows: usize,
    pub cols: usize,
    pub off: u64,
    pub nbytes: usize,
}

pub struct Pack {
    #[allow(dead_code)]
    pub path: String,
    pub data_start: u64,
    pub cfg: HashMap<String, String>,
    pub tensors: HashMap<String, TensorInfo>,
    pub experts: Vec<(u64, usize)>, // index layer * E + expert -> (offset, nbytes)
    pub vocab: (u64, usize),
    file: File,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Pack {
    pub fn open(path: &str) -> io::Result<Pack> {
        let mut file = File::open(path)?;
        let mut head = [0u8; 24];
        file.read_exact(&mut head)?;
        if &head[..8] != MAGIC {
            return Err(bad(format!("{path} is not a .moe pack")));
        }
        let hlen = u64::from_le_bytes(head[8..16].try_into().unwrap()) as usize;
        let data_start = u64::from_le_bytes(head[16..24].try_into().unwrap());
        let mut htext = vec![0u8; hlen];
        file.read_exact(&mut htext)?;
        let htext = String::from_utf8(htext).map_err(|e| bad(e.to_string()))?;

        let mut cfg = HashMap::new();
        let mut tensors = HashMap::new();
        let mut elist = Vec::new();
        let mut vocab = (0, 0);
        for line in htext.lines() {
            let p: Vec<&str> = line.split_whitespace().collect();
            let num = |i: usize| -> io::Result<u64> {
                p.get(i).and_then(|v| v.parse().ok()).ok_or_else(|| bad(format!("bad header line: {line}")))
            };
            match p.first().copied() {
                Some("cfg") if p.len() >= 3 => {
                    cfg.insert(p[1].to_string(), p[2].to_string());
                }
                Some("t") => {
                    let fmt = match p.get(2).copied() {
                        Some("f32") => Fmt::F32,
                        Some("bf16") => Fmt::Bf16,
                        Some("q8") => Fmt::Q8,
                        Some("q4") => Fmt::Q4,
                        Some("q4b") => Fmt::Q4b,
                        _ => return Err(bad(format!("bad format in: {line}"))),
                    };
                    tensors.insert(
                        p[1].to_string(),
                        TensorInfo { fmt, rows: num(3)? as usize, cols: num(4)? as usize, off: num(5)?, nbytes: num(6)? as usize },
                    );
                }
                Some("e") => elist.push((num(1)? as usize, num(2)? as usize, num(3)?, num(4)? as usize)),
                Some("vocab") => vocab = (num(1)?, num(2)? as usize),
                _ => {}
            }
        }
        let mut pk = Pack { path: path.to_string(), data_start, cfg, tensors, experts: Vec::new(), vocab, file };
        let (l, e) = (pk.cfg_usize("num_hidden_layers")?, pk.cfg_usize("num_experts")?);
        pk.experts = vec![(0, 0); l * e];
        for (li, ei, off, n) in elist {
            if li >= l || ei >= e {
                return Err(bad(format!("expert {li}/{ei} out of range")));
            }
            pk.experts[li * e + ei] = (off, n);
        }
        if pk.experts.iter().any(|x| x.1 == 0) {
            return Err(bad("pack is missing expert blocks".into()));
        }
        Ok(pk)
    }

    pub fn cfg_str(&self, k: &str) -> io::Result<&str> {
        self.cfg.get(k).map(|s| s.as_str()).ok_or_else(|| bad(format!("pack header has no cfg {k}")))
    }
    pub fn cfg_usize(&self, k: &str) -> io::Result<usize> {
        self.cfg_str(k)?.parse().map_err(|_| bad(format!("cfg {k} is not an integer")))
    }
    pub fn cfg_f64(&self, k: &str) -> io::Result<f64> {
        self.cfg_str(k)?.parse().map_err(|_| bad(format!("cfg {k} is not a number")))
    }

    pub fn tensor(&self, name: &str) -> io::Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| bad(format!("pack has no tensor {name}")))
    }

    pub fn read_at(&self, off: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.file.read_exact_at(&mut buf, self.data_start + off)?;
        Ok(buf)
    }

    pub fn tensor_bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let t = self.tensor(name)?;
        self.read_at(t.off, t.nbytes)
    }

    pub fn vocab(&self) -> io::Result<Vec<Vec<u8>>> {
        let blob = self.read_at(self.vocab.0, self.vocab.1)?;
        let mut out = Vec::new();
        let mut i = 0;
        while i + 4 <= blob.len() {
            let n = u32::from_le_bytes(blob[i..i + 4].try_into().unwrap()) as usize;
            out.push(blob[i + 4..i + 4 + n].to_vec());
            i += 4 + n;
        }
        Ok(out)
    }

    /// The pack's open file (buffered reads for small data such as embedding rows).
    pub fn file(&self) -> &File {
        &self.file
    }
}
