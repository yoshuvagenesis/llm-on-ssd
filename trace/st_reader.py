"""Minimal safetensors reader (numpy only, no torch).

Reads one tensor at a time straight from the shard files, so a 60 GB model can
be processed layer by layer on a 16 GB machine. BF16/F16/F32 are returned as
float32 numpy arrays.
"""
import json
import os
import struct

import numpy as np

_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}


class SafeTensors:
    def __init__(self, model_dir):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.meta = {}  # name -> (path, dtype, shape, abs_offset, nbytes)
        for fn in files:
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            base = 8 + n
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                dt = info["dtype"]
                if dt not in _DTYPES:
                    raise ValueError(f"{name}: unsupported dtype {dt}")
                s, e = info["data_offsets"]
                self.meta[name] = (path, dt, tuple(info["shape"]), base + s, e - s)
        self._fh = {}

    def __contains__(self, name):
        return name in self.meta

    def _file(self, path):
        if path not in self._fh:
            self._fh[path] = open(path, "rb")
        return self._fh[path]

    def raw(self, name):
        """Return (uint16-or-float array with original bits, dtype string, shape)."""
        path, dt, shape, off, nbytes = self.meta[name]
        f = self._file(path)
        f.seek(off)
        buf = f.read(nbytes)
        if dt == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        elif dt == "F16":
            arr = np.frombuffer(buf, dtype=np.float16)
        else:
            arr = np.frombuffer(buf, dtype=np.uint16)
        return arr.reshape(shape), dt

    @staticmethod
    def to_f32(arr, dt):
        if dt == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)

    def get(self, name):
        arr, dt = self.raw(name)
        return self.to_f32(arr, dt)

    def rows(self, name, idx):
        """Gather rows (e.g. embedding lookup) without converting the whole matrix."""
        arr, dt = self.raw(name)
        return self.to_f32(arr[idx], dt)


def write_safetensors(path, tensors):
    """Write {name: float32 array} as BF16 safetensors (used only by the self-test)."""
    header, chunks, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        bits = a.view(np.uint32)
        # round-to-nearest-even to bf16
        bf = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        b = bf.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        chunks.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for c in chunks:
            f.write(c)
