"""End-to-end test: tiny random Qwen3-MoE -> convert -> engine -> compare with ref_pack.py.

  python test_engine.py [path/to/moe-engine]

Checks that the engine's logits match the independent reference at every step,
under different cache sizes (incl. strict mode), thread counts, prefetch settings
and with or without O_DIRECT.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "trace"))
import selftest  # noqa: E402  (tiny-model builder)

import ref_pack  # noqa: E402

ENGINE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "target", "release", "moe-engine")

CFG = dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
           num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64, rms_norm_eps=1e-6,
           rope_theta=10000.0, vocab_size=512, norm_topk_prob=True, tie_word_embeddings=False, model_type="qwen3_moe")

# Cache size, prefetching, read mode, I/O threads, pack copies and compute threads
# must not change a single bit of output.
RUNS = [
    ("big cache, 4 threads", ["--cache-gb", "1", "--threads", "4"]),
    ("1 compute thread", ["--cache-gb", "1", "--threads", "1"]),
    ("strict: no cache", ["--cache-gb", "0", "--threads", "3"]),
    ("tiny cache, 1 io thread", ["--cache-gb", "0.0004", "--io-threads", "1", "--threads", "4"]),
    ("no prefetch", ["--prefetch-width", "0", "--threads", "2"]),
    ("lookahead 3, buffered reads", ["--lookahead", "3", "--buffered", "--threads", "4"]),
    ("reads spread over a mirror copy", ["--mirror", "{mirror}", "--threads", "4"]),
    ("prompt token by token", ["--prefill-batch", "1", "--threads", "4"]),
    ("prompt in batches of 4", ["--prefill-batch", "4", "--threads", "2"]),
    ("lookahead 1, wide guesses", ["--lookahead", "1", "--prefetch-width", "12", "--threads", "4"]),
    ("second prediction (--refine), tiny cache", ["--refine", "--cache-gb", "0.0004", "--threads", "4"]),
    ("second prediction, token by token, 1 io thread", ["--refine", "--prefill-batch", "1", "--io-threads", "1", "--threads", "3"]),
    ("RAM cap and short context", ["--ram-gb", "0.5", "--max-context", "256", "--threads", "4"]),
]


def engine(pack, prompt, d, extra):
    lg, ids = os.path.join(d, "lg.bin"), os.path.join(d, "ids.txt")
    r = subprocess.run([ENGINE, "--pack", pack, "--prompt-ids", prompt, "--max-new", "12",
                        "--dump-logits", lg, "--out-ids", ids, *extra], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"engine failed: {extra}")
    stats = [ln for ln in r.stderr.splitlines() if ln.startswith(("expert blocks", "SSD:"))]
    return np.fromfile(lg, np.float32).reshape(-1, CFG["vocab_size"]), open(ids).read(), stats


def compare(a, b):
    n = min(len(a), len(b))
    err = np.abs(a[:n] - b[:n]).max(1) / np.abs(b[:n]).max(1)
    return err, (a[:n].argmax(1) == b[:n].argmax(1)).mean()


def main():
    ok = True
    for fmt, dense in (("q4b", "q8"), ("q4", "q8"), ("q4b", "q4b")):
        print(f"===== experts {fmt}, attention/output head {dense}")
        ok &= run_format(fmt, dense)
    print("ENGINE TEST PASSED" if ok else "ENGINE TEST FAILED")
    sys.exit(0 if ok else 1)


def run_format(fmt, dense):
    ok = True
    with tempfile.TemporaryDirectory() as d:
        selftest.CFG.clear()
        selftest.CFG.update(CFG)
        selftest.build(d, np.random.default_rng(11))
        pack = os.path.join(d, "tiny.moe")
        subprocess.run([sys.executable, os.path.join(HERE, "convert.py"), "--model", d, "--out", pack,
                        "--expert-fmt", fmt, "--dense-fmt", dense], check=True, stdout=subprocess.DEVNULL)
        mirror = os.path.join(d, "tiny_copy.moe")
        shutil.copyfile(pack, mirror)
        prompt = ",".join(str(x) for x in np.random.default_rng(5).integers(0, 512, 9))

        print("1) engine vs independent reference (teacher-forced on the engine's own output)")
        for th in ("1", "4"):
            eng, ids, _ = engine(pack, prompt, d, ["--threads", th])
            ref = ref_pack.Ref(pack)
            seq = [int(x) for x in ids.split(",")]
            R = np.stack([ref.forward(t) for t in seq[:-1]]).astype(np.float32)
            err, same = compare(eng, R)
            exact = int(np.argmax(err > 1e-5)) if (err > 1e-5).any() else len(err)
            good = bool(err[0] < 1e-5 and err.max() < 5e-2 and same >= 0.9 and len(eng) == len(R))
            ok &= good
            print(f"   {'OK ' if good else 'FAIL'} threads {th}: {len(eng)} steps, exact (<1e-5) for the first {exact}, "
                  f"max error {err.max():.1e}, same next token {same * 100:.0f}%")
        print("   (after the first int8 rounding flip, small differences carry forward; see README)")

        print("2) cache / prefetch / I/O / thread settings must give bit-identical results")
        base = None
        for name, extra in RUNS:
            eng, ids, stats = engine(pack, prompt, d, [a.replace("{mirror}", mirror) for a in extra])
            if base is None:
                base = (eng, ids)
            same = eng.shape == base[0].shape and np.array_equal(eng, base[0]) and ids == base[1]
            ok &= same
            print(f"   {'OK ' if same else 'FAIL'} {name:<30} " + " | ".join(s.split("|")[1].strip() if "|" in s else s for s in stats[:1]))
            for s in stats[1:]:
                print("        " + s)
    return ok


if __name__ == "__main__":
    main()
