"""Quality of "simpler math": run the model with each weight/activation format
and compare perplexity against full precision. Run on the Mac.

Usage:
  python quant_sweep.py --model ~/models/Qwen3-30B-A3B --text corpus.txt               # formats
  python quant_sweep.py --model ~/models/Qwen3-30B-A3B --text corpus.txt --set neuron  # neuron skipping
  python quant_sweep.py --model ~/models/Qwen3-30B-A3B --text corpus.txt --set group   # real rule + groups
Each config takes about as long as one trace with --num-seqs (default 4).
Results go to <set>_sweep_results.txt.
"""
import argparse
import os
import re
import subprocess
import sys
import time

# name: (expert bits, dense bits, activation bits, neuron skip, what it corresponds to)
SETS = {
    "quant": [
        ("full", "16", "16", 0, 0.0, "bf16 reference"),
        ("e8", "8", "16", 0, 0.0, "q8 multiply kernel, experts"),
        ("e4", "4", "16", 0, 0.0, "q4 multiply kernel, experts"),
        ("e4_a6", "4", "16", 6, 0.0, "lut4 kernel (6-bit activations)"),
        ("e3", "3", "16", 0, 0.0, "3-bit experts"),
        ("e2", "2", "16", 0, 0.0, "2-bit experts"),
        ("e2_a6", "2", "16", 6, 0.0, "lut2 kernel (6-bit activations)"),
        ("etern", "tern", "16", 0, 0.0, "ternary experts, no retraining"),
        ("plan", "3", "4", 8, 0.0, "engine plan: 3b experts, 4b dense, 8b acts"),
    ],
    # oracle = a perfect predictor of which neurons matter; the upper bound for the idea
    "neuron": [
        ("full", "16", "16", 0, 0.0, "bf16 reference (+ neuron statistics)"),
        ("s30", "16", "16", 0, 0.3, "skip 30% of neurons per expert (oracle)"),
        ("s50", "16", "16", 0, 0.5, "skip 50% (oracle)"),
        ("s70", "16", "16", 0, 0.7, "skip 70% (oracle)"),
        ("s85", "16", "16", 0, 0.85, "skip 85% (oracle)"),
        ("e4_s50", "4", "16", 0, 0.5, "4-bit experts + skip 50% (oracle)"),
    ],
    # real (gate-first) skipping, and skipping whole groups of neurons = SSD read units
    "group": [
        ("full", "16", "16", 0, 0.0, "bf16 reference"),
        ("gate30", "16", "16", 0, 0.3, "gate-first, skip 30%, single neurons", ["--skip-mode", "gate"]),
        ("gate50", "16", "16", 0, 0.5, "gate-first, skip 50%, single neurons", ["--skip-mode", "gate"]),
        ("o50_g32p", "16", "16", 0, 0.5, "oracle 50%, groups of 32, original order",
         ["--neuron-group", "32", "--group-order", "plain"]),
        ("o50_g32c", "16", "16", 0, 0.5, "oracle 50%, groups of 32, clustered",
         ["--neuron-group", "32", "--group-order", "cluster"]),
        ("g50_g16c", "16", "16", 0, 0.5, "gate-first 50%, groups of 16, clustered",
         ["--skip-mode", "gate", "--neuron-group", "16", "--group-order", "cluster"]),
        ("g50_g32c", "16", "16", 0, 0.5, "gate-first 50%, groups of 32, clustered",
         ["--skip-mode", "gate", "--neuron-group", "32", "--group-order", "cluster"]),
        ("g50_g64c", "16", "16", 0, 0.5, "gate-first 50%, groups of 64, clustered",
         ["--skip-mode", "gate", "--neuron-group", "64", "--group-order", "cluster"]),
        ("e4_g50_g32c", "4", "16", 0, 0.5, "4-bit + gate-first 50%, groups of 32, clustered",
         ["--skip-mode", "gate", "--neuron-group", "32", "--group-order", "cluster"]),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--ids", help=".npy token ids instead of text")
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--set", default="quant", choices=sorted(SETS), help="which experiment")
    ap.add_argument("--only", default="", help="comma-separated config names to run")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    todo = [c for c in SETS[args.set] if not args.only or c[0] in args.only.split(",")]
    results = []
    out_path = f"{args.set}_sweep_results.txt"
    for i, (name, eb, db, ab, skip, what, *extra) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name}: {what}", flush=True)
        t0 = time.time()
        cmd = [sys.executable, os.path.join(here, "trace_routing.py"), "--model", args.model,
               *(["--text", args.text] if args.text else ["--ids", args.ids]), "--seq-len", str(args.seq_len), "--num-seqs", str(args.num_seqs),
               "--expert-bits", eb, "--dense-bits", db, "--act-bits", str(ab),
               "--neuron-skip", str(skip), "--out", f"trace_{args.set}_{name}.npz", *(extra[0] if extra else [])]
        if args.set == "neuron" and name == "full":
            cmd.append("--neuron-stats")
        p = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(r"PPL=([0-9.]+|nan|inf)", p.stdout)
        if p.returncode != 0 or not m:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f"config {name} failed")
        ppl = float(m.group(1))
        results.append((name, what, ppl))
        print(f"   perplexity {ppl:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

        base = next((r[2] for r in results if r[0] == "full"), None)
        lines = [f"{'config':<12} {'what':<50} {'perplexity':>10} {'vs full':>8}"]
        for n, w, v in results:
            rel = f"{(v / base - 1) * 100:+.1f}%" if base else ""
            lines.append(f"{n:<12} {w:<50} {v:>10.3f} {rel:>8}")
        lines.append("\nRough guide: under +3% is hard to notice, +5-10% is noticeably weaker,"
                     "\nover +20% is a clearly worse model.")
        if args.set == "quant":
            lines.append("Weight rounding here is the simple kind; better quantizers lose less, especially at 2-3 bits.")
        elif args.set == "group":
            lines.append("gate-first is a REAL rule: it computes the gate projection (1/3 of the work) for all neurons"
                         "\nand skips up/down for the rest. Groups = neurons stored together on the SSD; at 4 bits one"
                         "\nneuron is ~3 KB, so groups of 16/32/64 are reads of ~48/96/192 KB. Clustered groups are"
                         "\nlearned from the first half of the sequences.")
        else:
            lines.append("Skipping is an ORACLE (it peeks at the true activations): a real predictor can only do worse."
                         "\nSkipped neurons need no SSD read and no compute: 50% skip = half the expert bytes and work.")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    print("\n" + open(out_path).read())


if __name__ == "__main__":
    main()
