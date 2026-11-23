#!/usr/bin/env bash
# Runs inside the container. Results go to /bench/results_<time>/
#   MODE=cpu     CPU kernel benchmark only
#   MODE=ssd     SSD benchmark only (needs /bench on the NVMe)
#   MODE=engine  full engine speed test with a random-weight pack of real size (~20 GB on /bench)
#   MODE=all     cpu + ssd (default)
#   SSD_GB=16    size of the SSD test file;  CACHE_GB=4  engine expert cache
# Extra arguments are passed to int8bench (cpu) or moe-engine (engine).
set -uo pipefail
MODE="${MODE:-all}"
SSD_GB="${SSD_GB:-16}"
CACHE_GB="${CACHE_GB:-4}"
mkdir -p /bench
OUT="/bench/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

if [[ "$MODE" == cpu || "$MODE" == all ]]; then
  echo "===== CPU benchmark (a few minutes; keep the NAS otherwise idle)"
  int8bench "$@" | tee "$OUT/int8bench.txt"
  echo; echo "===== engine's own expert kernels (cache layout, huge pages if available)"
  moe-engine --bench-kernels --threads 4 | tee -a "$OUT/int8bench.txt"
  moe-engine --bench-kernels --threads 1 | tee -a "$OUT/int8bench.txt"
fi

if [[ "$MODE" == ssd || "$MODE" == all ]]; then
  echo; echo "===== SSD benchmark"
  ( cd "$OUT" && /kit/ssd/bench_ssd.sh /bench/data "$SSD_GB" )
  rm -f /bench/data/moe_bench.dat; echo "(test file deleted)"
fi

if [[ "$MODE" == engine ]]; then
  PACK=/bench/bench48.moe
  if [[ ! -f "$PACK" ]]; then
    echo "===== writing a 20 GB random-weight pack with Qwen3-30B-A3B's shape (one time)"
    python3 /kit/engine/tools/make_bench_pack.py --out "$PACK"
  fi
  for cg in "$CACHE_GB" 0; do
    echo; echo "===== engine, expert cache ${cg} GB"
    moe-engine --pack "$PACK" --prompt-ids 1,2,3,4,5,6,7,8 --max-new 24 --cache-gb "$cg" "$@" \
      2>&1 | sed 's/<[0-9]*>//g' | tee "$OUT/engine_cache${cg}.txt"
  done
  echo "(the pack stays at $PACK for later runs; delete it to free 20 GB)"
fi
echo; echo "All results are in $OUT on the NAS (the folder you mounted as /bench)."
