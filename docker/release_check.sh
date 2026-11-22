#!/bin/sh
# Release check: engine self-test, then speed runs with the same 512-token prompt and
# 128 generated tokens, inside a container limited to 7 GB of RAM.
#
#   MODELS=/path/to/packs [MIRROR=/path/on/a/second/drive] sh docker/release_check.sh
#
# MODELS must hold prompt512.txt and the packs qwen3-q4b-d4.moe (4-bit attention) and
# qwen3-30b-a3b.moe (8-bit attention); see docs/REPRODUCE.md. MIRROR (optional) holds a
# copy of qwen3-30b-a3b.moe on a second SSD. Run as root (it drops the OS file cache
# between runs so every run starts cold).
MODELS=${MODELS:-/volume1/moe}
MIRROR=${MIRROR:-}
OUT=${OUT:-$MODELS/release_check_results.txt}
: > "$OUT"

echo "== engine self-test (tiny model, checks correctness on this CPU)" | tee -a "$OUT"
docker run --rm --entrypoint sh moe-bench -c "cd /kit/engine && python3 tools/test_engine.py /usr/local/bin/moe-engine" 2>&1 \
  | grep -E "FAIL|PASSED" | tee -a "$OUT"

run() {
  name=$1; shift
  echo "" | tee -a "$OUT"; echo "== $name" | tee -a "$OUT"
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "(could not drop the file cache: not root)"
  mirror_mount=""
  [ -n "$MIRROR" ] && mirror_mount="-v $MIRROR:/models2"
  docker run --rm --memory 7g --memory-swap 7g --entrypoint moe-engine \
    -v "$MODELS":/models $mirror_mount moe-bench \
    "$@" --prompt-file /models/prompt512.txt --max-new 128 --ram-gb 7 2>&1 \
    | grep -E "^(RAM|prefill|decode|expert blocks|heat|SSD:|error|note)" | tee -a "$OUT"
}

run "4-bit attention pack, 1 drive" --pack /models/qwen3-q4b-d4.moe
run "4-bit attention pack, 1 drive, with --refine (second prediction)" --pack /models/qwen3-q4b-d4.moe --refine
run "8-bit attention pack (best quality), 1 drive" --pack /models/qwen3-30b-a3b.moe
if [ -n "$MIRROR" ]; then
  run "8-bit attention pack (best quality), 2 drives" \
      --pack /models/qwen3-30b-a3b.moe --mirror /models2/qwen3-30b-a3b.moe
fi

echo ""
echo "All results: $OUT"
