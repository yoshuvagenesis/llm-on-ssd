#!/bin/sh
# Release check: engine self-test, speed runs (512-token prompt, 128 generated tokens)
# and a perplexity check for each pack found, inside a container limited to 7 GB of RAM.
#
#   MODELS=/path/to/packs [MIRROR=/path/on/a/second/drive] sh docker/release_check.sh
#
# Packs looked for in MODELS (missing ones are skipped):
#   qwen3-q4b-d4.moe, qwen3-30b-a3b.moe        Qwen3-30B-A3B, with tokenizer.json
#   qwen3.6-35b-a3b.moe, qwen3.6-35b-a3b-d4.moe Qwen3.6-35B-A3B, with tokenizer-qwen3.6.json
# The Qwen3 runs use prompt512.txt if it is there (comparable with earlier results),
# otherwise the engine builds the same kind of 512-token prompt itself. MIRROR (optional)
# holds a copy of qwen3-30b-a3b.moe on a second SSD. Run as root (it drops the OS file
# cache between runs so every run starts cold).
MODELS=${MODELS:-/volume1/moe}
MIRROR=${MIRROR:-}
OUT=${OUT:-$MODELS/release_check_results.txt}
: > "$OUT"

echo "== engine self-test (tiny models, checks correctness on this CPU)" | tee -a "$OUT"
docker run --rm --entrypoint sh moe-bench -c "cd /kit/engine && python3 tools/test_engine.py /usr/local/bin/moe-engine" 2>&1 \
  | grep -E "FAIL|PASSED" | tee -a "$OUT"

engine() {
  mirror_mount=""
  [ -n "$MIRROR" ] && mirror_mount="-v $MIRROR:/models2"
  docker run --rm --memory 7g --memory-swap 7g --entrypoint moe-engine \
    -v "$MODELS":/models $mirror_mount moe-bench --ram-gb 7 "$@" 2>&1
}

speed() {  # name, pack, tokenizer, extra args...
  name=$1; pack=$2; tok=$3; shift 3
  [ -f "$MODELS/$pack" ] || return 0
  echo "" | tee -a "$OUT"; echo "== $name" | tee -a "$OUT"
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "(could not drop the file cache: not root)"
  if [ "$tok" = tokenizer.json ] && [ -f "$MODELS/prompt512.txt" ]; then
    p="--prompt-file /models/prompt512.txt"
  else
    p="--tokenizer /models/$tok --bench-prompt 512 --bench-text /kit/trace/corpus.txt"
  fi
  engine --pack "/models/$pack" $p --max-new 128 "$@" \
    | grep -E "^(model|RAM|prefill|decode|expert blocks|heat|SSD:|error|note)" | tee -a "$OUT"
}

quality() {  # pack, tokenizer
  [ -f "$MODELS/$1" ] && [ -f "$MODELS/$2" ] || return 0
  echo "" | tee -a "$OUT"; echo "== perplexity on trace/corpus.txt (1024 tokens): $1" | tee -a "$OUT"
  engine --pack "/models/$1" --tokenizer "/models/$2" --perplexity /kit/trace/corpus.txt --max-tokens 1024 \
    | grep -E "^(perplexity|error)" | tee -a "$OUT"
}

T36=tokenizer-qwen3.6.json
speed "Qwen3-30B-A3B, 4-bit attention pack, 1 drive" qwen3-q4b-d4.moe tokenizer.json
speed "Qwen3-30B-A3B, 8-bit attention pack, 1 drive" qwen3-30b-a3b.moe tokenizer.json
if [ -n "$MIRROR" ]; then
  speed "Qwen3-30B-A3B, 8-bit attention pack, 2 drives" qwen3-30b-a3b.moe tokenizer.json \
    --mirror /models2/qwen3-30b-a3b.moe
fi
speed "Qwen3.6-35B-A3B, 8-bit dense pack, 1 drive" qwen3.6-35b-a3b.moe $T36
speed "Qwen3.6-35B-A3B, 4-bit dense pack, 1 drive" qwen3.6-35b-a3b-d4.moe $T36
quality qwen3-q4b-d4.moe tokenizer.json
quality qwen3-30b-a3b.moe tokenizer.json
quality qwen3.6-35b-a3b.moe $T36
quality qwen3.6-35b-a3b-d4.moe $T36

echo ""
echo "All results: $OUT"
