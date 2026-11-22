#!/usr/bin/env bash
# Step 0 / SSD benchmark for the SSD-MoE engine.
# Run on the N5105 box (Linux). Needs: fio (sudo apt install fio), python3.
#
# Usage:  ./bench_ssd.sh /path/on/the/nvme/testdir   [size_gb, default 16]
#
# What it measures (all with O_DIRECT, i.e. no page cache, like the engine):
#   1. PCIe link actually negotiated by the NVMe drive (Gen / lanes)
#   2. Random reads at expert-sized blocks (1, 2, 4 MiB) across queue depths
#   3. Random 64 KiB reads (KV-cache page size candidates)
#   4. A 120 s sustained run to expose thermal throttling
# It writes a real test file first (reading never-written blocks gives fake
# speeds). That is one write of SIZE_GB; it does not meaningfully wear the SSD.

set -euo pipefail

DIR="${1:-}"
SIZE_GB="${2:-16}"
if [[ -z "$DIR" ]]; then
  echo "usage: $0 /path/on/nvme/testdir [size_gb]" >&2; exit 1
fi
command -v fio >/dev/null || { echo "fio not found: sudo apt install fio" >&2; exit 1; }
mkdir -p "$DIR"
FILE="$DIR/moe_bench.dat"
OUT="$(pwd)/ssd_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# --- filesystem checks: results are only meaningful if reads really hit the SSD
FS=$(stat -f -c %T "$DIR" 2>/dev/null || echo unknown)
case "$FS" in
  overlay*|tmpfs|ramfs)
    echo "ERROR: $DIR is on '$FS' (container or RAM filesystem), not on the NVMe." >&2
    echo "In a container, bind-mount a folder on the NVMe, e.g. -v /mnt/nvme/moe-bench:/bench" >&2
    exit 1 ;;
  zfs)
    echo "NOTE: ZFS detected. ZFS caches reads in RAM (ARC), which inflates results."
    echo "      Best: run on a dataset with 'zfs set primarycache=metadata <pool>/<dataset>'."
    echo "      Otherwise use a test file of at least 3x RAM (second argument, in GB)." ;;
  fuse*|nfs*|cifs|smb*)
    echo "WARNING: $DIR is on a network/FUSE filesystem ($FS); this measures that layer, not the SSD." ;;
esac

# --- pick an I/O engine that actually works here (Docker's default seccomp profile blocks io_uring)
probe() { fio --name=probe --filename="$DIR/.moe_probe" --size=4M --rw=write --bs=64k --direct=1 \
              --ioengine="$1" --output=/dev/null >/dev/null 2>&1; }
ENGINE=""
for e in io_uring libaio psync; do
  if probe "$e"; then ENGINE=$e; break; fi
done
rm -f "$DIR/.moe_probe"
if [[ -z "$ENGINE" ]]; then
  echo "ERROR: O_DIRECT reads/writes fail on $DIR ($FS). Use an ext4/xfs folder on the NVMe." >&2
  exit 1
fi
[[ "$ENGINE" != io_uring ]] && echo "NOTE: io_uring not available (in Docker: add --security-opt seccomp=unconfined); using $ENGINE."

echo "== System" | tee "$OUT/summary.txt"
{
  uname -r
  grep -m1 "model name" /proc/cpuinfo || true
  echo "RAM: $(awk '/MemTotal/ {printf "%.1f GB", $2/1e6}' /proc/meminfo) | test dir filesystem: $FS | io engine: $ENGINE"
  for d in /sys/class/nvme/nvme*; do
    [[ -e "$d" ]] || continue
    n=$(basename "$d")
    model=$(cat "$d/model" 2>/dev/null | xargs)
    spd=$(cat "$d/device/current_link_speed" 2>/dev/null || echo "?")
    wid=$(cat "$d/device/current_link_width" 2>/dev/null || echo "?")
    mspd=$(cat "$d/device/max_link_speed" 2>/dev/null || echo "?")
    mwid=$(cat "$d/device/max_link_width" 2>/dev/null || echo "?")
    echo "$n: $model | link now: $spd x$wid | drive max: $mspd x$mwid"
  done
  echo "(8.0 GT/s = PCIe Gen3, 16.0 GT/s = Gen4. x2 roughly halves throughput vs x4.)"
} | tee -a "$OUT/summary.txt"

nvme_temp() {
  for h in /sys/class/nvme/nvme0/hwmon*/temp1_input /sys/class/nvme/nvme0/device/hwmon/hwmon*/temp1_input; do
    [[ -r "$h" ]] && { echo "$(( $(cat "$h") / 1000 ))C"; return; }
  done
  echo "n/a"
}

echo "== Laying out ${SIZE_GB} GiB test file (one-time write)..." | tee -a "$OUT/summary.txt"
fio --name=layout --filename="$FILE" --size="${SIZE_GB}G" --rw=write --bs=4M \
    --direct=1 --ioengine="$ENGINE" --iodepth=16 --output=/dev/null

run() { # name bs iodepth runtime
  local name=$1 bs=$2 qd=$3 rt=$4
  fio --name="$name" --filename="$FILE" --size="${SIZE_GB}G" --rw=randread \
      --bs="$bs" --direct=1 --ioengine="$ENGINE" --iodepth="$qd" --numjobs=1 \
      --time_based --runtime="$rt" --ramp_time=2 --norandommap --randrepeat=0 \
      --output-format=json --output="$OUT/$name.json" \
      ${EXTRA:-}
  python3 - "$OUT/$name.json" "$name" "$(nvme_temp)" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys
j = json.load(open(sys.argv[1]))["jobs"][0]["read"]
bw = j["bw_bytes"] / 1e9
lat = j.get("clat_ns", {})
p50 = lat.get("percentile", {}).get("50.000000", 0) / 1e6
p99 = lat.get("percentile", {}).get("99.000000", 0) / 1e6
print(f"{sys.argv[2]:<22} {bw:6.2f} GB/s   p50 {p50:7.2f} ms   p99 {p99:7.2f} ms   nvme {sys.argv[3]}")
PY
}

echo "== Expert-sized random reads (GB/s = 1e9 bytes/s)" | tee -a "$OUT/summary.txt"
for bs in 1M 2M 4M; do
  for qd in 1 4 16 32 64; do
    run "rand_${bs}_qd${qd}" "$bs" "$qd" 15
  done
done

echo "== Small random reads (KV page candidates)" | tee -a "$OUT/summary.txt"
for qd in 8 32; do run "rand_64K_qd${qd}" 64K "$qd" 15; done

echo "== Sustained: 6 back-to-back 20 s runs at 2M / QD32 (falling GB/s = throttling)" | tee -a "$OUT/summary.txt"
for i in 1 2 3 4 5 6; do run "sustained_2M_qd32_part$i" 2M 32 20; done

echo
echo "Done. Results in $OUT  (send me summary.txt)"
echo "Test file left at $FILE; delete it yourself when finished."
