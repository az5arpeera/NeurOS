#!/usr/bin/env bash
# Download one dataset as parallel chunks of people, then one full run (runbook step 5).
#
#   bash download-chunks.sh DATASET N_CHUNKS
#
# For hosts that throttle each connection: Zenodo gave ~0.8-2.5 MB/s per file on 5 Oct, so
# GuttmannFlury2025's 26 GB would take ~9 h in one stream; 4 chunks ran at ~10 MB/s together.
# Each chunk is its own `neuros download --subjects ...`. Chunks would overwrite each other's
# manifest, so a final full run finds every file already present (nothing is fetched twice),
# fetches whatever a chunk failed to get, verifies archives against the publisher's checksums and
# writes the complete manifest. Same approvals and licence rule as download-lane.sh.
set -uo pipefail

DATASET="${1:?usage: download-chunks.sh DATASET N_CHUNKS}"
CHUNKS="${2:?usage: download-chunks.sh DATASET N_CHUNKS}"
if ! [[ "$CHUNKS" =~ ^[1-9][0-9]?$ ]]; then
  echo "ERROR: N_CHUNKS must be a whole number from 1 to 99, got '$CHUNKS'" >&2
  exit 2
fi

REPO="${NEUROS_REPO_DIR:-$HOME/NeurOS}"
MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
LOGS="$MOUNT/logs"
SUMMARY="$LOGS/downloads-summary.txt"
export PATH="$HOME/.local/bin:$PATH"   # uv, installed by bootstrap.sh
export PYTHONUNBUFFERED=1
export TMPDIR="$MOUNT/tmp"             # the root disk is small
mkdir -p "$TMPDIR" "$LOGS"
cd "$REPO" || { echo "ERROR: no repository at $REPO" >&2; exit 1; }
[ -L data ] || { echo "ERROR: data/ isn't linked to the volume: run bootstrap.sh first" >&2; exit 1; }
DOWNLOAD=(uv run --no-sync neuros --config configs/data.yaml download --dataset "$DATASET")

# The registry's people in CHUNKS contiguous groups, one group per line.
mapfile -t groups < <(uv run --no-sync python - "$DATASET" "$CHUNKS" <<'PY'
import sys

from neuros.data.registry import get_spec

subjects = list(get_spec(sys.argv[1]).subjects)
size = -(-len(subjects) // int(sys.argv[2]))  # ceiling division
for start in range(0, len(subjects), size):
    print(" ".join(str(s) for s in subjects[start:start + size]))
PY
)
if [ "${#groups[@]}" -lt 1 ]; then
  echo "ERROR: no people found for $DATASET (is it in the registry?)" >&2
  exit 1
fi

start=$(date +%s)
echo "lane $DATASET started $(date -u +%FT%TZ) at commit $(git rev-parse --short HEAD):" \
  "${#groups[@]} parallel chunks" | tee -a "$SUMMARY"
pids=()
for group in "${groups[@]}"; do
  first="${group%% *}"
  # shellcheck disable=SC2086  # the group is a space-separated list of numbers
  "${DOWNLOAD[@]}" --subjects $group >"$LOGS/download-$DATASET-from-$first.log" 2>&1 &
  pids+=("$!")
  sleep 5  # stagger the first requests
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$(( failed + 1 ))
done
echo "lane $DATASET: chunks done ($failed failed); full run writes the manifest" | tee -a "$SUMMARY"

"${DOWNLOAD[@]}" >"$LOGS/download-$DATASET.log" 2>&1
status=$?
minutes=$(( ($(date +%s) - start) / 60 ))
free="$(df -h --output=avail "$MOUNT" | tail -1 | tr -d ' ')"
echo "lane $DATASET: $DATASET exit=$status minutes=$minutes free=$free" | tee -a "$SUMMARY"
