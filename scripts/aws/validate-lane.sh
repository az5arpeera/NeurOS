#!/usr/bin/env bash
# Validate each dataset as soon as its download has finished (runbook step 5, last command).
#
#   bash validate-lane.sh DATASET [DATASET ...]
#
# Waits until the download summary reports the dataset done (exit 0, from any lane style), then
# runs `neuros validate` (harmonise + validate every recording, JSON report) at low CPU priority
# (nice 10), so training running alongside keeps the CPU. A dataset whose download hasn't finished
# within MAX_WAIT_H hours (default 8) is skipped and reported. Run inside tmux.
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: validate-lane.sh DATASET [DATASET ...]" >&2; exit 2; }
REPO="${NEUROS_REPO_DIR:-$HOME/NeurOS}"
MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
LOGS="$MOUNT/logs"
DOWNLOADS="$LOGS/downloads-summary.txt"
SUMMARY="$LOGS/validate-summary.txt"
MAX_WAIT_H="${MAX_WAIT_H:-8}"
[[ "$MAX_WAIT_H" =~ ^[1-9][0-9]?$ ]] || { echo "ERROR: MAX_WAIT_H must be 1-99" >&2; exit 2; }

export PATH="$HOME/.local/bin:$PATH"   # uv, installed by bootstrap.sh
export PYTHONUNBUFFERED=1
export TMPDIR="$MOUNT/tmp"
mkdir -p "$TMPDIR" "$LOGS"
cd "$REPO" || { echo "ERROR: no repository at $REPO" >&2; exit 1; }

# Done lines: "lane A: Cho2017 exit=0", "lane GuttmannFlury2025: GuttmannFlury2025 exit=0" or
# "lane Dreyer2023: attempt 2 exit=0".
downloaded() {
  grep -aqE "^lane [^:]+: $1 exit=0|^lane $1: attempt [0-9]+ exit=0" "$DOWNLOADS" 2>/dev/null
}

for dataset in "$@"; do
  waited=0
  until downloaded "$dataset"; do
    if [ "$waited" -ge $(( MAX_WAIT_H * 3600 )) ]; then
      echo "validate $dataset: download not finished after $MAX_WAIT_H h; skipped" \
        | tee -a "$SUMMARY"
      continue 2
    fi
    sleep 60
    waited=$(( waited + 60 ))
  done
  start=$(date +%s)
  echo "== validate $dataset ($(date -u +%FT%TZ)) at commit $(git rev-parse --short HEAD)"
  nice -n 10 uv run --no-sync neuros --config configs/data.yaml validate --dataset "$dataset" \
    >"$LOGS/validate-$dataset.log" 2>&1
  status=$?
  minutes=$(( ($(date +%s) - start) / 60 ))
  echo "validate $dataset exit=$status minutes=$minutes" | tee -a "$SUMMARY"
done
echo "validation lane finished $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
