#!/usr/bin/env bash
# Run leave-one-person-out experiments on the GPU, one after another (runbook step 7).
#
#   bash loso-queue.sh EXPERIMENT [EXPERIMENT ...]      # names of configs/<EXPERIMENT>.yaml
#   WORKERS=3 bash loso-queue.sh e0b_eegnet_bnci2014_001_4_30 e0b_eegnet_bnci2014_001_8_30
#
# Each run logs to <volume>/logs/train-<name>.log; results land in results/<name>/ on the data
# volume; the summary records exit status and minutes. WORKERS parallel folds (default 3) share
# the one GPU, 2 PyTorch threads each: on an 8-vCPU g6.2xlarge, 3 leaves room for downloads or
# validation running alongside. Locked test people are refused by `neuros loso` itself. It
# refuses to start on uncommitted code, which provenance would flag as dirty. Run inside tmux.
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: loso-queue.sh EXPERIMENT [EXPERIMENT ...]" >&2; exit 2; }
REPO="${NEUROS_REPO_DIR:-$HOME/NeurOS}"
MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
LOGS="$MOUNT/logs"
SUMMARY="$LOGS/train-summary.txt"
WORKERS="${WORKERS:-3}"
[[ "$WORKERS" =~ ^[1-9][0-9]?$ ]] || { echo "ERROR: WORKERS must be 1-99" >&2; exit 2; }

export PATH="$HOME/.local/bin:$PATH"   # uv, installed by bootstrap.sh
export PYTHONUNBUFFERED=1
cd "$REPO" || { echo "ERROR: no repository at $REPO" >&2; exit 1; }
mkdir -p "$LOGS"
if ! git diff --quiet HEAD -- src configs pyproject.toml uv.lock; then
  echo "ERROR: uncommitted changes in src/configs/pyproject/uv.lock: commit first" >&2
  exit 1
fi
for name in "$@"; do  # all configs must exist before the first run starts
  [ -f "configs/$name.yaml" ] || { echo "ERROR: configs/$name.yaml is missing" >&2; exit 2; }
done

for name in "$@"; do
  start=$(date +%s)
  echo "== $name ($(date -u +%FT%TZ)) at commit $(git rev-parse --short HEAD), $WORKERS workers"
  uv run --no-sync neuros --config configs/data.yaml loso --experiment "configs/$name.yaml" \
    --device cuda --workers "$WORKERS" --threads 2 >"$LOGS/train-$name.log" 2>&1
  status=$?
  minutes=$(( ($(date +%s) - start) / 60 ))
  echo "$name exit=$status minutes=$minutes" | tee -a "$SUMMARY"
done
echo "queue finished $(date -u +%FT%TZ): $*" | tee -a "$SUMMARY"
