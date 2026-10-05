#!/usr/bin/env bash
# Download datasets one after another into the data volume: one "lane" (runbook step 5).
#
#   bash download-lane.sh LANE DATASET [DATASET ...]
#
# Only datasets the founders approved (runbook step 5). Licence-gated datasets (Shin2017A) are
# refused by `neuros download` itself unless run with --accept-licence, which this script never
# passes: run those by hand, by the person who accepts the licence.
#
# Run several lanes in parallel, grouped by host (e.g. GigaDB, Figshare, Zenodo/OSF), so one slow
# server doesn't hold up the rest; see download-chunks.sh and download-retry.sh for throttling
# hosts. Within a lane, one dataset failing doesn't stop the next. Each dataset has its own log;
# the shared summary records lane, exit status, duration and free space. Run inside tmux.
set -uo pipefail

LANE="${1:?usage: download-lane.sh LANE DATASET [DATASET ...]}"
shift
[ "$#" -ge 1 ] || { echo "ERROR: no datasets given" >&2; exit 2; }

REPO="${NEUROS_REPO_DIR:-$HOME/NeurOS}"
MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
LOGS="$MOUNT/logs"
SUMMARY="$LOGS/downloads-summary.txt"

export PATH="$HOME/.local/bin:$PATH"   # uv, installed by bootstrap.sh
export PYTHONUNBUFFERED=1
# Temporary files on the data volume: the root disk (~48 GB free) is smaller than one archive.
export TMPDIR="$MOUNT/tmp"
mkdir -p "$TMPDIR" "$LOGS"
cd "$REPO" || { echo "ERROR: no repository at $REPO" >&2; exit 1; }
[ -L data ] || { echo "ERROR: data/ isn't linked to the volume: run bootstrap.sh first" >&2; exit 1; }

echo "lane $LANE started $(date -u +%FT%TZ) at commit $(git rev-parse --short HEAD): $*" \
  | tee -a "$SUMMARY"
for dataset in "$@"; do
  log="$LOGS/download-$dataset.log"
  start=$(date +%s)
  echo "== $dataset ($(date -u +%FT%TZ)); log: $log"
  uv run --no-sync neuros --config configs/data.yaml download --dataset "$dataset" >"$log" 2>&1
  status=$?
  minutes=$(( ($(date +%s) - start) / 60 ))
  free="$(df -h --output=avail "$MOUNT" | tail -1 | tr -d ' ')"
  echo "lane $LANE: $dataset exit=$status minutes=$minutes free=$free" | tee -a "$SUMMARY"
done
echo "lane $LANE finished $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
