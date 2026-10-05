#!/usr/bin/env bash
# Download one dataset, retrying politely when its host rate-limits us (runbook step 5).
#
#   bash download-retry.sh DATASET [MAX_ATTEMPTS] [FIRST_WAIT_MIN]
#
# OSF (Dreyer2023's host) answered a burst with "429 Too Many Requests" on 5 Oct after ~3 min.
# Files already on disk are skipped on the next attempt, so each retry continues where the last
# one stopped. Waits double (10, 20, 40, 60, 60 ... minutes by default). While waiting, it holds
# the idle auto-stop off (~/.neuros-keep-awake), or the machine could shut down between attempts.
# Any other failure stops at once. Same approvals and licence rule as download-lane.sh.
set -uo pipefail

DATASET="${1:?usage: download-retry.sh DATASET [MAX_ATTEMPTS] [FIRST_WAIT_MIN]}"
MAX_ATTEMPTS="${2:-12}"
WAIT_MIN="${3:-10}"
for value in "$MAX_ATTEMPTS" "$WAIT_MIN"; do
  if ! [[ "$value" =~ ^[1-9][0-9]{0,2}$ ]]; then
    echo "ERROR: attempts and wait must be whole numbers from 1 to 999, got '$value'" >&2
    exit 2
  fi
done

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

start=$(date +%s)
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  log="$LOGS/download-$DATASET-attempt-$attempt.log"
  uv run --no-sync neuros --config configs/data.yaml download --dataset "$DATASET" >"$log" 2>&1
  status=$?
  minutes=$(( ($(date +%s) - start) / 60 ))
  free="$(df -h --output=avail "$MOUNT" | tail -1 | tr -d ' ')"
  echo "lane $DATASET: attempt $attempt exit=$status minutes=$minutes free=$free" \
    | tee -a "$SUMMARY"
  if [ "$status" -eq 0 ]; then
    cp "$log" "$LOGS/download-$DATASET.log"
    exit 0
  fi
  if ! grep -aq "Too Many Requests" "$log"; then
    echo "lane $DATASET: not a rate limit (see $log); stopping" | tee -a "$SUMMARY"
    exit "$status"
  fi
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    touch "$HOME/.neuros-keep-awake"   # holds the idle auto-stop for 3 h
    echo "lane $DATASET: rate-limited; next attempt in $WAIT_MIN min" | tee -a "$SUMMARY"
    sleep $(( WAIT_MIN * 60 ))
    WAIT_MIN=$(( WAIT_MIN * 2 > 60 ? 60 : WAIT_MIN * 2 ))
  fi
done
echo "lane $DATASET: still rate-limited after $MAX_ATTEMPTS attempts" | tee -a "$SUMMARY"
exit 1
