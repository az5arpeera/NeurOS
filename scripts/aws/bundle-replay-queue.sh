#!/usr/bin/env bash
# Build a shipping bundle and replay it through the real engine on the 34 never-seen people
# of the 10 Oct replay check (knowledge-base/results/2026-10-10-mvp-v0-replay.md).
#
#   usage: bundle-replay-queue.sh NAME EXPERIMENT
#     NAME        the bundle's name, e.g. mvp-v1 (never overwritten)
#     EXPERIMENT  the recipe, e.g. configs/a6_mirepnet.yaml (A6's MIRepNet)
#
#   bundle-NAME   `neuros bundle`: the experiment's recipe on all 7 training datasets (GPU).
#   replay-NAME   `neuros simulate` on the 34 development people of BNCI2014_004 and
#                 GuttmannFlury2025 (never trained the bundle), first session, as fast as possible.
#
# The 10 Oct mvp-v0 queue (/mnt/neuros/aws_queue_bundle.sh, uncommitted) with the bundle and
# experiment as arguments; the people and steps are unchanged. Logs go to
# /mnt/neuros/logs/<step>.log, and "<step> exit=N minutes=M" lines to the train summary.
# Refuses uncommitted code (provenance).
set -uo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 NAME EXPERIMENT" >&2
  exit 2
fi
NAME=$1
EXPERIMENT=$2
case "$NAME" in
  *[!A-Za-z0-9._-]*|"") echo "ERROR: bundle name must be letters, digits, '.', '_' or '-'" >&2; exit 2 ;;
esac

REPO="$HOME/NeurOS"
LOGS=/mnt/neuros/logs
SUMMARY="$LOGS/train-summary.txt"
BUNDLE=/mnt/neuros/bundles/$NAME
REPLAYS=/mnt/neuros/results/replay/$NAME
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 TMPDIR=/mnt/neuros/tmp
# MNE/MOABB must never fall back to ~/mne_data and re-download (10 Oct incident).
export MNE_DATA=/mnt/neuros/data/raw MOABB_DOWNLOAD_PROVIDER=upstream
cd "$REPO" || { echo "ERROR: no repository at $REPO" >&2; exit 1; }
if [ ! -f "$EXPERIMENT" ]; then
  echo "ERROR: no experiment file $EXPERIMENT" >&2
  exit 1
fi
if [ -e "$BUNDLE/bundle.json" ]; then
  echo "ERROR: $BUNDLE already holds a bundle" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- src configs pyproject.toml uv.lock; then
  echo "ERROR: uncommitted changes in src/configs/pyproject/uv.lock: commit first" >&2
  exit 1
fi
mkdir -p /mnt/neuros/bundles "$REPLAYS" /mnt/neuros/tmp

start=$(date +%s)
echo "== bundle-$NAME ($(date -u +%FT%TZ)) at commit $(git rev-parse --short HEAD), $EXPERIMENT" | tee -a "$SUMMARY"
uv run --no-sync neuros --config configs/data.yaml bundle --experiment "$EXPERIMENT" \
  --out "$BUNDLE" --device cuda >"$LOGS/bundle-$NAME.log" 2>&1
status=$?
echo "bundle-$NAME exit=$status minutes=$(( ($(date +%s) - start) / 60 ))" | tee -a "$SUMMARY"
touch "$HOME/.neuros-keep-awake"
if [ "$status" -ne 0 ]; then
  echo "bundle queue stopped: the bundle failed $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
  exit 1
fi

# One replay per person; a failed replay is logged and counted (pass bar 1), never fatal.
start=$(date +%s)
failed=0
: >"$LOGS/replay-$NAME.log"
echo "== replay-$NAME ($(date -u +%FT%TZ))" | tee -a "$SUMMARY"
replay() {  # dataset, subject
  echo "--- $1 $2" >>"$LOGS/replay-$NAME.log"
  if ! uv run --no-sync neuros --config configs/data.yaml simulate --bundle "$BUNDLE" \
      --dataset "$1" --subject "$2" --out "$REPLAYS/$1_$2.json" >>"$LOGS/replay-$NAME.log" 2>&1; then
    echo "REPLAY FAILED $1 $2" >>"$LOGS/replay-$NAME.log"
    failed=$((failed + 1))
  fi
}
for s in 1 2 3 4 5 6 7 8 9; do replay BNCI2014_004 "$s"; done
for s in 1 2 4 5 7 8 9 10 14 15 16 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
  replay GuttmannFlury2025 "$s"
  touch "$HOME/.neuros-keep-awake"
done
echo "replay-$NAME exit=$failed minutes=$(( ($(date +%s) - start) / 60 )) failed=$failed" | tee -a "$SUMMARY"
echo "bundle queue finished $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
