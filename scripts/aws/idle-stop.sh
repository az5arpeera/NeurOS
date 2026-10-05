#!/usr/bin/env bash
# NeurOS idle auto-stop: shut the training machine down after a stretch with nothing to do.
#
# Why. The likeliest way to waste credits is an instance left running overnight with nothing on
# it: a g6.4xlarge costs about $32 a day. The budget guard (infra/aws/budget-guard.yaml) caps the
# month, but it acts on billing figures that lag by hours. This acts within minutes, and costs
# nothing.
#
# How. A systemd timer runs this every 5 minutes (installed by scripts/aws/bootstrap.sh). Each
# run decides whether the machine is busy. It counts as busy if any of these holds:
#   - a NeurOS command is running (training, downloads, validation, audits, tests);
#   - the GPU is in use (utilisation above IDLE_GPU_PERCENT);
#   - the CPU is in use (1-minute load above IDLE_LOAD_PER_CORE x cores);
#   - someone typed in an SSH terminal in the last IDLE_TTY_MINUTES;
#   - someone ran `touch ~/.neuros-keep-awake` in the last KEEP_AWAKE_HOURS.
# After IDLE_MINUTES of continuous idleness it shuts down. For an EBS-backed instance with the
# default "Shutdown behavior: Stop", shutting down STOPS it: the disks and their data stay, and
# billing drops to storage only. It never terminates anything.
#
#   idle-stop.sh            one check (what the timer runs)
#   idle-stop.sh --status   print the decision and its reasons; never shuts down
#
# Settings can be overridden in /etc/default/neuros-idle-stop (KEY=value lines).
set -euo pipefail

IDLE_MINUTES=45          # continuous idleness before shutting down
IDLE_GPU_PERCENT=5       # GPU utilisation at or below this counts as idle
IDLE_LOAD_PER_CORE=0.10  # 1-minute load at or below this x cores counts as idle
IDLE_TTY_MINUTES=20      # an SSH terminal typed in more recently than this keeps it awake
KEEP_AWAKE_HOURS=3       # `touch ~/.neuros-keep-awake` holds the machine this long
STATE_FILE=/var/lib/neuros-idle-stop/idle-since
KEEP_AWAKE_GLOB="/home/*/.neuros-keep-awake"
# shellcheck disable=SC1091
[ -r /etc/default/neuros-idle-stop ] && . /etc/default/neuros-idle-stop

status_only=0
[ "${1:-}" = "--status" ] && status_only=1
reasons=()

# 1. A NeurOS command running (python -m neuros, the neuros entry point, pytest on the repo).
if pgrep -f -- "(neuros(\.exe)? |bin/neuros |-m neuros|pytest)" >/dev/null 2>&1; then
  reasons+=("a NeurOS command is running")
fi

# 2. The GPU in use. A driver that doesn't answer counts as busy: never stop on a broken probe.
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$( { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
           || echo unknown; } | sort -n | tail -1 | tr -d ' ' )
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    reasons+=("GPU state unknown (nvidia-smi: ${gpu:-no output})")
  elif [ "$gpu" -gt "$IDLE_GPU_PERCENT" ]; then
    reasons+=("GPU at ${gpu}%")
  fi
fi

# 3. The CPU in use.
cores=$(nproc)
load=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$load" -v c="$cores" -v t="$IDLE_LOAD_PER_CORE" 'BEGIN { exit !(l > c * t) }'; then
  reasons+=("load ${load} on ${cores} cores")
fi

# 4. Someone typing in an SSH terminal ("who -u" idle column: "." = active this minute).
while read -r idle; do
  case "$idle" in
    .) reasons+=("an SSH terminal is active"); break ;;
    old) ;;
    *:*)
      minutes=$(( 10#${idle%%:*} * 60 + 10#${idle##*:} ))
      if [ "$minutes" -lt "$IDLE_TTY_MINUTES" ]; then
        reasons+=("an SSH terminal was used ${minutes} min ago")
        break
      fi ;;
  esac
done < <(who -u 2>/dev/null | awk '{ print $5 }')

# 5. A keep-awake file touched recently.
for file in $KEEP_AWAKE_GLOB; do
  [ -e "$file" ] || continue
  age_min=$(( ( $(date +%s) - $(stat -c %Y "$file") ) / 60 ))
  if [ "$age_min" -lt $(( KEEP_AWAKE_HOURS * 60 )) ]; then
    reasons+=("keep-awake set ${age_min} min ago by ${file}")
  fi
done

now=$(date +%s)
since=$now
if [ -s "$STATE_FILE" ] && [[ "$(cat "$STATE_FILE" 2>/dev/null)" =~ ^[0-9]+$ ]]; then
  since=$(cat "$STATE_FILE")
fi

# --status reports and changes nothing (it may run as a normal user).
if [ "$status_only" -eq 1 ]; then
  if [ "${#reasons[@]}" -gt 0 ]; then
    printf 'busy: %s\n' "${reasons[@]}"
  else
    echo "idle for $(( (now - since) / 60 )) min (shuts down at ${IDLE_MINUTES})"
  fi
  exit 0
fi

mkdir -p "$(dirname "$STATE_FILE")"
if [ "${#reasons[@]}" -gt 0 ]; then
  rm -f "$STATE_FILE"   # busy: the idle clock starts again
  exit 0
fi
[ -s "$STATE_FILE" ] || echo "$now" > "$STATE_FILE"   # idle from now on
idle_min=$(( (now - since) / 60 ))
if [ "$idle_min" -ge "$IDLE_MINUTES" ]; then
  logger -t neuros-idle-stop "idle for ${idle_min} min: shutting down (the instance stops)"
  rm -f "$STATE_FILE"
  /sbin/shutdown -h now "NeurOS idle auto-stop: idle for ${idle_min} minutes"
fi
