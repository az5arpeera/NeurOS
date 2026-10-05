#!/usr/bin/env bash
# NeurOS on an AWS GPU instance (D-032). Idempotent: safe to run again after a reboot or update.
#
# Run it ON the instance, as its default user, after the founders have:
#   - launched it (Ubuntu with an NVIDIA driver, e.g. a Deep Learning Base AMI),
#   - attached and mounted the data volume (default /mnt/neuros; see the runbook),
#   - signed GitHub in on the instance (the repo is private): `gh auth login`, or a read-only
#     deploy key in ~/.ssh.
#
# It never touches AWS credentials, keys or budgets: those stay with the founders.
#
#   bash scripts/aws/bootstrap.sh            # from a clone, or
#   NEUROS_DATA_MOUNT=/data bash bootstrap.sh
#
# What it does:
#   1. checks the GPU and the data volume,
#   2. installs uv (Astral's official installer) if missing,
#   3. clones or updates the repository,
#   4. creates the environment with PyTorch's CUDA 12.6 build (`ml-cuda`), exactly as locked,
#   5. points the repo's data/, cache/ and results/ at the data volume,
#   6. runs the test suite, including the GPU test that is skipped on laptops,
#   7. installs the idle auto-stop (scripts/aws/idle-stop.sh): the machine shuts itself down
#      after 45 minutes with nothing running. NEUROS_IDLE_STOP=0 skips it.
set -euo pipefail

DATA_MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
REPO_DIR="${NEUROS_REPO_DIR:-$HOME/NeurOS}"
REPO_URL="${NEUROS_REPO_URL:-https://github.com/az5arpeera/NeurOS.git}"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "1/7 GPU and data volume"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: use an AMI with the NVIDIA driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || die "the NVIDIA driver doesn't answer"
driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "${driver_major:-0}" -ge 525 ] || die "driver ${driver_major}.x is too old for CUDA 12.6 (needs >= 525)"
[ -d "$DATA_MOUNT" ] || die "$DATA_MOUNT doesn't exist: mount the data volume first (runbook step 3)"
[ -w "$DATA_MOUNT" ] || die "$DATA_MOUNT isn't writable by $(whoami)"
df -h "$DATA_MOUNT"

say "2/7 uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi
uv --version

say "3/7 code"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR" || die "clone failed: is GitHub signed in on this machine?"
fi
cd "$REPO_DIR"
git log --oneline -1

say "4/7 environment (CUDA build, exactly as locked)"
uv sync --extra dev --extra ml-cuda --locked
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA not usable'; \
print('torch', torch.__version__, 'CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

say "5/7 data, cache and results on the volume"
for dir in data cache results; do
  mkdir -p "$DATA_MOUNT/$dir"
  if [ -L "$dir" ]; then
    [ "$(readlink "$dir")" = "$DATA_MOUNT/$dir" ] || die "$dir links elsewhere: $(readlink "$dir")"
  elif [ -e "$dir" ]; then
    die "$dir exists and isn't a link: move its contents to $DATA_MOUNT/$dir first"
  else
    ln -s "$DATA_MOUNT/$dir" "$dir"
  fi
done
ls -l data cache results

say "6/7 tests (the GPU test runs here)"
uv run --no-sync pytest -q -m "not slow"
uv run --no-sync neuros datasets

say "7/7 idle auto-stop (the machine shuts itself down after 45 min with nothing running)"
if [ "${NEUROS_IDLE_STOP:-1}" = "1" ]; then
  # A copy outside the repo, so a `git pull` can't change the script while the timer runs it.
  sudo install -m 0755 scripts/aws/idle-stop.sh /usr/local/sbin/neuros-idle-stop
  sudo tee /etc/systemd/system/neuros-idle-stop.service >/dev/null <<'UNIT'
[Unit]
Description=NeurOS idle auto-stop check (scripts/aws/idle-stop.sh)

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/neuros-idle-stop
UNIT
  sudo tee /etc/systemd/system/neuros-idle-stop.timer >/dev/null <<'UNIT'
[Unit]
Description=Run the NeurOS idle auto-stop check every 5 minutes

[Timer]
OnBootSec=15min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now neuros-idle-stop.timer
  /usr/local/sbin/neuros-idle-stop --status
  echo "keep it awake during long interactive work: touch ~/.neuros-keep-awake (3 h)"
else
  echo "skipped (NEUROS_IDLE_STOP=0)"
fi

say "ready: downloads only after approval (knowledge-base/plan/aws-runbook.md, step 5)"
