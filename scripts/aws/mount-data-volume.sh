#!/usr/bin/env bash
# Format and mount the NeurOS data volume (runbook step 3d). Run ON the instance, once.
#
#   bash scripts/aws/mount-data-volume.sh
#
# mkfs erases a disk, so every check runs first. It refuses unless exactly one disk is an EBS
# volume of NEUROS_DATA_GIB GiB (default 500) that is blank: no partitions, no filesystem, no
# signature of any kind. g6 machines also carry a local NVMe "instance store" (model "Amazon EC2
# NVMe Instance Storage", wiped at every stop; the AMI mounts it at /opt/dlami/nvme): the model
# check keeps it out. Idempotent: if the mount point is already mounted, it only reports.
set -euo pipefail

MOUNT="${NEUROS_DATA_MOUNT:-/mnt/neuros}"
GIB="${NEUROS_DATA_GIB:-500}"
[[ "$GIB" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "ERROR: NEUROS_DATA_GIB must be 1-99999" >&2; exit 2; }
SIZE=$(( GIB * 1024 * 1024 * 1024 ))  # bytes, as lsblk -b reports them

if mountpoint -q "$MOUNT"; then
  echo "$MOUNT is already mounted; nothing to do"
  df -h "$MOUNT"
  exit 0
fi

mapfile -t candidates < <(lsblk -dnpb -o NAME,SIZE,MODEL \
  | awk -v size="$SIZE" '$2 == size && /Amazon Elastic Block Store/ {print $1}')
if [ "${#candidates[@]}" -ne 1 ]; then
  echo "REFUSING: expected exactly one $GIB GiB EBS disk, found: ${candidates[*]:-none}" >&2
  lsblk -o NAME,SIZE,MODEL,FSTYPE,MOUNTPOINT >&2
  exit 1
fi
DEV="${candidates[0]}"
echo "candidate: $DEV"

# It must be blank.
if [ "$(lsblk -nrp -o NAME "$DEV" | wc -l)" -ne 1 ]; then
  echo "REFUSING: $DEV has partitions" >&2
  exit 1
fi
if [ -n "$(lsblk -dn -o FSTYPE "$DEV")" ]; then
  echo "REFUSING: $DEV already has a filesystem" >&2
  exit 1
fi
if sudo blkid -p "$DEV" >/dev/null 2>&1; then
  echo "REFUSING: $DEV carries a signature (blkid found one)" >&2
  exit 1
fi
if ! sudo file -s "$DEV" | grep -q ': data$'; then
  echo "REFUSING: $DEV doesn't read as blank: $(sudo file -s "$DEV")" >&2
  exit 1
fi
echo "checks passed: $DEV is blank"

command -v mkfs.xfs >/dev/null || sudo apt-get install -y -q xfsprogs
sudo mkfs.xfs -q -L neuros-data "$DEV"

UUID="$(sudo blkid -s UUID -o value "$DEV")"
if [ -z "$UUID" ]; then
  echo "ERROR: no UUID after mkfs" >&2
  exit 1
fi
sudo mkdir -p "$MOUNT"
if ! grep -q "UUID=$UUID" /etc/fstab; then
  sudo cp /etc/fstab /etc/fstab.before-neuros
  # nofail: the machine still boots if the volume is ever detached.
  echo "UUID=$UUID $MOUNT xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
fi
sudo systemctl daemon-reload
sudo mount "$MOUNT"  # through /etc/fstab, so the entry is proven now rather than at next boot
sudo chown "$(id -un):$(id -gn)" "$MOUNT"

echo "--- mounted"
df -h "$MOUNT"
grep " $MOUNT " /etc/fstab
touch "$MOUNT/.write-test" && rm "$MOUNT/.write-test" && echo "writable by $(id -un)"
