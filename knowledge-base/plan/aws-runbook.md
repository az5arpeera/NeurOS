# AWS runbook (D-032)

**Written:** 4 Oct 2026. Training moves to AWS, paid with credits. The school cluster is the fallback.

**Who does what:**
- **Founders:**
  - the account, credentials, region, quotas and budget alarm;
  - launching and stopping instances;
  - signing GitHub in on the instance;
  - approving each download.
- **Claude:**
  - the code, `scripts/aws/bootstrap.sh` and this runbook;
  - commands on the instance only if you give it SSH access through a host alias (step 6). It never sees, types or stores a key.

**Order:** steps 1–2 today (the quota can take days); steps 3–5 when the quota arrives.

## 1. Request the GPU quota (do this first)
1. Open **Service Quotas → Amazon EC2 → "Running On-Demand G and VT instances"** in the region you'll use (step 3).
2. Request **16 vCPUs**: one g6.4xlarge.
   - New accounts often start at 0, and approval can take from hours to a few days.
   - Ask for 32 if you might run two machines at once.

## 2. Set a budget alarm
In **Billing → Budgets**:
- Set a monthly cost budget at or below the credit amount.
- Add email alerts at 50%, 80% and 100%.
- Credits cover most services. The alarm catches a machine left running.

## 3. Launch the machine and its data volume
**Recommendation** (on-demand prices are approximate, from memory; check the console):

| Item | Choice | ≈ Cost | Why |
|---|---|---|---|
| Region | **us-east-1** | | PhysioNet's open-data mirror is there: fast, free in-region transfer |
| Instance | **g6.4xlarge**: 1× NVIDIA L4 (24 GB), 16 vCPU, 64 GB RAM | ≈ $1.3/h | EEGNet is small, so the GPU is rarely the bottleneck. CPUs for data preparation and parallel folds matter more |
| Alternative | g5.4xlarge: 1× A10G (24 GB), 16 vCPU, 64 GB | ≈ $1.6/h | If g6 capacity is short |
| Too small | g6.xlarge: 4 vCPU, 16 GB | ≈ $0.8/h | Data preparation at 1000 Hz would crawl |
| Image (AMI) | AWS **Deep Learning Base AMI (Ubuntu)**, with the NVIDIA driver installed | | `bootstrap.sh` checks for driver ≥ 525 (CUDA 12.6) |
| Root disk | 100 GB gp3 | ≈ $8/month | |
| Data volume | **500 GB gp3**, mounted at `/mnt/neuros` | ≈ $40/month | About 350 GB is used (step 5) |
| Network | Security group: SSH (port 22) from your IP only | | |

- **Stop the instance when it's idle.** A stopped instance costs only its disks, and the data volume keeps everything.
- **Spot instances** are about 60–70% cheaper, but can be interrupted mid-run. Use on-demand until runs can resume.

**Mount the data volume** (once, as the instance's user).

⚠️ `mkfs` erases the volume. Run it only on the new, empty data volume, never on the root disk.

```bash
lsblk                                   # find the new 500 GB volume, e.g. /dev/nvme1n1
sudo mkfs -t xfs /dev/nvme1n1           # ERASES it: new, empty volume only
sudo mkdir -p /mnt/neuros
sudo mount /dev/nvme1n1 /mnt/neuros
sudo chown "$USER" /mnt/neuros
echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1) /mnt/neuros xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

## 4. Set up the code
The repository is private.
1. Sign GitHub in on the instance, with `gh auth login` or a read-only deploy key.
2. Then:

```bash
git clone https://github.com/az5arpeera/NeurOS.git ~/NeurOS
bash ~/NeurOS/scripts/aws/bootstrap.sh
```

The script is idempotent, and stops with a clear message if anything is missing. It:
1. checks the GPU, the driver and the data volume;
2. installs uv;
3. creates the environment with PyTorch's CUDA build (`uv sync --extra dev --extra ml-cuda --locked`);
4. links `data/`, `cache/` and `results/` to the volume;
5. runs the tests. The GPU test runs here: on laptops it's skipped.

## 5. Downloads: each needs your approval first
Sizes were measured on 4 Oct from publisher metadata: HTTP headers and the Zenodo, Figshare and S3 APIs. Nothing was downloaded.

| # | Dataset | Source | Files | Size | Licence |
|---|---|---|---|---|---|
| 1 | BNCI2014_001 | bnci-horizon-2020.eu (`lampx.tugraz.at`) | 18 .mat | 0.75 GB | BNCI terms (Q7b) |
| 2 | BNCI2014_004 | same | 18 .mat | 0.45 GB | BNCI terms (Q7b) |
| 3 | PhysionetMI | PhysioNet's AWS open-data mirror, checked against `SHA256SUMS.txt` | 872 EDF | 1.9 GB | ODC-By 1.0 |
| 4 | EEGMAT | same mirror | 76 | 0.18 GB | ODC-By 1.0 |
| 5 | Cho2017 | GigaDB (Wasabi S3) | 52 .mat | 10.7 GB | GigaDB (to confirm, Q7b) |
| 6 | Lee2019_MI | GigaDB (Wasabi S3), motor-imagery files only | 108 .mat | 64.8 GB | GigaDB (to confirm, Q7b) |
| 7 | Dreyer2023 | OSF project `8tdk5` (MOABB's source); the Zenodo copy is 27.5 GB | per person | ≈ 27.5 GB | CC BY 4.0 (Zenodo, checked) |
| 8 | Stieger2021 | Figshare 13123148, **sessions 1–2 only** (registry) | 124 .mat | 77.5 GB (of 377 GB) | CC BY 4.0 (Figshare, checked) |
| 9 | **Yang2025** (D-033) | Figshare+ 22671172 | 1 ZIP | **65.6 GB** | CC BY 4.0 |
| 10 | **GuttmannFlury2025** (D-033) | Zenodo 18970793 | 31 zips | **26.2 GB** | CC0 (data) |
| 11 | **Shin2017A** (D-033) | doc.ml.tu-berlin.de/hBCI | 6 zips | **6.26 GB** | **GPL-3.0, accepted by the person running the download** |

**Total ≈ 282 GB**, plus about 66 GB to unpack Yang2025's ZIP, so about 350 GB on the 500 GB volume.
- After `neuros download` has checked the Yang2025 archive against Figshare's MD5, you may delete it to free 65.6 GB.
- Rows 1–3 are already on the laptop. On AWS they're downloaded again rather than copied: it's faster and needs no credentials.

**Integrity:**
- Every file is SHA-256-hashed into `data/manifests/<dataset>.json`.
- PhysioNet files are checked against PhysioNet's published SHA-256 sums.
- Stieger2021 files are checked against Figshare's MD5s (by MOABB).
- Yang2025 and GuttmannFlury2025 archives are checked against the publishers' MD5s (`src/neuros/data/checksums/`).

**Commands, in order (smallest first, so problems show up early):**

```bash
cd ~/NeurOS
N="uv run --no-sync neuros --config configs/data.yaml"
$N download --dataset BNCI2014_001 && $N download --dataset BNCI2014_004
$N download --dataset PhysionetMI && $N download --dataset EEGMAT
$N download --dataset Shin2017A --accept-licence      # you accept GPL-3.0 by running this
$N download --dataset Cho2017
$N download --dataset GuttmannFlury2025
$N download --dataset Dreyer2023
$N download --dataset Yang2025
$N download --dataset Lee2019_MI
$N download --dataset Stieger2021                     # sessions 1-2 only (registry)
$N validate --dataset <each>                          # then: harmonise + validate every recording
```

- **EEGMAT** needs its own loader, still to be written: download it, but `validate` comes later.
- **Long commands** go inside `tmux new -s dl`, so they survive a dropped SSH connection.

## 6. Optional: let Claude run commands on the instance
- **What you do:** add an entry to `~/.ssh/config` on the laptop, with the key file kept where you keep it:
  ```
  Host neuros-aws
      HostName <the instance's public DNS>
      User ubuntu
      IdentityFile ~/.ssh/<your key>.pem
  ```
- **What Claude then does:**
  - runs `ssh neuros-aws "cd NeurOS && ..."`, the same way it uses git through your credential manager;
  - starts long runs inside `tmux` on the instance, so they don't depend on the laptop staying awake.
- **What stays yours:** the key never appears in a command, and you can revoke access by deleting the entry.

## 7. Day-to-day
- **Before each session:** start the instance, then `git pull && uv sync --extra dev --extra ml-cuda --locked`. `bootstrap.sh` does both.
- **Training runs:** `neuros loso --experiment … --device cuda`. On a 16-vCPU machine the default is 8 fold workers × 2 threads, all sharing the one GPU.
- **Results:**
  - Results land in `results/` on the volume. Copy the JSON files you want to keep to the laptop with `scp`; they're small. The results log in the knowledge base is what gets committed.
  - Every result records the GPU, CUDA and cuDNN versions and whether deterministic kernels were on (the default).
- **When you're done:** stop the instance. Delete the volume only when the project no longer needs the data.
