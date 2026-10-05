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

**Order:** steps 1, 2 and 2b today (the quota can take days); steps 3–5 when the quota arrives.

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

### 2b. Cap the spending: the budget guard (free to run)
A budget on its own only sends emails. The guard stops the machines.
- **Template:** `infra/aws/budget-guard.yaml`, deployed once with CloudFormation.
- **What it does:** at 90% of the monthly cap, it stops every running EC2 instance in every region. It then keeps any instance that starts again in us-east-1 stopped for the rest of the month.
- **What it never does:** terminate an instance or delete a disk or snapshot. Your data stays; a stopped machine costs only its disks.
- **Cost:** nothing. It uses Lambda, SNS, EventBridge and Parameter Store within their free tiers, and monitoring budgets are free.
- **⚠️ Your existing budget probably counts credits.** That's AWS's default, and credits cancel out spending, so the budget may show about $0 while you burn credits. Edit it: Advanced options → Charge types → untick **Credits**. The guard's own budget is already set this way.

**Deploy, in us-east-1:**
1. CloudFormation → Create stack → "Upload a template file" → `infra/aws/budget-guard.yaml`.
2. Fill in the parameters:
   - `MonthlyBudgetUSD`: e.g. 300;
   - `GuardThresholdPercent`: 90. AWS updates spending figures only a few times a day, so stopping at 90% keeps the lag inside the cap;
   - `AlertEmail`.
3. Tick "I acknowledge that AWS CloudFormation might create IAM resources" → Create. It takes about a minute.
4. Test it: Lambda → `neuros-budget-guard` → Test, with the event `{"dry_run": true}`. It lists every instance it would stop, and stops nothing.
5. To lift the lock after raising the budget, delete `/neuros/budget-guard/locked-month` in Systems Manager → Parameter Store.

**The second safety net: idle auto-stop.**
- **What it does:** `bootstrap.sh` installs `scripts/aws/idle-stop.sh` as a 5-minute timer on the machine. After 45 minutes with no NeurOS command running, no GPU or CPU load and no one typing over SSH, the machine shuts itself down.
- **Why it matters:** forgetting to stop the machine is the likeliest way to waste credits, at about $32 a day. This catches it in minutes, where the budget guard acts on billing figures hours old.
- **To keep it awake** during long interactive work: `touch ~/.neuros-keep-awake` holds it for 3 hours.
- **To check it:** `neuros-idle-stop --status`.
- **When launching the machine**, keep "Shutdown behavior: Stop", the default. Then shutting down stops the instance and keeps its disks.

## Costs (estimated 5 Oct; us-east-1 on-demand prices)
- **Prices:** g6.4xlarge $1.323/h (Spot ≈ $0.96: not worth the interruptions); g6.2xlarge $0.978/h; gp3 storage ≈ $0.08 per GB-month.
- **Model run times are guesses.** The first E1 run measures them, and this table gets updated then.

| Work | When | Machine-hours | ≈ Cost |
|---|---|---|---|
| Setup, the 282 GB of downloads, checksums | Wed 7 | 5–8 | $7–11 |
| Validation, filling the cache for ~400 people, cue audits of 8 datasets | Thu 8 | 8–12 | $11–16 |
| E0b's EEGNet check | Fri 9 | 2–3 | $3–4 |
| E1: 52 models, one seed, plus one re-run | Fri 9–Sat 10 | 6–12 | $8–16 |
| E2 warm-up curves, H1, E3 test-time adaptation | Week 2 | 10–20 | $13–26 |
| E4–E7: heads, virtual headsets, EEGMAT, command sets | Week 3 | 15–30 | $20–40 |
| Freeze runs: locked test set, 10-fold over everyone, LODO, 3 seeds | Sat 24 | 10–20 | $13–26 |
| Machine on with no job running (reading results, debugging) | throughout | 30–60 | $40–80 |
| **Compute** | | **86–165** | **$114–218** |
| Disks: 100 GB root + 500 GB data, Oct 7 – Nov 1 | | | ≈ $40 |
| Public IPv4 address while running; the guard; data transfer in | | | ≈ $1 |
| **Total to 1 Nov** | | | **≈ $155–260** |

- **Suggested monthly budget:** $300, with the guard at 90%.
- **After 1 Nov:** the disks cost about $48 a month until you delete them, or snapshot them first at about $0.05 per GB-month.
- **Cheaper training phases:** once data preparation is done, a g6.2xlarge (8 vCPU, 32 GB) may be enough. Stop the machine, change its type, start it again: about 2 minutes, and about 26% cheaper per hour.
- **The worst case the guard and idle auto-stop prevent:** a forgotten g6.4xlarge for the rest of the month, about $32 a day, or roughly $850.

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

- **Stop the instance when it's idle.** A stopped instance costs only its disks, and the data volume keeps everything. The idle auto-stop (step 2b) does it for you after 45 minutes.
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
- **When you're done:** stop the instance. The idle auto-stop does it after 45 minutes if you forget, and the budget guard stops everything at 90% of the monthly cap.
- **Delete the volume** only when the project no longer needs the data.
