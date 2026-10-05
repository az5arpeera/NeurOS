# AWS runbook (D-032)

**Written:** 4 Oct 2026. Console steps detailed and checked against AWS's documentation on 5 Oct. Training moves to AWS, paid with credits. The school cluster is the fallback.

**Who does what:**
- **Founders:**
  - the account, credentials, region, quotas and budget alarm;
  - launching and stopping instances;
  - adding the machine's read-only deploy key on GitHub (step 4);
  - approving each download.
- **Claude:**
  - the code, `scripts/aws/bootstrap.sh` and this runbook;
  - commands on the instance only through a host alias you approve (step 6). It never sees, types or stores a key.

**Order:** steps 1, 2 and 2b today (the quota can take days); steps 3–5 when the quota arrives.

**Region:** everything runs in **us-east-1 (N. Virginia)**. Check the region picker (top right) before each step. Budgets and billing are global.

## 1. Request the GPU quota (do this first)
1. Open **Service Quotas → AWS services → Amazon Elastic Compute Cloud (Amazon EC2) → "Running On-Demand G and VT instances"** in us-east-1.
2. **Request increase at account level** → **16**. The quota counts vCPUs, and one g6.4xlarge has 16.
   - New accounts often start at 0, and approval can take from hours to a few days.
   - Ask for 32 if you might run two machines at once.
3. If AWS asks for a use case: "Training small EEG deep-learning models (EEGNet) for a research prototype: one g6.4xlarge, on demand, stopped when idle, under a budget cap."

## 2. Set a budget alarm
In **Billing and Cost Management → Budgets**:
- Set a monthly cost budget at or below the credit amount, with email alerts at 50%, 80% and 100%.
- **Verify the email.**
  - AWS Budgets now emails every new recipient a verification link, from an `@aws.com` address. The link is valid for 12 hours, and you must be signed in to the account to use it. Until you click it, no alerts are sent.
  - The budget's page shows each recipient's status:
    - Active: verified, nothing to do;
    - Pending: use "Resend verification";
    - Expiring or Inactive: use "Send Notification".
- Credits cover most services. The alarm catches a machine left running.

### 2b. Cap the spending: the budget guard (free to run)
A budget on its own only sends emails. The guard stops the machines.
- **Template:** `infra/aws/budget-guard.yaml`, deployed once with CloudFormation.
- **What it does:** at 90% of the monthly cap, it stops every running EC2 instance in every region. It then keeps any instance that starts again in us-east-1 stopped for the rest of the month.
- **What it never does:** terminate an instance or delete a disk or snapshot. Your data stays; a stopped machine costs only its disks.
- **Cost:** nothing. It uses Lambda, SNS, EventBridge and Parameter Store within their free tiers, and monitoring budgets are free.
- **⚠️ Your existing budget probably counts credits.**
  - AWS nets credits out by default, so the budget may show about $0 while you spend credits.
  - To fix it: Edit → Budget scope → Filters → Add filter → **Charge type: Usage** → Save.
  - Or leave it: the guard's own budget already excludes credits. It emails you at 80% (forecast) and 90% (actual).

**Deploy, in us-east-1:**
1. CloudFormation → Create stack → With new resources → "Upload a template file" → `infra/aws/budget-guard.yaml` → Next.
2. Stack name `neuros-budget-guard`. Parameters:
   - `MonthlyBudgetUSD`: e.g. 300;
   - `GuardThresholdPercent`: 90. AWS updates spending figures only a few times a day, so stopping at 90% keeps the lag inside the cap;
   - `AlertEmail`.
3. Tick "I acknowledge that AWS CloudFormation might create IAM resources" → Submit. After about a minute the status reads CREATE_COMPLETE.
4. **Click the verification email** that AWS Budgets sends to `AlertEmail` (step 2). Stopping the machines doesn't depend on it: the budget calls the guard directly.
5. **Test it:**
   1. Lambda → Functions → `neuros-budget-guard` → Test tab.
   2. Create a new event named `dry-run` with the JSON `{"dry_run": true}`, then run it.
   3. It lists every instance it would stop under `"stopped"` and stops nothing. The result should end with `"failed": {}`.
6. **Optional end-to-end test** (2 minutes; run it once the machine exists, before it's set up):
   1. Run the test event `{"simulate_alert": true}`. The machine stops.
   2. Start the machine again. Within about a minute the guard stops it again, which proves the lock works.
   3. Lift the lock (next point).
7. **To lift the lock** (after that test, or after raising the budget): Systems Manager → Parameter Store → `/neuros/budget-guard/locked-month` → Delete.
8. **To raise the cap:** CloudFormation → `neuros-budget-guard` → Update → use the existing template → change `MonthlyBudgetUSD`. Don't edit the budget in the Budgets console: it would no longer match the stack.

**The second safety net: idle auto-stop.**
- **What it does:** `bootstrap.sh` installs `scripts/aws/idle-stop.sh` as a 5-minute timer on the machine. After 45 minutes with no NeurOS command running, no GPU or CPU load and no one typing over SSH, the machine shuts itself down.
- **Why it matters:** forgetting to stop the machine is the likeliest way to waste credits, at about $32 a day. This catches it in minutes, where the budget guard acts on billing figures hours old.
- **To keep it awake** during long interactive work: `touch ~/.neuros-keep-awake` holds it for 3 hours.
- **To check it:** `neuros-idle-stop --status`.
- **When launching the machine**, keep "Shutdown behavior: Stop", the default. Then shutting down stops the instance and keeps its disks.

## Costs (estimated 5 Oct; us-east-1 on-demand prices)
- **Prices:**
  - g6.4xlarge: $1.323/h. Spot is ≈ $0.96, but not worth the interruptions.
  - g6.2xlarge: $0.978/h.
  - gp3 storage: ≈ $0.08 per GB-month.
  - Public IPv4 address: $0.005/h, even while the machine is stopped.
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
| Fixed public address (Elastic IP, step 3b); the guard; data transfer in | | | ≈ $4 |
| **Total to 1 Nov** | | | **≈ $160–265** |

- **Suggested monthly budget:** $300, with the guard at 90%.
- **After 1 Nov:** until you delete or release them, the disks cost about $48 a month and the Elastic IP $3.60 a month. To keep the data, snapshot the data volume first (about $0.05 per GB-month).
- **Cheaper training phases:**
  - Once data preparation is done, a g6.2xlarge (8 vCPU, 32 GB) may be enough.
  - Stop the machine, change its type, and start it again: about 2 minutes.
  - It's about 26% cheaper per hour.
- **The worst case the guard and idle auto-stop prevent:** a forgotten g6.4xlarge for the rest of the month, about $32 a day, or roughly $850.

## 3. Launch the machine and its data volume
**Recommendation:**

| Item | Choice | ≈ Cost | Why |
|---|---|---|---|
| Region | **us-east-1** | | PhysioNet's open-data mirror is there: fast, free in-region transfer |
| Instance | **g6.4xlarge**: 1× NVIDIA L4 (24 GB), 16 vCPU, 64 GB RAM | $1.323/h | EEGNet is small, so the GPU is rarely the bottleneck. CPUs for data preparation and parallel folds matter more |
| Alternative | g5.4xlarge: 1× A10G (24 GB), 16 vCPU, 64 GB | ≈ $1.6/h | If g6 capacity is short. Uses the same quota |
| Too small | g6.xlarge: 4 vCPU, 16 GB | ≈ $0.8/h | Data preparation at 1000 Hz would crawl |
| Image (AMI) | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**, AWS's own. Release 20261002 has NVIDIA driver 595 and supports G6 | | `bootstrap.sh` needs driver ≥ 525 (CUDA 12.6). The login user is `ubuntu` |
| Root disk | 100 GB gp3 | ≈ $8/month | |
| Data volume | **500 GB gp3**, mounted at `/mnt/neuros` | ≈ $40/month | About 350 GB is used (step 5) |
| Network | Security group: SSH (port 22) from your IP only | | |

**In the console** (EC2 → Launch instance, us-east-1):
1. **Name:** `neuros-gpu`.
2. **Image:** search for the AMI above and pick Amazon's newest.
3. **Instance type:** `g6.4xlarge`.
4. **Key pair:**
   - Create new key pair → `neuros-aws`, type ED25519, format `.pem` → Create.
   - The browser downloads `neuros-aws.pem`. AWS keeps no copy (step 3c).
5. **Network settings:**
   - Keep the default VPC and the automatic public IP.
   - Firewall: create a security group with **Allow SSH traffic from: My IP**, not Anywhere.
   - Leave HTTP and HTTPS unticked.
6. **Storage:**
   - Root volume: 100 GiB gp3.
   - **Add new volume:** 500 GiB gp3.
   - Switch the storage panel to Advanced and set the new volume's **Delete on termination: No**, so the data outlives the machine.
7. **Advanced details:**
   - Termination protection: **Enable**.
   - Shutdown behavior: **Stop**.
   - Stop protection: leave it off. The guard and the idle auto-stop must be able to stop the machine.
   - Spot: leave it unticked.
8. **Launch instance**, then wait for "2/2 checks passed".

- **Stop the instance when it's idle.** A stopped instance costs only its disks, and the data volume keeps everything. The idle auto-stop (step 2b) does it for you after 45 minutes.
- **Spot instances** are about 60–70% cheaper, but can be interrupted mid-run. Use on-demand until runs can resume.

**What we launched (5 Oct):**
- AWS granted only **8 vCPUs** of the 16 requested, so the machine is a **g6.2xlarge**: the same L4 GPU, 8 vCPUs, 30 GiB usable memory, $0.978/h.
  - Downloads aren't affected. Data preparation and parallel folds run about half as many at once.
  - Ask again for 16; to switch, stop the machine, change its type and start it again.
- The first launch failed with **"Insufficient capacity"**. Another subnet worked: the machine is in **us-east-1d**.
- Elastic IP **54.145.175.49**. The root disk has 48 GB free after the AMI. The instance store (419 GB) is mounted at `/opt/dlami/nvme`.

### 3b. A fixed address (recommended, ≈ $3.60 a month)
Without one, the machine's address changes every time it starts, and the SSH alias (step 6) needs the new one each session.
1. EC2 → Elastic IPs → **Allocate Elastic IP address** → Allocate.
2. Select it → Actions → **Associate Elastic IP address** → Instance `neuros-gpu` → Associate.
3. The instance's **Public IPv4 DNS** is now fixed. Release the address when the project ends: it bills even while the machine is stopped.

### 3c. The key file (on the laptop)
1. Move `neuros-aws.pem` into `~/.ssh/` (`C:\Users\<you>\.ssh\` on Windows). These two commands work in PowerShell and Git Bash:
   ```
   mkdir ~/.ssh
   mv ~/Downloads/neuros-aws.pem ~/.ssh/neuros-aws.pem
   ```
2. Never commit, share or email the key.
3. If SSH says the key's permissions are too open, run this in PowerShell:
   ```
   icacls "$HOME\.ssh\neuros-aws.pem" /inheritance:r /grant:r "${env:USERNAME}:R"
   ```

### 3d. Mount the data volume
Do this once. Claude does it with your OK.

⚠️ **Find the right disk first: `mkfs` erases whatever it's given.**
- The machine has three disks:
  - the root disk;
  - the 500 GB data volume;
  - a 600 GB local "instance store". The AMI mounts it at `/opt/dlami/nvme`, and it's wiped every time the machine stops, so never keep data on it.
- The data volume is the 500G disk whose model is "Amazon Elastic Block Store" and that has no filesystem.

`scripts/aws/mount-data-volume.sh` does it:
- It refuses unless exactly one disk is a blank 500 GiB EBS volume: no partitions, no filesystem, no signature.
- It formats it as XFS, adds it to `/etc/fstab` by UUID with `nofail`, and mounts it through fstab, so the boot entry is tested at once.
- It ran on 5 Oct: `/dev/nvme1n1` → `/mnt/neuros`.

## 4. Set up the code
The repository is private, so the machine gets a read-only **deploy key**. It opens this one repository and nothing else, and you can revoke it from GitHub at any time.
1. **On the machine:** `scripts/aws/deploy-key.sh`. Before the repository is cloned, copy the script over and run it there. It:
   - creates the key and prints its public line;
   - points `github.com` at that key;
   - trusts GitHub's host keys fetched over HTTPS, not on first use.
2. **On GitHub:**
   1. Open `az5arpeera/NeurOS` → Settings → Deploy keys → **Add deploy key**.
   2. Title it `neuros-aws` and paste the printed line.
   3. Leave **Allow write access** unticked → Add key.
   - Or, with the founders' OK, from the laptop: `gh repo deploy-key add KEY.pub --repo az5arpeera/NeurOS --title neuros-aws` (read-only by default; done this way on 5 Oct).
3. **On the machine:** `git clone git@github.com:az5arpeera/NeurOS.git ~/NeurOS`, then `bash ~/NeurOS/scripts/aws/bootstrap.sh`.

`gh auth login` works too, but it signs the machine in to every repository you can reach.

**5 Oct:** the first bootstrap stopped at its tests.
- The GPU-vs-CPU training test failed: dropout draws different random masks on the two devices.
- Fixed in 742c45e. The rerun passed 666 tests, and the idle auto-stop is active.

`bootstrap.sh` is idempotent, and stops with a clear message if anything is missing. It:
1. checks the GPU, the driver and the data volume;
2. installs uv;
3. creates the environment with PyTorch's CUDA build (`uv sync --extra dev --extra ml-cuda --locked`);
4. links `data/`, `cache/` and `results/` to the volume;
5. runs the tests. The GPU test runs here: on laptops it's skipped;
6. installs the idle auto-stop.

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

- **EEGMAT** needs its own loader, still to be written. `neuros download` refuses it for now ("not a MOABB dataset"); only E6 needs it.
- **Long commands** go inside `tmux new -s dl`, so they survive a dropped SSH connection.

**How the downloads actually ran (5 Oct, approved datasets 1–10).** Parallel lanes, grouped by host, all in the `dl` tmux session:
- **Lane commands:**
  - `scripts/aws/download-lane.sh A BNCI2014_001 BNCI2014_004 PhysionetMI EEGMAT Cho2017 Lee2019_MI`
  - `scripts/aws/download-lane.sh B Stieger2021 Yang2025`
  - `scripts/aws/download-chunks.sh GuttmannFlury2025 4`: Zenodo throttles each connection to ~0.8–2.5 MB/s, so one stream would take ~9 h. Four chunks of people run at ~10 MB/s together, then a full run writes the manifest.
  - `scripts/aws/download-retry.sh Dreyer2023`: OSF answered a burst with **429 Too Many Requests** after 3 minutes. The retry waits 10, 20, 40, then 60 minutes and resumes. Its first retry finished.
- **Measured speeds:**

  | Source | Speed |
  |---|---|
  | Figshare | ~55 MB/s |
  | GigaDB | 30–40 MB/s |
  | TU Graz | ~7 MB/s |
  | PhysioNet's AWS mirror | under a minute for 1.9 GB |

- **Sizes that differ from the table above:**
  - **Dreyer2023** from OSF is **7.5 GB** (520 files, 87 people), not the Zenodo copy's 27.5 GB.
  - **GuttmannFlury2025** unpacks to roughly another 26 GB, videos included, next to its zips.
- `scripts/aws/validate-lane.sh` validates each dataset once its download reports success, at `nice 10`.
- Each step logs to `/mnt/neuros/logs/`. `downloads-summary.txt` has one line per dataset (exit status, minutes, free space).

## 6. Let Claude run commands on the machine
- **The alias** goes in `~/.ssh/config` on the laptop. You add it, or Claude does with your OK:
  ```
  Host neuros-aws
      HostName <the machine's Public IPv4 DNS>
      User ubuntu
      IdentityFile ~/.ssh/neuros-aws.pem
      ServerAliveInterval 60
      StrictHostKeyChecking accept-new
  ```
  `accept-new` records the machine's fingerprint on the first connection, and refuses to connect if it ever changes.
- **5 Oct setup:**
  - The alias points at the Elastic IP's DNS name, `ec2-54-145-175-49.compute-1.amazonaws.com`.
  - Claude uses Windows' OpenSSH (`C:\Windows\System32\OpenSSH\ssh.exe`). It copies scripts to `~/neuros-scripts/` on the machine and runs them as files. Piping a script into `bash -s` breaks when a command inside it, such as `ssh`, reads stdin.
- **Permissions:**
  - Claude Code's auto mode blocks commands that change a remote machine.
  - To let Claude run them, give the session a permission rule or switch it out of auto mode. On 5 Oct the founders enabled bypass permissions.
- **What Claude then does:**
  - runs `ssh neuros-aws "cd NeurOS && ..."`, the same way it uses git through your credential manager;
  - starts long runs inside `tmux` on the instance, so they don't depend on the laptop staying awake.
- **What stays yours:** the key's contents never appear in a command, and you can revoke access by deleting the entry.

## 7. Day-to-day
- **Start:**
  1. EC2 → Instances → `neuros-gpu` → Instance state → Start.
  2. Then run `git pull && uv sync --extra dev --extra ml-cuda --locked`. `bootstrap.sh` does both.
- **Stop:**
  - Instance state → **Stop**. Never Terminate; termination protection refuses it anyway.
  - If you forget, the idle auto-stop does it after 45 minutes.
  - The budget guard stops everything at 90% of the monthly cap.
- **Training runs:**
  - Run `WORKERS=3 bash scripts/aws/loso-queue.sh NAME [NAME …]` inside tmux. It runs `configs/NAME.yaml` one after another with `--device cuda`, and refuses uncommitted code.
  - Parallel folds share the one GPU, at 2 threads each. Use 3 on the 8-vCPU g6.2xlarge while downloads run, up to 8 on a g6.4xlarge.
  - **Measured on 5 Oct:** BNCI2014-001 EEGNet + EA LOSO (9 folds, 300 epochs) took **6.2 min** on the L4 with 3 workers, against 36 min on the laptop's CPU.
- **Results:**
  - Results land in `results/` on the volume. Copy the JSON files you want to keep to the laptop with `scp`; they're small. The results log in the knowledge base is what gets committed.
  - Every result records the GPU, CUDA and cuDNN versions and whether deterministic kernels were on (the default).
- **Delete the volume** only when the project no longer needs the data.

## 8. If something goes wrong
- **"vCPU limit" error at launch:** the quota (step 1) isn't approved yet.
- **"Insufficient capacity":**
  - Network settings → Edit → pick another subnet (another availability zone), or try later.
  - A g5.4xlarge uses the same quota.
- **SSH times out:**
  - Your IP has probably changed, e.g. on another network.
  - EC2 → Security groups → the group from step 3 → Edit inbound rules → set the SSH source to **My IP** → Save.
- **SSH says the key's permissions are too open:** run the `icacls` line in step 3c.
- **The stack fails:** CloudFormation → the stack → Events. The first CREATE_FAILED row says why.
- **No budget emails:** the recipient isn't verified yet (step 2).
