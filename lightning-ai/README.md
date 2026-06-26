# Training Immortalite Zero on Lightning AI

Self-play training on a Lightning AI Studio GPU. Same workflow as Colab, but
checkpoints and Syzygy tablebases live in sibling folders you upload manually.

> **What you're doing:** Clone the repo in your studio, upload `results/` and
> `syzygy345/` next to it, run the notebook, then download the updated
> `results/` folder when the session ends.

---

## Workspace layout

Upload (or create) this structure in your Lightning AI studio:

```
parent/
├── immortalite-zero/     # git clone
│   └── lightning-ai/
│       ├── train.ipynb   # notebook workflow
│       ├── run_train.py  # background-friendly training script
│       └── run_gate.py   # manual checkpoint gate script
├── results/              # manual upload — checkpoints + metrics
│   ├── latest.pt
│   ├── metrics.csv
│   └── metrics_gates.csv
└── syzygy345/            # manual upload — 145 .rtbw files (~378 MB)
```

Training writes to `results/` every iteration (`latest.pt`, `metrics.csv`,
`metrics_gates.csv`, numbered snapshots). Re-upload `results/` before each
session to resume from `latest.pt`.

---

## Before you start

- A **Lightning AI** account with GPU studio access.
- `results/` and `syzygy345/` uploaded as siblings of the cloned repo.
- Push engine changes to GitHub so `git pull` in the notebook picks them up.

To build `syzygy345/` locally once:

```bash
python scripts/download_syzygy345.py --out syzygy345
```

---

## Step 1 — Set up the studio

1. Create a GPU studio on [Lightning AI](https://lightning.ai/).
2. Clone the repo into the workspace.
3. Upload `results/` and `syzygy345/` to the **parent** of the repo (same level
   as `immortalite-zero/`, not inside it).
4. Open `immortalite-zero/lightning-ai/train.ipynb`.

## Step 2 — Train (notebook or script)

Lightning AI disconnects after ~4 hours. **Prefer the script** so training keeps
running after you close the browser tab.

### Option A — Background script (recommended)

```bash
cd immortalite-zero
pip install -q python-chess numpy tqdm

# Edit TRAIN settings in lightning-ai/run_train.py if needed, then:
nohup python lightning-ai/run_train.py > ../results/train.log 2>&1 &

# Watch progress
tail -f ../results/train.log
```

Training writes to `../results/` (`latest.pt`, `metrics.csv`, …) every iteration.
Re-upload `results/` before each new studio session to resume.

### Option B — Notebook

Open `lightning-ai/train.ipynb` and run cells top to bottom. **Keep the browser
tab open** — the kernel stops if you close it.

### Manual gate (script)

Edit `CHECKPOINT_A` and `CHECKPOINT_B` at the top of `lightning-ai/run_gate.py`
(use an int iteration or `"latest"`), then:

```bash
cd immortalite-zero
python lightning-ai/run_gate.py
```

Results append to `../results/metrics_gates.csv`.

## Step 3 — Notebook cells (if using train.ipynb)

| Cell | What it does |
|------|--------------|
| 1 | Resolves `../results` and `../syzygy345`, `git pull`s latest code. |
| 2 | Installs `python-chess` (PyTorch should already be on the studio). |
| 3 | Confirms GPU + sets `--gpu` preset. |
| 4 | Verifies Syzygy upload (145 `.rtbw` files). |
| 5 | **Config + train** — flat 100 sims, gates every 20 iters. `resume: True` by default. |
| 6 | Optional manual gate between any two checkpoints. |
| 7 | Plots metrics + gate winrates from `results/metrics.csv`. |

## Step 4 — Know what "good" looks like

Each training line looks like:

```
iter  12 | sims 100 | games 128 | samples 5200 | buffer 40000 | policy_loss 1.85 | value_loss 0.21 | 180.0s
```

- **policy_loss** should trend **down** over time.
- **value_loss** should be meaningful (not ~0).
- The **metrics plot** (cell 7) is the clearest signal.

## Step 5 — Sessions and resuming

When a studio session ends, download the updated `results/` folder.

| Goal | What to do |
|------|------------|
| **Resume** next session | Re-upload `results/` (with `latest.pt`), re-run cells 1→5. |
| **Fresh run** | Upload an empty `results/` (no `latest.pt`), re-run cells 1→5. |

Numbered snapshots `ckpt_iter_0000.pt`, `ckpt_iter_0010.pt`, … are kept every
10 iterations (`save_every` in cell 5).

## Step 6 — Use the trained engine locally

1. Download `latest.pt` from your uploaded `results/` folder.
2. Verify encoding compatibility:

```bash
python -m engine.inspect_encoding results/latest.pt
```

3. Start the analysis server:

```bash
# Windows (PowerShell)
$env:IMMORTALITE_ZERO_CHECKPOINT="results\latest.pt"
python -m uvicorn server.app:app --port 8000
```

4. Open **http://localhost:8000/app/** .

---

## Troubleshooting

- **"CUDA available: False"** → select a GPU machine in Lightning AI and re-run cell 3.
- **Syzygy incomplete** → upload all 145 `.rtbw` files to `syzygy345/` next to the repo.
- **results/ not found** → folder must be a sibling of `immortalite-zero/`, not inside it.
- **Training seems stuck** → one line per iteration; can take 1–3 min each on the GPU preset.
- **Out of memory** → lower `filters`/`blocks` or `batch_size` in `engine/config.py`, push, pull, re-run.
