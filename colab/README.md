# Training Immortalite Zero on Google Colab

A step-by-step guide to training the engine on a free Colab GPU. No prior ML
experience needed — just follow the cells in order.

> **What you're doing:** Colab gives you a free cloud GPU. You'll clone the repo
> into Colab, run self-play training there, and save checkpoints to your Google
> Drive so nothing is lost when Colab disconnects. Then you download the trained
> file and run the analysis GUI on your own machine.

---

## Before you start

- A **Google account** (for Colab + Drive).
- The code pushed to GitHub (already done: `github.com/carlo-wong/immortalite-zero`).
- Whenever you change the engine locally, `git push` so Colab can pull it.

---

## Step 1 — Open the notebook in Colab

Click this link (it opens the notebook straight from GitHub):

**https://colab.research.google.com/github/carlo-wong/immortalite-zero/blob/main/colab/train.ipynb**

Or: go to [colab.research.google.com](https://colab.research.google.com) →
**File → Open notebook → GitHub** → paste `carlo-wong/immortalite-zero`.

## Step 2 — Turn on the GPU

Top menu: **Runtime → Change runtime type → Hardware accelerator → GPU → Save**.

This is essential — without it, training runs on a slow CPU.

## Step 3 — Run the cells top to bottom

Press **Shift+Enter** on each cell, or **Runtime → Run all**. Here's what each does:

| Cell | What it does |
|------|--------------|
| 1 | Clones the repo (and `git pull`s the latest each time you re-run). |
| 2 | Installs `python-chess` (PyTorch is already on Colab). |
| 3 | Mounts Drive → `MyDrive/immortalite_zero_checkpoints_v3` (fresh run; bump `RUN_TAG` in cell 3 for another clean slate). |
| 4 | Confirms GPU + sets `--gpu` preset. |
| 5 | Downloads Syzygy tablebases (local Colab disk). |
| 6 | **Config + train** — flat 100 sims, gates every 20 iters. `resume: False` (fresh start from iter 0). |
| 7 | Optional manual gate between any two checkpoints (same sims/Syzygy as training). |
| 8 | Plots metrics + gate winrates. |

## Step 4 — Know what "good" looks like

Each training line looks like:

```
iter  12 | sims 100 | games 64 | samples 5200 | buffer 40000 | policy_loss 1.85 | value_loss 0.21 | 180.0s
```

- **policy_loss** should trend **down** over time (the net is learning which moves matter).
- **value_loss** should be meaningful (not ~0) — that means games have real win/loss outcomes, not just timeouts.
- The **metrics plot** (cell 8) is the clearest signal: a downward loss curve = it's improving.

Strength emerges slowly. Pure self-play on a free GPU is a club-level engine at
best — the goal here is steady improvement, not Stockfish.

## Step 5 — Disconnects and resuming (important)

Free Colab disconnects after a while (idle, or ~12h max). **This is fine:**

- Checkpoints are saved to **Google Drive every iteration**, so you never lose more than one iteration.

| Goal | What to do |
|------|------------|
| **Resume** the same run | Set `resume: True` in cell 6 and re-run from cell 6 (or all cells). Training continues from `latest.pt` in the current `CKPT_DIR`. |
| **Fresh run** (clean slate) | Bump `RUN_TAG` in cell 3 (e.g. `v4`) so metrics and checkpoints start at iter 0. Leave `resume: False`. |

**Checkpoint history:**
In addition, a numbered snapshot `ckpt_iter_0000.pt`, `ckpt_iter_0005.pt`, ... is
kept every 10 iterations so you can compare or roll back to earlier versions.
Change the interval with `save_every` in cell 6 (`0` disables numbered snapshots).

Tips to stay connected longer: keep the browser tab open and interact
occasionally; don't close your laptop lid.

## Step 6 — Updating the code later

When you improve the engine on your machine:

```bash
git add -A
git commit -m "your change"
git push
```

Then in Colab just **re-run cell 1** (it does `git pull`) and continue training.

## Step 7 — Use the trained engine locally

1. In Google Drive, open `immortalite_zero_checkpoints_v3` (or your current `RUN_TAG` folder) and **download `latest.pt`**.
2. Put it in your local `checkpoints/` folder (or anywhere).
3. Verify encoding compatibility before starting the server:

```bash
python -m engine.inspect_encoding checkpoints/latest.pt
```

4. Start the analysis server pointing at it:

```bash
# Windows (PowerShell)
$env:IMMORTALITE_ZERO_CHECKPOINT="checkpoints\latest.pt"
python -m uvicorn server.app:app --port 8000
```

5. Open **http://localhost:8000/app/** and analyze.

> Re-download `latest.pt` and restart the server whenever you want the newest
> trained weights.

---

## Troubleshooting

- **"CUDA available: False"** → you skipped Step 2. Set the runtime to GPU and re-run.
- **Drive popup didn't appear / auth error** → re-run cell 3 and complete the Google login.
- **Training seems stuck** → it prints one line per iteration (can be 1–3 min each on the GPU preset). Give it a few minutes; check the loss plot.
- **Want it faster / a bigger net** → edit the `--gpu` preset values in `engine/train.py`, push, and `git pull` in Colab.
- **Out of memory** → lower `filters`/`blocks` or `batch_size` in `engine/config.py` (or the `--gpu` preset), push, pull, re-run.
