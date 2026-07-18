# Training Immortalite Zero on Kaggle

Step-by-step guide for the free Kaggle GPU workflow. Open `kaggle/train.ipynb` and run cells in order.

> **What you're doing:** clone the repo in a Kaggle notebook, run self-play training on a free GPU, write checkpoints to `/kaggle/working/results`, then persist them as a private Dataset so the next session can resume.

---

## Before you start

- A **Kaggle account** with phone verification (needed for GPU).
- Code on GitHub: `github.com/carlo-wong/immortalite-zero` — `git push` after local changes so Kaggle can `git pull`.
- Optional Datasets (Add Input in the notebook):
  - **Results** — previous `/kaggle/working/results` folder containing `latest.pt` (and ideally `metrics.csv`, shards).
  - **Syzygy** — 145 `.rtbw` files (~378 MB). Build locally with `python scripts/download_syzygy345.py --out syzygy345`, then upload as a Dataset.

---

## Step 1 — Create the notebook

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Upload `kaggle/train.ipynb` from this repo (or copy cells manually).
3. **Session options → Accelerator → GPU** (T4 / P100 / T4×2).
4. Enable **Internet**.
5. (Optional) **Add Input** → your results Dataset and/or Syzygy Dataset.

---

## Step 2 — Run cells

| Cell | What it does |
|------|--------------|
| 1 | Clone repo + `git pull` into `/kaggle/working/immortalite-zero` |
| 2 | Install `python-chess` |
| 3 | Set `CKPT_DIR=/kaggle/working/results`; copy `latest.pt` from an input Dataset if present |
| 4 | Confirm GPU, set `--gpu` preset |
| 5 | Syzygy tablebases (Dataset cache or download) |
| 6 | **Train** — edit `TRAIN` dict only; auto-stops at iters 260, 280, … |
| 7 | Optional **manual gate** (SPRT, 128 games / 100 sims) |
| 8 | Plot `metrics.csv` + gate results |
| 9 | Reminder to persist `/kaggle/working/results` |

## Step 3 — Current `TRAIN` defaults (cell 6)

Same recipe as Colab: iter **261+**, `sims=150`, workers **2**. See `TRAINING_CHANGELOG.md`.

| Key | Value | Notes |
|-----|-------|-------|
| `sims` | **150** | self-play; gate stays 100 |
| `move_temperature` / `move_temperature_plies` | **4.0** / **10** | early-ply sampling only |
| `value_target` | **root_q** | per-ply MCTS root Q labels |
| `games` / `concurrency` | 128 / 128 | keep equal |
| `selfplay_workers` / `gate_workers` | **2** / **2** | raise to 4 only if cell 4 shows enough vCPUs |
| `replay_buffer` / `replay_window` | **200k** | |
| `draw_penalty` | 1/3 | football 3-1-0 shaping |
| `resign` | False | off |
| `lr` / `lr_min` | **2.5e-4** | flat |
| `gate_games` / `gate_sims` | **128 / 100** | manual gate cell 7 only |
| `gate_openings` | **masters** | 64 lines × both colors |
| `save_every` | 10 | numbered snapshots |
| `resume` | True | loads `latest.pt` when present |

Training auto-stops after completing an iter that is a multiple of **20**. Re-run cell 6 for the next span.

## Step 4 — Persist and resume

`/kaggle/working` does **not** survive session end unless you save it.

| Goal | Action |
|------|--------|
| **After a run** | Download `/kaggle/working/results`, or create/update a private Dataset from that folder |
| **Resume** | Add Input → results Dataset → re-run cells 1→6 (`resume: True`). If multiple inputs have `latest.pt`, cell 3 picks the highest checkpoint iteration. |
| **Fresh run** | Start with no results Dataset (or empty `CKPT_DIR`) |
| **Quota** | ~30 GPU hours/week, ~12h max session — stop the session when idle |

Prefer **Save Version → Save & Run All** for long unattended blocks; interactive tabs still burn quota if left with GPU attached.

## Step 5 — Update code from your machine

```bash
git add -A && git commit -m "your change" && git push
```

In Kaggle, re-run **cell 1** (`git pull`) and continue training.

## Step 6 — Use locally

1. Download `latest.pt` from your results Dataset / working dir.
2. Verify encoding:

```bash
python -m engine.inspect_encoding checkpoints/latest.pt
```

3. Start server:

```bash
$env:IMMORTALITE_ZERO_CHECKPOINT="checkpoints\latest.pt"
python -m uvicorn server.app:app --port 8000
```

4. Open **http://localhost:8000/app/**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GPU greyed out | Phone-verify account; check weekly GPU quota |
| `CUDA available: False` | Session options → Accelerator → GPU, then restart |
| No resume | Results Dataset missing `latest.pt`, or not added as Input |
| Syzygy slow/missing | Upload a Syzygy Dataset once; cell 5 will copy instead of re-download |
| OOM | Lower `games` / `concurrency` together |
| Session died mid-run | Last complete iter should be in `latest.pt` if you Saved Version / Dataset-updated |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
