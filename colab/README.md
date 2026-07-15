# Training Immortalite Zero on Google Colab

Step-by-step guide for the free Colab GPU workflow. Open `colab/train.ipynb` and run cells in order.

> **What you're doing:** clone the repo in Colab, run self-play training on a free GPU, save checkpoints to Google Drive every iteration, and download `latest.pt` for local analysis.

---

## Before you start

- A **Google account** (Colab + Drive).
- Code on GitHub: `github.com/carlo-wong/immortalite-zero` — `git push` after local changes so Colab can `git pull`.
- **Syzygy:** cell 5 copies `syzygy345/` from Drive if present, or downloads once into your checkpoint folder.

---

## Step 1 — Open the notebook

**https://colab.research.google.com/github/carlo-wong/immortalite-zero/blob/main/colab/train.ipynb**

Or: [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** → `carlo-wong/immortalite-zero`.

## Step 2 — Enable GPU

**Runtime → Change runtime type → Hardware accelerator → GPU → Save.**

## Step 3 — Run cells

| Cell | What it does |
|------|--------------|
| 1 | Clone repo + `git pull` |
| 2 | Install `python-chess` |
| 3 | Mount Drive → `MyDrive/immortalite_zero_checkpoints` |
| 4 | Confirm GPU, set `--gpu` preset |
| 5 | Syzygy tablebases (Drive cache or download) |
| 6 | **Train** — edit `TRAIN` dict only; auto-stops at iters 160, 180, … |
| 7 | Optional **manual gate** (SPRT, 128 games / 100 sims) |
| 8 | Plot `metrics.csv` + gate results |

## Step 4 — Current `TRAIN` defaults (cell 6)

Bug-fix restart at iter **161** from `ckpt_iter_0160` — same as `lightning-ai/run_train.py`. See `TRAINING_CHANGELOG.md`.

| Key | Value | Notes |
|-----|-------|-------|
| `sims` | **100** | flat MCTS sims/move |
| `games` | 128 | full GPU batch width (`concurrency` matches) |
| `train_steps` | 800 | ~6× sample reuse at 128 games |
| `concurrency` | 128 | batched MCTS eval width (one GPU owner) |
| `selfplay_workers` / `gate_workers` | **2** / **2** | matches Colab 2 vCPU; separate CUDA process per worker |
| `replay_buffer` / `replay_window` | **200k** | ~12 iters at 128 games |
| `draw_penalty` | 1/3 | football 3-1-0 shaping |
| `resign` | False | off |
| `lr` / `lr_min` | **2.5e-4** | flat |
| `gate_games` / `gate_sims` | **128 / 100** | manual gate cell 7 only |
| `gate_exploration_moves` | **0** | after masters book (no temperature) |
| `gate_openings` | **masters** | 64 prefix-free lines × both colors (=128) |
| `save_every` | 10 | numbered snapshots |
| `resume` | True | loads `latest.pt` automatically |

Training auto-stops after completing an iter that is a multiple of **20** (160, 180, 200, …). Re-run cell 6 for the next span. No in-loop auto-gate.

## Step 5 — What good looks like

```
iter  40 | sims 100 | games 128 | samples 18500 | buffer 200000 | policy_loss 2.1 | value_loss 0.4 | lr 2.500e-04 | 420.0s
```

- **policy_loss** should trend down over many iterations (not every single iter).
- **value_loss** should stay meaningful — games need real outcomes, not only max-move truncations.
- **SPRT PASS** in a manual gate means significant improvement; **INCONCLUSIVE** is normal on short runs.
- Cell 8 plots are the clearest long-run signal.

Pure self-play on a free GPU targets club-level strength, not Stockfish.

## Step 6 — Disconnects and resuming

Checkpoints save to Drive **every iteration** (`latest.pt`, `metrics.csv`, sample shards).

| Goal | Action |
|------|--------|
| **Resume** after disconnect | Re-run cells 1→6. `resume: True` loads `latest.pt`. |
| **Fresh run** | Empty Drive checkpoint folder, re-run 1→6. |
| **Compare checkpoints** | Use cell 7 manual gate or download `ckpt_iter_XXXX.pt`. |

Numbered snapshots: `ckpt_iter_0000.pt`, `ckpt_iter_0010.pt`, … every `save_every` iters.

**metrics_gates.csv:** if you upgraded from an older recipe, delete or rotate the file — the header now includes the Fishtest-style SPRT (`llr`, `decision`, `verdict`) plus a logistic `elo` estimate with a 95% CI (`elo_lower`/`elo_upper`) and `los`.

## Step 7 — Update code from your machine

```bash
git add -A && git commit -m "your change" && git push
```

In Colab, re-run **cell 1** (`git pull`) and continue training.

## Step 8 — Use locally

1. Download `latest.pt` from Drive.
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
| `CUDA available: False` | Step 2 — set GPU runtime |
| Drive auth failed | Re-run cell 3 |
| Training "stuck" | One iter can take several minutes at 128 games; watch `metrics.csv` |
| OOM | Lower `games` / `concurrency` together, or reduce net in checkpoint (fresh start only) |
| SPRT always INCONCLUSIVE | Normal early; need more gate games or stronger signal |
| Old gate CSV garbled | Delete `metrics_gates.csv` and let it recreate |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
