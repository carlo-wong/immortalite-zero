# Training Immortalite Zero on Lightning AI

Self-play on a Lightning AI Studio GPU via `run_train.py` / `run_gate.py`. Same recipe as Colab (except 4 self-play workers), but checkpoints and Syzygy live in **sibling folders** you upload manually between sessions.

---

## Workspace layout

```
parent/
├── immortalite-zero/       # git clone
│   └── lightning-ai/
│       ├── run_train.py
│       ├── run_gate.py
│       └── paths.py
├── results/                # upload before each session
│   ├── latest.pt
│   ├── metrics.csv
│   ├── metrics_gates.csv
│   └── ckpt_iter_XXXX.pt
└── syzygy345/              # 145 .rtbw files (~378 MB)
```

Build Syzygy locally once:

```bash
python scripts/download_syzygy345.py --out syzygy345
```

---

## Before you start

- Lightning AI account with GPU studio.
- `results/` and `syzygy345/` as **siblings** of the repo (not inside it).
- Push engine changes to GitHub; `git pull` before `run_train.py`.

---

## Step 1 — Studio setup

1. Create a GPU studio.
2. Clone the repo.
3. Upload `results/` and `syzygy345/` next to the repo.
4. `pip install -q python-chess numpy tqdm`

## Step 2 — Train

Edit `TRAIN` in `lightning-ai/run_train.py` if needed, then:

```bash
cd immortalite-zero
nohup python lightning-ai/run_train.py > ../results/train.log 2>&1 &
tail -f ../results/train.log
```

Writes `latest.pt`, `metrics.csv`, shards every iteration to `../results/`. Training survives browser close (~4h studio limit still applies).

### Current `TRAIN` defaults

Same recipe as Colab except `selfplay_workers=4` (Lightning T4 has 4 vCPUs; Colab is 2). Bug-fix restart at iter **161** from `ckpt_iter_0160`. See `colab/README.md` and `TRAINING_CHANGELOG.md`.

| Key | Value |
|-----|-------|
| `sims` | 100 |
| `games` | 128 |
| `train_steps` | 800 |
| `concurrency` | 128 |
| `selfplay_workers` / `gate_workers` | 4 / 4 |
| `resign` | off |
| `replay_buffer` / `replay_window` | 200k |
| `gate_games` / `gate_sims` | 128 / 100 (manual gate only) |
| `gate_exploration_moves` / `gate_openings` | 0 / masters (64×2 colors) |
| `lr` / `lr_min` | 2.5e-4 flat |
| Training span | auto-stops at iters 160, 180, … (multiples of 20) |
| `RESET_OPTIMIZER` | `False` |

## Step 3 — Manual gate

Edit `CHECKPOINT_A` / `CHECKPOINT_B` in `lightning-ai/run_gate.py` (int or `"latest"`):

```bash
cd immortalite-zero
python lightning-ai/run_gate.py
```

Appends to `../results/metrics_gates.csv` with SPRT columns. Prints PASS / FAIL / INCONCLUSIVE.

## Step 4 — Sessions and resuming

When a studio ends, **download the updated `results/` folder**.

| Goal | Action |
|------|--------|
| Resume | Re-upload `results/` with `latest.pt`, re-run train |
| Fresh start | Empty `results/` (no `latest.pt`) |

Rotate old `metrics_gates.csv` if upgrading from pre-SPRT recipes.

## Step 5 — Use locally

```bash
python -m engine.inspect_encoding results/latest.pt
$env:IMMORTALITE_ZERO_CHECKPOINT="results\latest.pt"
python -m uvicorn server.app:app --port 8000
```

Open **http://localhost:8000/app/**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No CUDA | Select GPU machine, re-run |
| Syzygy incomplete | All 145 `.rtbw` in `syzygy345/` sibling folder |
| `results/` not found | Sibling of repo, not inside it |
| Slow self-play | Keep `concurrency` = `games`; `selfplay_workers=4` on Lightning (4 vCPUs). Colab bench only tested 2 |
| OOM | Lower `games` and `concurrency` together |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
