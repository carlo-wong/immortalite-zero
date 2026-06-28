# Training Immortalite Zero on Lightning AI

Self-play on a Lightning AI Studio GPU. Same recipe as Colab, but checkpoints and Syzygy live in **sibling folders** you upload manually between sessions.

> **Prefer `run_train.py`** over the notebook — training survives browser close (~4h studio limit still applies).

---

## Workspace layout

```
parent/
├── immortalite-zero/       # git clone
│   └── lightning-ai/
│       ├── train.ipynb
│       ├── run_train.py    # recommended
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
- Push engine changes to GitHub; `git pull` in cell 1 or before `run_train.py`.

---

## Step 1 — Studio setup

1. Create a GPU studio.
2. Clone the repo.
3. Upload `results/` and `syzygy345/` next to the repo.
4. `pip install -q python-chess numpy tqdm`

## Step 2 — Train (script recommended)

Edit `TRAIN` in `lightning-ai/run_train.py` if needed, then:

```bash
cd immortalite-zero
nohup python lightning-ai/run_train.py > ../results/train.log 2>&1 &
tail -f ../results/train.log
```

Writes `latest.pt`, `metrics.csv`, shards every iteration to `../results/`.

### Current `TRAIN` defaults

Same as Colab except **`selfplay_workers: 4`** (Lightning hosts tend to have more CPU cores). See `colab/README.md` for the full parameter table.

| Key | Lightning value |
|-----|-----------------|
| `games` | 256 |
| `train_steps` | 1600 |
| `selfplay_workers` | 4 |
| `gate_games` | 512 (SPRT cap) |
| `lr` / `lr_min` | 2.5e-4 (constant) |

## Step 3 — Notebook alternative

Open `lightning-ai/train.ipynb` — **keep the browser tab open** (kernel stops if closed).

| Cell | What it does |
|------|--------------|
| 1 | Resolve `../results`, `../syzygy345`, `git pull` |
| 2 | Install `python-chess` |
| 3 | GPU check + `--gpu` preset |
| 4 | Verify Syzygy (145 `.rtbw`) |
| 5 | Train (`TRAIN` dict, same as script) |
| 6 | Optional manual SPRT gate |
| 7 | Plot metrics |

## Step 4 — Manual gate

Edit `CHECKPOINT_A` / `CHECKPOINT_B` in `lightning-ai/run_gate.py` (int or `"latest"`):

```bash
cd immortalite-zero
python lightning-ai/run_gate.py
```

Appends to `../results/metrics_gates.csv` with SPRT columns. Prints PASS / FAIL / INCONCLUSIVE.

## Step 5 — Sessions and resuming

When a studio ends, **download the updated `results/` folder**.

| Goal | Action |
|------|--------|
| Resume | Re-upload `results/` with `latest.pt`, re-run train |
| Fresh start | Empty `results/` (no `latest.pt`) |

Rotate old `metrics_gates.csv` if upgrading from pre-SPRT recipes.

## Step 6 — Use locally

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
| Slow self-play | Raise `selfplay_workers` (watch CPU); ensure `concurrency` 128 |
| OOM | Lower `games` and `concurrency` together |

Recipe history: **[TRAINING_CHANGELOG.md](../TRAINING_CHANGELOG.md)**
