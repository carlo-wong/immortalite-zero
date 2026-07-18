# Immortalite Zero

A **lightweight, self-play chess engine** in the AlphaZero family (neural network + MCTS, not brute-force search), tuned for **decisive, attacking** play via draw shaping — wrapped in a Lichess-style web analysis tool and UCI front-end.

It learns purely from self-play (no human games, no Stockfish). Optional human-game pretraining is on the roadmap.

## What it does

- **Self-play training** with football-style draw shaping (`draw_penalty = 1/3`: a draw scores like one-third of a win).
- **Parallel self-play** across worker processes (`--selfplay-workers`) for higher throughput on multi-core GPU hosts.
- **SPRT strength gates** — sequential probability ratio tests with early stop (cap 128 games, H₀=0 Elo vs H₁=+25 Elo). Logs PASS / FAIL / INCONCLUSIVE; does not auto-reject checkpoints yet.
- **Beauty-bias move selection** (optional): among sound moves, prefer sacrificial / attacking / tactical / surprising lines.
- **Analysis GUI**: board, eval bar, best-move arrow, top-5 MultiPV lines, PGN/FEN import, move navigation.
- **UCI-compatible** for Arena, Cutechess, or Lichess local-engine mode.

## Project layout

```
engine/
  encoding.py   board ↔ tensor planes, move ↔ index (v2, 20 input planes)
  network.py    ResNet policy + HL-Gauss value head
  mcts.py       PUCT search + Gumbel completed-Q policy targets
  beauty.py     soundness gate + beauty scoring (optional)
  analyze.py    analysis API (eval, MultiPV)
  selfplay.py   self-play + parallel workers (play_games_parallel)
  sprt.py       Gaussian SPRT helpers for strength gates
  train.py      training loop, play_match, checkpointing
  config.py     defaults and tunables
  inspect_encoding.py   shard/checkpoint encoding audit
uci/uci_engine.py       UCI front-end
server/app.py           FastAPI /analyze + serves GUI
web/index.html          vanilla analysis GUI (no build step)
web/react/              optional React + Chessground GUI (/app-react/)
colab/train.ipynb       Google Colab training notebook
colab/README.md         Colab step-by-step guide
kaggle/train.ipynb      Kaggle GPU training notebook
kaggle/README.md        Kaggle step-by-step guide
lightning-ai/
  run_train.py          background training script
  run_gate.py           manual checkpoint gate script
  README.md             Lightning AI guide
scripts/download_syzygy345.py   one-time Syzygy download
tests/                  encoding, pipeline, gating, SPRT, parallel split
results/                local checkpoints + metrics (or sibling folder on Lightning)
```

## Quick start

```bash
pip install -r requirements.txt

# Tests (CPU, ~30s)
python -m pytest tests/ -q --ignore=tests/test_server.py

# Analysis server + GUI
python -m uvicorn server.app:app --port 8000
# http://localhost:8000/app/          vanilla GUI
# http://localhost:8000/app-react/    after npm run build in web/react/
```

The server auto-discovers `results/latest.pt`, then `checkpoints/latest.pt`, unless `IMMORTALITE_ZERO_CHECKPOINT` is set.

```bash
# Windows (PowerShell)
$env:IMMORTALITE_ZERO_CHECKPOINT="results\latest.pt"
python -m uvicorn server.app:app --port 8000
```

## Training

### Where to train

| Platform | Path | Notes |
|----------|------|-------|
| **Google Colab** | `colab/train.ipynb` | Free GPU, Drive checkpoints, 2 workers |
| **Kaggle** | `kaggle/train.ipynb` | Free GPU (~30h/week), Dataset persistence, 2 workers |
| **Lightning AI** | `lightning-ai/run_train.py` | ~4h sessions, background-friendly, 4 workers |
| **Local CPU** | `engine.train` | `--light` preset for smoke tests |

See `colab/README.md`, `kaggle/README.md`, and `lightning-ai/README.md` for full workflows.

### Current GPU recipe (Colab / Kaggle / Lightning)

These override the `--gpu` preset when passed on the CLI. Resume always keeps the **checkpoint architecture** (currently 8×96, 51 value bins).

| Setting | Colab / Kaggle | Lightning |
|---------|----------------|-----------|
| Games / iter | 128 | 128 |
| Train steps / iter | 800 | 800 |
| MCTS sims / move (self-play) | 150 | 150 |
| Concurrency | 128 | 128 |
| Self-play workers | 2 | 4 |
| Replay buffer / window | 200k | 200k |
| Draw penalty | 1/3 | 1/3 |
| Resign | off | off |
| LR | 2.5e-4 constant | 2.5e-4 constant |
| Training span | stop at iters 260, 280, … | stop at iters 260, 280, … |
| In-loop gate | off | off |
| Manual gate (SPRT cap) | 128 games | 128 games |
| Manual gate sims | 100 | 100 |
| Save snapshot | every 10 iters | every 10 iters |

**LR schedule:** cosine warmup/decay is built into `engine/train.py`, but both runners set `lr == lr_min` so the effective rate stays flat. When strength plateaus, manually drop both (e.g. to `6e-5`) and resume from `latest.pt`.

**Gates:** no in-loop auto-gate. Run the manual gate cell or `lightning-ai/run_gate.py` when you want SPRT (128-game cap, 100 sims). Logged to `metrics_gates.csv`: the SPRT (`llr`, `decision`, `verdict`) plus a logistic `elo` estimate with a 95% confidence interval (`elo_lower`/`elo_upper`) and `los`. Rotate or delete an old `metrics_gates.csv` if the header schema changed.

### Local smoke test

```bash
python -m engine.train --device cpu --light --games 8 --iterations 1 \
  --selfplay-workers 2 --train-steps 4 --gate-every 0 \
  --checkpoint-dir tmp/smoke --save-every 0
```

### Encoding compatibility

Canonical encoding uses **20 input planes** (side-to-move mirror + repetition + halfmove clock). Older 18-plane shards are ignored.

```bash
python -m engine.inspect_encoding --checkpoint-dir results
python -m engine.inspect_encoding results/latest.pt
```

## UCI engine

```bash
python -m uci.uci_engine results/latest.pt
```

Options: `Simulations`, `MultiPV`.

## Design notes

- **Strength:** a light net trained via pure self-play on free GPUs lands around club-amateur level — a characterful analysis tool, not a Stockfish rival.
- **Throughput:** on one GPU, use `selfplay_workers: 1` and set `concurrency` = `games` so MCTS batches at full width with `torch.compile` + FP16 in the main process. Multi-worker self-play duplicates CUDA contexts and is slower on a single GPU.
- **Research baked in:** Gumbel completed-Q policy targets, search draw-contempt, SPRT gating, Syzygy adjudication.

## Training history

See **[TRAINING_CHANGELOG.md](TRAINING_CHANGELOG.md)** for a summary of recipe changes across git history and when to resume vs restart.

## Relevant research

- Gumbel AlphaZero / MuZero — Danihelka et al., ICLR 2022
- Search-contempt — arXiv:2504.07757, 2025
- MiniZero — arXiv:2310.11305
- Maia / Maia-2 — KDD 2020 / NeurIPS 2024
- Grandmaster-Level Chess Without Search — DeepMind, NeurIPS 2024

## Roadmap

- Optional supervised pretraining on attacking-master games
- Full Gumbel sequential-halving budget allocation
- SPRT enforcement (auto-reject on FAIL)
- Central inference server for parallel self-play (single GPU copy)
