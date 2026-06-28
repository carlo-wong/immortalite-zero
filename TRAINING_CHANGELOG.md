# Training recipe changelog

Resume from the listed **start iter** with the `TRAIN` settings below (`colab/train.ipynb` cell 6 or `lightning-ai/run_train.py`). Training-parameter changes were aligned to **every 20 iterations** so each gate compares against a checkpoint trained on the same recipe.

Gates run every 20 iters vs the checkpoint **20 iters ago**. Edit only the `TRAIN` dict when moving to a new row.

| Start iter | Games | Train steps | Concurrency | Workers | Replay | Gate | LR | Notes |
|------------|-------|-------------|-------------|---------|--------|------|-----|-------|
| **0** | 64 | 400 | 64 | 1 | 50k | 64 games, winrate | cosine 6e-4→1e-4 | draw 1/3, 100 sims, resign off |
| **20** | 64 | 400 | 64 | 1 | 50k | 64 | (cosine) | same recipe |
| **40** | 64 | 400 | 64 | 1 | 50k | 64 | (cosine) | same recipe |
| **60** | 64 | 400 | **128** | 1 | **200k** | 64 | (cosine) | MCTS batch throughput |
| **61**† | **128** | **800** | 128 | 1 | 200k | 64 | (cosine) | scale games + steps with concurrency |
| **80** | 128 | 800 | 128 | 1 | 200k | 64 | **~6e-4 flat** | resume from `ckpt_iter_0080` |
| **100** | 128 | 800 | 128 | 1 | 200k | 64 | **2.5e-4 flat** | consolidate after hot LR |
| **120** | **256** | **1600** | 128 | **2** / **4**‡ | 200k | **512 SPRT** | 2.5e-4 flat | parallel self-play; Lightning uses 4 workers |

† First logged iter with 128 games / 800 steps in `metrics.csv` is **61** (resume timing). Treat **60** as the intended boundary when preparing the recipe.

‡ Colab **2** workers, Lightning **4**.

**Current row:** start **120** — 256×1600, SPRT cap 512 (H₀ 0 Elo / H₁ +25 Elo), LR 2.5e-4 constant. Rotate old `metrics_gates.csv` if upgrading from pre-SPRT runs.

Resume keeps **checkpoint net architecture** (8×96, 51 value bins). Fresh net only with a new `--checkpoint-dir`.

Last updated: 2026-06-28.
