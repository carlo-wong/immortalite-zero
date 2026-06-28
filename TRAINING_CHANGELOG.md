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
| **120** | **256** | **1600** | **256** | **1** | 200k | **512 SPRT** | 2.5e-4 flat | single GPU batch; concurrency=games |

**Current row:** start **120** — 256 games, concurrency 256, 1 worker, SPRT cap 512, LR 2.5e-4 constant. Multi-worker self-play reverted (slower on one GPU).

Resume keeps **checkpoint net architecture** (8×96, 51 value bins). Fresh net only with a new `--checkpoint-dir`.

---

## Major changes by start iter

### Iter 0 — baseline recipe

- Flat `TRAIN` dict in Colab; resume-on-by-default from `latest.pt`.
- **64 self-play games** per iteration, **400 train steps**, **128 batch** (~6× sample reuse).
- **100 MCTS sims** per move (training and gates); no sim ramp.
- **Draw penalty 1/3** in self-play (football 3-1-0); gates use normal WDL (draw contempt 0).
- **Resign off** during self-play and gates.
- **Syzygy** adjudication in self-play and gate matches.
- **Cosine LR** from 6e-4 down toward 1e-4 over the schedule horizon.
- **50k replay** buffer/window; grows to cap as shards accumulate.
- Auto-gate every 20 iters vs checkpoint 20 iters ago; **64 games**, winrate thresholds (~0.55 / 0.45).

### Iter 20 / 40 — hold steady

- No `TRAIN` dict changes; LR continues on the cosine schedule.
- Lets each 20-iter gate block compare nets trained on identical data scale.

### Iter 60 — throughput (label recipe unchanged)

- **Concurrency 128** so MCTS batches full GPU width while still playing 64 games/iter.
- **Replay buffer/window 200k** — more history without changing how positions are labeled.
- MCTS batching optimizations in the engine (faster eval throughput).
- Still 64 games / 400 steps until the scale-up row below.

### Iter 61 — data scale-up

- **128 games** and **800 train steps** scaled together with concurrency 128.
- Keeps ~6× sample reuse (more games → proportionally more train steps).
- Replay stays at 200k (~12 iters of history at 128 games/iter before later changes).

### Iter 80 — LR reset from anchor checkpoint

- Resume from **`ckpt_iter_0080`** with LR raised to a **flat ~6e-4** (end of previous cosine was too cold).
- Same 128 / 800 / 200k otherwise; grad clip 10.
- Intended as a consolidation block before the next LR drop.

### Iter 100 — cooler flat LR

- **LR and lr_min both 2.5e-4** — effective rate stays constant (no cosine decay).
- Same games, steps, replay, and 64-game winrate gates.
- Policy was still improving but loss/noise suggested the hotter 6e-4 block had run its course.

### Iter 120 — scale-up + SPRT gates (current)

- **256 games / 1600 train steps** — double games and steps vs 128/800; ~6× sample reuse at batch 128.
- **`selfplay_workers: 1`** — one GPU owner with `torch.compile` + FP16; multi-worker subprocess self-play reverted (contended CUDA on single GPU).
- **`concurrency: 256`** — matches games so every active position batches in one `evaluate_batch` call.
- **SPRT gates** replace winrate thresholds: cap **512 games**, early-stop when H₀ (0 Elo) or H₁ (+25 Elo) is decided; logs PASS / FAIL / INCONCLUSIVE. Not enforced yet (no auto-reject).
- `metrics_gates.csv` adds `llr`, `sprt_decision`, `games_played`, `elo0`, `elo1` — delete old CSV before first run on this recipe.
- Replay 200k holds ~**6 iters** of history at 256 games.
- LR unchanged at 2.5e-4; draw 1/3; resign off; gate sims 100; exploration moves 20.

Last updated: 2026-06-28.
