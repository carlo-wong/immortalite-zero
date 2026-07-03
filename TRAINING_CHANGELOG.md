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
| **120** | 256 | 1600 | 256 | 1 | 200k | 512 SPRT | 2.5e-4 flat | scale-up trial; reverted at 122 |
| **122** | **128** | **800** | **128** | **1** | 200k | **128 SPRT** | 2.5e-4 flat | faster s/game; gate cap matches batch |
| **161**‡ | **128** | **800** | **128** | **1** | **120k** | **128 SPRT** | **5e-4→2e-4** (161–196) | Phase 2A — **reverted** (regressed ~69 Elo vs 160); shards/checkpoints 161–180 removed |
| **161** | **128** | **800** | **128** | **1** | **200k** | **256 SPRT** | **2.5e-4 flat** | rewind to `ckpt_iter_0160`; **200 sims** only; resign off |

**Current row:** start **161** (from iter **160** checkpoint) — 200 sims self-play + gate, 256-game SPRT cap, 200k replay, 2.5e-4 flat LR. Resume from `latest.pt` (= `ckpt_iter_0160.pt`). Do **not** use `--reset-optimizer`.

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

### Iter 120 — 256-game scale-up (reverted at 122)

- **256 games / 1600 train steps** — trial; ~17% slower per game vs 128 on T4 (see iter 121 metrics).
- **`selfplay_workers: 1`**, **`concurrency: 256`**, SPRT cap 512.

### Iter 122 — back to 128 + smaller SPRT cap (current)

- **128 games / 800 train steps**, **concurrency 128**, **`selfplay_workers: 1`** — best measured s/game on single GPU.
- **SPRT gate cap 128** (was 512); still early-stops when H₀/H₁ decided.
- LR 2.5e-4 flat; replay 200k (~12 iters at 128 games); draw 1/3; resign off; gate sims 100.

### Iter 161 — Phase 2A (reverted)

- Bundled LR warm restart, optimizer reset, and 120k replay — gate 180 vs 160 **−69 Elo**; training metrics improved but strength collapsed (entropy collapse). Run discarded; resume from **`ckpt_iter_0160`**.

### Iter 161 — sims 200 experiment (current)

- Rewind to **`ckpt_iter_0160`**; delete shards/checkpoints/metrics for iters 161–180.
- **Only change vs 141–160 recipe:** self-play and gate **200 MCTS sims** (was 100).
- **256-game SPRT cap** (was 128) for tighter gate estimates.
- **200k replay**, **2.5e-4 flat LR**, resign off, optimizer state preserved.

Last updated: 2026-07-04.
