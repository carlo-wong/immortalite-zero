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
| **161**§ | **128** | **800** | **128** | **1** | **200k** | **256 SPRT** | **2.5e-4 flat** | rewind to `ckpt_iter_0160`; 200 sims only; resign off — **reverted** (one-hot policy targets from c_scale=1.0 bug + worst-child root-Q bug; iters 161–180 discarded) |
| **161**¶ | **128** | **800** | **128** | **2** | **200k** | **128 games** | **2.5e-4 flat** | claim_draw off in search — **reverted** (17× repetition draws corrupted value targets; gate 180 vs 160 −112 Elo FAIL; iters 161–180 discarded) |
| **161** | **128** | **800** | **128** | **2** | **200k** | **128 games** (Elo CI) | **2.5e-4 flat** | **100 sims**; resign off; **claim_draw on** in search; **value_target=root_q** (per-ply MCTS Q); Gumbel c_scale 0.1 + root-Q fixes; encoding vectorized |
| **241** | **128** | **800** | **128** | **2**/4 | **200k** | **128 games** (Elo CI) | **2.5e-4 flat** | same as 161 + **move_temperature=4.0** for first **10** plies (sampling only); log `metrics_first_moves.csv` |

**Current row:** start **241** — same as root_q recipe plus early-ply move temperature **T=4 / 10 plies** (policy targets untempered; gates unchanged). Resume from `latest.pt`. Do **not** use `--reset-optimizer`.

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

### Iter 122 — back to 128 + smaller SPRT cap

- **128 games / 800 train steps**, **concurrency 128**, **`selfplay_workers: 1`** — best measured s/game on single GPU.
- Gate cap **128**; LR 2.5e-4 flat; replay 200k (~12 iters at 128 games); draw 1/3; resign off; gate sims 100.

### Iter 161 — Phase 2A (reverted)

- Bundled LR warm restart, optimizer reset, and 120k replay — gate 180 vs 160 **−69 Elo**; training metrics improved but strength collapsed (entropy collapse). Run discarded; resume from **`ckpt_iter_0160`**.

### Iter 161 — sims 200 experiment (reverted)

- Rewind to **`ckpt_iter_0160`**; delete shards/checkpoints/metrics for iters 161–180.
- **Only change vs 141–160 recipe:** self-play and gate **200 MCTS sims** (was 100).
- **256-game SPRT cap** (was 128) for tighter gate estimates.
- **200k replay**, **2.5e-4 flat LR**, resign off, optimizer state preserved.

### Iter 161 — bug-fix restart with claim_draw=False (reverted)

- **Sims-200 run discarded:** two bugs introduced by the Jul 6 fix commit corrupted training targets. (1) Gumbel improved-policy collapsed to one-hot because `gumbel_c_scale` was set to 1.0 instead of 0.1 (argmax sigma dominates). (2) `searched_root_q` returned the worst child's value instead of the visit-weighted mean, corrupting truncation value labels. Both bugs are now fixed with regression tests.
- Rewound to **`ckpt_iter_0160`** with **100 sims**, workers **2**, vectorized encoding, Gumbel c_scale 0.1 + root-Q fixes.
- **`claim_draw=False` in MCTS search** (intended as a speedup) made search blind to threefold/fifty-move draws while the game loop still adjudicated them → 17× more repetition draws, corrupted value targets (−1/3 on winning positions), value_std collapse, gate 180 vs 160 **−112 Elo FAIL**.
- **Third discard of iters 161–180.** Resume from **`ckpt_iter_0160`**.

### Iter 161 — claim_draw restored (superseded by root_q labels)

- Same recipe as the bug-fix restart **except** search keeps **`claim_draw=True`** (Config default; no train.py override).
- Gate logging uses **Elo 95% CI verdict** (PASS if lower bound > 0, FAIL if upper bound < 0, else INCONCLUSIVE) — no H₀/H₁ LLR columns in `metrics_gates.csv`.
- **Resignation off**; workers 2; 200k replay; 2.5e-4 flat LR; 100 sims.

### Iter 161 — value_target=root_q

- **One change** vs claim_draw-restored recipe: self-play value labels use per-ply **`searched_root_q`** (`--value-target root_q`) instead of terminal game outcome (±1 / −draw_penalty).
- Policy targets unchanged (Gumbel improved policy). Search still uses `draw_contempt = draw_penalty` and `claim_draw=True`. Gates unchanged (WDL outcomes).
- Wired in `colab/train.ipynb` and `lightning-ai/run_train.py` TRAIN dicts.
- Abort watch: `value_std` should rise or stay high (not collapse toward 0); threefold count must stay ~2/128; do not trust train loss alone.

### Iter 241 — move temperature (current)

- Same recipe as root_q row, plus early-ply **move sampling temperature T=4 for first 10 plies** (`--move-temperature 4 --move-temperature-plies 10`).
- Sampling only during exploration plies; stored policy targets stay untempered. Gate exploration / gate temperature unchanged.
- Each self-play iter appends `metrics_first_moves.csv`:
  `iter,n,entropy,top1..top5_uci/share,main_share,flank_share`
  (`main`={e4,d4,Nf3,c4}; `flank`=wing/fianchetto set). An older CSV header is rotated to
  `metrics_first_moves_legacy.csv` on first write after upgrade.

Last updated: 2026-07-15.
