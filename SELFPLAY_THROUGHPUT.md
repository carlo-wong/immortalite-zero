# Self-play throughput notes

Mental note for future recipe changes. Goal: **fastest games/hour on one GPU** without changing search quality (current recipe: **200 sims/move**, same net, same MCTS).

**Always:** `selfplay_workers: 1`, `concurrency = games`. Multi-worker self-play spawns separate CUDA contexts and is slower on a single GPU.

---

## Seconds per game (mean)

Measured from `results/metrics.csv` where noted. Same hardware class (Colab/Lightning T4), 100 MCTS sims, Syzygy on, `workers=1`.

| Games / iter | Concurrency | Mean s/game | Games/hour | Source |
|--------------|-------------|-------------|------------|--------|
| **20** | 20 | **~22.0** | ~164 | *Estimated* — no long run logged; small batches underuse GPU |
| **64** | 64 | **20.49** | ~176 | iters 0–60 (61 iters) |
| **128** | 128 | **15.25** | ~236 | iters 61–120 (60 iters) |
| **256** | 256 | **16.62** | ~217 | iter 121 self-play only (256 scale-up trial) |

**Cross-scale mean** (20 + 64 + 128 + 256 buckets above): **~18.6 s/game** — useful as a rough “typical” anchor, not a target.

**Best recent 128 block** (iters 101–120, flat LR): mean **14.55 s/game**, median **14.66**, best **12.63** (iter 114).

---

## What we learned

1. **64 → 128** was a big win (~25% faster per game) — wider GPU batches without doubling CPU/MCTS stragglers as badly.
2. **128 → 256** was a loss (~17% slower per game vs recent 128) despite 2× batch width — CPU (MCTS, encoding, Syzygy) dominates; batch-256 adds overhead without enough GPU upside on T4.
3. **256 → 128** restored ~30 min/iter vs ~72 min at 256/1600 train steps.
4. **Below 64 games** (e.g. 20) is for smoke tests only — poor GPU fill.

---

## Recommended defaults (current recipe)

| Parameter | Value | Why |
|-----------|-------|-----|
| `games` | **128** | Best measured s/game on single GPU |
| `concurrency` | **128** | Must match `games` for full batch width |
| `selfplay_workers` | **1** | One GPU owner; compile + FP16 in main process |
| `train_steps` | **800** | ~6× sample reuse at batch 128 |
| `gate_games` | **256** | manual SPRT cap (200 sims, matches self-play) |

See iter **161+** in `TRAINING_CHANGELOG.md` (200 sims experiment).

---

## Before changing games/concurrency again

- [ ] Compare **s/game**, not just wall-clock per iter (doubling games doubles work).
- [ ] Keep `concurrency == games` and `selfplay_workers == 1` on one GPU unless testing multi-GPU.
- [ ] Log `selfplay X.Xs` from the iter line; consider adding `selfplay_seconds` to `metrics.csv`.
- [ ] Run at least **3 iters** after a change (compile warmup, variance).
- [ ] Suspect regression if s/game rises more than ~5% vs median of prior block.

---

## Not worth revisiting (unless hardware changes)

- `selfplay_workers > 1` on one GPU
- `games: 256` on T4-class GPU with current engine (tested iter 121)
- `gate_games: 512` when self-play is 128 — gates take too long for marginal SPRT precision

---

Last updated: 2026-06-28 (from metrics iters 0–120 + iter 121 log).
