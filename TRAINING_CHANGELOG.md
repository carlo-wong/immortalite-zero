# Training recipe changelog

When the Colab/Lightning `TRAIN` recipe changed. Edit training in `colab/train.ipynb` cell 6 or `lightning-ai/run_train.py`.

**Current (phase I):** 256 games × 1600 steps, 100 sims, 200k replay, draw 1/3, LR 2.5e-4 constant, SPRT gates (cap 512) every 20 iters vs checkpoint 20 iters ago, 2 workers (Colab) / 4 (Lightning).

| Phase | Approx. use | Games | Train steps | Concurrency | Workers | Gate games | LR | Other |
|-------|-------------|-------|-------------|-------------|---------|------------|-----|-------|
| **A** Early Colab | pre-Jun 2026 | 24 | 150 | 64 | 1 | 20 | cosine default | Syzygy added |
| **B** Resume tune | early Jun | 64 | 400 | 64 | 1 | 60 | cosine | fixed 100 sims |
| **C** Flat TRAIN | fresh ~iter 0 | 64 | 400 | 64 | 1 | 64 | default | draw 0.03 |
| **D** Football draws | fresh or resume | 64 | 400 | 64 | 1 | 64 | default | draw 1/3 |
| **E** Throughput 60 | resume ~iter 60+ | 64 | 400 | **128** | 1 | 64 | default | replay 200k |
| **F** Scale batch | resume ~iter 60+ | **128** | **800** | 128 | 1 | 64 | default | |
| **G** Long-run LR | resume mid-run | 128 | 800 | 128 | 1 | 64 | **6e-4→1e-4** | grad_clip |
| **H** Iter 100 LR | resume ~iter 100+ | 128 | 800 | 128 | 1 | 64 | **2.5e-4 const** | |
| **I** Parallel + SPRT | **current** | **256** | **1600** | 128 | **2–4** | **512 SPRT** | 2.5e-4 const | |

Resume keeps **checkpoint net architecture** (production run: 8×96). `--gpu` preset only applies on fresh starts.

**Resume vs restart:** same encoding + recipe → resume `latest.pt`; v1→v2 encoding or old `metrics_gates.csv` header → fresh dir or delete gate CSV.

Last updated: 2026-06-28.
