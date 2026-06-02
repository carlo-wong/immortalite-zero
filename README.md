# Immortalite

A **lightweight, self-play chess engine** in the AlphaZero family (neural network + MCTS, *not* brute-force search), tuned to play **beautiful-but-sound** chess — sacrifices, attacks, tactics, and surprising "alien" moves — wrapped in a Lichess-style web analysis tool.

It learns purely from self-play (no human games, no Stockfish). Optional human-game pretraining is on the roadmap for extra strength.

## What it does

- **Self-play training** with reward shaping that discourages dull draws (contempt) for more decisive, aggressive play.
- **Beauty-bias move selection**: among moves that are *sound* (within a small win-probability window of the best move), it prefers the most sacrificial / attacking / tactical / surprising one. Soundness is a hard gate; beauty only breaks ties within it.
- **Analysis GUI**: board, eval bar, best-move + beautiful-move arrows, top-3 candidate lines (MultiPV), PGN/FEN import, move navigation — and it shows the *cost* of the beautiful move vs. the objectively best one.
- **UCI-compatible** so the engine also runs in Arena, Cutechess, or Lichess's local-engine mode.

## Project layout

```
engine/
  encoding.py   board <-> tensor planes, move <-> index (AlphaZero 8x8x73 scheme)
  network.py    lightweight ResNet, policy + value heads
  mcts.py       PUCT search + Gumbel "completed-Q" improved policy
  beauty.py     soundness gate + beauty scoring  <-- the heart of the project
  analyze.py    high-level analysis API (eval, MultiPV, best vs beautiful)
  selfplay.py   self-play game generation (reward shaping)
  train.py      training loop with checkpointing + progressive simulations
  config.py     all tunables in one place
uci/uci_engine.py   UCI front-end
server/app.py       FastAPI /analyze backend + serves the GUI
web/index.html      self-contained analysis GUI (no build step)
colab/train.ipynb   Colab free-GPU training notebook
tests/              encoding round-trip + end-to-end pipeline smoke tests
```

## Quick start

```bash
pip install -r requirements.txt

# Run the tests (fast, CPU)
python -m tests.test_encoding
python -m tests.test_pipeline

# Start the analysis server + GUI
python -m uvicorn server.app:app --port 8000
# open http://localhost:8000/app/
```

To analyze with a trained net, point the server at a checkpoint:

```bash
# Windows
set IMMORTALITE_CHECKPOINT=checkpoints\latest.pt
python -m uvicorn server.app:app --port 8000
```

## Training

Local (slow, CPU):

```bash
python -m engine.train --iterations 20 --device cpu
```

Colab free GPU (recommended): open `colab/train.ipynb`, set the runtime to GPU, and run the cells. Checkpoints save to Google Drive every iteration so disconnects don't lose progress.

Note: the current canonical encoding uses 20 input planes (side-to-move mirrored + repetition + halfmove clock). Older 18-plane sample shards/checkpoints are intentionally ignored by the trainer; start this encoding with a fresh `--checkpoint-dir`.

You can inspect a folder before training/resume:

```bash
python -m engine.inspect_encoding --checkpoint-dir checkpoints
python -m engine.inspect_encoding --checkpoint-dir checkpoints --only-incompatible --json
```

## Use as a UCI engine

```bash
python -m uci.uci_engine checkpoints/latest.pt
```

Options: `Simulations` (search budget/move), `Beauty` (play beautiful vs best), `MultiPV`.

## Design notes & honest expectations

- **Strength**: a *light* model trained via *pure self-play* on free hardware lands around club-amateur level — this is a fun, characterful analysis tool, not a Stockfish/Leela rival. Strength comes from model + data scale, which we deliberately trade away for speed and style.
- **Tunable spice**: `engine/config.py -> BeautyConfig.soundness_window` is the main "how risky" dial. Wider window = more spectacular but slightly less sound; narrower = safer.
- **Research baked in**: Gumbel "completed-Q" improved policy (sample-efficient at low simulation counts) and MiniZero-style progressive simulations (start cheap, ramp up) help self-play actually improve within a free Colab budget. Draw-contempt nudges toward decisive, beautiful games.

## Relevant research

- Gumbel AlphaZero / MuZero — Danihelka et al., ICLR 2022 (sample-efficient low-sim search)
- Search-contempt — arXiv:2504.07757, 2025 (compute-efficient self-play on consumer GPUs)
- MiniZero — arXiv:2310.11305 (progressive simulation)
- Maia / Maia-2 — KDD 2020 / NeurIPS 2024 (human-like, characterful play)
- Grandmaster-Level Chess Without Search / ChessBench — DeepMind, NeurIPS 2024 (dataset for optional pretraining)

## Roadmap

- Optional supervised pretraining on attacking-master games (Tal, Kasparov, Morphy) for more strength + sharper style.
- Full Gumbel sequential-halving budget allocation in search.
- Upgrade GUI to React + Chessground if a richer UI is wanted.
```
