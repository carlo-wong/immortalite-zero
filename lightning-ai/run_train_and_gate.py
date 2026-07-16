#!/usr/bin/env python3
"""Train to the next multiple of 20, then gate vs the checkpoint 20 iters ago.

Combines ``run_train.py`` + ``run_gate.py`` in one terminal job.

Example (latest at iter 260):
  train iters 261..280, then gate ckpt 280 vs 260.

  cd immortalite-zero
  nohup python lightning-ai/run_train_and_gate.py > ../results/train_and_gate.log 2>&1 &
  tail -f ../results/train_and_gate.log

Uses the same TRAIN dict / STOP_INTERVAL as ``run_train.py``.
Gate uses TRAIN gate_* knobs and ``gate_sims`` (not self-play ``sims``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from paths import ensure_ckpt_dir, resolve_paths, validate_syzygy
from run_gate import run_gate_match
from run_train import (
    RESET_OPTIMIZER,
    RESIGN_MIN_MOVES,
    RESIGN_PLIES,
    RESIGN_THRESHOLD,
    STOP_INTERVAL,
    TRAIN,
    _training_span,
)


def main() -> None:
    paths = resolve_paths()
    ensure_ckpt_dir(paths)
    rtbw = validate_syzygy(paths.tb_dir)

    has_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    preset = ["--device", "cuda", "--gpu"] if has_cuda else ["--device", "cpu", "--light"]

    resume_path = os.path.join(paths.ckpt_dir, "latest.pt")
    resume_args: list[str] = []
    if TRAIN["resume"]:
        if os.path.exists(resume_path):
            resume_args = ["--resume", resume_path]
        else:
            print("WARNING: resume=True but no latest.pt — starting at iter 0")

    start_iter, end_iter, train_iterations = _training_span(
        resume_path, TRAIN["resume"], STOP_INTERVAL,
    )
    prev_iter = end_iter - STOP_INTERVAL
    if prev_iter < 0:
        raise SystemExit(
            f"Cannot gate: end_iter={end_iter} has no prev checkpoint "
            f"(need end_iter >= {STOP_INTERVAL})"
        )

    resign_args: list[str] = []
    if TRAIN["resign"]:
        resign_args = [
            "--resign-threshold", str(RESIGN_THRESHOLD),
            "--resign-plies", str(RESIGN_PLIES),
            "--resign-min-moves", str(RESIGN_MIN_MOVES),
        ]

    cmd = [
        sys.executable, "-m", "engine.train",
        "--iterations", str(train_iterations),
        *preset,
        "--games", str(TRAIN["games"]),
        "--train-steps", str(TRAIN["train_steps"]),
        "--concurrency", str(TRAIN["concurrency"]),
        "--selfplay-workers", str(TRAIN["selfplay_workers"]),
        "--replay-buffer", str(TRAIN["replay_buffer"]),
        "--replay-window", str(TRAIN["replay_window"]),
        "--sims", str(TRAIN["sims"]),
        "--draw-penalty", str(TRAIN["draw_penalty"]),
        "--value-target", str(TRAIN["value_target"]),
        *resign_args,
        "--syzygy-path", paths.tb_dir,
        "--save-every", str(TRAIN["save_every"]),
        "--gate-every", "0",
        "--quick-eval-games", "0",
        "--lr", str(TRAIN["lr"]),
        "--lr-min", str(TRAIN["lr_min"]),
        "--lr-total-iters", str(TRAIN["lr_total_iters"]),
        "--lr-warmup-iters", str(TRAIN["lr_warmup_iters"]),
        "--grad-clip", str(TRAIN["grad_clip"]),
        "--move-temperature", str(TRAIN["move_temperature"]),
        "--move-temperature-plies", str(TRAIN["move_temperature_plies"]),
        "--checkpoint-dir", paths.ckpt_dir,
        *resume_args,
    ]
    if RESET_OPTIMIZER:
        cmd.append("--reset-optimizer")

    print("repo:       ", paths.repo_dir)
    print("checkpoints:", paths.ckpt_dir)
    print("syzygy:     ", paths.tb_dir, f"({rtbw} .rtbw)")
    print("CUDA:       ", has_cuda, device)
    print("TRAIN:      ", TRAIN)
    print(
        f"training span: iters {start_iter}..{end_iter} "
        f"({train_iterations} iterations), stop_interval={STOP_INTERVAL}"
    )
    print(f"after train: gate {end_iter} vs {prev_iter} (gate_sims={TRAIN['gate_sims']})")
    print("command:    ", " ".join(cmd))
    print()

    os.chdir(paths.repo_dir)
    subprocess.run(cmd, check=True)

    # Confirm end checkpoint exists (save_every should hit multiples of 20).
    end_ckpt = os.path.join(paths.ckpt_dir, f"ckpt_iter_{end_iter:04d}.pt")
    if not os.path.exists(end_ckpt):
        raise FileNotFoundError(
            f"Expected {end_ckpt} after training; check save_every={TRAIN['save_every']}"
        )

    print()
    print("=" * 40)
    print(f"TRAINING DONE — gating {end_iter} vs {prev_iter}")
    print("=" * 40)
    print()

    run_gate_match(
        end_iter,
        prev_iter,
        gate_games=int(TRAIN["gate_games"]),
        gate_sims=int(TRAIN["gate_sims"]),
        gate_workers=int(TRAIN.get("gate_workers", TRAIN["selfplay_workers"])),
        gate_concurrency=int(TRAIN.get("gate_concurrency", TRAIN["concurrency"])),
        gate_exploration_moves=int(TRAIN["gate_exploration_moves"]),
        gate_openings=str(TRAIN["gate_openings"]),
        draw_penalty=float(TRAIN["draw_penalty"]),
    )


if __name__ == "__main__":
    main()
