#!/usr/bin/env python3
"""Self-play training for Lightning AI (notebook cell 5 as a script).

Runs in the terminal so training continues after you close the browser.
Same sibling-folder layout as train.ipynb: ../results and ../syzygy345.

Example (background, survives browser close):
  cd immortalite-zero
  nohup python lightning-ai/run_train.py > ../results/train.log 2>&1 &
  tail -f ../results/train.log
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

# --- edit training settings here (matches lightning-ai/train.ipynb cell 5) ---
TRAIN = {
    "sims": 100,
    "gate_sims": 100,
    "games": 256,
    "train_steps": 1600,
    "concurrency": 128,
    "selfplay_workers": 4,
    "replay_buffer": 200_000,
    "replay_window": 200_000,
    "draw_penalty": 1 / 3,
    "gate_every": 20,
    "gate_games": 512,
    "gate_exploration_moves": 20,
    "save_every": 10,
    "iterations": 1000,
    "resume": True,
    "resign": False,
    "lr": 2.5e-4,
    "lr_min": 2.5e-4,
    "lr_total_iters": 1000,
    "grad_clip": 10.0,
}
RESIGN_THRESHOLD = -0.90
RESIGN_PLIES = 3
RESIGN_MIN_MOVES = 20


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

    resign_args: list[str] = []
    if TRAIN["resign"]:
        resign_args = [
            "--resign-threshold", str(RESIGN_THRESHOLD),
            "--resign-plies", str(RESIGN_PLIES),
            "--resign-min-moves", str(RESIGN_MIN_MOVES),
        ]

    cmd = [
        sys.executable, "-m", "engine.train",
        "--iterations", str(TRAIN["iterations"]),
        *preset,
        "--games", str(TRAIN["games"]),
        "--train-steps", str(TRAIN["train_steps"]),
        "--concurrency", str(TRAIN["concurrency"]),
        "--selfplay-workers", str(TRAIN["selfplay_workers"]),
        "--replay-buffer", str(TRAIN["replay_buffer"]),
        "--replay-window", str(TRAIN["replay_window"]),
        "--sims", str(TRAIN["sims"]),
        "--draw-penalty", str(TRAIN["draw_penalty"]),
        *resign_args,
        "--syzygy-path", paths.tb_dir,
        "--save-every", str(TRAIN["save_every"]),
        "--gate-every", str(TRAIN["gate_every"]),
        "--gate-games", str(TRAIN["gate_games"]),
        "--gate-sims", str(TRAIN["gate_sims"]),
        "--gate-exploration-moves", str(TRAIN["gate_exploration_moves"]),
        "--lr", str(TRAIN["lr"]),
        "--lr-min", str(TRAIN["lr_min"]),
        "--lr-total-iters", str(TRAIN["lr_total_iters"]),
        "--grad-clip", str(TRAIN["grad_clip"]),
        "--checkpoint-dir", paths.ckpt_dir,
        *resume_args,
    ]

    print("repo:       ", paths.repo_dir)
    print("checkpoints:", paths.ckpt_dir)
    print("syzygy:     ", paths.tb_dir, f"({rtbw} .rtbw)")
    print("CUDA:       ", has_cuda, device)
    print("TRAIN:      ", TRAIN)
    print("command:    ", " ".join(cmd))
    print()

    os.chdir(paths.repo_dir)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
