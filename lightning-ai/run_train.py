#!/usr/bin/env python3
"""Self-play training for Lightning AI (terminal / nohup).

Runs in the terminal so training continues after you close the browser.
Sibling-folder layout: ../results and ../syzygy345.

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

# --- edit training settings here ---
STOP_INTERVAL = 20  # stop after completing iters 160, 180, 200, …

TRAIN = {
    "sims": 100,
    "gate_sims": 100,  # manual gate (run_gate.py / notebook gate cell) only
    "games": 128,
    "train_steps": 800,
    "concurrency": 128,
    "selfplay_workers": 4,  # Lightning T4 has 4 vCPUs; Colab bench was 2-vCPU only
    "replay_buffer": 200_000,
    "replay_window": 200_000,
    "draw_penalty": 1 / 3,
    # Per-ply MCTS root Q value labels (option B); gates still use WDL outcomes.
    "value_target": "root_q",
    "gate_games": 128,
    "gate_workers": 4,
    "gate_concurrency": 128,
    # Masters book (64 lines × 2 colors); 0 temperature after forced start.
    "gate_exploration_moves": 0,
    "gate_openings": "masters",
    "save_every": 10,
    "resume": True,
    "resign": False,
    "lr": 2.5e-4,
    "lr_min": 2.5e-4,
    "lr_total_iters": 10_000,
    "lr_warmup_iters": 0,
    "grad_clip": 10.0,
    # Early-ply move sampling temperature (policy targets stay untempered).
    "move_temperature": 4.0,
    "move_temperature_plies": 10,
}
RESET_OPTIMIZER = False
RESIGN_THRESHOLD = -0.90
RESIGN_PLIES = 3
RESIGN_MIN_MOVES = 20


def _training_span(resume_path: str, resume: bool, stop_interval: int) -> tuple[int, int, int]:
    """Return (start_iter, end_iter, num_iterations). Stops after completing end_iter."""
    start_iter = 0
    if resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        start_iter = int(state.get("iteration", -1)) + 1
    if start_iter % stop_interval == 0:
        end_iter = start_iter
    else:
        end_iter = ((start_iter // stop_interval) + 1) * stop_interval
    return start_iter, end_iter, end_iter - start_iter + 1


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
    print("command:    ", " ".join(cmd))
    print()

    os.chdir(paths.repo_dir)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
