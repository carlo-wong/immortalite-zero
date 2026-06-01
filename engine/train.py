"""Self-play training loop with checkpointing.

Run:  python -m engine.train --iterations 50 --device cpu
On Colab use --device cuda. Checkpoints are written every iteration so runs
survive disconnects (point --checkpoint-dir at a Google Drive folder).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, NetConfig
from .network import ChessNet, NetEvaluator
from .selfplay import Sample, play_game


def _sims_for_iteration(cfg: Config, it: int) -> int:
    t = cfg.train
    if it >= t.sims_ramp_iterations:
        return t.sims_end
    frac = it / max(1, t.sims_ramp_iterations)
    return int(round(t.sims_start + frac * (t.sims_end - t.sims_start)))


def train_step(net: ChessNet, optimizer, batch: list[Sample], device: str) -> tuple[float, float]:
    planes = torch.from_numpy(np.stack([s.planes for s in batch])).to(device)
    target_pi = torch.from_numpy(np.stack([s.policy for s in batch])).to(device)
    target_v = torch.tensor([s.value for s in batch], dtype=torch.float32, device=device)

    logits, value = net(planes)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_v)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def save_checkpoint(net: ChessNet, cfg: Config, path: str, iteration: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Store the net architecture so the loader can rebuild a matching model,
    # and the iteration so training resumes with a continuous step count.
    torch.save({"model": net.state_dict(), "net": asdict(cfg.net),
                "iteration": iteration}, path)


def _log_metrics(ckpt_dir: str, it: int, sims: int, samples: int,
                 pl: float, vl: float, dt: float) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics.csv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("iter,sims,samples,policy_loss,value_loss,seconds\n")
        f.write(f"{it},{sims},{samples},{pl:.4f},{vl:.4f},{dt:.1f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", default=None)
    # Per-iteration workload overrides (lower these for faster CPU iterations).
    parser.add_argument("--games", type=int, default=None, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=None, help="fixed MCTS sims/move (disables ramp)")
    parser.add_argument("--train-steps", type=int, default=None, help="optimizer steps per iteration")
    parser.add_argument("--light", action="store_true",
                        help="CPU-friendly preset: small net, few games/sims")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU preset: larger net, more games/sims per iteration")
    parser.add_argument("--save-every", type=int, default=5,
                        help="also save a numbered checkpoint every N iterations (0 = off)")
    args = parser.parse_args()

    cfg = Config()
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.light:
        cfg.net.blocks, cfg.net.filters = 4, 48
        cfg.train.games_per_iteration = 6
        cfg.train.train_steps_per_iteration = 80
        cfg.train.sims_start, cfg.train.sims_end = 24, 60
    if args.gpu:
        cfg.net.blocks, cfg.net.filters = 10, 128
        cfg.train.games_per_iteration = 40
        cfg.train.train_steps_per_iteration = 400
        cfg.train.sims_start, cfg.train.sims_end = 60, 160
    if args.games is not None:
        cfg.train.games_per_iteration = args.games
    if args.train_steps is not None:
        cfg.train.train_steps_per_iteration = args.train_steps
    if args.sims is not None:
        cfg.train.sims_start = cfg.train.sims_end = args.sims

    # When resuming, the checkpoint's own architecture wins over the CLI preset
    # (you cannot change net size mid-training). To train a fresh net of a
    # different size, resume from an empty folder instead.
    start_iter = 0
    state = None
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=args.device)
        if isinstance(state, dict) and "net" in state:
            cfg.net = NetConfig(**state["net"])
    net = ChessNet(cfg.net).to(args.device)
    if state is not None:
        net.load_state_dict(state["model"])
        start_iter = int(state.get("iteration", -1)) + 1
        print(f"resumed from {args.resume} at iteration {start_iter} "
              f"(net: {cfg.net.blocks}x{cfg.net.filters})")

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    buffer: deque[Sample] = deque(maxlen=cfg.train.replay_buffer_size)

    for local_it in range(args.iterations):
        it = start_iter + local_it
        sims = _sims_for_iteration(cfg, it)
        evaluator = NetEvaluator(net, device=args.device)

        t0 = time.time()
        new_samples = 0
        for _ in range(cfg.train.games_per_iteration):
            samples = play_game(evaluator, cfg, simulations=sims)
            buffer.extend(samples)
            new_samples += len(samples)

        net.train()
        p_losses, v_losses = [], []
        if len(buffer) >= cfg.train.batch_size:
            for _ in range(cfg.train.train_steps_per_iteration):
                batch = random.sample(buffer, cfg.train.batch_size)
                pl, vl = train_step(net, optimizer, batch, args.device)
                p_losses.append(pl)
                v_losses.append(vl)
        net.eval()

        save_checkpoint(net, cfg, os.path.join(cfg.train.checkpoint_dir, "latest.pt"), it)
        if args.save_every and it % args.save_every == 0:
            snap = os.path.join(cfg.train.checkpoint_dir, f"ckpt_iter_{it:04d}.pt")
            save_checkpoint(net, cfg, snap, it)

        dt = time.time() - t0
        pl = np.mean(p_losses) if p_losses else float("nan")
        vl = np.mean(v_losses) if v_losses else float("nan")
        print(f"iter {it:3d} | sims {sims:3d} | games {cfg.train.games_per_iteration} "
              f"| samples {new_samples:4d} | buffer {len(buffer):6d} "
              f"| policy_loss {pl:.3f} | value_loss {vl:.3f} | {dt:.1f}s", flush=True)
        _log_metrics(cfg.train.checkpoint_dir, it, sims, new_samples, pl, vl, dt)


if __name__ == "__main__":
    main()
