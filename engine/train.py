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
from collections import Counter, deque
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config, NetConfig
from .network import ChessNet, NetEvaluator
from .selfplay import GameResult, Sample, play_games_batched

_SAMPLE_SHARD_PREFIX = "samples_iter_"
_SAMPLE_SHARD_SUFFIX = ".npz"


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
                 pl: float, vl: float, dt: float,
                 termination_counts: dict[str, int]) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics.csv")
    new = not os.path.exists(path)
    terminations = ";".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("iter,sims,samples,policy_loss,value_loss,seconds,terminations\n")
        f.write(f"{it},{sims},{samples},{pl:.4f},{vl:.4f},{dt:.1f},{terminations}\n")


def _sample_shard_path(ckpt_dir: str, iteration: int) -> str:
    return os.path.join(ckpt_dir, f"{_SAMPLE_SHARD_PREFIX}{iteration:04d}{_SAMPLE_SHARD_SUFFIX}")


def _list_sample_shards(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    names = [
        name for name in os.listdir(ckpt_dir)
        if name.startswith(_SAMPLE_SHARD_PREFIX) and name.endswith(_SAMPLE_SHARD_SUFFIX)
    ]
    names.sort()
    return [os.path.join(ckpt_dir, name) for name in names]


def _save_sample_shard(ckpt_dir: str, iteration: int, samples: list[Sample]) -> None:
    if not samples:
        return
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = _sample_shard_path(ckpt_dir, iteration)
    planes = np.stack([s.planes for s in samples]).astype(np.uint8)
    policies = np.stack([s.policy for s in samples]).astype(np.float16)
    players = np.array([bool(s.player) for s in samples], dtype=np.bool_)
    values = np.array([s.value for s in samples], dtype=np.float32)
    np.savez_compressed(path, planes=planes, policies=policies, players=players, values=values)


def _count_samples_in_shard(path: str) -> int:
    with np.load(path) as data:
        key = "values" if "values" in data else "value"
        return int(data[key].shape[0])


def _load_sample_shard(path: str) -> list[Sample]:
    with np.load(path) as data:
        planes = data["planes"]
        policies = data["policies"] if "policies" in data else data["policy"]
        players = data["players"] if "players" in data else data["player"]
        values = data["values"] if "values" in data else data["value"]
        samples: list[Sample] = []
        for i in range(values.shape[0]):
            samples.append(
                Sample(
                    planes=planes[i].astype(np.float32),
                    policy=policies[i].astype(np.float32),
                    player=bool(players[i]),
                    value=float(values[i]),
                )
            )
        return samples


def _prune_sample_shards(ckpt_dir: str, keep_samples: int) -> None:
    if keep_samples <= 0:
        return
    shards = _list_sample_shards(ckpt_dir)
    if not shards:
        return

    keep: set[str] = set()
    total = 0
    for path in reversed(shards):
        keep.add(path)
        total += _count_samples_in_shard(path)
        if total >= keep_samples:
            break

    for path in shards:
        if path not in keep:
            os.remove(path)


def _warm_replay_buffer(buffer: deque[Sample], ckpt_dir: str, replay_window: int) -> int:
    if replay_window <= 0:
        return 0
    maxlen = buffer.maxlen if buffer.maxlen is not None else replay_window
    target = min(maxlen, replay_window)
    if target <= 0:
        return 0

    shards = _list_sample_shards(ckpt_dir)
    chosen_chunks: list[list[Sample]] = []
    remaining = target
    for path in reversed(shards):
        if remaining <= 0:
            break
        shard_samples = _load_sample_shard(path)
        if len(shard_samples) > remaining:
            shard_samples = shard_samples[-remaining:]
        chosen_chunks.append(shard_samples)
        remaining -= len(shard_samples)

    loaded = 0
    for chunk in reversed(chosen_chunks):
        for sample in chunk:
            buffer.append(sample)
            loaded += 1
    return loaded


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
    parser.add_argument("--concurrency", type=int, default=None,
                        help="concurrent self-play games to batch on GPU")
    parser.add_argument("--replay-window", type=int, default=None,
                        help="max persisted replay samples kept on disk")
    args = parser.parse_args()

    cfg = Config()
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.light:
        cfg.net.blocks, cfg.net.filters = 4, 48
        cfg.train.games_per_iteration = 6
        cfg.train.selfplay_concurrency = 4
        cfg.train.train_steps_per_iteration = 80
        cfg.train.sims_start, cfg.train.sims_end = 24, 60
    if args.gpu:
        cfg.net.blocks, cfg.net.filters = 8, 96
        cfg.train.games_per_iteration = 12
        cfg.train.selfplay_concurrency = 64
        cfg.train.train_steps_per_iteration = 300
        cfg.train.sims_start, cfg.train.sims_end = 48, 128
    if args.games is not None:
        cfg.train.games_per_iteration = args.games
    if args.train_steps is not None:
        cfg.train.train_steps_per_iteration = args.train_steps
    if args.sims is not None:
        cfg.train.sims_start = cfg.train.sims_end = args.sims
    if args.concurrency is not None:
        cfg.train.selfplay_concurrency = args.concurrency
    if args.replay_window is not None:
        cfg.train.replay_window = args.replay_window
    # Keep self-play search contempt aligned with the draw target shaping.
    cfg.mcts.draw_contempt = cfg.train.draw_penalty

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
    _prune_sample_shards(cfg.train.checkpoint_dir, cfg.train.replay_window)
    loaded = _warm_replay_buffer(buffer, cfg.train.checkpoint_dir, cfg.train.replay_window)
    if loaded:
        print(f"warmed replay buffer with {loaded} samples from shard files")

    for local_it in range(args.iterations):
        it = start_iter + local_it
        sims = _sims_for_iteration(cfg, it)
        evaluator = NetEvaluator(net, device=args.device)

        t0 = time.time()
        new_samples = 0
        iteration_samples: list[Sample] = []
        termination_counts: Counter[str] = Counter()
        n_games = cfg.train.games_per_iteration
        game_bar = tqdm(total=n_games, desc=f"iter {it} self-play", unit="game", leave=False)

        def _on_game(game: GameResult) -> None:
            nonlocal new_samples
            buffer.extend(game.samples)
            iteration_samples.extend(game.samples)
            new_samples += len(game.samples)
            termination_counts[game.termination] += 1
            game_bar.set_postfix(moves=len(game.samples), buffer=len(buffer))
            game_bar.update(1)

        play_games_batched(
            evaluator,
            cfg,
            simulations=sims,
            num_games=n_games,
            concurrency=cfg.train.selfplay_concurrency,
            on_game_finished=_on_game,
        )
        game_bar.close()

        _save_sample_shard(cfg.train.checkpoint_dir, it, iteration_samples)
        _prune_sample_shards(cfg.train.checkpoint_dir, cfg.train.replay_window)

        net.train()
        p_losses, v_losses = [], []
        if len(buffer) >= cfg.train.batch_size:
            train_bar = tqdm(range(cfg.train.train_steps_per_iteration),
                             desc=f"iter {it} train", unit="step", leave=False)
            for _ in train_bar:
                batch = random.sample(buffer, cfg.train.batch_size)
                pl, vl = train_step(net, optimizer, batch, args.device)
                p_losses.append(pl)
                v_losses.append(vl)
                train_bar.set_postfix(p=f"{pl:.3f}", v=f"{vl:.3f}")
        net.eval()

        save_checkpoint(net, cfg, os.path.join(cfg.train.checkpoint_dir, "latest.pt"), it)
        if args.save_every and it % args.save_every == 0:
            snap = os.path.join(cfg.train.checkpoint_dir, f"ckpt_iter_{it:04d}.pt")
            save_checkpoint(net, cfg, snap, it)

        dt = time.time() - t0
        pl = np.mean(p_losses) if p_losses else float("nan")
        vl = np.mean(v_losses) if v_losses else float("nan")
        term_summary = ", ".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
        print(f"iter {it:3d} | sims {sims:3d} | games {cfg.train.games_per_iteration} "
              f"| samples {new_samples:4d} | buffer {len(buffer):6d} "
              f"| policy_loss {pl:.3f} | value_loss {vl:.3f} | ends {term_summary} "
              f"| {dt:.1f}s", flush=True)
        _log_metrics(cfg.train.checkpoint_dir, it, sims, new_samples, pl, vl, dt,
                     dict(termination_counts))


if __name__ == "__main__":
    main()
