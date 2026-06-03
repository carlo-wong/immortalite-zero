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
import zipfile
from copy import deepcopy
from collections import Counter, deque
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config, NetConfig
from .encoding import ENCODING_VERSION
from .network import ChessNet, NetEvaluator
from .selfplay import GameResult, Sample, play_game_gen, play_games_batched

_SAMPLE_SHARD_PREFIX = "samples_iter_"
_SAMPLE_SHARD_SUFFIX = ".npz"


def _sims_for_iteration(cfg: Config, it: int) -> int:
    t = cfg.train
    if it >= t.sims_ramp_iterations:
        return t.sims_end
    frac = it / max(1, t.sims_ramp_iterations)
    return int(round(t.sims_start + frac * (t.sims_end - t.sims_start)))


def _gaussian_value_targets(target_v: torch.Tensor, support: torch.Tensor,
                            sigma: float) -> torch.Tensor:
    diff = support.unsqueeze(0) - target_v.unsqueeze(1)
    dist = torch.exp(-0.5 * (diff / sigma) ** 2)
    return dist / dist.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _load_matching_state_dict(module: torch.nn.Module, state_dict: dict,
                              *, label: str, verbose: bool = True) -> None:
    module_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state_dict.items():
        if key in module_state and module_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped += 1
    result = module.load_state_dict(matched, strict=False)
    if verbose:
        reinitialized = len(result.missing_keys)
        print(f"{label}: loaded {len(matched)} tensors, "
              f"reinitialized {reinitialized}, skipped {skipped}")


def train_step(net: ChessNet, optimizer, batch: list[Sample], device: str) -> dict[str, float]:
    planes = torch.from_numpy(np.stack([s.planes for s in batch])).to(device)
    target_pi = torch.from_numpy(np.stack([s.policy for s in batch])).to(device)
    target_v = torch.tensor([s.value for s in batch], dtype=torch.float32, device=device)

    logits, value_logits = net(planes)
    value = net.value_from_logits(value_logits)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_log_probs = F.log_softmax(value_logits, dim=1)
    bins = max(2, int(net.value_support.numel()))
    bin_width = 2.0 / (bins - 1)
    sigma = 0.75 * bin_width
    target_v_dist = _gaussian_value_targets(target_v, net.value_support, sigma)
    value_loss = -(target_v_dist * value_log_probs).sum(dim=1).mean()
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1e9)
    optimizer.step()

    with torch.no_grad():
        probs = log_probs.exp()
        policy_entropy = -(probs * log_probs).sum(dim=1).mean()
        value_sign_acc = (torch.sign(value) == torch.sign(target_v)).float().mean()
        policy_top1_agree = (torch.argmax(logits, dim=1) == torch.argmax(target_pi, dim=1)).float().mean()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "policy_entropy": float(policy_entropy.item()),
        "value_sign_acc": float(value_sign_acc.item()),
        "policy_top1_agree": float(policy_top1_agree.item()),
        "grad_norm": float(grad_norm),
    }


def save_checkpoint(net: ChessNet, cfg: Config, path: str, iteration: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Store the net architecture so the loader can rebuild a matching model,
    # and the iteration so training resumes with a continuous step count.
    torch.save(
        {
            "model": net.state_dict(),
            "net": asdict(cfg.net),
            "iteration": iteration,
            "encoding_version": ENCODING_VERSION,
        },
        path,
    )


def _log_metrics(ckpt_dir: str, it: int, sims: int, samples: int, dt: float, *,
                 policy_loss: float, value_loss: float,
                 policy_entropy: float, value_sign_acc: float,
                 policy_top1_agree: float, grad_norm: float,
                 mean_game_len: float, decisive_rate: float,
                 white_win_rate: float, draw_rate: float,
                 max_moves_trunc_rate: float, value_mean: float,
                 value_std: float, winrate_vs_prev: float,
                 termination_counts: dict[str, int]) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics.csv")
    new = not os.path.exists(path)
    terminations = ";".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "iter,sims,samples,seconds,policy_loss,value_loss,"
                "policy_entropy,value_sign_acc,policy_top1_agree,grad_norm,"
                "mean_game_len,decisive_rate,white_win_rate,draw_rate,"
                "max_moves_trunc_rate,value_mean,value_std,winrate_vs_prev,terminations\n"
            )
        f.write(
            f"{it},{sims},{samples},{dt:.1f},{policy_loss:.6f},{value_loss:.6f},"
            f"{policy_entropy:.6f},{value_sign_acc:.6f},{policy_top1_agree:.6f},{grad_norm:.6f},"
            f"{mean_game_len:.6f},{decisive_rate:.6f},{white_win_rate:.6f},{draw_rate:.6f},"
            f"{max_moves_trunc_rate:.6f},{value_mean:.6f},{value_std:.6f},{winrate_vs_prev:.6f},"
            f"{terminations}\n"
        )


def _log_step_metrics(ckpt_dir: str, it: int, step: int, metrics: dict[str, float]) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics_steps.csv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "iter,step,policy_loss,value_loss,policy_entropy,value_sign_acc,"
                "policy_top1_agree,grad_norm\n"
            )
        f.write(
            f"{it},{step},"
            f"{metrics['policy_loss']:.6f},{metrics['value_loss']:.6f},"
            f"{metrics['policy_entropy']:.6f},{metrics['value_sign_acc']:.6f},"
            f"{metrics['policy_top1_agree']:.6f},{metrics['grad_norm']:.6f}\n"
        )


def _winner_of(game: GameResult) -> int:
    """Return +1 for white win, -1 for black win, 0 for non-decisive result."""
    if game.termination not in {"checkmate", "resign"} or not game.samples:
        return 0
    first = game.samples[0]
    if first.value == 0.0:
        return 0
    winner_is_first_player = first.value > 0.0
    winner_is_white = bool(first.player) if winner_is_first_player else not bool(first.player)
    return 1 if winner_is_white else -1


def _latest_snapshot_before(ckpt_dir: str, iteration: int) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    best_iter = -1
    best_path: str | None = None
    for name in os.listdir(ckpt_dir):
        if not (name.startswith("ckpt_iter_") and name.endswith(".pt")):
            continue
        idx_text = name[len("ckpt_iter_"):-len(".pt")]
        if not idx_text.isdigit():
            continue
        idx = int(idx_text)
        if idx < iteration and idx > best_iter:
            best_iter = idx
            best_path = os.path.join(ckpt_dir, name)
    return best_path


def _play_match_game(cfg: Config, simulations: int,
                     white_eval: NetEvaluator, black_eval: NetEvaluator) -> GameResult:
    gen = play_game_gen(cfg, simulations, add_noise=False, exploration_moves=0)
    req = next(gen)
    while True:
        evaluator = white_eval if req.turn else black_eval
        logits, value = evaluator.evaluate(req)
        try:
            req = gen.send((logits, value))
        except StopIteration as stop:
            return stop.value


def play_match(net_a: ChessNet, net_b: ChessNet, cfg: Config,
               n_games: int, sims: int, device: str) -> float:
    if n_games <= 0:
        return float("nan")

    match_cfg = deepcopy(cfg)
    match_cfg.beauty.enabled = False

    eval_a = NetEvaluator(net_a, device=device)
    eval_b = NetEvaluator(net_b, device=device)
    score = 0.0
    gate_bar = tqdm(range(n_games), desc=f"gate ({n_games} games)", unit="game", leave=False)
    for game_idx in gate_bar:
        a_is_white = (game_idx % 2 == 0)
        white_eval = eval_a if a_is_white else eval_b
        black_eval = eval_b if a_is_white else eval_a
        game = _play_match_game(match_cfg, sims, white_eval, black_eval)
        winner = _winner_of(game)
        if winner == 0:
            score += 0.5
        elif (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            score += 1.0
        gate_bar.set_postfix(score=f"{score / (game_idx + 1):.3f}")
    gate_bar.close()
    return score / n_games


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
    planes = np.stack([s.planes for s in samples]).astype(np.float16)
    policies = np.stack([s.policy for s in samples]).astype(np.float16)
    players = np.array([bool(s.player) for s in samples], dtype=np.bool_)
    values = np.array([s.value for s in samples], dtype=np.float32)
    np.savez_compressed(
        path,
        planes=planes,
        policies=policies,
        players=players,
        values=values,
        encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
    )


def _shard_encoding_version(data: np.lib.npyio.NpzFile) -> int:
    if "encoding_version" not in data:
        return 1
    version_raw = np.asarray(data["encoding_version"]).reshape(-1)
    if version_raw.size == 0:
        return 1
    return int(version_raw[0])


def _load_sample_shard(path: str) -> list[Sample]:
    try:
        with np.load(path) as data:
            if _shard_encoding_version(data) != ENCODING_VERSION:
                return []
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
    except (ValueError, OSError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        print(f"warning: skipping unreadable shard {os.path.basename(path)} ({exc})")
        return []


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
        if not shard_samples:
            continue
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
    parser.add_argument(
        "--c1-stage",
        choices=["off", "24", "32", "48", "64"],
        default="off",
        help="throughput ramp profile: sets games/train-steps/concurrency for C1",
    )
    parser.add_argument("--max-game-moves", type=int, default=None,
                        help="self-play truncation cap (C2 shaping)")
    parser.add_argument("--draw-penalty", type=float, default=None,
                        help="target for draw outcomes (C2 shaping)")
    parser.add_argument("--resign-threshold", type=float, default=None,
                        help="enable resignation when root value <= threshold")
    parser.add_argument("--resign-plies", type=int, default=None,
                        help="consecutive plies below threshold before resignation")
    parser.add_argument("--resign-min-moves", type=int, default=None,
                        help="minimum plies before resignation can trigger")
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
    parser.add_argument("--gate-every", type=int, default=0,
                        help="run strength gate every N iterations (0 = off)")
    parser.add_argument("--gate-games", type=int, default=20,
                        help="games per strength gate (current net vs previous snapshot)")
    args = parser.parse_args()

    # Fall back to CPU if CUDA was requested but isn't available in this runtime.
    # This also lets torch.load map CUDA-saved checkpoints onto the CPU.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to --device cpu")
        args.device = "cpu"

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
    c1_profiles = {
        "24": (24, 480, 24),
        "32": (32, 640, 32),
        "48": (48, 960, 48),
        "64": (64, 1280, 64),
    }
    if args.c1_stage != "off":
        games, steps, concurrency = c1_profiles[args.c1_stage]
        cfg.train.games_per_iteration = games
        cfg.train.train_steps_per_iteration = steps
        cfg.train.selfplay_concurrency = concurrency
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
    if args.max_game_moves is not None:
        cfg.train.max_game_moves = args.max_game_moves
    if args.draw_penalty is not None:
        cfg.train.draw_penalty = args.draw_penalty
    if args.resign_threshold is not None:
        cfg.train.resign_threshold = args.resign_threshold
    if args.resign_plies is not None:
        cfg.train.resign_plies = args.resign_plies
    if args.resign_min_moves is not None:
        cfg.train.resign_min_moves = args.resign_min_moves
    # Keep self-play search contempt aligned with the draw target shaping.
    cfg.mcts.draw_contempt = cfg.train.draw_penalty
    print(
        "config: "
        f"games={cfg.train.games_per_iteration} "
        f"steps={cfg.train.train_steps_per_iteration} "
        f"concurrency={cfg.train.selfplay_concurrency} "
        f"max_moves={cfg.train.max_game_moves} "
        f"draw_penalty={cfg.train.draw_penalty:.3f} "
        f"resign_threshold={cfg.train.resign_threshold:.3f} "
        f"resign_plies={cfg.train.resign_plies} "
        f"resign_min_moves={cfg.train.resign_min_moves}"
    )

    # When resuming, the checkpoint's own architecture wins over the CLI preset
    # (you cannot change net size mid-training). To train a fresh net of a
    # different size, resume from an empty folder instead.
    start_iter = 0
    state = None
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=args.device)
        ckpt_encoding_version = 1
        if isinstance(state, dict):
            ckpt_encoding_version = int(state.get("encoding_version", 1))
        if ckpt_encoding_version != ENCODING_VERSION:
            raise ValueError(
                f"checkpoint encoding version {ckpt_encoding_version} does not match "
                f"current encoding version {ENCODING_VERSION}; start with a fresh "
                f"--checkpoint-dir for this encoding"
            )
        if isinstance(state, dict) and "net" in state:
            cfg.net = NetConfig(**state["net"])
    net = ChessNet(cfg.net).to(args.device)
    if state is not None:
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        _load_matching_state_dict(net, model_state, label="resume load")
        start_iter = int(state.get("iteration", -1)) + 1
        print(f"resumed from {args.resume} at iteration {start_iter} "
              f"(net: {cfg.net.blocks}x{cfg.net.filters})")

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    buffer: deque[Sample] = deque(maxlen=cfg.train.replay_buffer_size)
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
        game_lengths: list[int] = []
        game_outcomes: list[int] = []
        n_games = cfg.train.games_per_iteration
        game_bar = tqdm(total=n_games, desc=f"iter {it} self-play", unit="game", leave=False)

        def _on_game(game: GameResult) -> None:
            nonlocal new_samples
            buffer.extend(game.samples)
            iteration_samples.extend(game.samples)
            new_samples += len(game.samples)
            termination_counts[game.termination] += 1
            game_lengths.append(len(game.samples))
            game_outcomes.append(_winner_of(game))
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

        net.train()
        step_metrics: dict[str, list[float]] = {}
        if len(buffer) >= cfg.train.batch_size:
            train_bar = tqdm(range(cfg.train.train_steps_per_iteration),
                             desc=f"iter {it} train", unit="step", leave=False)
            for step in train_bar:
                batch = random.sample(buffer, cfg.train.batch_size)
                step_result = train_step(net, optimizer, batch, args.device)
                for name, value in step_result.items():
                    step_metrics.setdefault(name, []).append(value)
                _log_step_metrics(cfg.train.checkpoint_dir, it, int(step), step_result)
                train_bar.set_postfix(
                    p=f"{step_result['policy_loss']:.3f}",
                    v=f"{step_result['value_loss']:.3f}",
                    g=f"{step_result['grad_norm']:.2f}",
                )
        net.eval()

        save_checkpoint(net, cfg, os.path.join(cfg.train.checkpoint_dir, "latest.pt"), it)
        if args.save_every and it % args.save_every == 0:
            snap = os.path.join(cfg.train.checkpoint_dir, f"ckpt_iter_{it:04d}.pt")
            save_checkpoint(net, cfg, snap, it)

        def _mean_metric(name: str) -> float:
            values = step_metrics.get(name)
            return float(np.mean(values)) if values else float("nan")

        dt = time.time() - t0
        pl = _mean_metric("policy_loss")
        vl = _mean_metric("value_loss")
        policy_entropy = _mean_metric("policy_entropy")
        value_sign_acc = _mean_metric("value_sign_acc")
        policy_top1_agree = _mean_metric("policy_top1_agree")
        grad_norm = _mean_metric("grad_norm")

        total_games = len(game_lengths)
        white_wins = sum(1 for o in game_outcomes if o == 1)
        black_wins = sum(1 for o in game_outcomes if o == -1)
        draws_or_other = total_games - white_wins - black_wins
        mean_game_len = float(np.mean(game_lengths)) if game_lengths else float("nan")
        decisive_games = termination_counts.get("checkmate", 0) + termination_counts.get("resign", 0)
        decisive_rate = decisive_games / total_games if total_games else float("nan")
        white_win_rate = white_wins / total_games if total_games else float("nan")
        draw_rate = draws_or_other / total_games if total_games else float("nan")
        max_moves_trunc_rate = (
            termination_counts.get("max_moves", 0) / total_games if total_games else float("nan")
        )
        value_targets = np.array([s.value for s in iteration_samples], dtype=np.float32)
        value_mean = float(value_targets.mean()) if value_targets.size else float("nan")
        value_std = float(value_targets.std()) if value_targets.size else float("nan")

        winrate_vs_prev = float("nan")
        if args.gate_every > 0 and it > 0 and it % args.gate_every == 0:
            previous_snapshot = _latest_snapshot_before(cfg.train.checkpoint_dir, it)
            if previous_snapshot is None:
                print(f"gate iter {it}: skipped (no prior numbered snapshot found)")
            else:
                prev_state = torch.load(previous_snapshot, map_location=args.device)
                if isinstance(prev_state, dict) and "model" in prev_state:
                    prev_encoding_version = int(prev_state.get("encoding_version", 1))
                    if prev_encoding_version != ENCODING_VERSION:
                        print(
                            f"gate iter {it}: skipped ({os.path.basename(previous_snapshot)} "
                            f"encoding {prev_encoding_version} != {ENCODING_VERSION})"
                        )
                    else:
                        prev_net_cfg = cfg.net
                        if "net" in prev_state:
                            prev_net_cfg = NetConfig(**prev_state["net"])
                        prev_net = ChessNet(prev_net_cfg).to(args.device)
                        prev_model = (
                            prev_state["model"]
                            if isinstance(prev_state, dict) and "model" in prev_state
                            else prev_state
                        )
                        _load_matching_state_dict(prev_net, prev_model, label="gate load", verbose=False)
                        winrate_vs_prev = play_match(
                            net, prev_net, cfg, n_games=args.gate_games, sims=sims, device=args.device
                        )
                        print(
                            f"gate iter {it}: current vs {os.path.basename(previous_snapshot)} "
                            f"-> {winrate_vs_prev:.3f}"
                        )
                else:
                    print(f"gate iter {it}: skipped (invalid snapshot {previous_snapshot})")

        term_summary = ", ".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
        print(f"iter {it:3d} | sims {sims:3d} | games {cfg.train.games_per_iteration} "
              f"| samples {new_samples:4d} | buffer {len(buffer):6d} "
              f"| policy_loss {pl:.3f} | value_loss {vl:.3f} "
              f"| ent {policy_entropy:.3f} | sign_acc {value_sign_acc:.3f} "
              f"| decisive {decisive_rate:.3f} | gate {winrate_vs_prev:.3f} "
              f"| ends {term_summary} "
              f"| {dt:.1f}s", flush=True)
        _log_metrics(
            cfg.train.checkpoint_dir,
            it,
            sims,
            new_samples,
            dt,
            policy_loss=pl,
            value_loss=vl,
            policy_entropy=policy_entropy,
            value_sign_acc=value_sign_acc,
            policy_top1_agree=policy_top1_agree,
            grad_norm=grad_norm,
            mean_game_len=mean_game_len,
            decisive_rate=decisive_rate,
            white_win_rate=white_win_rate,
            draw_rate=draw_rate,
            max_moves_trunc_rate=max_moves_trunc_rate,
            value_mean=value_mean,
            value_std=value_std,
            winrate_vs_prev=winrate_vs_prev,
            termination_counts=dict(termination_counts),
        )


if __name__ == "__main__":
    main()
