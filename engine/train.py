"""Self-play training loop with checkpointing.

Run:  python -m engine.train --iterations 50 --device cpu
On Colab use --device cuda. Checkpoints are written every iteration so runs
survive disconnects (point --checkpoint-dir at a Google Drive folder).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import tempfile
import time
import zipfile
from copy import deepcopy
from collections import Counter, deque
from dataclasses import asdict
from typing import Callable

import chess
import chess.syzygy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config, NetConfig
from .encoding import ENCODING_VERSION
from .first_move_stats import (
    CSV_COLUMNS as FIRST_MOVE_CSV_COLUMNS,
    summarize_first_moves,
    summarize_first_moves_from_shard,
)
from .network import ChessNet, NetEvaluator
from .sprt import (
    ALPHA,
    BETA,
    ELO0,
    ELO1,
    sprt_bounds,
    sprt_decision,
    sprt_elo,
    sprt_llr,
)
from .selfplay import (
    EvalRequest,
    GATE_OPENING_PLIES,
    GameResult,
    Sample,
    _opening_row,
    play_game_gen,
    play_games_batched,
    play_games_parallel,
    play_match_parallel,
)

_SAMPLE_SHARD_PREFIX = "samples_iter_"
_SAMPLE_SHARD_SUFFIX = ".npz"


def _tqdm_sync_progress(bar: tqdm, completed: int) -> None:
    """Advance a tqdm bar to `completed` (safe for multi-worker polling in notebooks)."""
    delta = int(completed) - int(bar.n)
    if delta > 0:
        bar.update(delta)


def _net_cfg_from_module(net: ChessNet) -> NetConfig:
    return NetConfig(
        blocks=len(net.tower),
        filters=int(net.stem[0].out_channels),
        value_bins=int(net.value_bins),
    )


def _lr_for_iteration(cfg: Config, it: int) -> float:
    peak_lr = float(cfg.train.learning_rate)
    min_lr = float(cfg.train.lr_min)
    if peak_lr <= 0.0:
        return peak_lr
    if min_lr < 0.0:
        min_lr = 0.0
    if min_lr > peak_lr:
        min_lr = peak_lr

    warmup_iters = max(0, int(cfg.train.lr_warmup_iters))
    total_iters = max(1, int(cfg.train.lr_total_iters))

    if warmup_iters > 0 and it < warmup_iters:
        warmup_frac = (it + 1) / warmup_iters
        return min_lr + (peak_lr - min_lr) * warmup_frac

    if total_iters <= warmup_iters:
        return min_lr

    decay_span = total_iters - warmup_iters
    decay_step = min(max(it - warmup_iters, 0), decay_span)
    decay_frac = decay_step / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_frac))
    return min_lr + (peak_lr - min_lr) * cosine


def _finite_median(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    finite.sort()
    mid = len(finite) // 2
    if len(finite) % 2 == 1:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2.0


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


def train_step(net: ChessNet, optimizer, batch: list[Sample], device: str,
               scaler: torch.cuda.amp.GradScaler | None = None,
               grad_clip: float = 10.0) -> dict[str, float]:
    use_cuda = device.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        planes = torch.from_numpy(np.stack([s.planes for s in batch])).to(device).float()
        target_pi = torch.from_numpy(np.stack([s.policy for s in batch])).to(device).float()
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
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()

    with torch.no_grad():
        probs = log_probs.float().exp()
        log_probs_f = log_probs.float()
        value_f = value.float()
        policy_entropy = -(probs * log_probs_f).sum(dim=1).mean()
        value_sign_acc = (torch.sign(value_f) == torch.sign(target_v)).float().mean()
        policy_top1_agree = (torch.argmax(logits, dim=1) == torch.argmax(target_pi, dim=1)).float().mean()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "policy_entropy": float(policy_entropy.item()),
        "value_sign_acc": float(value_sign_acc.item()),
        "policy_top1_agree": float(policy_top1_agree.item()),
        "grad_norm": float(grad_norm),
    }


def save_checkpoint(net: torch.nn.Module, cfg: Config, path: str, iteration: int = 0,
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    ckpt_dir = os.path.dirname(path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    # Store the net architecture so the loader can rebuild a matching model,
    # and the iteration so training resumes with a continuous step count.
    model_module = getattr(net, "_orig_mod", net)
    payload = {
        "model": model_module.state_dict(),
        "net": asdict(cfg.net),
        "iteration": iteration,
        "encoding_version": ENCODING_VERSION,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _log_metrics(ckpt_dir: str, it: int, sims: int, samples: int, dt: float, *,
                 selfplay_seconds: float = float("nan"),
                 train_seconds: float = float("nan"),
                 policy_loss: float, value_loss: float,
                 policy_entropy: float, value_sign_acc: float,
                 policy_top1_agree: float, grad_norm: float,
                 mean_game_len: float, decisive_rate: float,
                 white_win_rate: float, draw_rate: float,
                 max_moves_trunc_rate: float, value_mean: float,
                 value_std: float, winrate_vs_prev: float,
                 learning_rate: float, games: int, train_steps: int,
                 batch_size: int, buffer_size: int,
                 termination_counts: dict[str, int],
                 winrate_quick: float = float("nan"),
                 gpu_util_pct: float = float("nan"),
                 buffer_min_iter: int = 0,
                 buffer_max_iter: int = 0) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics.csv")
    new = not os.path.exists(path)
    terminations = ";".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "iter,sims,samples,seconds,selfplay_seconds,train_seconds,policy_loss,value_loss,"
                "policy_entropy,value_sign_acc,policy_top1_agree,grad_norm,"
                "mean_game_len,decisive_rate,white_win_rate,draw_rate,"
                "max_moves_trunc_rate,value_mean,value_std,winrate_vs_prev,"
                "learning_rate,games,train_steps,batch_size,buffer_size,terminations,"
                "winrate_quick,gpu_util_pct,buffer_min_iter,buffer_max_iter\n"
            )
        f.write(
            f"{it},{sims},{samples},{dt:.1f},{selfplay_seconds:.1f},{train_seconds:.1f},"
            f"{policy_loss:.6f},{value_loss:.6f},"
            f"{policy_entropy:.6f},{value_sign_acc:.6f},{policy_top1_agree:.6f},{grad_norm:.6f},"
            f"{mean_game_len:.6f},{decisive_rate:.6f},{white_win_rate:.6f},{draw_rate:.6f},"
            f"{max_moves_trunc_rate:.6f},{value_mean:.6f},{value_std:.6f},{winrate_vs_prev:.6f},"
            f"{learning_rate:.6e},{games},{train_steps},{batch_size},{buffer_size},"
            f"{terminations},"
            f"{winrate_quick:.6f},{gpu_util_pct:.6f},{buffer_min_iter},{buffer_max_iter}\n"
        )


def _update_metrics_winrate_vs_prev(ckpt_dir: str, it: int, winrate: float) -> None:
    """Patch winrate_vs_prev on the most recent metrics row for *it*."""
    path = os.path.join(ckpt_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return
    header = lines[0].strip().split(",")
    try:
        winrate_idx = header.index("winrate_vs_prev")
    except ValueError:
        return
    for i in range(len(lines) - 1, 0, -1):
        parts = lines[i].rstrip("\n").split(",")
        if parts and parts[0] == str(it):
            parts[winrate_idx] = f"{winrate:.6f}"
            lines[i] = ",".join(parts) + "\n"
            break
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


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


def _log_first_move_metrics(ckpt_dir: str, it: int, stats: dict) -> None:
    """Append first-move diversity row (top-5 + main/flank) and print a one-liner.

    If an older CSV header is present, rename it to ``*_legacy.csv`` and start fresh.
    """
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics_first_moves.csv")
    header = ",".join(FIRST_MOVE_CSV_COLUMNS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = f.readline().strip()
        if existing and existing != header:
            legacy = os.path.join(ckpt_dir, "metrics_first_moves_legacy.csv")
            os.replace(path, legacy)
            print(f"first-moves: rotated old CSV → {legacy}", flush=True)

    n = int(stats.get("n", 0) or 0)
    entropy = float(stats.get("entropy", float("nan")))
    main = float(stats.get("main_share", float("nan")))
    flank = float(stats.get("flank_share", float("nan")))
    tops: list[tuple[str, float]] = []
    for i in range(1, 6):
        tops.append((
            str(stats.get(f"top{i}_uci", "") or ""),
            float(stats.get(f"top{i}_share", float("nan"))),
        ))

    def _fmt(x: float) -> str:
        return f"{x:.6f}" if math.isfinite(x) else "nan"

    def _top_note(uci: str, share: float) -> str:
        if uci and math.isfinite(share):
            return f"{uci}:{share:.2f}"
        return "none"

    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(header + "\n")
        parts = [str(it), str(n), _fmt(entropy)]
        for uci, share in tops:
            parts.extend([uci, _fmt(share)])
        parts.extend([_fmt(main), _fmt(flank)])
        f.write(",".join(parts) + "\n")
    top_bits = " ".join(
        f"top{i}={_top_note(uci, share)}" for i, (uci, share) in enumerate(tops, start=1)
    )
    print(
        f"first-moves: H={_fmt(entropy)} {top_bits} "
        f"main={_fmt(main)} flank={_fmt(flank)} (n={n})",
        flush=True,
    )


def _flush_step_metrics(ckpt_dir: str, it: int, rows: list[tuple[int, dict[str, float]]]) -> None:
    if not rows:
        return
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics_steps.csv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "iter,step,policy_loss,value_loss,policy_entropy,value_sign_acc,"
                "policy_top1_agree,grad_norm\n"
            )
        for step, metrics in rows:
            f.write(
                f"{it},{step},"
                f"{metrics['policy_loss']:.6f},{metrics['value_loss']:.6f},"
                f"{metrics['policy_entropy']:.6f},{metrics['value_sign_acc']:.6f},"
                f"{metrics['policy_top1_agree']:.6f},{metrics['grad_norm']:.6f}\n"
            )


GATE_CSV_HEADER = (
    "iter,prev_iter,games,games_played,"
    "wins_as_white,wins_as_black,losses_as_white,losses_as_black,"
    "draws_as_white,draws_as_black,"
    "winrate,wins,draws,losses,mean_game_len,terminations,"
    "elo,elo_lower,elo_upper,los,verdict\n"
)

GATE_OPENINGS_CSV_HEADER = (
    "iter,prev_iter,game_idx,a_is_white,opening_uci,result,termination,plies\n"
)


def _elo_ci_verdict(elo_lower: float, elo_upper: float) -> str:
    """Gate label from the Elo 95% CI (no SPRT H0/H1 band)."""
    if elo_upper < 0:
        return "FAIL"
    if elo_lower > 0:
        return "PASS"
    return "INCONCLUSIVE"

ANCHOR_CSV_HEADER = (
    "iter,anchor_iter,games,winrate,elo,elo_lower,elo_upper,los\n"
)


def _read_gpu_util_pct() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(out.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError, OSError):
        return float("nan")


def _buffer_iter_range(buffer: deque[Sample]) -> tuple[int, int]:
    if not buffer:
        return 0, 0
    iters = [s.source_iter for s in buffer]
    return min(iters), max(iters)


def _load_net_from_snapshot(snapshot_path: str, cfg: Config, device: str) -> ChessNet | None:
    state = torch.load(snapshot_path, map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        return None
    prev_encoding_version = int(state.get("encoding_version", 1))
    if prev_encoding_version != ENCODING_VERSION:
        return None
    prev_net_cfg = cfg.net
    if "net" in state:
        prev_net_cfg = NetConfig(**state["net"])
    prev_net = ChessNet(prev_net_cfg).to(device)
    _load_matching_state_dict(prev_net, state["model"], label="snapshot load", verbose=False)
    prev_net.eval()
    return prev_net


def _log_anchor_metrics(ckpt_dir: str, it: int, anchor_iter: int,
                        metrics: dict, games: int) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics_anchors.csv")
    new = not os.path.exists(path)
    wins = metrics["wins_as_white"] + metrics["wins_as_black"]
    losses = metrics["losses_as_white"] + metrics["losses_as_black"]
    draws = metrics["draws_as_white"] + metrics["draws_as_black"]
    elo, elo_lower, elo_upper, los = sprt_elo(wins, draws, losses)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(ANCHOR_CSV_HEADER)
        f.write(
            f"{it},{anchor_iter},{games},"
            f"{metrics['winrate']:.6f},"
            f"{elo:.2f},{elo_lower:.2f},{elo_upper:.2f},{los:.6f}\n"
        )


def _log_gate_metrics(ckpt_dir: str, it: int, prev_it: int, metrics: dict, games: int) -> None:
    os.makedirs(ckpt_dir or ".", exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics_gates.csv")
    new = not os.path.exists(path)

    wins = metrics["wins_as_white"] + metrics["wins_as_black"]
    losses = metrics["losses_as_white"] + metrics["losses_as_black"]
    draws = metrics["draws_as_white"] + metrics["draws_as_black"]
    games_played = int(metrics.get("games_played", wins + draws + losses))
    elo, elo_lower, elo_upper, los = sprt_elo(wins, draws, losses)
    verdict = _elo_ci_verdict(elo_lower, elo_upper)

    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(GATE_CSV_HEADER)
        f.write(
            f"{it},{prev_it},{games},{games_played},"
            f"{metrics['wins_as_white']},{metrics['wins_as_black']},"
            f"{metrics['losses_as_white']},{metrics['losses_as_black']},"
            f"{metrics['draws_as_white']},{metrics['draws_as_black']},"
            f"{metrics['winrate']:.6f},{wins},{draws},{losses},"
            f"{metrics['mean_game_len']:.2f},{metrics['terminations']},"
            f"{elo:.2f},{elo_lower:.2f},{elo_upper:.2f},{los:.6f},"
            f"{verdict}\n"
        )

    openings = list(metrics.get("openings") or [])
    openings.sort(key=lambda row: int(row.get("game_idx", 0)))
    if openings:
        openings_path = os.path.join(ckpt_dir, "metrics_gates_openings.csv")
        openings_new = not os.path.exists(openings_path)
        with open(openings_path, "a", encoding="utf-8") as f:
            if openings_new:
                f.write(GATE_OPENINGS_CSV_HEADER)
            for row in openings:
                f.write(
                    f"{it},{prev_it},{row['game_idx']},{row['a_is_white']},"
                    f"{row['opening_uci']},{row['result']},{row['termination']},"
                    f"{row['plies']}\n"
                )
    book_lines = metrics.get("book_lines")
    print(
        _opening_summary(
            openings,
            book_lines=int(book_lines) if book_lines is not None else None,
        ),
        flush=True,
    )


def _opening_summary(openings: list[dict], *, book_lines: int | None = None) -> str:
    if not openings:
        return "gate openings: none"
    prefixes = [str(o.get("opening_uci", "")) for o in openings]
    counts = Counter(prefixes)
    unique = len(counts)
    top_prefix, top_n = counts.most_common(1)[0]
    top_share = top_n / len(openings)
    first_moves = Counter(
        (p.split()[0] if p else "") for p in prefixes
    )
    first_hist = ",".join(f"{m}:{c}" for m, c in first_moves.most_common())
    book_note = f" book_lines={book_lines}" if book_lines is not None else ""
    return (
        f"gate openings: n={len(openings)} unique={unique} "
        f"top_share={top_share:.2f} top={top_prefix!r} "
        f"first=[{first_hist}] (plies={GATE_OPENING_PLIES}){book_note}"
    )


def _winner_of(game: GameResult) -> int:
    """Return +1 for white win, -1 for black win, 0 for non-decisive result."""
    if game.termination not in {"checkmate", "resign", "tablebase_win"}:
        return 0
    if game.winner is None:
        return 0
    return 1 if game.winner == chess.WHITE else -1


def _snapshot_at_iter(ckpt_dir: str, iteration: int) -> str | None:
    if iteration < 0:
        return None
    path = os.path.join(ckpt_dir, f"ckpt_iter_{iteration:04d}.pt")
    return path if os.path.exists(path) else None


def play_match(net_a: ChessNet, net_b: ChessNet, cfg: Config,
               n_games: int, sims: int, device: str,
               exploration_moves: int = 10,
               tablebase: chess.syzygy.Tablebase | None = None,
               sprt: bool = False,
               sprt_elo0: float = ELO0,
               sprt_elo1: float = ELO1,
               sprt_alpha: float = ALPHA,
               sprt_beta: float = BETA,
               workers: int = 1,
               on_progress: Callable[[int], None] | None = None,
               concurrency: int | None = None,
               openings: list[list[str]] | None = None,
               ) -> dict:
    if n_games <= 0:
        return {
            "winrate": float("nan"),
            "wins_as_white": 0,
            "wins_as_black": 0,
            "losses_as_white": 0,
            "losses_as_black": 0,
            "draws_as_white": 0,
            "draws_as_black": 0,
            "mean_game_len": 0.0,
            "terminations": "",
            "llr": float("nan"),
            "sprt_decision": "continue",
            "sprt_lower": float("nan"),
            "sprt_upper": float("nan"),
            "games_played": 0,
            "elo0": sprt_elo0,
            "elo1": sprt_elo1,
            "openings": [],
            "book_lines": len(openings) if openings else None,
        }

    match_cfg = deepcopy(cfg)
    match_cfg.beauty.enabled = False
    # Strength gates use normal chess: draws score 0.5, search treats draws as 0.
    match_cfg.mcts.draw_contempt = 0.0
    # Pin claim_draw so gates always match self-play search (Config default True).
    match_cfg.mcts.claim_draw = True
    # No artificial ply cap — Syzygy, 50-move, threefold, etc. end games naturally.
    match_cfg.train.max_game_moves = 10_000
    # Disable resignation during strength evaluation matches
    match_cfg.train.resign_threshold = -1.1
    match_cfg.train.resign_plies = 0
    if concurrency is not None:
        match_cfg.train.selfplay_concurrency = concurrency

    gate_bar = tqdm(total=n_games, desc=f"gate ({n_games} games)", unit="game", leave=False)

    if workers > 1:
        ckpt_dir = match_cfg.train.checkpoint_dir or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        weights_path_a = os.path.join(ckpt_dir, "_gate_net_a.pt")
        weights_path_b = os.path.join(ckpt_dir, "_gate_net_b.pt")
        net_a_module = getattr(net_a, "_orig_mod", net_a)
        net_b_module = getattr(net_b, "_orig_mod", net_b)
        torch.save({"model": net_a_module.state_dict()}, weights_path_a)
        torch.save({"model": net_b_module.state_dict()}, weights_path_b)

        def _gate_progress(completed: int) -> None:
            _tqdm_sync_progress(gate_bar, completed)
            if on_progress is not None:
                on_progress(completed)

        merged = play_match_parallel(
            match_cfg,
            _net_cfg_from_module(net_a),
            _net_cfg_from_module(net_b),
            weights_path_a,
            weights_path_b,
            n_games,
            sims,
            workers,
            device,
            exploration_moves,
            syzygy_path=match_cfg.train.syzygy_path,
            on_progress=_gate_progress,
            openings=openings,
        )
        _tqdm_sync_progress(gate_bar, n_games)
        gate_bar.close()

        wins_w = merged.wins_w
        wins_b = merged.wins_b
        losses_w = merged.losses_w
        losses_b = merged.losses_b
        draws_w = merged.draws_w
        draws_b = merged.draws_b
        game_lengths = merged.game_lengths
        termination_counts = merged.termination_counts
        score = merged.score
        games_played = len(game_lengths)
        winrate = score / games_played if games_played > 0 else float("nan")
        total_wins = wins_w + wins_b
        total_losses = losses_w + losses_b
        total_draws = draws_w + draws_b
        sprt_lower, sprt_upper = sprt_bounds(sprt_alpha, sprt_beta) if sprt else (float("nan"), float("nan"))
        current_llr = (
            sprt_llr(total_wins, total_draws, total_losses, elo0=sprt_elo0, elo1=sprt_elo1)
            if sprt else float("nan")
        )
        current_sprt_decision = (
            sprt_decision(current_llr, sprt_lower, sprt_upper) if sprt else "continue"
        )
        terminations_str = ";".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
        elo, elo_lower, elo_upper, los = sprt_elo(total_wins, total_draws, total_losses)
        return {
            "winrate": winrate,
            "wins_as_white": wins_w,
            "wins_as_black": wins_b,
            "losses_as_white": losses_w,
            "losses_as_black": losses_b,
            "draws_as_white": draws_w,
            "draws_as_black": draws_b,
            "mean_game_len": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "terminations": terminations_str,
            "llr": current_llr,
            "sprt_decision": current_sprt_decision,
            "sprt_lower": sprt_lower,
            "sprt_upper": sprt_upper,
            "elo": elo,
            "elo_lower": elo_lower,
            "elo_upper": elo_upper,
            "los": los,
            "games_played": games_played,
            "elo0": sprt_elo0 if sprt else float("nan"),
            "elo1": sprt_elo1 if sprt else float("nan"),
            "openings": list(merged.openings or []),
            "book_lines": len(openings) if openings else None,
        }

    eval_a = NetEvaluator(net_a, device=device)
    eval_b = NetEvaluator(net_b, device=device)
    score = 0.0
    wins_w = 0
    wins_b = 0
    losses_w = 0
    losses_b = 0
    draws_w = 0
    draws_b = 0
    game_lengths = []
    termination_counts: Counter[str] = Counter()
    opening_rows: list[dict] = []

    def _record_game_result(game: GameResult, a_is_white: bool, game_idx: int) -> None:
        nonlocal score, wins_w, wins_b, losses_w, losses_b, draws_w, draws_b
        winner = _winner_of(game)
        game_lengths.append(len(game.samples))
        termination_counts[game.termination] += 1
        opening_rows.append(_opening_row(game_idx, a_is_white, game, winner))

        if winner == 0:
            score += 0.5
            if a_is_white:
                draws_w += 1
            else:
                draws_b += 1
        elif (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
            score += 1.0
            if a_is_white:
                wins_w += 1
            else:
                wins_b += 1
        else:
            if a_is_white:
                losses_w += 1
            else:
                losses_b += 1

    from engine.openings import opening_for_game

    concurrency = max(1, min(n_games, cfg.train.selfplay_concurrency))
    active: list[tuple[object, EvalRequest, bool, int]] = []
    launched = 0
    completed = 0
    sprt_lower, sprt_upper = sprt_bounds(sprt_alpha, sprt_beta) if sprt else (float("nan"), float("nan"))
    sprt_decided = False
    current_llr = 0.0
    current_sprt_decision = "continue"

    while active or (launched < n_games and not sprt_decided):
        while launched < n_games and len(active) < concurrency and not sprt_decided:
            a_is_white = (launched % 2 == 0)
            gen = play_game_gen(
                match_cfg,
                sims,
                add_noise=False,
                exploration_moves=exploration_moves,
                tablebase=tablebase,
                start_moves=opening_for_game(openings, launched),
            )
            active.append((gen, next(gen), a_is_white, launched))
            launched += 1

        if not active:
            break

        a_indices: list[int] = []
        a_boards: list[chess.Board] = []
        b_indices: list[int] = []
        b_boards: list[chess.Board] = []
        for idx, (_, pending, a_is_white, _) in enumerate(active):
            search_uses_a = (pending.search_turn == a_is_white)
            if search_uses_a:
                a_indices.append(idx)
                a_boards.append(pending.board)
            else:
                b_indices.append(idx)
                b_boards.append(pending.board)

        pending_eval: dict[int, tuple[np.ndarray, float]] = {}
        if a_boards:
            a_logits_batch, a_values_batch = eval_a.evaluate_batch(a_boards)
            for idx, logits, value in zip(a_indices, a_logits_batch, a_values_batch):
                pending_eval[idx] = (logits, float(value))
        if b_boards:
            b_logits_batch, b_values_batch = eval_b.evaluate_batch(b_boards)
            for idx, logits, value in zip(b_indices, b_logits_batch, b_values_batch):
                pending_eval[idx] = (logits, float(value))

        next_active: list[tuple[object, EvalRequest, bool, int]] = []
        for idx, (gen, _, a_is_white, game_idx) in enumerate(active):
            logits, value = pending_eval[idx]
            try:
                pending = gen.send((logits, value))
                next_active.append((gen, pending, a_is_white, game_idx))
            except StopIteration as stop:
                _record_game_result(stop.value, a_is_white, game_idx)
                completed += 1
                gate_bar.update(1)
                gate_bar.set_postfix(score=f"{score / completed:.3f}")
                if sprt:
                    total_wins = wins_w + wins_b
                    total_losses = losses_w + losses_b
                    total_draws = draws_w + draws_b
                    current_llr = sprt_llr(
                        total_wins, total_draws, total_losses,
                        elo0=sprt_elo0, elo1=sprt_elo1,
                    )
                    current_sprt_decision = sprt_decision(current_llr, sprt_lower, sprt_upper)
                    if current_sprt_decision != "continue":
                        sprt_decided = True
        active = next_active
    gate_bar.close()

    total_wins = wins_w + wins_b
    total_losses = losses_w + losses_b
    total_draws = draws_w + draws_b
    terminations_str = ";".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
    games_played = completed if completed > 0 else 0
    winrate = score / games_played if games_played > 0 else float("nan")
    elo, elo_lower, elo_upper, los = sprt_elo(total_wins, total_draws, total_losses)

    return {
        "winrate": winrate,
        "wins_as_white": wins_w,
        "wins_as_black": wins_b,
        "losses_as_white": losses_w,
        "losses_as_black": losses_b,
        "draws_as_white": draws_w,
        "draws_as_black": draws_b,
        "mean_game_len": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "terminations": terminations_str,
        "llr": current_llr if sprt else float("nan"),
        "sprt_decision": current_sprt_decision if sprt else "continue",
        "sprt_lower": sprt_lower,
        "sprt_upper": sprt_upper,
        "elo": elo,
        "elo_lower": elo_lower,
        "elo_upper": elo_upper,
        "los": los,
        "games_played": games_played,
        "elo0": sprt_elo0 if sprt else float("nan"),
        "elo1": sprt_elo1 if sprt else float("nan"),
        "openings": opening_rows,
        "book_lines": len(openings) if openings else None,
    }


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
    source_iters = np.array([s.source_iter for s in samples], dtype=np.int32)
    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            planes=planes,
            policies=policies,
            players=players,
            values=values,
            source_iters=source_iters,
            encoding_version=np.array([ENCODING_VERSION], dtype=np.int16),
        )
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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
            source_iters = data["source_iters"] if "source_iters" in data else None
            samples: list[Sample] = []
            for i in range(values.shape[0]):
                source_iter = int(source_iters[i]) if source_iters is not None else 0
                samples.append(
                    Sample(
                        planes=planes[i],
                        policy=policies[i],
                        player=bool(players[i]),
                        value=float(values[i]),
                        source_iter=source_iter,
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
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="do not restore Adam state from checkpoint (fresh optimizer on resume)")
    parser.add_argument("--checkpoint-dir", default=None)
    # Per-iteration workload overrides (lower these for faster CPU iterations).
    parser.add_argument("--games", type=int, default=None, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=None, help="MCTS sims/move")
    parser.add_argument("--train-steps", type=int, default=None, help="optimizer steps per iteration")
    parser.add_argument("--max-game-moves", type=int, default=None,
                        help="self-play truncation cap (C2 shaping)")
    parser.add_argument("--draw-penalty", type=float, default=None,
                        help="target for draw outcomes (C2 shaping)")
    parser.add_argument("--value-target", choices=("outcome", "root_q"), default=None,
                        help="self-play value labels: terminal outcome or per-ply MCTS root Q")
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
    parser.add_argument("--selfplay-workers", type=int, default=1,
                        help="parallel self-play worker processes (spawn); 1 = single process")
    parser.add_argument("--replay-window", type=int, default=None,
                        help="max persisted replay samples kept on disk")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override training minibatch size")
    parser.add_argument("--replay-buffer", type=int, default=None,
                        help="override in-memory replay buffer size")
    parser.add_argument("--gate-every", type=int, default=0,
                        help="run strength gate every N iterations (0 = off)")
    parser.add_argument("--gate-games", type=int, default=30,
                        help="games per strength gate (current net vs checkpoint N iters ago)")
    parser.add_argument("--gate-sims", type=int, default=None,
                        help="override sims/move for gate matches (defaults to self-play sims)")
    parser.add_argument("--gate-exploration-moves", type=int, default=10,
                        help="sample moves for first N plies in gate games "
                             "(counts from start, including book plies; use 0 with masters book)")
    parser.add_argument(
        "--gate-openings",
        type=str,
        default="masters",
        help="gate opening book: 'masters' (default prefix-free top-64), 'none', or TSV path",
    )
    parser.add_argument("--gate-anchor-iter", type=int, default=60,
                        help="also gate vs this frozen checkpoint iter (0 = off)")
    parser.add_argument("--gate-anchor-games", type=int, default=30,
                        help="games for anchor checkpoint gate")
    parser.add_argument("--quick-eval-games", type=int, default=0,
                        help="lightweight eval games each iter vs iter-lag snapshot (0 = off)")
    parser.add_argument("--quick-eval-lag", type=int, default=10,
                        help="compare current net to checkpoint at iter minus this lag")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning rate")
    parser.add_argument("--lr-min", type=float, default=None,
                        help="minimum learning rate for cosine schedule")
    parser.add_argument("--lr-total-iters", type=int, default=None,
                        help="iterations spanned by cosine decay")
    parser.add_argument("--lr-warmup-iters", type=int, default=None,
                        help="iterations before cosine decay begins (also short warmup ramp if small)")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="gradient clip norm (default: cfg.train.grad_clip_norm)")
    parser.add_argument("--syzygy-path", type=str, default=None,
                        help="path to Syzygy WDL tablebase directory for self-play adjudication")
    parser.add_argument("--move-temperature", type=float, default=1.0,
                        help="early-ply move sampling temperature (1.0 = off)")
    parser.add_argument("--move-temperature-plies", type=int, default=0,
                        help="apply move temperature for first N plies (0 = disabled)")
    args = parser.parse_args()

    # Fall back to CPU if CUDA was requested but isn't available in this runtime.
    # This also lets torch.load map CUDA-saved checkpoints onto the CPU.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to --device cpu")
        args.device = "cpu"
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if args.gate_sims is not None and args.gate_sims <= 0:
        raise ValueError("--gate-sims must be >= 1")
    if args.selfplay_workers < 1:
        raise ValueError("--selfplay-workers must be >= 1")

    cfg = Config()
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.light:
        cfg.net.blocks, cfg.net.filters = 4, 48
        cfg.train.games_per_iteration = 6
        cfg.train.selfplay_concurrency = 4
        cfg.train.train_steps_per_iteration = 80
        cfg.train.sims_per_move = 48
    if args.gpu:
        cfg.net.blocks, cfg.net.filters = 8, 96
        cfg.train.games_per_iteration = 64
        cfg.train.selfplay_concurrency = 64
        cfg.train.train_steps_per_iteration = 400
        cfg.train.sims_per_move = 100
    if args.games is not None:
        cfg.train.games_per_iteration = args.games
    if args.train_steps is not None:
        cfg.train.train_steps_per_iteration = args.train_steps
    if args.sims is not None:
        cfg.train.sims_per_move = args.sims
    if args.concurrency is not None:
        cfg.train.selfplay_concurrency = args.concurrency
    if args.replay_window is not None:
        cfg.train.replay_window = args.replay_window
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.replay_buffer is not None:
        cfg.train.replay_buffer_size = args.replay_buffer
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.lr_min is not None:
        cfg.train.lr_min = args.lr_min
    if args.lr_total_iters is not None:
        cfg.train.lr_total_iters = args.lr_total_iters
    if args.lr_warmup_iters is not None:
        cfg.train.lr_warmup_iters = args.lr_warmup_iters
    if args.grad_clip is not None:
        cfg.train.grad_clip_norm = args.grad_clip
    if args.syzygy_path:
        cfg.train.syzygy_path = args.syzygy_path
    if args.max_game_moves is not None:
        cfg.train.max_game_moves = args.max_game_moves
    if args.draw_penalty is not None:
        cfg.train.draw_penalty = args.draw_penalty
    if args.value_target is not None:
        cfg.train.value_target = args.value_target
    if args.resign_threshold is not None:
        cfg.train.resign_threshold = args.resign_threshold
    if args.resign_plies is not None:
        cfg.train.resign_plies = args.resign_plies
    if args.resign_min_moves is not None:
        cfg.train.resign_min_moves = args.resign_min_moves
    cfg.train.move_temperature = float(args.move_temperature)
    cfg.train.move_temperature_plies = max(0, int(args.move_temperature_plies))
    args.gate_exploration_moves = max(0, int(args.gate_exploration_moves))
    from engine.openings import load_default_gate_openings, load_opening_book

    gate_openings_spec = str(args.gate_openings or "none").strip()
    if gate_openings_spec.lower() in {"", "none", "off"}:
        gate_openings: list[list[str]] | None = None
    elif gate_openings_spec.lower() == "masters":
        gate_openings = load_default_gate_openings()
    else:
        gate_openings = load_opening_book(gate_openings_spec)
    # Keep self-play search contempt aligned with the draw target shaping.
    cfg.mcts.draw_contempt = cfg.train.draw_penalty
    print(
        "config: "
        f"games={cfg.train.games_per_iteration} "
        f"steps={cfg.train.train_steps_per_iteration} "
        f"concurrency={cfg.train.selfplay_concurrency} "
        f"selfplay_workers={args.selfplay_workers} "
        f"max_moves={cfg.train.max_game_moves} "
        f"lr={cfg.train.learning_rate:.6f} "
        f"lr_min={cfg.train.lr_min:.6f} "
        f"lr_warmup_iters={cfg.train.lr_warmup_iters} "
        f"lr_total_iters={cfg.train.lr_total_iters} "
        f"tb_max_pieces={cfg.train.tb_max_pieces} "
        f"syzygy_path={cfg.train.syzygy_path or 'off'} "
        f"draw_penalty={cfg.train.draw_penalty:.3f} "
        f"value_target={cfg.train.value_target} "
        f"resign_threshold={cfg.train.resign_threshold:.3f} "
        f"resign_plies={cfg.train.resign_plies} "
        f"resign_min_moves={cfg.train.resign_min_moves} "
        f"move_temperature={cfg.train.move_temperature:.3f} "
        f"move_temperature_plies={cfg.train.move_temperature_plies} "
        f"gate_games={args.gate_games} "
        f"gate_sims={args.gate_sims if args.gate_sims is not None else 'match-selfplay'} "
        f"gate_exploration_moves={args.gate_exploration_moves} "
        f"gate_openings={gate_openings_spec}"
        f"{f'({len(gate_openings)} lines)' if gate_openings else ''} "
        f"gate_anchor_iter={args.gate_anchor_iter} "
        f"quick_eval_games={args.quick_eval_games} "
        f"quick_eval_lag={args.quick_eval_lag}"
    )
    step_metrics_path = os.path.join(cfg.train.checkpoint_dir, "metrics_steps.csv")
    print(f"step metrics: {step_metrics_path}")

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
    if args.device.startswith("cuda"):
        if hasattr(torch, "compile"):
            net = torch.compile(net, dynamic=True)
        else:
            print("warning: torch.compile is unavailable in this runtime; continuing without compile")

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    if (state is not None and isinstance(state, dict) and "optimizer" in state
            and not args.reset_optimizer):
        optimizer.load_state_dict(state["optimizer"])
        print("resumed optimizer state")
    elif args.reset_optimizer:
        print("reset optimizer state (fresh Adam)")
    scaler: torch.cuda.amp.GradScaler | None = None
    if args.device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    buffer: deque[Sample] = deque(maxlen=cfg.train.replay_buffer_size)
    loaded = _warm_replay_buffer(buffer, cfg.train.checkpoint_dir, cfg.train.replay_window)
    if loaded:
        print(f"warmed replay buffer with {loaded} samples from shard files")

    tablebase = None
    if cfg.train.syzygy_path:
        if not os.path.isdir(cfg.train.syzygy_path):
            raise ValueError(f"Syzygy path does not exist or is not a directory: {cfg.train.syzygy_path}")
        tablebase = chess.syzygy.open_tablebase(cfg.train.syzygy_path)
        print(f"syzygy: enabled ({cfg.train.syzygy_path})")

    if args.selfplay_workers == 1:
        stale_worker_weights = os.path.join(cfg.train.checkpoint_dir, "_worker_net.pt")
        if os.path.isfile(stale_worker_weights):
            os.remove(stale_worker_weights)

    # Hoist: NetEvaluator holds a live reference to net (no weight snapshot), so
    # in-place updates from training are always visible; construct once to avoid
    # pinned-memory re-allocation on every iteration.
    evaluator = NetEvaluator(net, device=args.device)

    try:
        for local_it in range(args.iterations):
            it = start_iter + local_it
            current_lr = _lr_for_iteration(cfg, it)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            sims = cfg.train.sims_per_move

            t0 = time.time()
            new_samples = 0
            iteration_samples: list[Sample] = []
            termination_counts: Counter[str] = Counter()
            game_lengths: list[int] = []
            game_outcomes: list[int] = []
            first_moves: list[str] = []
            n_games = cfg.train.games_per_iteration
            game_bar = tqdm(total=n_games, desc=f"iter {it} self-play", unit="game", leave=False)

            def _tag_source_iter(samples: list[Sample]) -> None:
                for s in samples:
                    s.source_iter = it

            def _on_game(game: GameResult) -> None:
                nonlocal new_samples
                _tag_source_iter(game.samples)
                buffer.extend(game.samples)
                iteration_samples.extend(game.samples)
                new_samples += len(game.samples)
                termination_counts[game.termination] += 1
                game_lengths.append(len(game.samples))
                game_outcomes.append(_winner_of(game))
                if game.moves:
                    first_moves.append(game.moves[0])
                game_bar.set_postfix(moves=len(game.samples), buffer=len(buffer))
                game_bar.update(1)

            if args.selfplay_workers > 1:
                worker_weights_path = os.path.join(cfg.train.checkpoint_dir, "_worker_net.pt")
                os.makedirs(cfg.train.checkpoint_dir or ".", exist_ok=True)
                model_module = getattr(net, "_orig_mod", net)
                torch.save({"model": model_module.state_dict()}, worker_weights_path)

                def _parallel_selfplay_progress(completed: int) -> None:
                    _tqdm_sync_progress(game_bar, completed)

                parallel_samples, parallel_terms, parallel_lengths, parallel_outcomes = (
                    play_games_parallel(
                        cfg,
                        cfg.net,
                        worker_weights_path,
                        simulations=sims,
                        num_games=n_games,
                        workers=args.selfplay_workers,
                        device=args.device,
                        syzygy_path=cfg.train.syzygy_path,
                        on_progress=_parallel_selfplay_progress,
                    )
                )
                buffer.extend(parallel_samples)
                iteration_samples = parallel_samples
                new_samples = len(parallel_samples)
                _tag_source_iter(parallel_samples)
                termination_counts = Counter(parallel_terms)
                game_lengths = parallel_lengths
                game_outcomes = parallel_outcomes
                _tqdm_sync_progress(game_bar, n_games)
            else:
                play_games_batched(
                    evaluator,
                    cfg,
                    simulations=sims,
                    num_games=n_games,
                    concurrency=cfg.train.selfplay_concurrency,
                    on_game_finished=_on_game,
                    tablebase=tablebase,
                )
            game_bar.close()
            selfplay_dt = time.time() - t0
            gpu_util_pct = _read_gpu_util_pct() if args.device.startswith("cuda") else float("nan")
            buffer_min_iter, buffer_max_iter = _buffer_iter_range(buffer)

            _save_sample_shard(cfg.train.checkpoint_dir, it, iteration_samples)
            if args.selfplay_workers > 1:
                shard_path = _sample_shard_path(cfg.train.checkpoint_dir, it)
                fm_stats = (
                    summarize_first_moves_from_shard(shard_path)
                    if os.path.isfile(shard_path)
                    else summarize_first_moves([])
                )
            else:
                fm_stats = summarize_first_moves(first_moves)
            _log_first_move_metrics(cfg.train.checkpoint_dir, it, fm_stats)

            net.train()
            step_metrics: dict[str, list[float]] = {}
            step_rows: list[tuple[int, dict[str, float]]] = []
            t_train = time.time()
            if len(buffer) >= cfg.train.batch_size:
                # Snapshot buffer once; random.sample on a deque is O(N) per call.
                sample_pool = list(buffer)
                train_bar = tqdm(range(cfg.train.train_steps_per_iteration),
                                 desc=f"iter {it} train", unit="step", leave=False)
                for step in train_bar:
                    batch = random.sample(sample_pool, cfg.train.batch_size)
                    step_result = train_step(net, optimizer, batch, args.device, scaler=scaler,
                                             grad_clip=cfg.train.grad_clip_norm)
                    for name, value in step_result.items():
                        step_metrics.setdefault(name, []).append(value)
                    step_rows.append((int(step), step_result))
                    train_bar.set_postfix(
                        p=f"{step_result['policy_loss']:.3f}",
                        v=f"{step_result['value_loss']:.3f}",
                        g=f"{step_result['grad_norm']:.2f}",
                    )
            train_dt = time.time() - t_train
            _flush_step_metrics(cfg.train.checkpoint_dir, it, step_rows)
            net.eval()

            save_checkpoint(net, cfg, os.path.join(cfg.train.checkpoint_dir, "latest.pt"), it,
                            optimizer=optimizer)
            if args.save_every and it % args.save_every == 0:
                snap = os.path.join(cfg.train.checkpoint_dir, f"ckpt_iter_{it:04d}.pt")
                save_checkpoint(net, cfg, snap, it, optimizer=optimizer)

            def _mean_metric(name: str) -> float:
                values = step_metrics.get(name)
                return float(np.mean(values)) if values else float("nan")

            dt = time.time() - t0
            pl = _mean_metric("policy_loss")
            vl = _mean_metric("value_loss")
            policy_entropy = _mean_metric("policy_entropy")
            value_sign_acc = _mean_metric("value_sign_acc")
            policy_top1_agree = _mean_metric("policy_top1_agree")
            grad_norm = _finite_median(step_metrics.get("grad_norm", []))

            total_games = len(game_lengths)
            white_wins = sum(1 for o in game_outcomes if o == 1)
            black_wins = sum(1 for o in game_outcomes if o == -1)
            draws_or_other = total_games - white_wins - black_wins
            mean_game_len = float(np.mean(game_lengths)) if game_lengths else float("nan")
            decisive_games = (
                termination_counts.get("checkmate", 0)
                + termination_counts.get("resign", 0)
                + termination_counts.get("tablebase_win", 0)
            )
            decisive_rate = decisive_games / total_games if total_games else float("nan")
            white_win_rate = white_wins / total_games if total_games else float("nan")
            draw_rate = draws_or_other / total_games if total_games else float("nan")
            max_moves_trunc_rate = (
                termination_counts.get("max_moves", 0) / total_games if total_games else float("nan")
            )
            value_targets = np.array([s.value for s in iteration_samples], dtype=np.float32)
            value_mean = float(value_targets.mean()) if value_targets.size else float("nan")
            value_std = float(value_targets.std()) if value_targets.size else float("nan")

            winrate_quick = float("nan")
            if args.quick_eval_games > 0 and it > args.quick_eval_lag:
                quick_it = it - args.quick_eval_lag
                quick_snapshot = _snapshot_at_iter(cfg.train.checkpoint_dir, quick_it)
                if quick_snapshot is not None:
                    try:
                        quick_net = _load_net_from_snapshot(quick_snapshot, cfg, args.device)
                        if quick_net is not None:
                            quick_sims = args.gate_sims if args.gate_sims is not None else sims
                            quick_metrics = play_match(
                                net,
                                quick_net,
                                cfg,
                                n_games=args.quick_eval_games,
                                sims=quick_sims,
                                device=args.device,
                                exploration_moves=args.gate_exploration_moves,
                                tablebase=tablebase,
                                sprt=False,
                                openings=gate_openings,
                            )
                            winrate_quick = float(quick_metrics["winrate"])
                    except Exception as exc:
                        print(f"quick eval iter {it}: failed ({exc})", flush=True)

            term_summary = ", ".join(f"{k}:{v}" for k, v in sorted(termination_counts.items()))
            print(f"iter {it:3d} | sims {sims:3d} | games {cfg.train.games_per_iteration} "
                  f"| samples {new_samples:4d} | buffer {len(buffer):6d} "
                  f"| policy_loss {pl:.3f} | value_loss {vl:.3f} "
                  f"| ent {policy_entropy:.3f} | sign_acc {value_sign_acc:.3f} "
                  f"| lr {current_lr:.3e} "
                  f"| decisive {decisive_rate:.3f} "
                  f"| ends {term_summary} "
                  f"| selfplay {selfplay_dt:.1f}s train {train_dt:.1f}s"
                  f"| {dt:.1f}s", flush=True)

            # Log training metrics before the gate so OOM during gating still
            # preserves this iteration's row (winrate_vs_prev patched after gate).
            _log_metrics(
                cfg.train.checkpoint_dir,
                it,
                sims,
                new_samples,
                dt,
                selfplay_seconds=selfplay_dt,
                train_seconds=train_dt,
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
                winrate_vs_prev=float("nan"),
                learning_rate=current_lr,
                games=cfg.train.games_per_iteration,
                train_steps=cfg.train.train_steps_per_iteration,
                batch_size=cfg.train.batch_size,
                buffer_size=len(buffer),
                termination_counts=dict(termination_counts),
                winrate_quick=winrate_quick,
                gpu_util_pct=gpu_util_pct,
                buffer_min_iter=buffer_min_iter,
                buffer_max_iter=buffer_max_iter,
            )

            if args.gate_every > 0 and it > 0 and it % args.gate_every == 0:
                gate_sims = args.gate_sims if args.gate_sims is not None else sims
                try:
                    prev_it = it - args.gate_every
                    previous_snapshot = _snapshot_at_iter(cfg.train.checkpoint_dir, prev_it)
                    if previous_snapshot is None:
                        print(f"gate iter {it}: skipped (no snapshot at iter {prev_it})")
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
                                gate_metrics = play_match(
                                    net,
                                    prev_net,
                                    cfg,
                                    n_games=args.gate_games,
                                    sims=gate_sims,
                                    device=args.device,
                                    exploration_moves=args.gate_exploration_moves,
                                    tablebase=tablebase,
                                    sprt=True,
                                    sprt_elo0=ELO0,
                                    sprt_elo1=ELO1,
                                    sprt_alpha=ALPHA,
                                    sprt_beta=BETA,
                                    openings=gate_openings,
                                )
                                winrate_vs_prev = float(gate_metrics["winrate"])
                                total_wins = gate_metrics["wins_as_white"] + gate_metrics["wins_as_black"]
                                total_losses = gate_metrics["losses_as_white"] + gate_metrics["losses_as_black"]
                                total_draws = gate_metrics["draws_as_white"] + gate_metrics["draws_as_black"]
                                games_played = int(gate_metrics["games_played"])
                                ci_verdict = _elo_ci_verdict(
                                    float(gate_metrics["elo_lower"]),
                                    float(gate_metrics["elo_upper"]),
                                )
                                print(
                                    f"gate iter {it}: current vs {os.path.basename(previous_snapshot)} "
                                    f"-> {gate_metrics['winrate']:.3f} (Wins: {total_wins}, Losses: {total_losses}, "
                                    f"Draws: {total_draws}, games: {games_played}) "
                                    f"{ci_verdict} "
                                    f"Elo {gate_metrics['elo']:+.1f} "
                                    f"[95% {gate_metrics['elo_lower']:+.1f}, {gate_metrics['elo_upper']:+.1f}] "
                                    f"LOS {gate_metrics['los'] * 100:.1f}%"
                                )
                                _log_gate_metrics(
                                    cfg.train.checkpoint_dir,
                                    it,
                                    prev_it,
                                    gate_metrics,
                                    args.gate_games,
                                )
                                _update_metrics_winrate_vs_prev(
                                    cfg.train.checkpoint_dir, it, winrate_vs_prev,
                                )
                                if args.gate_anchor_iter > 0:
                                    anchor_snapshot = _snapshot_at_iter(
                                        cfg.train.checkpoint_dir, args.gate_anchor_iter,
                                    )
                                    if anchor_snapshot is None:
                                        print(
                                            f"anchor gate iter {it}: skipped "
                                            f"(no snapshot at iter {args.gate_anchor_iter})"
                                        )
                                    else:
                                        try:
                                            anchor_net = _load_net_from_snapshot(
                                                anchor_snapshot, cfg, args.device,
                                            )
                                            if anchor_net is None:
                                                print(
                                                    f"anchor gate iter {it}: skipped "
                                                    f"(invalid snapshot {anchor_snapshot})"
                                                )
                                            else:
                                                anchor_metrics = play_match(
                                                    net,
                                                    anchor_net,
                                                    cfg,
                                                    n_games=args.gate_anchor_games,
                                                    sims=gate_sims,
                                                    device=args.device,
                                                    exploration_moves=args.gate_exploration_moves,
                                                    openings=gate_openings,
                                                    tablebase=tablebase,
                                                    sprt=False,
                                                )
                                                _log_anchor_metrics(
                                                    cfg.train.checkpoint_dir,
                                                    it,
                                                    args.gate_anchor_iter,
                                                    anchor_metrics,
                                                    args.gate_anchor_games,
                                                )
                                                print(
                                                    f"anchor gate iter {it}: vs iter "
                                                    f"{args.gate_anchor_iter} -> "
                                                    f"{anchor_metrics['winrate']:.3f} "
                                                    f"Elo {anchor_metrics['elo']:+.1f}"
                                                )
                                        except Exception as exc:
                                            print(
                                                f"anchor gate iter {it}: failed ({exc}); continuing",
                                                flush=True,
                                            )
                        else:
                            print(f"gate iter {it}: skipped (invalid snapshot {previous_snapshot})")
                except Exception as exc:
                    print(f"gate iter {it}: failed ({exc}); continuing", flush=True)
    finally:
        if tablebase is not None:
            tablebase.close()


if __name__ == "__main__":
    main()
