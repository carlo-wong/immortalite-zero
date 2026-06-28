"""Self-play game generation for training."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Generator

import chess
import chess.syzygy
import numpy as np
import torch

from .config import BeautyConfig, Config, MCTSConfig, NetConfig, TrainConfig
from .encoding import POLICY_SIZE, board_to_planes
from .mcts import MCTS
from .network import ChessNet, NetEvaluator

_EXPLORATION_MOVES = 20  # sample from the policy for this many plies, then argmax


@dataclass
class Sample:
    planes: np.ndarray      # (20, 8, 8)
    policy: np.ndarray      # (POLICY_SIZE,)
    player: chess.Color
    value: float = 0.0      # filled in once the game finishes


@dataclass
class GameResult:
    samples: list[Sample]
    termination: str


_DRAW_TERMINATION_NAMES = {
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.THREEFOLD_REPETITION: "threefold_repetition",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
    chess.Termination.FIFTY_MOVES: "fifty_moves",
    chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
}
_DRAW_TERMINATION_SET = set(_DRAW_TERMINATION_NAMES.values())


@dataclass
class _ActiveGame:
    gen: Generator[chess.Board, tuple[np.ndarray, float], GameResult]
    pending_board: chess.Board


def _tablebase_adjudication(board: chess.Board, tablebase: Any | None, max_pieces: int
                            ) -> tuple[str | None, chess.Color | None]:
    if tablebase is None or chess.popcount(board.occupied) > max_pieces:
        return None, None
    try:
        wdl = int(tablebase.probe_wdl(board))
    except (KeyError, chess.syzygy.MissingTableError):
        return None, None
    if wdl == 2:
        return "tablebase_win", board.turn
    if wdl == -2:
        return "tablebase_win", not board.turn
    if wdl in {-1, 0, 1}:
        return "tablebase_draw", None
    return None, None


def play_game_gen(cfg: Config, simulations: int, *, add_noise: bool = True,
                  exploration_moves: int = _EXPLORATION_MOVES,
                  tablebase: Any | None = None,
                  ) -> Generator[chess.Board, tuple[np.ndarray, float], GameResult]:
    board = chess.Board()
    mcts = MCTS(None, cfg.mcts)
    samples: list[Sample] = []
    move_count = 0
    no_legal_moves = False
    resigned_winner: chess.Color | None = None
    tablebase_winner: chess.Color | None = None
    tablebase_draw = False
    low_value_streak = 0
    last_root_value = 0.0

    while not board.is_game_over(claim_draw=True) and move_count < cfg.train.max_game_moves:
        tb_termination, tb_winner = _tablebase_adjudication(board, tablebase, cfg.train.tb_max_pieces)
        if tb_termination == "tablebase_win":
            tablebase_winner = tb_winner
            break
        if tb_termination == "tablebase_draw":
            tablebase_draw = True
            break

        search = mcts.search_gen(board, simulations=simulations, add_noise=add_noise)
        req = next(search)
        while True:
            logits, value = yield req
            try:
                req = search.send((logits, value))
            except StopIteration as stop:
                result = stop.value
                break
        if not result.moves:
            no_legal_moves = True
            break

        improved = result.improved_policy()
        last_root_value = float(result.root_value)
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, p in zip(result.indices, improved):
            policy[idx] = p
        samples.append(Sample(board_to_planes(board).astype(np.float16), policy.astype(np.float16), board.turn))

        resign_enabled = cfg.train.resign_plies > 0 and cfg.train.resign_threshold >= -1.0
        if resign_enabled and move_count >= cfg.train.resign_min_moves:
            if result.root_value <= cfg.train.resign_threshold:
                low_value_streak += 1
            else:
                low_value_streak = 0
            if low_value_streak >= cfg.train.resign_plies:
                # Side to move resigns; opponent is the winner.
                resigned_winner = not board.turn
                break
        else:
            low_value_streak = 0

        if move_count < exploration_moves:
            choice = np.random.choice(len(result.moves), p=improved / improved.sum())
            move = result.moves[choice]
        else:
            move = result.best_move()

        board.push(move)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    resigned = resigned_winner is not None
    tablebase_win = tablebase_winner is not None
    hit_max_moves = move_count >= cfg.train.max_game_moves and outcome is None and not resigned
    termination = _termination_reason(outcome, hit_max_moves=hit_max_moves,
                                      no_legal_moves=no_legal_moves, resigned=resigned,
                                      tablebase_win=tablebase_win, tablebase_draw=tablebase_draw)
    winner_override = tablebase_winner if tablebase_winner is not None else resigned_winner
    _assign_values(
        samples,
        outcome,
        termination,
        cfg,
        move_count,
        winner_override=winner_override,
        truncation_bootstrap=last_root_value,
    )
    return GameResult(samples=samples, termination=termination)


def play_game(evaluator: NetEvaluator, cfg: Config, simulations: int,
              tablebase: Any | None = None) -> GameResult:
    gen = play_game_gen(cfg, simulations, tablebase=tablebase)
    req = next(gen)
    while True:
        logits, value = evaluator.evaluate(req)
        try:
            req = gen.send((logits, value))
        except StopIteration as stop:
            return stop.value


def play_games_batched(evaluator: NetEvaluator, cfg: Config, simulations: int,
                       num_games: int, concurrency: int,
                       on_game_finished: Callable[[GameResult], None] | None = None,
                       on_step: Callable[[int], None] | None = None,
                       tablebase: Any | None = None,
                       ) -> list[GameResult]:
    if num_games <= 0:
        return []
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")

    active: list[_ActiveGame] = []
    launched = 0
    completed: list[GameResult] = []

    while len(completed) < num_games:
        while launched < num_games and len(active) < concurrency:
            gen = play_game_gen(cfg, simulations, tablebase=tablebase)
            active.append(_ActiveGame(gen=gen, pending_board=next(gen)))
            launched += 1

        boards = [state.pending_board for state in active]
        logits_batch, values_batch = evaluator.evaluate_batch(boards)
        if on_step is not None:
            on_step(len(active))

        next_active: list[_ActiveGame] = []
        for state, logits, value in zip(active, logits_batch, values_batch):
            try:
                pending = state.gen.send((logits, float(value)))
                next_active.append(_ActiveGame(gen=state.gen, pending_board=pending))
            except StopIteration as stop:
                game = stop.value
                completed.append(game)
                if on_game_finished is not None:
                    on_game_finished(game)

        active = next_active

    return completed


def _config_from_dict(data: dict) -> Config:
    return Config(
        net=NetConfig(**data["net"]),
        mcts=MCTSConfig(**data["mcts"]),
        beauty=BeautyConfig(**data["beauty"]),
        train=TrainConfig(**data["train"]),
    )


def _config_to_dict(cfg: Config) -> dict:
    return {
        "net": {
            "blocks": cfg.net.blocks,
            "filters": cfg.net.filters,
            "value_bins": cfg.net.value_bins,
        },
        "mcts": {
            "simulations": cfg.mcts.simulations,
            "c_puct": cfg.mcts.c_puct,
            "dirichlet_alpha": cfg.mcts.dirichlet_alpha,
            "dirichlet_epsilon": cfg.mcts.dirichlet_epsilon,
            "gumbel_c_visit": cfg.mcts.gumbel_c_visit,
            "gumbel_c_scale": cfg.mcts.gumbel_c_scale,
            "draw_contempt": cfg.mcts.draw_contempt,
            "claim_draw": cfg.mcts.claim_draw,
        },
        "beauty": {
            "enabled": cfg.beauty.enabled,
            "soundness_window": cfg.beauty.soundness_window,
            "w_sacrifice": cfg.beauty.w_sacrifice,
            "w_attack": cfg.beauty.w_attack,
            "w_tactical": cfg.beauty.w_tactical,
            "w_surprise": cfg.beauty.w_surprise,
        },
        "train": {
            "games_per_iteration": cfg.train.games_per_iteration,
            "selfplay_concurrency": cfg.train.selfplay_concurrency,
            "train_steps_per_iteration": cfg.train.train_steps_per_iteration,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "lr_min": cfg.train.lr_min,
            "lr_warmup_iters": cfg.train.lr_warmup_iters,
            "lr_total_iters": cfg.train.lr_total_iters,
            "weight_decay": cfg.train.weight_decay,
            "replay_buffer_size": cfg.train.replay_buffer_size,
            "replay_window": cfg.train.replay_window,
            "max_game_moves": cfg.train.max_game_moves,
            "syzygy_path": cfg.train.syzygy_path,
            "tb_max_pieces": cfg.train.tb_max_pieces,
            "draw_penalty": cfg.train.draw_penalty,
            "resign_threshold": cfg.train.resign_threshold,
            "resign_plies": cfg.train.resign_plies,
            "resign_min_moves": cfg.train.resign_min_moves,
            "fast_mate_bonus": cfg.train.fast_mate_bonus,
            "sims_per_move": cfg.train.sims_per_move,
            "checkpoint_dir": cfg.train.checkpoint_dir,
            "grad_clip_norm": cfg.train.grad_clip_norm,
        },
    }


def _split_games(num_games: int, workers: int) -> list[int]:
    if workers <= 0:
        raise ValueError("workers must be >= 1")
    base = num_games // workers
    remainder = num_games % workers
    counts = [base + (1 if i < remainder else 0) for i in range(workers)]
    return [c for c in counts if c > 0]


def _selfplay_worker(payload: dict) -> tuple[
    list[Sample],
    dict[str, int],
    list[int],
    list[int],
]:
    worker_id = int(payload["worker_id"])
    n_games = int(payload["n_games"])
    weights_path = str(payload["weights_path"])
    net_cfg = NetConfig(**payload["net_cfg"])
    cfg = _config_from_dict(payload["cfg_dict"])
    sims = int(payload["sims"])
    device = str(payload["device"])
    syzygy_path = payload.get("syzygy_path")
    seed = int(payload["seed"])

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    tablebase = None
    if syzygy_path:
        tablebase = chess.syzygy.open_tablebase(syzygy_path)

    try:
        net = ChessNet(net_cfg)
        state = torch.load(weights_path, map_location=device)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        net.load_state_dict(model_state, strict=True)
        net.to(device).eval()

        evaluator = NetEvaluator(net, device=device)
        samples: list[Sample] = []
        termination_counts: Counter[str] = Counter()
        game_lengths: list[int] = []
        game_outcomes: list[int] = []

        def _on_game(game: GameResult) -> None:
            samples.extend(game.samples)
            termination_counts[game.termination] += 1
            game_lengths.append(len(game.samples))
            game_outcomes.append(_winner_of_worker(game))

        play_games_batched(
            evaluator,
            cfg,
            simulations=sims,
            num_games=n_games,
            concurrency=n_games,
            on_game_finished=_on_game,
            tablebase=tablebase,
        )
        return samples, dict(termination_counts), game_lengths, game_outcomes
    finally:
        if tablebase is not None:
            tablebase.close()


def _winner_of_worker(game: GameResult) -> int:
    if game.termination not in {"checkmate", "resign", "tablebase_win"} or not game.samples:
        return 0
    first = game.samples[0]
    if first.value == 0.0:
        return 0
    winner_is_first_player = first.value > 0.0
    winner_is_white = bool(first.player) if winner_is_first_player else not bool(first.player)
    return 1 if winner_is_white else -1


def play_games_parallel(
    cfg: Config,
    net_cfg: NetConfig,
    weights_path: str,
    simulations: int,
    num_games: int,
    workers: int,
    device: str,
    syzygy_path: str | None = None,
) -> tuple[list[Sample], Counter[str], list[int], list[int]]:
    if num_games <= 0:
        return [], Counter(), [], []
    if workers <= 1:
        raise ValueError("play_games_parallel requires workers > 1")

    game_counts = _split_games(num_games, workers)
    cfg_dict = _config_to_dict(cfg)
    net_cfg_dict = {
        "blocks": net_cfg.blocks,
        "filters": net_cfg.filters,
        "value_bins": net_cfg.value_bins,
    }

    payloads = []
    for worker_id, n_worker_games in enumerate(game_counts):
        seed = (os.getpid() * 2654435761 + worker_id * 1597334677) & 0xFFFFFFFF
        payloads.append({
            "worker_id": worker_id,
            "n_games": n_worker_games,
            "weights_path": weights_path,
            "net_cfg": net_cfg_dict,
            "cfg_dict": cfg_dict,
            "sims": simulations,
            "device": device,
            "syzygy_path": syzygy_path,
            "seed": seed,
        })

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(payloads)) as pool:
        results = pool.map(_selfplay_worker, payloads)

    all_samples: list[Sample] = []
    termination_counts: Counter[str] = Counter()
    game_lengths: list[int] = []
    game_outcomes: list[int] = []
    for samples, term_dict, lengths, outcomes in results:
        all_samples.extend(samples)
        termination_counts.update(term_dict)
        game_lengths.extend(lengths)
        game_outcomes.extend(outcomes)

    return all_samples, termination_counts, game_lengths, game_outcomes


def _termination_reason(outcome: chess.Outcome | None, *,
                        hit_max_moves: bool, no_legal_moves: bool, resigned: bool = False,
                        tablebase_win: bool = False, tablebase_draw: bool = False) -> str:
    if tablebase_win:
        return "tablebase_win"
    if tablebase_draw:
        return "tablebase_draw"
    if resigned:
        return "resign"
    if outcome is not None:
        if outcome.termination == chess.Termination.CHECKMATE:
            return "checkmate"
        draw_name = _DRAW_TERMINATION_NAMES.get(outcome.termination)
        if draw_name is not None:
            return draw_name
        if outcome.winner is None:
            return outcome.termination.name.lower()
    if hit_max_moves:
        return "max_moves"
    if no_legal_moves:
        return "no_legal_moves"
    return "no_legal_moves"


def _assign_values(samples: list[Sample], outcome: chess.Outcome | None,
                   termination: str, cfg: Config, move_count: int,
                   winner_override: chess.Color | None = None,
                   truncation_bootstrap: float = 0.0) -> None:
    if termination == "max_moves" and samples:
        # Max-move truncation is a cutoff, not a terminal chess result. Bootstrap
        # from the final root value so long games don't collapse to all-zero labels.
        final_player = samples[-1].player
        target = float(truncation_bootstrap)
        for s in samples:
            s.value = target if s.player == final_player else -target
        return

    winner = winner_override if winner_override is not None else (outcome.winner if outcome is not None else None)
    if termination in {"checkmate", "resign", "tablebase_win"} and winner is not None:
        target = 1.0
        if termination == "checkmate" and cfg.train.fast_mate_bonus > 0.0:
            target += cfg.train.fast_mate_bonus / max(1, move_count)
    elif termination in _DRAW_TERMINATION_SET or termination == "tablebase_draw":
        # Contempt: a small negative target discourages dull draws,
        # nudging the net toward decisive, imbalanced positions.
        target = -cfg.train.draw_penalty
    else:
        # Truncation at max_game_moves is a training cutoff, not a chess draw.
        target = 0.0

    for s in samples:
        if termination in {"checkmate", "resign", "tablebase_win"} and winner is not None:
            s.value = target if s.player == winner else -target
        else:
            s.value = target
