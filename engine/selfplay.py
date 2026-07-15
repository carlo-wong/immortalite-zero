"""Self-play game generation for training."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
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
GATE_OPENING_PLIES = 8


@dataclass
class Sample:
    planes: np.ndarray      # (20, 8, 8)
    policy: np.ndarray      # (POLICY_SIZE,)
    player: chess.Color
    value: float = 0.0      # filled in once the game finishes
    source_iter: int = 0
    root_q: float = 0.0     # STM-POV searched_root_q at this ply (for value_target=root_q)


@dataclass
class EvalRequest:
    board: chess.Board
    search_turn: chess.Color


@dataclass
class GameResult:
    samples: list[Sample]
    termination: str
    # Actual game winner (None for draws / truncations). Must not be inferred from
    # sample.value — that breaks when value_target=root_q (per-ply search Q).
    winner: chess.Color | None = None
    moves: list[str] = field(default_factory=list)  # UCI moves played


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
    gen: Generator[EvalRequest, tuple[np.ndarray, float], GameResult]
    pending: EvalRequest


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
                  source_iter: int = 0,
                  start_moves: list[str] | None = None,
                  ) -> Generator[EvalRequest, tuple[np.ndarray, float], GameResult]:
    board = chess.Board()
    mcts = MCTS(None, cfg.mcts)
    samples: list[Sample] = []
    moves: list[str] = []
    move_count = 0
    no_legal_moves = False
    resigned_winner: chess.Color | None = None
    tablebase_winner: chess.Color | None = None
    tablebase_draw = False
    low_value_streak = {chess.WHITE: 0, chess.BLACK: 0}
    last_root_value = 0.0

    if start_moves:
        for uci in start_moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"illegal start move {uci!r} in position {board.fen()}")
            board.push(move)
            moves.append(uci)
            move_count += 1

    while not board.is_game_over(claim_draw=True) and move_count < cfg.train.max_game_moves:
        tb_termination, tb_winner = _tablebase_adjudication(board, tablebase, cfg.train.tb_max_pieces)
        if tb_termination == "tablebase_win":
            tablebase_winner = tb_winner
            break
        if tb_termination == "tablebase_draw":
            tablebase_draw = True
            break

        search_turn = board.turn
        search = mcts.search_gen(board, simulations=simulations, add_noise=add_noise)
        req = next(search)
        while True:
            logits, value = yield EvalRequest(req, search_turn)
            try:
                req = search.send((logits, value))
            except StopIteration as stop:
                result = stop.value
                break
        if not result.moves:
            no_legal_moves = True
            break

        improved = result.improved_policy()
        last_root_value = float(result.searched_root_q)
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, p in zip(result.indices, improved):
            policy[idx] = p
        samples.append(Sample(
            board_to_planes(board).astype(np.float16),
            policy.astype(np.float16),
            board.turn,
            source_iter=source_iter,
            root_q=last_root_value,
        ))

        resign_enabled = cfg.train.resign_plies > 0 and cfg.train.resign_threshold >= -1.0
        if resign_enabled and move_count >= cfg.train.resign_min_moves:
            player = board.turn
            if result.searched_root_q <= cfg.train.resign_threshold:
                low_value_streak[player] += 1
            else:
                low_value_streak[player] = 0
            if low_value_streak[player] >= cfg.train.resign_plies:
                resigned_winner = not player
                break

        if move_count < exploration_moves:
            choice = np.random.choice(len(result.moves), p=improved / improved.sum())
            move = result.moves[choice]
        else:
            move = result.best_move()

        board.push(move)
        moves.append(move.uci())
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
    if termination in {"checkmate", "resign", "tablebase_win"}:
        if winner_override is not None:
            winner = winner_override
        elif outcome is not None:
            winner = outcome.winner
        else:
            winner = None
    else:
        winner = None
    return GameResult(samples=samples, termination=termination, winner=winner, moves=moves)


def _gate_result_letter(winner: int, a_is_white: bool) -> str:
    if winner == 0:
        return "D"
    if (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
        return "W"
    return "L"


def _opening_row(game_idx: int, a_is_white: bool, game: GameResult, winner: int) -> dict[str, Any]:
    return {
        "game_idx": game_idx,
        "a_is_white": int(a_is_white),
        "opening_uci": " ".join(game.moves[:GATE_OPENING_PLIES]),
        "result": _gate_result_letter(winner, a_is_white),
        "termination": game.termination,
        "plies": len(game.samples),
    }


def play_game(evaluator: NetEvaluator, cfg: Config, simulations: int,
              tablebase: Any | None = None, source_iter: int = 0) -> GameResult:
    gen = play_game_gen(cfg, simulations, tablebase=tablebase, source_iter=source_iter)
    req = next(gen)
    while True:
        logits, value = evaluator.evaluate(req.board)
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
            active.append(_ActiveGame(gen=gen, pending=next(gen)))
            launched += 1

        boards = [state.pending.board for state in active]
        logits_batch, values_batch = evaluator.evaluate_batch(boards)
        if on_step is not None:
            on_step(len(active))

        next_active: list[_ActiveGame] = []
        for state, logits, value in zip(active, logits_batch, values_batch):
            try:
                pending = state.gen.send((logits, float(value)))
                next_active.append(_ActiveGame(gen=state.gen, pending=pending))
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
            "value_target": cfg.train.value_target,
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
        # Match train.py workers=1 path: compile after load. Pool respawns each
        # iter, so compile warms once per worker process (~once per train iter).
        if device.startswith("cuda") and hasattr(torch, "compile"):
            net = torch.compile(net, dynamic=True)

        evaluator = NetEvaluator(net, device=device)
        samples: list[Sample] = []
        termination_counts: Counter[str] = Counter()
        game_lengths: list[int] = []
        game_outcomes: list[int] = []

        games_done = payload.get("games_done")

        def _on_game(game: GameResult) -> None:
            samples.extend(game.samples)
            termination_counts[game.termination] += 1
            game_lengths.append(len(game.samples))
            game_outcomes.append(_winner_of_worker(game))
            if games_done is not None:
                games_done.value += 1

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
    """Map GameResult → +1 white win / -1 black win / 0 draw-or-other."""
    if game.termination not in {"checkmate", "resign", "tablebase_win"}:
        return 0
    if game.winner is None:
        return 0
    return 1 if game.winner == chess.WHITE else -1


def play_games_parallel(
    cfg: Config,
    net_cfg: NetConfig,
    weights_path: str,
    simulations: int,
    num_games: int,
    workers: int,
    device: str,
    syzygy_path: str | None = None,
    on_progress: Callable[[int], None] | None = None,
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
    with mp.Manager() as manager:
        games_done = manager.Value("i", 0)
        for payload in payloads:
            payload["games_done"] = games_done

        with ctx.Pool(processes=len(payloads)) as pool:
            async_result = pool.map_async(_selfplay_worker, payloads)
            while not async_result.ready():
                if on_progress is not None:
                    on_progress(int(games_done.value))
                time.sleep(0.2)
            results = async_result.get()
            if on_progress is not None:
                on_progress(num_games)

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


@dataclass
class _MatchChunkStats:
    score: float = 0.0
    wins_w: int = 0
    wins_b: int = 0
    losses_w: int = 0
    losses_b: int = 0
    draws_w: int = 0
    draws_b: int = 0
    game_lengths: list[int] | None = None
    termination_counts: Counter[str] | None = None
    openings: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.game_lengths is None:
            self.game_lengths = []
        if self.termination_counts is None:
            self.termination_counts = Counter()
        if self.openings is None:
            self.openings = []


def _record_match_game(
    stats: _MatchChunkStats,
    game: GameResult,
    a_is_white: bool,
    game_idx: int,
) -> None:
    winner = _winner_of_worker(game)
    stats.game_lengths.append(len(game.samples))
    stats.termination_counts[game.termination] += 1
    stats.openings.append(_opening_row(game_idx, a_is_white, game, winner))
    if winner == 0:
        stats.score += 0.5
        if a_is_white:
            stats.draws_w += 1
        else:
            stats.draws_b += 1
    elif (winner == 1 and a_is_white) or (winner == -1 and not a_is_white):
        stats.score += 1.0
        if a_is_white:
            stats.wins_w += 1
        else:
            stats.wins_b += 1
    else:
        if a_is_white:
            stats.losses_w += 1
        else:
            stats.losses_b += 1


def _run_match_games(
    eval_a: NetEvaluator,
    eval_b: NetEvaluator,
    match_cfg: Config,
    sims: int,
    n_games: int,
    exploration_moves: int,
    tablebase: Any | None,
    *,
    start_game_index: int = 0,
    on_game_finished: Callable[[], None] | None = None,
    openings: list[list[str]] | None = None,
) -> _MatchChunkStats:
    from engine.openings import opening_for_game

    stats = _MatchChunkStats()
    concurrency = max(1, min(n_games, match_cfg.train.selfplay_concurrency))
    active: list[tuple[object, EvalRequest, bool, int]] = []
    launched = 0

    while launched < n_games or active:
        while launched < n_games and len(active) < concurrency:
            game_idx = start_game_index + launched
            a_is_white = (game_idx % 2) == 0
            gen = play_game_gen(
                match_cfg,
                sims,
                add_noise=False,
                exploration_moves=exploration_moves,
                tablebase=tablebase,
                start_moves=opening_for_game(openings, game_idx),
            )
            active.append((gen, next(gen), a_is_white, game_idx))
            launched += 1

        if not active:
            break

        a_indices: list[int] = []
        a_boards: list[chess.Board] = []
        b_indices: list[int] = []
        b_boards: list[chess.Board] = []
        for idx, (_, pending, a_is_white, _) in enumerate(active):
            if pending.search_turn == a_is_white:
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
                _record_match_game(stats, stop.value, a_is_white, game_idx)
                if on_game_finished is not None:
                    on_game_finished()
        active = next_active

    return stats


def _match_worker(payload: dict) -> dict[str, Any]:
    worker_id = int(payload["worker_id"])
    n_games = int(payload["n_games"])
    start_game_index = int(payload["start_game_index"])
    weights_path_a = str(payload["weights_path_a"])
    weights_path_b = str(payload["weights_path_b"])
    net_cfg_a = NetConfig(**payload["net_cfg_a"])
    net_cfg_b = NetConfig(**payload["net_cfg_b"])
    cfg = _config_from_dict(payload["cfg_dict"])
    sims = int(payload["sims"])
    device = str(payload["device"])
    syzygy_path = payload.get("syzygy_path")
    exploration_moves = int(payload["exploration_moves"])
    seed = int(payload["seed"])
    openings = payload.get("openings")

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    tablebase = None
    if syzygy_path:
        tablebase = chess.syzygy.open_tablebase(syzygy_path)

    try:
        net_a = ChessNet(net_cfg_a)
        state_a = torch.load(weights_path_a, map_location=device)
        model_a = state_a["model"] if isinstance(state_a, dict) and "model" in state_a else state_a
        net_a.load_state_dict(model_a, strict=True)
        net_a.to(device).eval()

        net_b = ChessNet(net_cfg_b)
        state_b = torch.load(weights_path_b, map_location=device)
        model_b = state_b["model"] if isinstance(state_b, dict) and "model" in state_b else state_b
        net_b.load_state_dict(model_b, strict=True)
        net_b.to(device).eval()

        eval_a = NetEvaluator(net_a, device=device)
        eval_b = NetEvaluator(net_b, device=device)
        games_done = payload.get("games_done")

        def _on_game() -> None:
            if games_done is not None:
                games_done.value += 1

        stats = _run_match_games(
            eval_a,
            eval_b,
            cfg,
            sims,
            n_games,
            exploration_moves,
            tablebase,
            start_game_index=start_game_index,
            on_game_finished=_on_game,
            openings=openings,
        )
        return {
            "score": stats.score,
            "wins_w": stats.wins_w,
            "wins_b": stats.wins_b,
            "losses_w": stats.losses_w,
            "losses_b": stats.losses_b,
            "draws_w": stats.draws_w,
            "draws_b": stats.draws_b,
            "game_lengths": stats.game_lengths,
            "termination_counts": dict(stats.termination_counts),
            "openings": stats.openings,
        }
    finally:
        if tablebase is not None:
            tablebase.close()


def play_match_parallel(
    match_cfg: Config,
    net_cfg_a: NetConfig,
    net_cfg_b: NetConfig,
    weights_path_a: str,
    weights_path_b: str,
    n_games: int,
    sims: int,
    workers: int,
    device: str,
    exploration_moves: int,
    syzygy_path: str | None = None,
    on_progress: Callable[[int], None] | None = None,
    openings: list[list[str]] | None = None,
) -> _MatchChunkStats:
    if n_games <= 0:
        return _MatchChunkStats()
    if workers <= 1:
        raise ValueError("play_match_parallel requires workers > 1")

    game_counts = _split_games(n_games, workers)
    cfg_dict = _config_to_dict(match_cfg)
    net_cfg_a_dict = {
        "blocks": net_cfg_a.blocks,
        "filters": net_cfg_a.filters,
        "value_bins": net_cfg_a.value_bins,
    }
    net_cfg_b_dict = {
        "blocks": net_cfg_b.blocks,
        "filters": net_cfg_b.filters,
        "value_bins": net_cfg_b.value_bins,
    }

    payloads = []
    start_index = 0
    for worker_id, n_worker_games in enumerate(game_counts):
        seed = (os.getpid() * 2654435761 + worker_id * 1597334677) & 0xFFFFFFFF
        payloads.append({
            "worker_id": worker_id,
            "n_games": n_worker_games,
            "start_game_index": start_index,
            "weights_path_a": weights_path_a,
            "weights_path_b": weights_path_b,
            "net_cfg_a": net_cfg_a_dict,
            "net_cfg_b": net_cfg_b_dict,
            "cfg_dict": cfg_dict,
            "sims": sims,
            "device": device,
            "syzygy_path": syzygy_path,
            "exploration_moves": exploration_moves,
            "seed": seed,
            "openings": openings,
        })
        start_index += n_worker_games

    merged = _MatchChunkStats()
    ctx = mp.get_context("spawn")
    with mp.Manager() as manager:
        games_done = manager.Value("i", 0)
        for payload in payloads:
            payload["games_done"] = games_done

        with ctx.Pool(processes=len(payloads)) as pool:
            async_result = pool.map_async(_match_worker, payloads)
            while not async_result.ready():
                if on_progress is not None:
                    on_progress(int(games_done.value))
                time.sleep(0.2)
            results = async_result.get()
            if on_progress is not None:
                on_progress(n_games)

    for chunk in results:
        merged.score += float(chunk["score"])
        merged.wins_w += int(chunk["wins_w"])
        merged.wins_b += int(chunk["wins_b"])
        merged.losses_w += int(chunk["losses_w"])
        merged.losses_b += int(chunk["losses_b"])
        merged.draws_w += int(chunk["draws_w"])
        merged.draws_b += int(chunk["draws_b"])
        merged.game_lengths.extend(chunk["game_lengths"])
        merged.termination_counts.update(chunk["termination_counts"])
        merged.openings.extend(chunk.get("openings") or [])

    return merged


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
    # Per-ply MCTS value labels: keep each position's searched_root_q (STM POV).
    # Search already applies draw_contempt at terminals; do not overwrite with z.
    if cfg.train.value_target == "root_q":
        for s in samples:
            s.value = float(s.root_q)
        return

    if cfg.train.value_target != "outcome":
        raise ValueError(
            f"unknown value_target={cfg.train.value_target!r}; "
            "expected 'outcome' or 'root_q'"
        )

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
