"""Self-play game generation for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator

import chess
import numpy as np

from .config import Config
from .encoding import POLICY_SIZE, board_to_planes
from .mcts import MCTS
from .network import NetEvaluator

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


def play_game_gen(cfg: Config, simulations: int, *, add_noise: bool = True,
                  exploration_moves: int = _EXPLORATION_MOVES
                  ) -> Generator[chess.Board, tuple[np.ndarray, float], GameResult]:
    board = chess.Board()
    mcts = MCTS(None, cfg.mcts)
    samples: list[Sample] = []
    move_count = 0
    no_legal_moves = False

    while not board.is_game_over(claim_draw=True) and move_count < cfg.train.max_game_moves:
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
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, p in zip(result.indices, improved):
            policy[idx] = p
        samples.append(Sample(board_to_planes(board), policy, board.turn))

        if move_count < exploration_moves:
            choice = np.random.choice(len(result.moves), p=improved / improved.sum())
            move = result.moves[choice]
        else:
            move = result.best_move()

        board.push(move)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    hit_max_moves = move_count >= cfg.train.max_game_moves and outcome is None
    termination = _termination_reason(outcome, hit_max_moves=hit_max_moves,
                                      no_legal_moves=no_legal_moves)
    _assign_values(samples, outcome, termination, cfg, move_count)
    return GameResult(samples=samples, termination=termination)


def play_game(evaluator: NetEvaluator, cfg: Config, simulations: int) -> GameResult:
    gen = play_game_gen(cfg, simulations)
    req = next(gen)
    while True:
        logits, value = evaluator.evaluate(req)
        try:
            req = gen.send((logits, value))
        except StopIteration as stop:
            return stop.value


def play_games_batched(evaluator: NetEvaluator, cfg: Config, simulations: int,
                       num_games: int, concurrency: int,
                       on_game_finished: Callable[[GameResult], None] | None = None
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
            gen = play_game_gen(cfg, simulations)
            active.append(_ActiveGame(gen=gen, pending_board=next(gen)))
            launched += 1

        boards = [state.pending_board for state in active]
        logits_batch, values_batch = evaluator.evaluate_batch(boards)

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


def _termination_reason(outcome: chess.Outcome | None, *,
                        hit_max_moves: bool, no_legal_moves: bool) -> str:
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
                   termination: str, cfg: Config, move_count: int) -> None:
    winner = outcome.winner if outcome is not None else None
    if termination == "checkmate" and winner is not None:
        target = 1.0
        if cfg.train.fast_mate_bonus > 0.0:
            target += cfg.train.fast_mate_bonus / max(1, move_count)
    elif termination in _DRAW_TERMINATION_SET:
        # Contempt: a small negative target discourages dull draws,
        # nudging the net toward decisive, imbalanced positions.
        target = -cfg.train.draw_penalty
    else:
        # Truncation at max_game_moves is a training cutoff, not a chess draw.
        target = 0.0

    for s in samples:
        if termination == "checkmate" and winner is not None:
            s.value = target if s.player == winner else -target
        else:
            s.value = target
