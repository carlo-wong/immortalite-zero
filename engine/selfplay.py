"""Self-play game generation for training."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from .config import Config
from .encoding import POLICY_SIZE, board_to_planes
from .mcts import MCTS
from .network import NetEvaluator

_EXPLORATION_MOVES = 20  # sample from the policy for this many plies, then argmax


@dataclass
class Sample:
    planes: np.ndarray      # (18, 8, 8)
    policy: np.ndarray      # (POLICY_SIZE,)
    player: chess.Color
    value: float = 0.0      # filled in once the game finishes


def play_game(evaluator: NetEvaluator, cfg: Config, simulations: int) -> list[Sample]:
    board = chess.Board()
    mcts = MCTS(evaluator, cfg.mcts)
    samples: list[Sample] = []
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < cfg.train.max_game_moves:
        result = mcts.run(board, simulations=simulations, add_noise=True)
        if not result.moves:
            break

        improved = result.improved_policy()
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, p in zip(result.indices, improved):
            policy[idx] = p
        samples.append(Sample(board_to_planes(board), policy, board.turn))

        if move_count < _EXPLORATION_MOVES:
            choice = np.random.choice(len(result.moves), p=improved / improved.sum())
            move = result.moves[choice]
        else:
            move = result.best_move()

        board.push(move)
        move_count += 1

    _assign_values(samples, board, cfg)
    return samples


def _assign_values(samples: list[Sample], board: chess.Board, cfg: Config) -> None:
    result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        winner = chess.WHITE
    elif result_str == "0-1":
        winner = chess.BLACK
    else:
        winner = None

    for s in samples:
        if winner is None:
            # Contempt: a small negative target discourages dull draws,
            # nudging the net toward decisive, imbalanced positions.
            s.value = -cfg.train.draw_penalty
        else:
            s.value = 1.0 if s.player == winner else -1.0
