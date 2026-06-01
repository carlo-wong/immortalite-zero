"""Beauty-bias move selection.

Philosophy ("balanced"): among the moves that are *sound* (within a small
win-probability window of the engine's best move), prefer the one that is most
beautiful -- sacrificial, attacking, tactical, and surprising. Soundness is a
hard gate; beauty only breaks ties within it. So the engine plays for
fireworks whenever doing so is not objectively losing.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from .config import BeautyConfig
from .mcts import SearchResult

_PIECE_VALUE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass
class BeautyBreakdown:
    sacrifice: float
    attack: float
    tactical: float
    surprise: float
    total: float


def _captured_value(board: chess.Board, move: chess.Move) -> float:
    if board.is_en_passant(move):
        return _PIECE_VALUE[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return _PIECE_VALUE[victim.piece_type] if victim else 0.0


def _king_zone_pressure(board_after: chess.Board, attacker: chess.Color) -> int:
    enemy_king = board_after.king(not attacker)
    if enemy_king is None:
        return 0
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king])
    return sum(1 for sq in ring if board_after.is_attacked_by(attacker, sq))


def compute_beauty(board: chess.Board, move: chess.Move, prior: float,
                   max_prior: float, cfg: BeautyConfig) -> BeautyBreakdown:
    mover = board.piece_at(move.from_square)
    mover_value = _PIECE_VALUE[mover.piece_type] if mover else 0.0
    captured = _captured_value(board, move)
    gives_check = board.gives_check(move)

    board_after = board.copy()
    board_after.push(move)

    # Sacrifice: we leave material en prise beyond what we captured.
    invested = 0.0
    if board_after.is_attacked_by(not board.turn, move.to_square):
        invested = max(0.0, mover_value - captured)
    sacrifice = invested

    # Attack: pressure on the enemy king zone, plus a check bonus.
    attack = _king_zone_pressure(board_after, board.turn) + (2.0 if gives_check else 0.0)

    # Tactical: captures, checks, and promotions are sharp.
    tactical = (1.0 if board.is_capture(move) else 0.0)
    tactical += 1.0 if gives_check else 0.0
    tactical += 1.0 if move.promotion else 0.0

    # Surprise: a strong move the raw policy underrates ("alien" feel).
    surprise = max(0.0, 1.0 - (prior / max_prior)) if max_prior > 0 else 0.0

    total = (
        cfg.w_sacrifice * sacrifice
        + cfg.w_attack * attack
        + cfg.w_tactical * tactical
        + cfg.w_surprise * surprise
    )
    return BeautyBreakdown(sacrifice, attack, tactical, surprise, total)


def _win_prob(q: float) -> float:
    return (q + 1.0) / 2.0


def select_beautiful_move(board: chess.Board, result: SearchResult,
                          cfg: BeautyConfig) -> tuple[chess.Move, BeautyBreakdown | None]:
    """Return (chosen_move, beauty_breakdown). breakdown is None if beauty
    selection was not applied (disabled or only one sound option)."""
    best_visit_move = result.best_move()
    if not cfg.enabled or len(result.moves) == 0:
        return best_visit_move, None

    best_wp = _win_prob(float(result.q_values.max()))
    max_prior = float(result.priors.max()) if result.priors.size else 0.0

    best_total = -float("inf")
    chosen = best_visit_move
    chosen_bd: BeautyBreakdown | None = None
    sound_count = 0
    for i, move in enumerate(result.moves):
        wp = _win_prob(float(result.q_values[i]))
        if best_wp - wp > cfg.soundness_window:
            continue  # not sound enough
        sound_count += 1
        bd = compute_beauty(board, move, float(result.priors[i]), max_prior, cfg)
        if bd.total > best_total:
            best_total = bd.total
            chosen = move
            chosen_bd = bd

    if sound_count <= 1:
        return best_visit_move, None
    return chosen, chosen_bd
