"""Outcome classification and terminal-value tests."""

import os

import chess
import chess.syzygy
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import NUM_INPUT_PLANES, POLICY_SIZE
from engine.mcts import MCTS
from engine.selfplay import (
    Sample,
    _assign_values,
    _tablebase_adjudication,
    _termination_reason,
    play_game,
)


class FakeEvaluator:
    def __init__(self, value: float = 0.0, *, value_for_turn: dict[chess.Color, float] | None = None):
        self.value = float(value)
        self.value_for_turn = value_for_turn
        self.calls = 0

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        self.calls += 1
        if self.value_for_turn is not None:
            value = float(self.value_for_turn[board.turn])
        else:
            value = self.value
        return np.zeros(POLICY_SIZE, dtype=np.float32), value


def _sample_pair() -> list[Sample]:
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    return [
        Sample(planes.copy(), policy.copy(), chess.WHITE),
        Sample(planes.copy(), policy.copy(), chess.BLACK),
    ]


def _threefold_board() -> chess.Board:
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board.push_uci(uci)
    return board


def _fifty_move_board() -> chess.Board:
    board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 99 1")
    board.push_uci("b1b2")
    return board


def _syzygy_path_or_skip() -> str:
    path = os.environ.get("IMMORTALITE_ZERO_SYZYGY_PATH")
    if not path:
        pytest.skip("Set IMMORTALITE_ZERO_SYZYGY_PATH to run Syzygy adjudication tests.")
    if not os.path.isdir(path):
        pytest.skip(f"IMMORTALITE_ZERO_SYZYGY_PATH does not exist: {path}")
    return path


def test_checkmate_values_are_plus_minus_one() -> None:
    cfg = Config()
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    assert outcome.termination == chess.Termination.CHECKMATE

    samples = _sample_pair()
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=len(board.move_stack))

    assert term == "checkmate"
    assert samples[0].value == -1.0
    assert samples[1].value == 1.0


def test_stalemate_gets_draw_penalty() -> None:
    cfg = Config()
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    assert outcome.termination == chess.Termination.STALEMATE

    samples = _sample_pair()
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=0)

    assert term == "stalemate"
    assert all(s.value == -cfg.train.draw_penalty for s in samples)


def test_threefold_repetition_gets_draw_penalty() -> None:
    cfg = Config()
    board = _threefold_board()
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    assert outcome.termination == chess.Termination.THREEFOLD_REPETITION

    samples = _sample_pair()
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=len(board.move_stack))

    assert term == "threefold_repetition"
    assert all(s.value == -cfg.train.draw_penalty for s in samples)


def test_fifty_move_rule_gets_draw_penalty() -> None:
    cfg = Config()
    board = _fifty_move_board()
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None
    assert outcome.termination == chess.Termination.FIFTY_MOVES

    samples = _sample_pair()
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=len(board.move_stack))

    assert term == "fifty_moves"
    assert all(s.value == -cfg.train.draw_penalty for s in samples)


def test_max_game_moves_bootstraps_from_final_root_value() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 1
    cfg.mcts.simulations = 4
    game = play_game(FakeEvaluator(value=0.8), cfg, simulations=4)

    assert game.termination == "max_moves"
    assert len(game.samples) == 1
    assert np.isclose(game.samples[0].value, -0.8)


def test_max_game_moves_bootstrap_flips_by_side_to_move() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 2
    cfg.mcts.simulations = 4
    game = play_game(FakeEvaluator(value=0.6), cfg, simulations=4)

    assert game.termination == "max_moves"
    assert len(game.samples) == 2
    assert game.samples[0].player == chess.WHITE
    assert game.samples[1].player == chess.BLACK
    assert np.isclose(game.samples[0].value, 0.6)
    assert np.isclose(game.samples[1].value, -0.6)


def test_mcts_treats_claimable_draws_as_terminal() -> None:
    cfg = Config()
    cfg.mcts.simulations = 8
    for board in (_threefold_board(), _fifty_move_board()):
        evaluator = FakeEvaluator(value=0.9)
        result = MCTS(evaluator, cfg.mcts).run(board, simulations=cfg.mcts.simulations)

        assert result.root_value == -cfg.mcts.draw_contempt
        assert len(result.moves) == 0
        assert evaluator.calls == 0


def test_tablebase_win_adjudication_assigns_plus_minus_one() -> None:
    cfg = Config()
    path = _syzygy_path_or_skip()
    board = chess.Board("k7/8/8/8/8/8/8/KQ6 w - - 0 1")

    tablebase = chess.syzygy.open_tablebase(path)
    try:
        termination, winner = _tablebase_adjudication(board, tablebase, max_pieces=5)
    finally:
        tablebase.close()

    assert termination == "tablebase_win"
    assert winner == chess.WHITE

    samples = _sample_pair()
    _assign_values(samples, None, termination, cfg, move_count=0, winner_override=winner)
    assert samples[0].value == 1.0
    assert samples[1].value == -1.0


def test_tablebase_draw_adjudication_gets_draw_penalty() -> None:
    cfg = Config()
    path = _syzygy_path_or_skip()
    board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")

    tablebase = chess.syzygy.open_tablebase(path)
    try:
        termination, winner = _tablebase_adjudication(board, tablebase, max_pieces=5)
    finally:
        tablebase.close()

    assert termination == "tablebase_draw"
    assert winner is None

    samples = _sample_pair()
    _assign_values(samples, None, termination, cfg, move_count=0)
    assert all(s.value == -cfg.train.draw_penalty for s in samples)


def test_resign_streak_counts_per_player_plies() -> None:
    cfg = Config()
    cfg.train.resign_threshold = -0.9
    cfg.train.resign_plies = 2
    cfg.train.resign_min_moves = 0
    cfg.train.max_game_moves = 40
    cfg.mcts.simulations = 2

    # BLACK: 0.95 means Black is winning (0.95) from Black's perspective, so the
    # backed-up q for White's explored children is -0.95, below the resign threshold.
    # Previously BLACK: 0.5 only worked because np.min() pulled in unvisited children
    # whose fallback q=root_value=-0.95; the corrected visit-weighted mean requires the
    # visited children themselves to have q < -0.9.
    evaluator = FakeEvaluator(
        value_for_turn={chess.WHITE: -0.95, chess.BLACK: 0.95},
    )
    game = play_game(evaluator, cfg, simulations=2)

    assert game.termination == "resign"
    assert len(game.samples) == 3


def test_searched_root_q_used_for_max_moves_bootstrap() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 1
    cfg.mcts.simulations = 4

    class ShiftingEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
            del board
            self.calls += 1
            value = 0.9 if self.calls == 1 else -0.4
            return np.zeros(POLICY_SIZE, dtype=np.float32), value

    game = play_game(ShiftingEvaluator(), cfg, simulations=4)

    assert game.termination == "max_moves"
    assert len(game.samples) == 1
    assert np.isclose(game.samples[0].value, 0.0)
    assert not np.isclose(game.samples[0].value, 0.9)


def test_root_q_value_target_keeps_per_ply_search_q() -> None:
    """value_target=root_q must not overwrite with terminal ±1 / draw_penalty."""
    cfg = Config()
    cfg.train.value_target = "root_q"
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None

    samples = _sample_pair()
    samples[0].root_q = 0.42
    samples[1].root_q = -0.77
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=len(board.move_stack))

    assert term == "checkmate"
    assert samples[0].value == pytest.approx(0.42)
    assert samples[1].value == pytest.approx(-0.77)


def test_root_q_value_target_ignores_draw_penalty_overwrite() -> None:
    cfg = Config()
    cfg.train.value_target = "root_q"
    cfg.train.draw_penalty = 1 / 3
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None

    samples = _sample_pair()
    samples[0].root_q = 0.15
    samples[1].root_q = -0.05
    term = _termination_reason(outcome, hit_max_moves=False, no_legal_moves=False)
    _assign_values(samples, outcome, term, cfg, move_count=0)

    assert term == "stalemate"
    assert samples[0].value == pytest.approx(0.15)
    assert samples[1].value == pytest.approx(-0.05)
    assert samples[0].value != -cfg.train.draw_penalty


def test_root_q_mode_play_game_stores_search_labels() -> None:
    cfg = Config()
    cfg.train.value_target = "root_q"
    cfg.train.max_game_moves = 3
    cfg.mcts.simulations = 8
    cfg.mcts.dirichlet_epsilon = 0.0

    class ConstantEvaluator:
        def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
            # Side-to-move POV: White optimistic, Black pessimistic.
            value = 0.6 if board.turn == chess.WHITE else -0.4
            return np.zeros(POLICY_SIZE, dtype=np.float32), value

    game = play_game(ConstantEvaluator(), cfg, simulations=8)
    assert game.termination == "max_moves"
    assert len(game.samples) == 3
    for s in game.samples:
        assert s.value == pytest.approx(s.root_q)
        assert -1.0 <= s.value <= 1.0
    # Not collapsed to a single terminal outcome label.
    assert not all(s.value == -cfg.train.draw_penalty for s in game.samples)
    assert not all(abs(s.value) == 1.0 for s in game.samples)


def test_unknown_value_target_raises() -> None:
    cfg = Config()
    cfg.train.value_target = "bogus"
    samples = _sample_pair()
    with pytest.raises(ValueError, match="unknown value_target"):
        _assign_values(samples, None, "stalemate", cfg, move_count=0)
