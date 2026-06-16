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
    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.calls = 0

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        self.calls += 1
        return np.zeros(POLICY_SIZE, dtype=np.float32), self.value


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
    assert all(np.isclose(s.value, 0.8) for s in game.samples)


def test_max_game_moves_bootstrap_flips_by_side_to_move() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 2
    cfg.mcts.simulations = 4
    game = play_game(FakeEvaluator(value=0.6), cfg, simulations=4)

    assert game.termination == "max_moves"
    assert len(game.samples) == 2
    assert game.samples[0].player == chess.WHITE
    assert game.samples[1].player == chess.BLACK
    assert np.isclose(game.samples[0].value, -0.6)
    assert np.isclose(game.samples[1].value, 0.6)


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
