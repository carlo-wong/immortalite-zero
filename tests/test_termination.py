"""Outcome classification and terminal-value tests."""

import chess
import numpy as np

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.mcts import MCTS
from engine.selfplay import Sample, _assign_values, _termination_reason, play_game


class FakeEvaluator:
    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.calls = 0

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        self.calls += 1
        return np.zeros(POLICY_SIZE, dtype=np.float32), self.value


def _sample_pair() -> list[Sample]:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
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


def test_max_game_moves_is_neutral_not_draw_penalty() -> None:
    cfg = Config()
    cfg.train.max_game_moves = 1
    cfg.mcts.simulations = 4
    game = play_game(FakeEvaluator(value=0.8), cfg, simulations=4)

    assert game.termination == "max_moves"
    assert len(game.samples) == 1
    assert all(s.value == 0.0 for s in game.samples)


def test_mcts_treats_claimable_draws_as_terminal() -> None:
    cfg = Config()
    cfg.mcts.simulations = 8
    for board in (_threefold_board(), _fifty_move_board()):
        evaluator = FakeEvaluator(value=0.9)
        result = MCTS(evaluator, cfg.mcts).run(board, simulations=cfg.mcts.simulations)

        assert result.root_value == 0.0
        assert len(result.moves) == 0
        assert evaluator.calls == 0
