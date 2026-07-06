"""Regression tests for MCTS training-target bugs."""

import chess
import numpy as np

from engine.config import Config
from engine.encoding import POLICY_SIZE, legal_move_indices
from engine.mcts import MCTS, SearchResult, _Node, _softmax


class FixedLogitEvaluator:
    def __init__(self, logits: np.ndarray, value: float = 0.0):
        self.logits = logits.astype(np.float32)
        self.value = float(value)

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        del board
        return self.logits.copy(), self.value


def _expected_improved_policy(result: SearchResult) -> np.ndarray:
    max_n = result.visits.max() if result.visits.size else 0.0
    sigma = (result._cfg.gumbel_c_visit + max_n) * result._cfg.gumbel_c_scale
    q = result.q_values
    q_span = float(q.max() - q.min())
    q_norm = (q - q.min()) / q_span if q_span > 0 else np.zeros_like(q)
    logits = np.log(np.clip(result.clean_priors, 1e-9, None)) + sigma * q_norm
    return _softmax(logits)


def test_improved_policy_ignores_dirichlet_noise() -> None:
    cfg = Config()
    cfg.mcts.simulations = 8
    cfg.mcts.dirichlet_epsilon = 0.25

    board = chess.Board()
    mapping = legal_move_indices(board)
    logits = np.full(POLICY_SIZE, -4.0, dtype=np.float32)
    for idx in mapping:
        logits[idx] = 0.0

    np.random.seed(123)
    noisy = MCTS(FixedLogitEvaluator(logits), cfg.mcts).run(board, add_noise=True)
    np.random.seed(123)
    clean = MCTS(FixedLogitEvaluator(logits), cfg.mcts).run(board, add_noise=False)

    assert not np.allclose(noisy.priors, clean.priors)
    assert np.allclose(noisy.clean_priors, clean.clean_priors)
    assert np.allclose(noisy.improved_policy(), _expected_improved_policy(noisy))
    assert np.allclose(clean.improved_policy(), _expected_improved_policy(clean))
    assert np.allclose(noisy.improved_policy(), clean.improved_policy())


def test_improved_policy_normalizes_q_values() -> None:
    cfg = Config()
    cfg.mcts.gumbel_c_visit = 50.0
    cfg.mcts.gumbel_c_scale = 1.0
    board = chess.Board()
    root = _Node(0.0)

    base = SearchResult(
        moves=[],
        indices=[10, 20],
        visits=np.array([3.0, 1.0]),
        q_values=np.array([0.1, -0.1]),
        priors=np.array([0.5, 0.5]),
        clean_priors=np.array([0.5, 0.5]),
        root_value=0.0,
        _root=root,
        _board=board,
        _cfg=cfg.mcts,
    )
    scaled = SearchResult(
        moves=[],
        indices=[10, 20],
        visits=np.array([3.0, 1.0]),
        q_values=np.array([0.8, 0.6]),
        priors=np.array([0.5, 0.5]),
        clean_priors=np.array([0.5, 0.5]),
        root_value=0.0,
        _root=root,
        _board=board,
        _cfg=cfg.mcts,
    )

    assert not np.allclose(base.q_values, scaled.q_values)
    assert np.allclose(base.improved_policy(), scaled.improved_policy())
