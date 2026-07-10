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


def test_searched_root_q_is_visit_weighted_mean() -> None:
    """searched_root_q must return visit-weighted mean, not the minimum."""
    board = chess.Board()
    root = _Node(0.0)
    root.N = 100

    visits = np.array([90.0, 8.0, 2.0])
    q_values = np.array([0.3, -0.5, -0.9])
    expected = float(np.dot(visits, q_values) / visits.sum())  # 0.212

    result = SearchResult(
        moves=[],
        indices=[10, 20, 30],
        visits=visits,
        q_values=q_values,
        priors=np.array([0.9, 0.08, 0.02]),
        clean_priors=np.array([0.9, 0.08, 0.02]),
        root_value=0.0,
        _root=root,
        _board=board,
        _cfg=Config().mcts,
    )

    assert abs(result.searched_root_q - expected) < 1e-9, (
        f"expected visit-weighted mean {expected:.4f}, got {result.searched_root_q:.4f}"
    )
    # Old code returned np.min(q_values) = -0.9; assert that is wrong.
    assert abs(result.searched_root_q - float(np.min(q_values))) > 0.1, (
        "searched_root_q should NOT equal the minimum q_value"
    )


def test_improved_policy_entropy_and_prior_ordering() -> None:
    """improved_policy must retain meaningful entropy (requires c_scale=0.1 fix)."""
    board = chess.Board()
    root = _Node(0.0)
    n_moves = 20
    cfg = Config()
    # Confirm the config default was fixed; with c_scale=1.0 entropy collapses to ~0.
    assert cfg.mcts.gumbel_c_scale == 0.1, (
        f"gumbel_c_scale must be 0.1 after fix, got {cfg.mcts.gumbel_c_scale}"
    )

    # Uniform priors, linearly spread q values (span 0.3), visits up to 50.
    visits = np.linspace(1.0, 50.0, n_moves)
    q_values = np.linspace(-0.15, 0.15, n_moves)
    clean_priors = np.full(n_moves, 1.0 / n_moves)

    result = SearchResult(
        moves=[],
        indices=list(range(n_moves)),
        visits=visits,
        q_values=q_values,
        priors=clean_priors.copy(),
        clean_priors=clean_priors.copy(),
        root_value=0.0,
        _root=root,
        _board=board,
        _cfg=cfg.mcts,
    )

    policy = result.improved_policy()
    entropy = float(-np.sum(policy * np.log(np.clip(policy, 1e-12, None))))
    assert entropy > 0.5, (
        f"policy entropy {entropy:.4f} nats is too low; policy collapsed to near one-hot. "
        "This fails with c_scale=1.0 where sigma~100 swamps prior logits."
    )
    assert float(policy.max()) < 0.9, (
        f"max policy prob {policy.max():.4f} too high; policy collapsed. "
        "This fails with c_scale=1.0 where best-Q move absorbs all probability."
    )

    # With equal q_values, improved_policy ranking must follow priors.
    skewed_priors = np.exp(np.linspace(-1.0, 1.0, n_moves))
    skewed_priors /= skewed_priors.sum()
    equal_q = np.zeros(n_moves)
    result_eq = SearchResult(
        moves=[],
        indices=list(range(n_moves)),
        visits=visits,
        q_values=equal_q,
        priors=skewed_priors.copy(),
        clean_priors=skewed_priors.copy(),
        root_value=0.0,
        _root=root,
        _board=board,
        _cfg=cfg.mcts,
    )
    policy_eq = result_eq.improved_policy()
    assert int(np.argmax(policy_eq)) == int(np.argmax(skewed_priors)), (
        "With equal q_values, improved_policy argmax must match prior argmax"
    )
