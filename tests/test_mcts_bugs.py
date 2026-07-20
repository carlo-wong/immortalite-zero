"""Regression tests for MCTS training-target bugs."""

import chess
import numpy as np

from engine.config import Config
from engine.encoding import POLICY_SIZE, legal_move_indices
from engine.mcts import MCTS, SearchResult, _Node, _softmax


# Fool's mate: black to move is checkmated.
_FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# Stalemate (black to move, no legal moves, not in check).
_STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


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


def test_expand_from_eval_priors_match_taken_logits() -> None:
    """np.take path must match the prior softmax of legal-move logits."""
    board = chess.Board()
    mapping = legal_move_indices(board)
    logits = np.linspace(-2.0, 2.0, POLICY_SIZE, dtype=np.float32)

    mcts = MCTS(None, Config().mcts)
    root = _Node(0.0)
    value = mcts._expand_from_eval(root, board, logits, 0.25)

    assert value == 0.25
    assert set(root.children) == set(mapping)
    idxs = list(mapping.keys())
    expected = _softmax(np.array([logits[i] for i in idxs], dtype=np.float32))
    got = np.array([root.children[i].prior for i in idxs], dtype=np.float32)
    assert np.allclose(got, expected, atol=1e-6)
    for i, move in mapping.items():
        assert root.children[i].move == move


def test_expand_from_eval_empty_mapping_is_noop() -> None:
    board = chess.Board(_STALEMATE_FEN)
    assert not any(board.legal_moves)
    mcts = MCTS(None, Config().mcts)
    root = _Node(0.0)
    logits = np.zeros(POLICY_SIZE, dtype=np.float32)
    assert mcts._expand_from_eval(root, board, logits, -0.5) == -0.5
    assert root.children == {}


def test_search_gen_terminal_root_returns_without_yield() -> None:
    board = chess.Board(_FOOLS_MATE_FEN)
    assert board.is_checkmate()
    mcts = MCTS(None, Config().mcts)
    gen = mcts.search_gen(board, simulations=8, add_noise=False)
    try:
        next(gen)
        raise AssertionError("terminal root must not yield for NN eval")
    except StopIteration as stop:
        result = stop.value
    assert result.moves == []
    assert result.root_value == -1.0


def test_terminal_eval_calls_board_outcome_not_is_game_over(monkeypatch) -> None:
    """_terminal_eval must resolve terminal state with a single board.outcome call."""
    board = chess.Board(_FOOLS_MATE_FEN)
    node = _Node(0.0)
    mcts = MCTS(None, Config().mcts)

    is_game_over_calls = {"n": 0}
    original_is_game_over = chess.Board.is_game_over

    def counted_is_game_over(self, *args, **kwargs):
        is_game_over_calls["n"] += 1
        return original_is_game_over(self, *args, **kwargs)

    monkeypatch.setattr(chess.Board, "is_game_over", counted_is_game_over)
    is_terminal, value = mcts._terminal_eval(node, board, chess.WHITE)

    assert is_terminal
    assert value == -1.0
    assert is_game_over_calls["n"] == 0


def test_collect_uses_child_move_not_index_to_move() -> None:
    cfg = Config()
    cfg.mcts.simulations = 8
    board = chess.Board()
    logits = np.zeros(POLICY_SIZE, dtype=np.float32)
    mcts = MCTS(FixedLogitEvaluator(logits, value=0.0), cfg.mcts)
    result = mcts.run(board, simulations=8, add_noise=False)
    assert result.moves
    for move, idx in zip(result.moves, result.indices):
        assert result._root.children[idx].move == move


def test_search_backs_up_checkmate_leaf_value() -> None:
    """Leaf mate must backup -1; each node runs the uncached terminal check once."""
    # White mates with Qa8#.
    board = chess.Board("6k1/8/6K1/8/8/8/8/7Q w - - 0 1")
    assert not board.is_game_over()
    mapping = legal_move_indices(board)
    logits = np.full(POLICY_SIZE, -8.0, dtype=np.float32)
    mate_idx = next(i for i, m in mapping.items() if m.uci() == "h1a8")
    logits[mate_idx] = 8.0

    mcts = MCTS(FixedLogitEvaluator(logits, value=0.0), Config().mcts)
    uncached_nodes: list[int] = []
    real_terminal_eval = mcts._terminal_eval

    def counting_terminal_eval(node, b, root_turn):
        if not node.terminal_checked:
            uncached_nodes.append(id(node))
        return real_terminal_eval(node, b, root_turn)

    mcts._terminal_eval = counting_terminal_eval  # type: ignore[method-assign]
    result = mcts.run(board, simulations=16, add_noise=False)

    assert len(uncached_nodes) == len(set(uncached_nodes)), (
        "uncached _terminal_eval must run at most once per node"
    )

    mate_child = result._root.children[mate_idx]
    assert mate_child.is_terminal
    assert mate_child.terminal_value == -1.0
    assert mate_child.N > 0
    assert mate_child.W < 0.0
    assert id(mate_child) in uncached_nodes


def test_terminal_eval_once_per_sim_on_forced_mate_child() -> None:
    """Deduped loop: 1 root check + 1 leaf check per sim (no post-while re-check)."""
    board = chess.Board("6k1/8/6K1/8/8/8/8/7Q w - - 0 1")
    mapping = legal_move_indices(board)
    logits = np.full(POLICY_SIZE, -8.0, dtype=np.float32)
    mate_idx = next(i for i, m in mapping.items() if m.uci() == "h1a8")
    logits[mate_idx] = 8.0

    sims = 12
    mcts = MCTS(FixedLogitEvaluator(logits, value=0.0), Config().mcts)
    calls = {"n": 0}
    real_terminal_eval = mcts._terminal_eval
    real_select = mcts._select_child

    def counting_terminal_eval(node, b, root_turn):
        calls["n"] += 1
        return real_terminal_eval(node, b, root_turn)

    def always_mate(node):
        return mate_idx, node.children[mate_idx]

    mcts._terminal_eval = counting_terminal_eval  # type: ignore[method-assign]
    mcts._select_child = always_mate  # type: ignore[method-assign]
    result = mcts.run(board, simulations=sims, add_noise=False)
    # Restore for safety if the instance is reused (it isn't).
    mcts._select_child = real_select  # type: ignore[method-assign]

    assert calls["n"] == 1 + sims
    assert result._root.children[mate_idx].N == sims


def test_search_stalemate_leaf_uses_draw_contempt() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    mapping = legal_move_indices(board)
    stalemate_move = chess.Move.from_uci("f7e6")
    assert stalemate_move in mapping.values()
    logits = np.full(POLICY_SIZE, -8.0, dtype=np.float32)
    stale_idx = next(i for i, m in mapping.items() if m == stalemate_move)
    logits[stale_idx] = 8.0

    cfg = Config().mcts
    cfg.draw_contempt = 0.0
    result = MCTS(FixedLogitEvaluator(logits), cfg).run(
        board, simulations=12, add_noise=False
    )
    child = result._root.children[stale_idx]
    assert child.is_terminal
    assert child.terminal_value == 0.0
    assert child.N > 0
    assert abs(child.W) < 1e-9
