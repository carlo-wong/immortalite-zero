"""MCTS Batch B: push/pop board reuse and cached child moves."""

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.mcts import MCTS
from engine.network import ChessNet, NetEvaluator


@pytest.fixture
def mcts() -> MCTS:
    cfg = Config()
    cfg.net.blocks = 2
    cfg.net.filters = 16
    cfg.mcts.simulations = 24
    return MCTS(NetEvaluator(ChessNet(cfg.net)), cfg.mcts)


def test_search_restores_root_board(mcts: MCTS) -> None:
    board = chess.Board()
    fen_before = board.fen()
    stack_before = len(board.move_stack)

    mcts.run(board, simulations=24, add_noise=True)

    assert board.fen() == fen_before
    assert len(board.move_stack) == stack_before


def test_search_gen_matches_run_for_fixed_rng(mcts: MCTS) -> None:
    board = chess.Board()
    evaluator = mcts.evaluator
    assert evaluator is not None

    np.random.seed(7)
    run_result = mcts.run(board, simulations=24, add_noise=True)

    np.random.seed(7)
    gen = mcts.search_gen(board, simulations=24, add_noise=True)
    req = next(gen)
    while True:
        logits, value = evaluator.evaluate(req)
        try:
            req = gen.send((logits, value))
        except StopIteration as stop:
            gen_result = stop.value
            break

    assert np.array_equal(run_result.visits, gen_result.visits)
    assert np.allclose(run_result.q_values, gen_result.q_values)


def test_child_nodes_cache_moves(mcts: MCTS) -> None:
    board = chess.Board()
    result = mcts.run(board, simulations=8, add_noise=False)
    for idx, child in result._root.children.items():
        assert child.move is not None
        assert child.move in board.legal_moves
        assert child.move == result.moves[result.indices.index(idx)]
