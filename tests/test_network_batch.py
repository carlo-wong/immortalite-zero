"""NetEvaluator batch inference tests."""

import chess
import numpy as np
import pytest

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet, NetEvaluator


@pytest.fixture
def evaluator() -> NetEvaluator:
    cfg = Config()
    cfg.net.blocks = 2
    cfg.net.filters = 16
    net = ChessNet(cfg.net)
    net.eval()
    return NetEvaluator(net, device="cpu")


def test_evaluate_batch_matches_single_evaluate(evaluator: NetEvaluator) -> None:
    boards = [
        chess.Board(),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ]
    batch_logits, batch_values = evaluator.evaluate_batch(boards)
    assert batch_logits.shape == (3, batch_logits.shape[1])
    assert batch_values.shape == (3,)

    for i, board in enumerate(boards):
        logits, value = evaluator.evaluate(board)
        np.testing.assert_allclose(batch_logits[i], logits, rtol=0, atol=1e-5)
        assert batch_values[i] == pytest.approx(value, abs=1e-5)


def test_evaluate_batch_empty(evaluator: NetEvaluator) -> None:
    logits, values = evaluator.evaluate_batch([])
    assert logits.shape == (0, POLICY_SIZE)
    assert values.shape == (0,)
