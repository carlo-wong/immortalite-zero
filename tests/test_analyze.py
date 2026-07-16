"""Tests for inference checkpoint loading and MultiPV principal variations."""

from dataclasses import asdict

import chess
import pytest
import torch

from engine.analyze import Analyzer, load_evaluator
from engine.config import Config
from engine.encoding import ENCODING_VERSION


def test_load_evaluator_rejects_stale_encoding_version(tmp_path) -> None:
    cfg = Config()
    ckpt_path = tmp_path / "stale.pt"
    torch.save(
        {
            "model": {},
            "net": asdict(cfg.net),
            "iteration": 0,
            "encoding_version": 1,
        },
        ckpt_path,
    )
    assert ENCODING_VERSION != 1
    with pytest.raises(ValueError, match="encoding version 1"):
        load_evaluator(str(ckpt_path), cfg)


def test_analyze_multipv_lines_have_continuations() -> None:
    """MultiPV side lines should include PV moves beyond the candidate itself."""
    analyzer = Analyzer(None, Config())
    board = chess.Board()
    analysis = analyzer.analyze(board, multipv=5, simulations=48)
    assert len(analysis.lines) == 5
    long_pvs = [line for line in analysis.lines if len(line["pv"]) >= 2]
    assert len(long_pvs) >= 2
    for line in analysis.lines:
        assert line["pv"][0] == line["move"]
        assert isinstance(line["visits"], int) and line["visits"] >= 0
        assert 0.0 <= float(line["visit_pct"]) <= 100.0


def test_analyze_pv_len_respected() -> None:
    analyzer = Analyzer(None, Config())
    board = chess.Board()
    short = analyzer.analyze(board, multipv=1, simulations=32, pv_len=4)
    long = analyzer.analyze(board, multipv=1, simulations=32, pv_len=12)
    assert len(short.lines[0]["pv"]) <= 4
    assert len(long.lines[0]["pv"]) >= len(short.lines[0]["pv"])
