"""Tests for inference checkpoint loading."""

from dataclasses import asdict

import pytest
import torch

from engine.analyze import load_evaluator
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
