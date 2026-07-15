"""Regression tests for early-ply move-selection temperature (sampling only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.config import Config, TrainConfig
from engine.first_move_stats import (
    summarize_first_moves,
    summarize_first_moves_from_shard,
)
from engine.network import ChessNet, NetEvaluator
from engine.selfplay import (
    _config_from_dict,
    _config_to_dict,
    play_game,
    tempered_policy,
)

_REPO = Path(__file__).resolve().parents[1]
_SHARD_0240 = _REPO / "results" / "samples_iter_0240.npz"


def _applies_temperature(move_count: int, temperature: float, temp_plies: int) -> bool:
    """Mirror of the gate in play_game_gen (sampling path only)."""
    return (
        temp_plies > 0
        and move_count < temp_plies
        and abs(float(temperature) - 1.0) > 1e-12
    )


def test_tempered_policy_flattens_peaked_distribution() -> None:
    improved = np.array([0.70, 0.15, 0.10, 0.05], dtype=np.float64)
    t4 = tempered_policy(improved, 4.0)
    assert t4.shape == improved.shape
    assert abs(t4.sum() - 1.0) < 1e-9
    assert t4.max() < improved.max()
    assert t4.min() > improved.min()


def test_tempered_policy_t1_is_identity() -> None:
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    out = tempered_policy(p, 1.0)
    assert np.allclose(out, p)


def test_tempered_policy_sampling_not_always_peak() -> None:
    """With T=4 on a peaked policy, many draws must not all hit the peak move."""
    improved = np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float64)
    probs = tempered_policy(improved, 4.0)
    rng = np.random.default_rng(0)
    draws = rng.choice(len(probs), size=200, p=probs)
    assert not np.all(draws == 0)
    assert (draws == 0).mean() < 0.95


def test_stored_sample_policy_is_untempered() -> None:
    """Sample.policy must remain the improved (T=1) distribution, not tempered."""
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 8
    cfg.train.max_game_moves = 4
    cfg.train.move_temperature = 4.0
    cfg.train.move_temperature_plies = 10
    evaluator = NetEvaluator(ChessNet(cfg.net), device="cpu")
    game = play_game(evaluator, cfg, simulations=8)
    assert game.samples, "expected at least one sample"
    stored = game.samples[0].policy.astype(np.float64)
    mass = stored[stored > 0]
    assert mass.size >= 2
    assert abs(mass.sum() - 1.0) < 1e-3
    retempered = tempered_policy(mass, 4.0)
    assert not np.allclose(mass / mass.sum(), retempered, atol=1e-4)
    assert mass.max() > retempered.max()


def test_defaults_are_noop() -> None:
    train = TrainConfig()
    assert train.move_temperature == 1.0
    assert train.move_temperature_plies == 0
    cfg = Config()
    assert cfg.train.move_temperature == 1.0
    assert cfg.train.move_temperature_plies == 0
    assert not _applies_temperature(0, cfg.train.move_temperature, cfg.train.move_temperature_plies)


def test_temperature_only_early_plies() -> None:
    T, plies = 4.0, 10
    assert _applies_temperature(0, T, plies)
    assert _applies_temperature(9, T, plies)
    assert not _applies_temperature(10, T, plies)
    assert not _applies_temperature(0, 1.0, plies)
    assert not _applies_temperature(0, T, 0)
    assert not _applies_temperature(5, 1.0, 0)


def test_first_move_stats_missing_path() -> None:
    stats = summarize_first_moves_from_shard(_REPO / "results" / "does_not_exist.npz")
    assert stats["n"] == 0
    assert stats["counts"] == {}


def test_first_move_stats_empty_list() -> None:
    stats = summarize_first_moves([])
    assert stats["n"] == 0


@pytest.mark.skipif(not _SHARD_0240.is_file(), reason="samples_iter_0240.npz not present")
def test_first_move_stats_from_shard_0240() -> None:
    stats = summarize_first_moves_from_shard(_SHARD_0240)
    assert stats["n"] > 0
    assert stats["entropy"] >= 0.0
    for key in (
        "n",
        "entropy",
        "d3_share",
        "a4_share",
        "main_share",
        "top1_uci",
        "top1_share",
        "counts",
    ):
        assert key in stats
    assert isinstance(stats["counts"], dict)
    assert stats["top1_uci"]


def test_config_round_trip_preserves_move_temperature() -> None:
    cfg = Config()
    cfg.train.move_temperature = 4.0
    cfg.train.move_temperature_plies = 10
    restored = _config_from_dict(_config_to_dict(cfg))
    assert restored.train.move_temperature == 4.0
    assert restored.train.move_temperature_plies == 10

    defaults = _config_from_dict(_config_to_dict(Config()))
    assert defaults.train.move_temperature == 1.0
    assert defaults.train.move_temperature_plies == 0
