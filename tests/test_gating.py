import os
import tempfile
from unittest.mock import patch

import chess
import pytest
import pandas as pd

from engine.config import Config
from engine.network import ChessNet
from engine.selfplay import GameResult
from engine.train import (
    play_match,
    _log_gate_metrics,
    _log_metrics,
    _snapshot_at_iter,
    _update_metrics_winrate_vs_prev,
)


def _tiny_nets() -> tuple[ChessNet, ChessNet, Config]:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.mcts.simulations = 2
    net_a = ChessNet(cfg.net)
    net_b = ChessNet(cfg.net)
    net_a.eval()
    net_b.eval()
    return net_a, net_b, cfg


def _make_fake_play_game_gen(recorded: list[dict]):
    def fake_play_game_gen(*args, **kwargs):
        recorded.append({"cfg": args[0], "kwargs": dict(kwargs)})
        board = chess.Board()
        result = GameResult(samples=[], termination="stalemate")

        def inner():
            yield board
            return result

        return inner()

    return fake_play_game_gen


def test_play_match_returns_expected_metrics_dict() -> None:
    net_a, net_b, cfg = _tiny_nets()

    n_games = 2
    sims = 2
    metrics = play_match(net_a, net_b, cfg, n_games=n_games, sims=sims, device="cpu")

    assert isinstance(metrics, dict)
    assert "winrate" in metrics
    assert "wins_as_white" in metrics
    assert "wins_as_black" in metrics
    assert "losses_as_white" in metrics
    assert "losses_as_black" in metrics
    assert "draws_as_white" in metrics
    assert "draws_as_black" in metrics
    assert "mean_game_len" in metrics
    assert "terminations" in metrics

    total_games = (
        metrics["wins_as_white"] + metrics["wins_as_black"] +
        metrics["losses_as_white"] + metrics["losses_as_black"] +
        metrics["draws_as_white"] + metrics["draws_as_black"]
    )
    assert total_games == n_games
    assert 0.0 <= metrics["winrate"] <= 1.0


def test_log_gate_metrics_writes_correct_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        it = 10
        prev_it = 5
        games = 20
        metrics = {
            "winrate": 0.65,
            "wins_as_white": 6,
            "wins_as_black": 5,
            "losses_as_white": 3,
            "losses_as_black": 2,
            "draws_as_white": 2,
            "draws_as_black": 2,
            "mean_game_len": 120.5,
            "terminations": "checkmate:16;threefold_repetition:4"
        }

        _log_gate_metrics(tmpdir, it, prev_it, metrics, games)

        csv_path = os.path.join(tmpdir, "metrics_gates.csv")
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 1
        expected_cols = [
            "iter", "prev_iter", "winrate", "wins_as_white", "wins_as_black",
            "losses_as_white", "losses_as_black", "draws_as_white", "draws_as_black",
            "mean_game_len", "games", "terminations"
        ]
        assert list(df.columns) == expected_cols

        row = df.iloc[0]
        assert int(row["iter"]) == it
        assert int(row["prev_iter"]) == prev_it
        assert float(row["winrate"]) == pytest.approx(metrics["winrate"])
        assert int(row["wins_as_white"]) == metrics["wins_as_white"]
        assert int(row["wins_as_black"]) == metrics["wins_as_black"]
        assert int(row["losses_as_white"]) == metrics["losses_as_white"]
        assert int(row["losses_as_black"]) == metrics["losses_as_black"]
        assert int(row["draws_as_white"]) == metrics["draws_as_white"]
        assert int(row["draws_as_black"]) == metrics["draws_as_black"]
        assert float(row["mean_game_len"]) == pytest.approx(metrics["mean_game_len"])
        assert int(row["games"]) == games
        assert row["terminations"] == metrics["terminations"]


def test_log_metrics_winrate_vs_prev_nan_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _log_metrics(
            tmpdir, 1, 24, 50, 10.0,
            policy_loss=1.0, value_loss=0.5,
            policy_entropy=2.0, value_sign_acc=0.5,
            policy_top1_agree=0.4, grad_norm=3.0,
            mean_game_len=80.0, decisive_rate=0.5,
            white_win_rate=0.5, draw_rate=0.2,
            max_moves_trunc_rate=0.1, value_mean=0.0,
            value_std=0.5, winrate_vs_prev=float("nan"),
            learning_rate=1e-3, games=4, train_steps=10,
            batch_size=32, buffer_size=100,
            termination_counts={"checkmate": 2},
        )
        df = pd.read_csv(os.path.join(tmpdir, "metrics.csv"))
        assert pd.isna(df.iloc[0]["winrate_vs_prev"])


def test_log_metrics_winrate_vs_prev_after_gate() -> None:
    """Gate winrate patches metrics.csv on gate iterations (log first, update after)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gate_winrate = 0.55
        gate_metrics = {
            "winrate": gate_winrate,
            "wins_as_white": 3,
            "wins_as_black": 2,
            "losses_as_white": 1,
            "losses_as_black": 1,
            "draws_as_white": 1,
            "draws_as_black": 1,
            "mean_game_len": 90.0,
            "terminations": "checkmate:8;stalemate:2",
        }
        it = 10
        _log_metrics(
            tmpdir, it, 24, 50, 10.0,
            policy_loss=1.0, value_loss=0.5,
            policy_entropy=2.0, value_sign_acc=0.5,
            policy_top1_agree=0.4, grad_norm=3.0,
            mean_game_len=80.0, decisive_rate=0.5,
            white_win_rate=0.5, draw_rate=0.2,
            max_moves_trunc_rate=0.1, value_mean=0.0,
            value_std=0.5, winrate_vs_prev=float("nan"),
            learning_rate=1e-3, games=4, train_steps=10,
            batch_size=32, buffer_size=100,
            termination_counts={"checkmate": 2},
        )
        _log_gate_metrics(tmpdir, it, 5, gate_metrics, games=10)
        _update_metrics_winrate_vs_prev(tmpdir, it, gate_winrate)
        gates_df = pd.read_csv(os.path.join(tmpdir, "metrics_gates.csv"))
        metrics_df = pd.read_csv(os.path.join(tmpdir, "metrics.csv"))
        assert float(gates_df.iloc[0]["winrate"]) == pytest.approx(gate_winrate)
        assert float(metrics_df.iloc[0]["winrate_vs_prev"]) == pytest.approx(gate_winrate)


def test_play_match_passes_tablebase_when_configured() -> None:
    recorded: list[dict] = []
    net_a, net_b, cfg = _tiny_nets()
    mock_tb = object()

    with patch("engine.train.play_game_gen", side_effect=_make_fake_play_game_gen(recorded)):
        play_match(net_a, net_b, cfg, n_games=1, sims=2, device="cpu", tablebase=mock_tb)

    assert len(recorded) == 1
    assert recorded[0]["kwargs"]["tablebase"] is mock_tb


def test_play_match_uses_zero_draw_contempt_for_normal_chess_gates() -> None:
    recorded: list[dict] = []
    net_a, net_b, cfg = _tiny_nets()
    cfg.train.draw_penalty = 1 / 3
    cfg.mcts.draw_contempt = 1 / 3

    with patch("engine.train.play_game_gen", side_effect=_make_fake_play_game_gen(recorded)):
        play_match(net_a, net_b, cfg, n_games=1, sims=2, device="cpu")

    assert len(recorded) == 1
    assert recorded[0]["cfg"].mcts.draw_contempt == pytest.approx(0.0)


def test_play_match_uncaps_max_game_moves_for_gates() -> None:
    recorded: list[dict] = []
    net_a, net_b, cfg = _tiny_nets()
    cfg.train.max_game_moves = 200

    with patch("engine.train.play_game_gen", side_effect=_make_fake_play_game_gen(recorded)):
        play_match(net_a, net_b, cfg, n_games=1, sims=2, device="cpu")

    assert len(recorded) == 1
    assert recorded[0]["cfg"].train.max_game_moves == 10_000


def test_snapshot_at_iter_returns_path_when_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = os.path.join(tmpdir, "ckpt_iter_0010.pt")
        with open(snap_path, "wb") as f:
            f.write(b"")
        assert _snapshot_at_iter(tmpdir, 10) == snap_path


def test_snapshot_at_iter_returns_none_when_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        assert _snapshot_at_iter(tmpdir, 10) is None
        assert _snapshot_at_iter(tmpdir, -1) is None
