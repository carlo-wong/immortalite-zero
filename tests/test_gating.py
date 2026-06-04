import os
import tempfile
import pytest
import pandas as pd

from engine.config import Config
from engine.network import ChessNet
from engine.train import play_match, _log_gate_metrics


def test_play_match_returns_correct_dict() -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.mcts.simulations = 2

    # Create two tiny ChessNet instances
    net_a = ChessNet(cfg.net)
    net_b = ChessNet(cfg.net)
    net_a.eval()
    net_b.eval()

    # Run a small match of 2 games
    n_games = 2
    sims = 2
    metrics = play_match(net_a, net_b, cfg, n_games=n_games, sims=sims, device="cpu")

    # Verify return types and constraints
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

        # Log metrics to the temporary directory
        _log_gate_metrics(tmpdir, it, prev_it, metrics, games)

        csv_path = os.path.join(tmpdir, "metrics_gates.csv")
        assert os.path.exists(csv_path)

        # Read and verify CSV contents
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
