import os
import tempfile
import pytest
import pandas as pd

from engine.config import Config
from engine.network import ChessNet
from engine.train import play_match, _log_gate_metrics


def test_play_match_returns_correct_tuple() -> None:
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
    winrate, wins, losses, draws = play_match(net_a, net_b, cfg, n_games=n_games, sims=sims, device="cpu")

    # Verify return types and constraints
    assert isinstance(winrate, float)
    assert isinstance(wins, int)
    assert isinstance(losses, int)
    assert isinstance(draws, int)
    assert wins + losses + draws == n_games
    assert 0.0 <= winrate <= 1.0
    assert winrate == (wins + 0.5 * draws) / n_games


def test_log_gate_metrics_writes_correct_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        it = 10
        prev_it = 5
        winrate = 0.65
        wins = 11
        losses = 5
        draws = 4
        games = 20

        # Log metrics to the temporary directory
        _log_gate_metrics(tmpdir, it, prev_it, winrate, wins, losses, draws, games)

        csv_path = os.path.join(tmpdir, "metrics_gates.csv")
        assert os.path.exists(csv_path)

        # Read and verify CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert list(df.columns) == ["iter", "prev_iter", "winrate", "wins", "losses", "draws", "games"]
        
        row = df.iloc[0]
        assert int(row["iter"]) == it
        assert int(row["prev_iter"]) == prev_it
        assert float(row["winrate"]) == pytest.approx(winrate)
        assert int(row["wins"]) == wins
        assert int(row["losses"]) == losses
        assert int(row["draws"]) == draws
        assert int(row["games"]) == games
