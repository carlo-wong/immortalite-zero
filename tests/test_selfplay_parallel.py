import time

import torch

from engine.config import Config
from engine.network import ChessNet
from engine.selfplay import _split_games, play_games_parallel


def _fake_selfplay_worker(payload: dict):
    games_done = payload["games_done"]
    for _ in range(int(payload["n_games"])):
        time.sleep(0.15)
        games_done.value += 1
    return [], {}, [], []


def test_split_games_even() -> None:
    assert _split_games(256, 4) == [64, 64, 64, 64]


def test_split_games_colab_two_workers() -> None:
    assert _split_games(256, 2) == [128, 128]


def test_split_games_with_remainder() -> None:
    assert sum(_split_games(10, 4)) == 10
    assert len(_split_games(10, 4)) == 4


def test_play_games_parallel_reports_progress(tmp_path, monkeypatch) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    weights_path = tmp_path / "worker_net.pt"
    torch.save({"model": ChessNet(cfg.net).state_dict()}, weights_path)

    monkeypatch.setattr("engine.selfplay._selfplay_worker", _fake_selfplay_worker)
    seen: list[int] = []

    play_games_parallel(
        cfg,
        cfg.net,
        str(weights_path),
        simulations=2,
        num_games=4,
        workers=2,
        device="cpu",
        on_progress=seen.append,
    )

    assert seen
    assert seen[-1] == 4
    assert max(seen) == 4
    assert len(seen) >= 2
