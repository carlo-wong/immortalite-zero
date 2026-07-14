import time
from dataclasses import asdict
from unittest.mock import MagicMock

import torch

from engine.config import Config, NetConfig
from engine.network import ChessNet
from engine.selfplay import _config_to_dict, _selfplay_worker, _split_games, play_games_parallel


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


def _worker_payload(device: str) -> dict:
    cfg = Config()
    cfg.net = NetConfig(blocks=1, filters=4, value_bins=51)
    return {
        "worker_id": 0,
        "n_games": 1,
        "weights_path": "unused.pt",
        "net_cfg": asdict(cfg.net),
        "cfg_dict": _config_to_dict(cfg),
        "sims": 2,
        "device": device,
        "syzygy_path": None,
        "seed": 0,
    }


class _FakeNet(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def load_state_dict(self, *_args, **_kwargs):
        return None

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


def test_selfplay_worker_compiles_on_cuda(monkeypatch) -> None:
    compile_calls: list[dict] = []

    monkeypatch.setattr("engine.selfplay.ChessNet", _FakeNet)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"model": {}})
    monkeypatch.setattr(
        torch,
        "compile",
        lambda net, dynamic=True: compile_calls.append({"dynamic": dynamic}) or net,
    )
    monkeypatch.setattr("engine.selfplay.NetEvaluator", MagicMock())
    monkeypatch.setattr("engine.selfplay.play_games_batched", MagicMock())

    _selfplay_worker(_worker_payload("cuda"))
    assert compile_calls == [{"dynamic": True}]


def test_selfplay_worker_skips_compile_on_cpu(monkeypatch) -> None:
    compile_calls: list[dict] = []

    monkeypatch.setattr("engine.selfplay.ChessNet", _FakeNet)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"model": {}})
    monkeypatch.setattr(
        torch,
        "compile",
        lambda net, dynamic=True: compile_calls.append({"dynamic": dynamic}) or net,
    )
    monkeypatch.setattr("engine.selfplay.NetEvaluator", MagicMock())
    monkeypatch.setattr("engine.selfplay.play_games_batched", MagicMock())

    _selfplay_worker(_worker_payload("cpu"))
    assert compile_calls == []
