import time
from dataclasses import asdict
from unittest.mock import MagicMock

import torch

from engine.config import Config, NetConfig
from engine.network import ChessNet
from engine.selfplay import (
    SelfplayWorkerPool,
    _config_to_dict,
    _selfplay_worker,
    _selfplay_worker_init,
    _selfplay_worker_run,
    _split_games,
    play_games_parallel,
)


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
    builds = 0

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        _FakeNet.builds += 1

    def load_state_dict(self, *_args, **_kwargs):
        return None

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


def test_selfplay_worker_compiles_on_cuda(monkeypatch) -> None:
    compile_calls: list[dict] = []
    _FakeNet.builds = 0

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
    assert _FakeNet.builds == 1


def test_selfplay_worker_skips_compile_on_cpu(monkeypatch) -> None:
    compile_calls: list[dict] = []
    _FakeNet.builds = 0

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
    assert _FakeNet.builds == 1


def test_persistent_worker_init_then_run_reuses_net(monkeypatch, tmp_path) -> None:
    """Init builds the net once; subsequent runs only reload weights."""
    _FakeNet.builds = 0
    compile_calls: list[dict] = []
    load_calls = {"n": 0}

    monkeypatch.setattr("engine.selfplay.ChessNet", _FakeNet)
    monkeypatch.setattr("engine.selfplay.NetEvaluator", MagicMock())
    monkeypatch.setattr("engine.selfplay.play_games_batched", MagicMock())
    monkeypatch.setattr(
        torch,
        "compile",
        lambda net, dynamic=True: compile_calls.append({"dynamic": dynamic}) or net,
    )

    def fake_load(*_a, **_k):
        load_calls["n"] += 1
        return {"model": {}}

    monkeypatch.setattr(torch, "load", fake_load)

    cfg = Config()
    cfg.net = NetConfig(blocks=1, filters=4, value_bins=51)
    net_cfg = asdict(cfg.net)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"unused")

    _selfplay_worker_init(net_cfg, "cuda", None)
    assert _FakeNet.builds == 1
    assert compile_calls == [{"dynamic": True}]

    payload = {
        "worker_id": 0,
        "n_games": 1,
        "weights_path": str(weights),
        "cfg_dict": _config_to_dict(cfg),
        "sims": 2,
        "seed": 1,
    }
    _selfplay_worker_run(payload)
    _selfplay_worker_run(payload)

    assert _FakeNet.builds == 1, "net must not be reconstructed between runs"
    assert len(compile_calls) == 1, "compile must run once"
    assert load_calls["n"] == 2, "weights must reload each run"


class _Value:
    def __init__(self, value: int = 0):
        self.value = value


class _FakeAsyncResult:
    def __init__(self, games_done: _Value, num_games: int, n_workers: int):
        self._games_done = games_done
        self._num_games = num_games
        self._n_workers = n_workers
        self._ticks = 0

    def ready(self) -> bool:
        self._ticks += 1
        if self._ticks == 1:
            self._games_done.value = max(1, self._num_games // 2)
            return False
        self._games_done.value = self._num_games
        return True

    def get(self):
        return [([], {}, [], []) for _ in range(self._n_workers)]


class _FakePool:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def map_async(self, fn, payloads):
        games_done = payloads[0]["games_done"]
        num_games = sum(int(p["n_games"]) for p in payloads)
        return _FakeAsyncResult(games_done, num_games, len(payloads))

    def close(self) -> None:
        self.closed = True

    def join(self) -> None:
        return None


class _FakeManager:
    def Value(self, _type: str, value: int) -> _Value:
        return _Value(value)

    def shutdown(self) -> None:
        return None


class _FakeCtx:
    def Manager(self) -> _FakeManager:
        return _FakeManager()

    def Pool(self, *args, **kwargs) -> _FakePool:
        return _FakePool(*args, **kwargs)


def test_selfplay_worker_pool_reports_progress_and_resets(monkeypatch, tmp_path) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    weights_path = tmp_path / "worker_net.pt"
    torch.save({"model": ChessNet(cfg.net).state_dict()}, weights_path)

    monkeypatch.setattr("engine.selfplay.mp.get_context", lambda _name: _FakeCtx())

    seen: list[int] = []
    with SelfplayWorkerPool(workers=2, net_cfg=cfg.net, device="cpu") as pool:
        pool.run(
            cfg,
            str(weights_path),
            simulations=2,
            num_games=4,
            on_progress=seen.append,
        )
        seen2: list[int] = []
        pool.run(
            cfg,
            str(weights_path),
            simulations=2,
            num_games=4,
            on_progress=seen2.append,
        )

    assert seen
    assert seen[-1] == 4
    assert seen2
    assert seen2[-1] == 4


def test_selfplay_worker_pool_close_is_idempotent(monkeypatch) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    monkeypatch.setattr("engine.selfplay.mp.get_context", lambda _name: _FakeCtx())
    pool = SelfplayWorkerPool(workers=2, net_cfg=cfg.net, device="cpu")
    pool.close()
    pool.close()  # must not raise


def test_selfplay_worker_pool_rejects_workers_one() -> None:
    cfg = Config()
    try:
        SelfplayWorkerPool(workers=1, net_cfg=cfg.net, device="cpu")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "workers > 1" in str(exc)
