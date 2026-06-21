import numpy as np
import pytest
import torch

from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.network import ChessNet
from engine.selfplay import Sample
from engine.train import (
    _load_sample_shard,
    _save_sample_shard,
    save_checkpoint,
)


def _tiny_net_and_optimizer() -> tuple[ChessNet, torch.optim.Adam, Config]:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    net = ChessNet(cfg.net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, optimizer, cfg


def _fake_sample() -> Sample:
    return Sample(
        planes=np.zeros((20, 8, 8), dtype=np.float32),
        policy=np.zeros(POLICY_SIZE, dtype=np.float32),
        player=True,
        value=0.5,
    )


def test_save_checkpoint_round_trip_with_optimizer(tmp_path) -> None:
    net, optimizer, cfg = _tiny_net_and_optimizer()
    x = torch.randn(1, 20, 8, 8)
    loss = net(x)[0].sum()
    loss.backward()
    optimizer.step()
    expected_opt_state = optimizer.state_dict()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(net, cfg, path, iteration=7, optimizer=optimizer)

    state = torch.load(path, map_location="cpu")
    assert state["iteration"] == 7
    assert "optimizer" in state

    net2 = ChessNet(cfg.net)
    net2.load_state_dict(state["model"])
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    optimizer2.load_state_dict(state["optimizer"])
    loaded_opt_state = optimizer2.state_dict()
    assert loaded_opt_state["param_groups"] == expected_opt_state["param_groups"]
    assert loaded_opt_state["state"].keys() == expected_opt_state["state"].keys()
    for key in loaded_opt_state["state"]:
        for field, tensor in loaded_opt_state["state"][key].items():
            assert torch.allclose(tensor, expected_opt_state["state"][key][field])


def test_save_checkpoint_without_optimizer_backward_compat(tmp_path) -> None:
    net, _, cfg = _tiny_net_and_optimizer()
    path = str(tmp_path / "legacy.pt")
    save_checkpoint(net, cfg, path, iteration=3)

    state = torch.load(path, map_location="cpu")
    assert "optimizer" not in state
    assert state["iteration"] == 3

    net2 = ChessNet(cfg.net)
    net2.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(net2.parameters(), lr=1e-3)
    assert optimizer.state_dict()["state"] == {}


def test_sample_shard_atomic_round_trip(tmp_path) -> None:
    ckpt_dir = str(tmp_path)
    samples = [_fake_sample(), _fake_sample()]
    _save_sample_shard(ckpt_dir, 12, samples)

    path = tmp_path / "samples_iter_0012.npz"
    assert path.exists()
    loaded = _load_sample_shard(str(path))
    assert len(loaded) == 2
    assert loaded[0].value == pytest.approx(0.5)
