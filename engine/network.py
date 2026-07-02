"""Lightweight ResNet with policy and value heads (AlphaZero-style)."""

from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetConfig
from .encoding import NUM_INPUT_PLANES, POLICY_SIZE, board_to_planes, fill_planes_batch


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        f = cfg.filters
        self.value_bins = cfg.value_bins

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(f) for _ in range(cfg.blocks)])

        # Policy head.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(f, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Value head.
        self.value_conv = nn.Sequential(
            nn.Conv2d(f, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True)
        )
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, self.value_bins)
        self.register_buffer("value_support", torch.linspace(-1.0, 1.0, self.value_bins))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)  # logits

        v = self.value_conv(x).flatten(1)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_fc2(v)
        return p, value_logits

    def value_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(value_logits.float(), dim=-1)
        return torch.sum(probs * self.value_support, dim=-1)


class NetEvaluator:
    """Wraps a ChessNet for single-position inference used by MCTS."""

    def __init__(self, net: ChessNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device
        self._use_cuda_autocast = str(device).startswith("cuda")
        self._batch_cap = 0
        self._planes_buf: np.ndarray | None = None
        self._host_input: torch.Tensor | None = None

    def _ensure_batch_buffers(self, batch_size: int) -> None:
        if batch_size <= self._batch_cap and self._planes_buf is not None:
            return
        new_cap = max(batch_size, self._batch_cap, 128)
        self._batch_cap = new_cap
        self._planes_buf = np.zeros(
            (new_cap, NUM_INPUT_PLANES, 8, 8), dtype=np.float32,
        )
        if self._use_cuda_autocast:
            self._host_input = torch.empty(
                (new_cap, NUM_INPUT_PLANES, 8, 8),
                dtype=torch.float32,
                pin_memory=True,
            )
        else:
            self._host_input = None

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Return (policy_logits over POLICY_SIZE, value in [-1, 1])."""
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
        else:
            logits, value_logits = self.net(x)
        value = self.net.value_from_logits(value_logits)
        return logits[0].float().cpu().numpy(), float(value[0].float().cpu())

    @torch.inference_mode()
    def evaluate_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        n = len(boards)
        if n == 0:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32)
        self._ensure_batch_buffers(n)
        assert self._planes_buf is not None
        fill_planes_batch(boards, self._planes_buf[:n])
        if self._host_input is not None:
            self._host_input[:n].copy_(torch.from_numpy(self._planes_buf[:n]))
            x = self._host_input[:n].to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(self._planes_buf[:n]).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value_logits = self.net(x)
        else:
            logits, value_logits = self.net(x)
        value = self.net.value_from_logits(value_logits)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()
