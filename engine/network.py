"""Lightweight ResNet with policy and value heads (AlphaZero-style)."""

from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetConfig
from .encoding import NUM_INPUT_PLANES, POLICY_SIZE, board_to_planes


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
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)  # logits

        v = self.value_conv(x).flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [-1, 1] from side-to-move perspective
        return p, v.squeeze(-1)


class NetEvaluator:
    """Wraps a ChessNet for single-position inference used by MCTS."""

    def __init__(self, net: ChessNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device
        self._use_cuda_autocast = str(device).startswith("cuda")

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Return (policy_logits over POLICY_SIZE, value in [-1, 1])."""
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value = self.net(x)
        else:
            logits, value = self.net(x)
        return logits[0].float().cpu().numpy(), float(value[0].float().cpu())

    @torch.inference_mode()
    def evaluate_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([board_to_planes(b) for b in boards])
        x = torch.from_numpy(x).to(self.device)
        if self._use_cuda_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, value = self.net(x)
        else:
            logits, value = self.net(x)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()
