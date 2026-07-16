"""Recover and summarize White first-move distributions from self-play shards.

Matches startpos planes → next-position fingerprint (same approach as
tmp/analyze_selfplay_first_moves.py). Used for per-iteration diversity logging.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import numpy as np

from .encoding import board_to_planes

# Classical first moves (human book-ish).
MAIN = frozenset({"e2e4", "d2d4", "g1f3", "c2c4"})
# Wing / fianchetto / rook-pawn starts (excludes MAIN and quiet d2d3/e2e3).
FLANK = frozenset({
    "a2a3", "a2a4", "h2h3", "h2h4",
    "b2b3", "b2b4", "g2g3", "g2g4",
    "b1a3", "b1c3", "g1h3",
})

# Schema: top-5 moves + category shares (no move-specific collapse columns).
CSV_COLUMNS = (
    "iter",
    "n",
    "entropy",
    "top1_uci",
    "top1_share",
    "top2_uci",
    "top2_share",
    "top3_uci",
    "top3_share",
    "top4_uci",
    "top4_share",
    "top5_uci",
    "top5_share",
    "main_share",
    "flank_share",
)

_START_PLANES: np.ndarray | None = None
_NEXT_FPS: dict[str, np.ndarray] | None = None


def _empty_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": 0,
        "entropy": float("nan"),
        "main_share": float("nan"),
        "flank_share": float("nan"),
        "counts": {},
    }
    for i in range(1, 6):
        out[f"top{i}_uci"] = ""
        out[f"top{i}_share"] = float("nan")
    return out


def shannon(counts: Counter) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return h


def _fingerprints() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global _START_PLANES, _NEXT_FPS
    if _START_PLANES is None or _NEXT_FPS is None:
        start = chess.Board()
        _START_PLANES = board_to_planes(start).astype(np.float16)
        fps: dict[str, np.ndarray] = {}
        for m in start.legal_moves:
            b = start.copy(stack=False)
            b.push(m)
            fps[m.uci()] = board_to_planes(b).astype(np.float16)
        _NEXT_FPS = fps
    return _START_PLANES, _NEXT_FPS


def summarize_first_moves(moves: list[str | None]) -> dict[str, Any]:
    """Summarize a list of UCI first moves (None / empty ignored)."""
    valid = [m for m in moves if m]
    if not valid:
        return _empty_stats()
    counts = Counter(valid)
    n = len(valid)
    ranked = counts.most_common(5)
    while len(ranked) < 5:
        ranked.append(("", 0))
    out: dict[str, Any] = {
        "n": n,
        "entropy": shannon(counts),
        "main_share": sum(counts[m] for m in MAIN) / n,
        "flank_share": sum(counts[m] for m in FLANK) / n,
        "counts": dict(counts),
    }
    for i, (uci, cnt) in enumerate(ranked, start=1):
        out[f"top{i}_uci"] = uci
        out[f"top{i}_share"] = cnt / n
    return out


def recover_first_moves_from_shard(path: str | Path) -> list[str | None]:
    """Recover White's first UCI move per startpos sample in an NPZ shard."""
    path = Path(path)
    if not path.is_file():
        return []
    start_planes, fps = _fingerprints()
    try:
        with np.load(path) as data:
            if "planes" not in data:
                return []
            planes = data["planes"]
            diffs = np.abs(
                planes.astype(np.float32) - start_planes.astype(np.float32)
            ).reshape(len(planes), -1).sum(axis=1)
            starts = np.where(diffs < 1e-3)[0]
            moves: list[str | None] = []
            for i in starts:
                if i + 1 >= len(planes):
                    moves.append(None)
                    continue
                nxt = planes[i + 1].astype(np.float32)
                hit = None
                for uci, fp in fps.items():
                    if np.abs(nxt - fp.astype(np.float32)).sum() < 1e-2:
                        hit = uci
                        break
                moves.append(hit)
            return moves
    except (OSError, ValueError, KeyError, EOFError):
        return []


def summarize_first_moves_from_shard(path: str | Path) -> dict[str, Any]:
    """Recover first moves from a sample shard and summarize diversity stats."""
    return summarize_first_moves(recover_first_moves_from_shard(path))
