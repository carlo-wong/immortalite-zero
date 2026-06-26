"""Resolve Lightning AI sibling-folder layout (results/, syzygy345/)."""

from __future__ import annotations

import os
from dataclasses import dataclass

EXPECTED_RTBW = 145


@dataclass(frozen=True)
class LightningPaths:
    repo_dir: str
    ckpt_dir: str
    tb_dir: str


def resolve_paths() -> LightningPaths:
    """Repo is immortalite-zero/; results/ and syzygy345/ are its siblings."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, ".."))
    if os.path.isdir(os.path.join(candidate, "engine")):
        repo_dir = candidate
    else:
        cwd = os.path.abspath(os.getcwd())
        if os.path.isdir(os.path.join(cwd, "engine")):
            repo_dir = cwd
        elif os.path.isdir(os.path.join(cwd, "..", "engine")):
            repo_dir = os.path.abspath(os.path.join(cwd, ".."))
        else:
            raise RuntimeError(
                "Could not find repo root (expected engine/ package). "
                "Run from immortalite-zero/ or lightning-ai/."
            )

    parent = os.path.dirname(repo_dir)
    return LightningPaths(
        repo_dir=repo_dir,
        ckpt_dir=os.path.join(parent, "results"),
        tb_dir=os.path.join(parent, "syzygy345"),
    )


def ensure_ckpt_dir(paths: LightningPaths) -> None:
    os.makedirs(paths.ckpt_dir, exist_ok=True)


def validate_syzygy(tb_dir: str) -> int:
    if not os.path.isdir(tb_dir):
        raise RuntimeError(
            f"Syzygy folder missing: {tb_dir}\n"
            "Upload syzygy345/ as a sibling of the repo (145 .rtbw files)."
        )
    count = len([f for f in os.listdir(tb_dir) if f.endswith(".rtbw")])
    if count < EXPECTED_RTBW:
        raise RuntimeError(
            f"Incomplete syzygy345 upload ({count}/{EXPECTED_RTBW}).\n"
            "Locally: python scripts/download_syzygy345.py --out syzygy345"
        )
    return count
