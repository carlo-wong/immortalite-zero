"""FastAPI analysis backend for the web GUI.

Run:  python -m uvicorn server.app:app --reload --port 8000
Optional env var IMMORTALITE_ZERO_CHECKPOINT points at a trained checkpoint.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.analyze import Analyzer
from engine.config import Config

app = FastAPI(title="Immortalite Zero Analysis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = Config()
# Use IMMORTALITE_ZERO_CHECKPOINT if set, otherwise fall back to the latest
# checkpoint under results/ so a plain `uvicorn server.app:app` run picks up
# trained weights automatically.
_repo_root = pathlib.Path(__file__).resolve().parent.parent
_default_checkpoint_candidates = (
    _repo_root / "checkpoints" / "latest.pt",
    _repo_root / "results" / "immortalite_checkpoints_v3" / "latest.pt",
    _repo_root / "results" / "immortalite_zero_checkpoints" / "latest.pt",
    _repo_root / "results" / "immortalite_checkpoints_v2" / "latest.pt",
)
_default_checkpoint = next(
    (str(path) for path in _default_checkpoint_candidates if path.exists()),
    str(_default_checkpoint_candidates[0]),
)
_checkpoint = os.environ.get("IMMORTALITE_ZERO_CHECKPOINT") or (
    _default_checkpoint if os.path.exists(_default_checkpoint) else None
)
_analyzer = Analyzer(_checkpoint, _cfg)


class AnalyzeRequest(BaseModel):
    fen: str
    multipv: int = 3
    simulations: int | None = None
    beauty: bool = True


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "checkpoint": _checkpoint or "untrained"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid FEN")

    _analyzer.cfg.beauty.enabled = req.beauty
    analysis = _analyzer.analyze(board, multipv=req.multipv, simulations=req.simulations)
    return asdict(analysis)


# Serve the static analysis GUI at /app  ->  http://localhost:8000/app/
_web_dir = pathlib.Path(__file__).resolve().parent.parent / "web"
app.mount("/app", StaticFiles(directory=str(_web_dir), html=True), name="web")
