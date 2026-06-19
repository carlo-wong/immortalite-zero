"""FastAPI analysis backend for the web GUI.

Run:  python -m uvicorn server.app:app --reload --port 8000
Optional env var IMMORTALITE_ZERO_CHECKPOINT points at a trained checkpoint.
"""

from __future__ import annotations

import os
import pathlib
import time

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.analyze import Analyzer
from engine.config import Config
from engine.encoding import ENCODING_VERSION

app = FastAPI(title="Immortalite Zero Analysis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = Config()
_repo_root = pathlib.Path(__file__).resolve().parent.parent
_default_checkpoint_candidates = (
    _repo_root / "results" / "latest.pt",
    _repo_root / "checkpoints" / "latest.pt",
    _repo_root / "results" / "immortalite_zero_checkpoints" / "latest.pt",
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
    multipv: int = 5
    simulations: int | None = None


def _checkpoint_status() -> dict:
    trained = bool(_checkpoint and os.path.exists(_checkpoint))
    return {
        "checkpoint": _checkpoint if trained else "untrained",
        "trained": trained,
        "encoding_version": ENCODING_VERSION,
    }


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **_checkpoint_status()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid FEN")

    t0 = time.perf_counter()
    analysis = _analyzer.analyze(board, multipv=req.multipv, simulations=req.simulations)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "fen": analysis.fen,
        "eval_cp": analysis.eval_cp,
        "win_prob": analysis.win_prob,
        "best_move": analysis.best_move,
        "lines": analysis.lines,
        "elapsed_ms": elapsed_ms,
        "game_over": board.is_game_over(claim_draw=_cfg.mcts.claim_draw),
    }


# Serve the static analysis GUI at /app  ->  http://localhost:8000/app/
_web_dir = _repo_root / "web"
app.mount("/app", StaticFiles(directory=str(_web_dir), html=True), name="web")

_react_build = _web_dir / "react" / "dist"
_react_dir = _react_build if _react_build.is_dir() else _web_dir / "react"
if _react_dir.is_dir():
    app.mount("/app-react", StaticFiles(directory=str(_react_dir), html=True), name="react")
