"""FastAPI analysis backend for the web GUI.

Run:  python -m uvicorn server.app:app --reload --port 8000
Optional env var IMMORTALITE_ZERO_CHECKPOINT points at a trained checkpoint.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

import chess
from fastapi import FastAPI, HTTPException, Query
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
    pv_len: int = 8


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

    sims = req.simulations if req.simulations is not None else _cfg.mcts.simulations
    pv_len = max(1, min(int(req.pv_len), 24))
    t0 = time.perf_counter()
    analysis = _analyzer.analyze(
        board, multipv=req.multipv, simulations=sims, pv_len=pv_len,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "fen": analysis.fen,
        "eval_cp": analysis.eval_cp,
        "win_prob": analysis.win_prob,
        "best_move": analysis.best_move,
        "lines": analysis.lines,
        "elapsed_ms": elapsed_ms,
        "game_over": board.is_game_over(claim_draw=_cfg.mcts.claim_draw),
        "simulations": sims,
        "depth": sims,
    }


_EXPLORER_URLS = {
    "masters": "https://explorer.lichess.ovh/masters",
    "lichess": "https://explorer.lichess.ovh/lichess",
}

# fen -> {moves: {san: games}, openings: [(games, eco, name), ...]}
_local_book: dict[str, dict] | None = None


def _normalize_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen.strip()


def _load_local_book() -> dict[str, dict]:
    """Index prefix-free masters openings by FEN for offline explorer fallback."""
    global _local_book
    if _local_book is not None:
        return _local_book

    book: dict[str, dict] = {}
    csv_path = _repo_root / "tmp" / "masters_prefix_free_top64.csv"
    if not csv_path.is_file():
        _local_book = book
        return book

    import csv

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                games = int(row.get("games") or 0)
            except ValueError:
                continue
            pgn = (row.get("pgn") or "").strip()
            if not pgn or games <= 0:
                continue
            eco = (row.get("eco") or "").strip() or None
            name = (row.get("name") or "").strip() or None
            tokens = [
                t for t in pgn.replace("\n", " ").split()
                if t
                and not t.endswith(".")
                and t not in {"1-0", "0-1", "1/2-1/2", "*"}
                and not t.startswith("{")
            ]
            board = chess.Board()
            for san in tokens:
                key = _normalize_fen(board.fen())
                entry = book.setdefault(key, {"moves": {}, "openings": []})
                entry["moves"][san] = entry["moves"].get(san, 0) + games
                if eco or name:
                    entry["openings"].append((games, eco, name))
                try:
                    board.push_san(san)
                except ValueError:
                    break
            key = _normalize_fen(board.fen())
            entry = book.setdefault(key, {"moves": {}, "openings": []})
            if eco or name:
                entry["openings"].append((games, eco, name))

    _local_book = book
    return book


def _local_explorer(fen: str) -> dict:
    book = _load_local_book()
    entry = book.get(_normalize_fen(fen))
    if not entry:
        return {"opening": None, "white": 0, "draws": 0, "black": 0, "moves": []}

    openings = sorted(entry["openings"], key=lambda x: -x[0])
    opening = None
    if openings:
        _g, eco, name = openings[0]
        opening = {"eco": eco, "name": name}

    moves_out = []
    for san, games in sorted(entry["moves"].items(), key=lambda kv: -kv[1]):
        # Offline book has game counts only — put mass in draws for the W/D/B bar.
        moves_out.append({
            "san": san,
            "uci": None,
            "white": 0,
            "draws": games,
            "black": 0,
            "averageRating": None,
        })
    total = sum(m["draws"] for m in moves_out)
    return {
        "opening": opening,
        "white": 0,
        "draws": total,
        "black": 0,
        "moves": moves_out,
        "source": "local_masters",
    }


@app.get("/explorer")
def explorer(
    fen: str = Query(...),
    database: Literal["masters", "lichess"] = "masters",
) -> dict:
    """Proxy Lichess opening explorer; fall back to local masters book on failure."""
    params: dict[str, str] = {"fen": fen}
    if database == "lichess":
        params["speeds"] = "blitz,rapid,classical"
        params["ratings"] = "1600,1800,2000,2200,2500"
    url = f"{_EXPLORER_URLS[database]}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ImmortaliteZero/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return _local_explorer(fen)


# Serve the static analysis GUI at /app  ->  http://localhost:8000/app/
_web_dir = _repo_root / "web"
app.mount("/app", StaticFiles(directory=str(_web_dir), html=True), name="web")

_react_build = _web_dir / "react" / "dist"
_react_dir = _react_build if _react_build.is_dir() else _web_dir / "react"
if _react_dir.is_dir():
    app.mount("/app-react", StaticFiles(directory=str(_react_dir), html=True), name="react")
