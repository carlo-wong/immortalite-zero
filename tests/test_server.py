"""Smoke tests for the FastAPI analysis server."""

from __future__ import annotations

import chess
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)
START = chess.STARTING_FEN


def test_health_returns_checkpoint_info() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "checkpoint" in data
    assert "trained" in data
    assert "encoding_version" in data
    assert "beauty_enabled" not in data


def test_analyze_start_position_schema() -> None:
    res = client.post("/analyze", json={"fen": START, "multipv": 5, "simulations": 8})
    assert res.status_code == 200
    data = res.json()
    assert data["fen"] == START
    assert "eval_cp" in data
    assert "best_move" in data
    assert "beautiful_move" not in data
    assert "beauty_cost_cp" not in data
    assert "elapsed_ms" in data
    assert data["game_over"] is False
    assert len(data["lines"]) == 5
    for line in data["lines"]:
        assert {"move", "eval_cp", "win_prob", "pv"} <= set(line.keys())


def test_analyze_invalid_fen() -> None:
    res = client.post("/analyze", json={"fen": "not-a-fen"})
    assert res.status_code == 400


def test_analyze_game_over() -> None:
    # Fool's mate finished position
    fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    res = client.post("/analyze", json={"fen": fen, "simulations": 4})
    assert res.status_code == 200
    data = res.json()
    assert data["game_over"] is True
    assert data["best_move"] is None


def test_root_redirects_to_app() -> None:
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (307, 308)
    assert res.headers["location"].endswith("/app/")


if __name__ == "__main__":
    test_health_returns_checkpoint_info()
    test_analyze_start_position_schema()
    test_analyze_invalid_fen()
    test_analyze_game_over()
    test_root_redirects_to_app()
    print("ALL SERVER CHECKS PASSED")
