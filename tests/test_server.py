"""Smoke tests for the FastAPI analysis server."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

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
    assert data["simulations"] == 8
    assert data["depth"] == 8
    assert len(data["lines"]) == 5
    visit_sum = 0
    for line in data["lines"]:
        assert {"move", "eval_cp", "win_prob", "pv", "visits", "visit_pct"} <= set(line.keys())
        assert isinstance(line["visits"], int)
        assert line["visits"] >= 0
        assert isinstance(line["visit_pct"], (int, float))
        assert 0.0 <= float(line["visit_pct"]) <= 100.0
        visit_sum += line["visits"]
    assert visit_sum > 0
    assert abs(sum(float(l["visit_pct"]) for l in data["lines"]) - 100.0) < 1.0 or (
        # MultiPV may be a subset of root children; pct is over all root visits.
        sum(float(l["visit_pct"]) for l in data["lines"]) <= 100.0 + 1e-6
    )


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


def test_explorer_proxies_masters_json() -> None:
    payload = {"white": 1, "draws": 0, "black": 0, "moves": []}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch("server.app.urllib.request.urlopen", return_value=_Resp()) as mock_open:
        res = client.get("/explorer", params={"fen": START, "database": "masters"})
    assert res.status_code == 200
    assert res.json() == payload
    called_url = mock_open.call_args[0][0].full_url
    assert called_url.startswith("https://explorer.lichess.ovh/masters?")
    assert "fen=" in called_url


def test_explorer_local_fallback_on_upstream_error() -> None:
    with patch(
        "server.app.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            url="https://explorer.lichess.ovh/masters",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        ),
    ):
        res = client.get("/explorer", params={"fen": START, "database": "masters"})
    assert res.status_code == 200
    data = res.json()
    assert "moves" in data
    assert data.get("source") == "local_masters"
    sans = {m["san"] for m in data["moves"]}
    assert "e4" in sans or "d4" in sans


def test_explorer_rejects_bad_database() -> None:
    res = client.get("/explorer", params={"fen": START, "database": "stockfish"})
    assert res.status_code == 422


if __name__ == "__main__":
    test_health_returns_checkpoint_info()
    test_analyze_start_position_schema()
    test_analyze_invalid_fen()
    test_analyze_game_over()
    test_root_redirects_to_app()
    test_explorer_proxies_masters_json()
    test_explorer_local_fallback_on_upstream_error()
    test_explorer_rejects_bad_database()
    print("ALL SERVER CHECKS PASSED")
