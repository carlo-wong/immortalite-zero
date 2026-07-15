"""Tests for masters opening book loader and gate start moves."""
from __future__ import annotations

import chess
import pytest

from engine.openings import (
    DEFAULT_MASTERS_OPENINGS_PATH,
    load_default_gate_openings,
    load_opening_book,
    opening_for_game,
    pgn_to_uci,
)
from engine.selfplay import play_game_gen, GATE_OPENING_PLIES
from engine.config import Config
from engine.network import ChessNet
from engine.train import play_match


def test_default_masters_book_path_exists() -> None:
    assert DEFAULT_MASTERS_OPENINGS_PATH.is_file()


def test_load_default_gate_openings() -> None:
    openings = load_default_gate_openings()
    assert len(openings) == 64
    assert openings[0][:2] == ["e2e4", "c7c5"]  # Sicilian
    assert openings[1][:4] == ["e2e4", "e7e5", "g1f3", "b8c6"]  # Ruy setup
    # All lines legal and nonempty
    for line in openings:
        assert 1 <= len(line) <= GATE_OPENING_PLIES
        board = chess.Board()
        for uci in line:
            board.push_uci(uci)


def test_pgn_to_uci_rejects_illegal() -> None:
    with pytest.raises(ValueError):
        pgn_to_uci("1. e4 e5 2. Nf3 Ke2")


def test_opening_for_game_color_pairs() -> None:
    openings = [["e2e4", "e7e5"], ["d2d4", "d7d5"]]
    assert opening_for_game(openings, 0) == openings[0]
    assert opening_for_game(openings, 1) == openings[0]
    assert opening_for_game(openings, 2) == openings[1]
    assert opening_for_game(openings, 3) == openings[1]
    assert opening_for_game(None, 0) is None


def test_play_game_gen_applies_start_moves() -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.mcts.simulations = 2
    cfg.train.max_game_moves = 4
    start = ["e2e4", "e7e5", "g1f3"]
    gen = play_game_gen(
        cfg,
        simulations=2,
        add_noise=False,
        exploration_moves=0,
        start_moves=start,
    )
    # First yield should be from the post-book position (Black to move after Nf3? wait Nf3 is White's 2nd move, Black to move)
    req = next(gen)
    assert req.board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R")
    # Drain generator with dummy evals
    try:
        while True:
            logits = __import__("numpy").zeros(4672, dtype="float32")
            req = gen.send((logits, 0.0))
    except StopIteration as stop:
        game = stop.value
    assert game.moves[:3] == start


def test_play_match_uses_book_for_both_colors() -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.mcts.simulations = 2
    cfg.train.max_game_moves = 8
    cfg.train.selfplay_concurrency = 1
    net_a = ChessNet(cfg.net)
    net_b = ChessNet(cfg.net)
    net_a.eval()
    net_b.eval()
    book = [["e2e4", "c7c5", "g1f3"]]
    metrics = play_match(
        net_a,
        net_b,
        cfg,
        n_games=2,
        sims=2,
        device="cpu",
        exploration_moves=0,
        openings=book,
    )
    assert metrics["book_lines"] == 1
    assert len(metrics["openings"]) == 2
    for row in metrics["openings"]:
        uci = row["opening_uci"].split()
        assert uci[:3] == book[0]
    assert int(metrics["openings"][0]["a_is_white"]) == 1
    assert int(metrics["openings"][1]["a_is_white"]) == 0


def test_load_opening_book_custom_path() -> None:
    openings = load_opening_book(DEFAULT_MASTERS_OPENINGS_PATH)
    assert len(openings) == 64


def test_play_match_parallel_preserves_book_pairing(tmp_path) -> None:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.filters = 4
    cfg.mcts.simulations = 2
    cfg.train.max_game_moves = 6
    cfg.train.selfplay_concurrency = 2
    cfg.train.checkpoint_dir = str(tmp_path)
    net_a = ChessNet(cfg.net)
    net_b = ChessNet(cfg.net)
    net_a.eval()
    net_b.eval()
    book = load_default_gate_openings()[:2]
    metrics = play_match(
        net_a,
        net_b,
        cfg,
        n_games=4,
        sims=2,
        device="cpu",
        exploration_moves=0,
        openings=book,
        workers=2,
        concurrency=2,
    )
    rows = sorted(metrics["openings"], key=lambda r: int(r["game_idx"]))
    assert len(rows) == 4
    assert rows[0]["opening_uci"].split()[: len(book[0])] == book[0]
    assert rows[1]["opening_uci"].split()[: len(book[0])] == book[0]
    assert rows[2]["opening_uci"].split()[: len(book[1])] == book[1]
    assert rows[3]["opening_uci"].split()[: len(book[1])] == book[1]
