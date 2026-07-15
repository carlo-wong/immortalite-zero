"""Masters opening book for strength gates (prefix-free top-64)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import chess

DEFAULT_MASTERS_OPENINGS_PATH = (
    Path(__file__).resolve().parent / "data" / "masters_prefix_free_top64.tsv"
)

_SAN_MOVE_RE = re.compile(r"[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?|O-O-O|O-O")


def _parse_pgn_sans(pgn: str) -> list[str]:
    """Extract SAN tokens from a numbered PGN fragment (no result / comments)."""
    clean = re.sub(r"\d+\.+", " ", pgn)
    clean = re.sub(r"[()#]", " ", clean)
    return [tok for tok in clean.split() if _SAN_MOVE_RE.fullmatch(tok)]


def pgn_to_uci(pgn: str) -> list[str]:
    """Convert a short PGN move list to UCI. Raises ValueError if illegal."""
    board = chess.Board()
    uci: list[str] = []
    for san in _parse_pgn_sans(pgn):
        try:
            move = board.parse_san(san)
        except ValueError as exc:
            raise ValueError(f"illegal SAN {san!r} in {pgn!r}: {exc}") from exc
        board.push(move)
        uci.append(move.uci())
    if not uci:
        raise ValueError(f"no moves parsed from {pgn!r}")
    return uci


def load_opening_book(path: str | Path | None = None) -> list[list[str]]:
    """Load prefix-free masters TSV → list of UCI move lists (one per opening)."""
    book_path = Path(path) if path is not None else DEFAULT_MASTERS_OPENINGS_PATH
    if not book_path.is_file():
        raise FileNotFoundError(f"opening book not found: {book_path}")

    openings: list[list[str]] = []
    with book_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "pgn" not in reader.fieldnames:
            raise ValueError(f"opening book missing 'pgn' column: {book_path}")
        for row in reader:
            pgn = (row.get("pgn") or "").strip()
            if not pgn:
                continue
            openings.append(pgn_to_uci(pgn))
    if not openings:
        raise ValueError(f"opening book empty: {book_path}")
    return openings


def load_default_gate_openings() -> list[list[str]]:
    """64 masters lines for gate_games=128 (each line × both colors)."""
    return load_opening_book(DEFAULT_MASTERS_OPENINGS_PATH)


def opening_for_game(openings: list[list[str]] | None, game_idx: int) -> list[str] | None:
    """Map game index → opening: opening_idx = game_idx // 2 (color-paired)."""
    if not openings:
        return None
    return openings[(game_idx // 2) % len(openings)]
