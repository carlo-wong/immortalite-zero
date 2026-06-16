"""Minimal UCI front-end so the engine works in our GUI and external ones
(Arena, Cutechess, Lichess local engine, ...).

Run:  python -m uci.uci_engine [path/to/checkpoint.pt]

Custom options:
  setoption name Simulations value 400   ; search budget per move
  setoption name Beauty value true        ; play the beautiful move vs the best
  setoption name MultiPV value 3
"""

from __future__ import annotations

import sys

import chess

from engine.analyze import Analyzer
from engine.config import Config


class UCIEngine:
    def __init__(self, checkpoint: str | None):
        self.cfg = Config()
        self.checkpoint = checkpoint
        self.board = chess.Board()
        self.analyzer = Analyzer(checkpoint, self.cfg)
        self.beauty = True
        self.multipv = 1

    def run(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if line == "uci":
                self._id()
            elif line == "isready":
                self._send("readyok")
            elif line == "ucinewgame":
                self.board = chess.Board()
            elif line.startswith("setoption"):
                self._set_option(line)
            elif line.startswith("position"):
                self._set_position(line)
            elif line.startswith("go"):
                self._go()
            elif line in ("quit", "exit"):
                break

    def _id(self) -> None:
        self._send("id name Immortalite Zero")
        self._send("id author self-play")
        self._send("option name Simulations type spin default 100 min 1 max 100000")
        self._send("option name Beauty type check default true")
        self._send("option name MultiPV type spin default 1 min 1 max 5")
        self._send("uciok")

    def _set_option(self, line: str) -> None:
        parts = line.split()
        if "name" not in parts or "value" not in parts:
            return
        name = parts[parts.index("name") + 1].lower()
        value = parts[parts.index("value") + 1]
        if name == "simulations":
            self.cfg.mcts.simulations = int(value)
        elif name == "beauty":
            self.beauty = value.lower() == "true"
            self.cfg.beauty.enabled = self.beauty
        elif name == "multipv":
            self.multipv = int(value)

    def _set_position(self, line: str) -> None:
        tokens = line.split()
        if "startpos" in tokens:
            self.board = chess.Board()
            moves_idx = tokens.index("startpos") + 1
        elif "fen" in tokens:
            fen_idx = tokens.index("fen") + 1
            fen = " ".join(tokens[fen_idx:fen_idx + 6])
            self.board = chess.Board(fen)
            moves_idx = fen_idx + 6
        else:
            return
        if moves_idx < len(tokens) and tokens[moves_idx] == "moves":
            for mv in tokens[moves_idx + 1:]:
                self.board.push_uci(mv)

    def _go(self) -> None:
        analysis = self.analyzer.analyze(self.board, multipv=max(1, self.multipv))
        for i, line in enumerate(analysis.lines, start=1):
            pv = " ".join(line["pv"])
            self._send(f"info multipv {i} depth 1 score cp {line['eval_cp']} pv {pv}")
        move = analysis.beautiful_move if self.beauty else analysis.best_move
        move = move or analysis.best_move
        if move is None:  # no legal move
            self._send("bestmove 0000")
        else:
            self._send(f"bestmove {move}")

    @staticmethod
    def _send(msg: str) -> None:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    UCIEngine(ckpt).run()
