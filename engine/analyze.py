"""High-level analysis API used by the UCI wrapper and the web server."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import chess
import numpy as np
import torch

from .beauty import compute_beauty, select_beautiful_move
from .config import Config, NetConfig
from .mcts import MCTS, SearchResult
from .network import ChessNet, NetEvaluator


def value_to_cp(q: float) -> int:
    """Map a value in [-1, 1] to centipawns (Leela-style logistic)."""
    q = max(-0.999, min(0.999, q))
    return int(round(111.7 * math.tan(1.5620688421 * q)))


def load_evaluator(checkpoint_path: str | None, cfg: Config, device: str = "cpu") -> NetEvaluator:
    net_cfg = cfg.net
    state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "net" in state:
            net_cfg = NetConfig(**state["net"])  # rebuild matching architecture
    net = ChessNet(net_cfg)
    if state is not None:
        model_state = state["model"] if "model" in state else state
        net_state = net.state_dict()
        matched = {
            k: v for k, v in model_state.items()
            if k in net_state and net_state[k].shape == v.shape
        }
        net.load_state_dict(matched, strict=False)
    return NetEvaluator(net, device=device)


@dataclass
class Line:
    move: str
    eval_cp: int
    win_prob: float
    pv: list[str]


@dataclass
class Analysis:
    fen: str
    eval_cp: int               # from side-to-move perspective
    win_prob: float            # side-to-move perspective, [0, 1]
    best_move: str | None
    beautiful_move: str | None
    beauty_cost_cp: int        # eval given up by playing the beautiful move
    beauty: dict | None
    lines: list[dict]          # top-N candidate lines (MultiPV)


class Analyzer:
    def __init__(self, checkpoint_path: str | None = None, cfg: Config | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or Config()
        self.evaluator = load_evaluator(checkpoint_path, self.cfg, device)
        self.mcts = MCTS(self.evaluator, self.cfg.mcts)

    def analyze(self, board: chess.Board, multipv: int = 3,
                simulations: int | None = None) -> Analysis:
        if board.is_game_over(claim_draw=self.cfg.mcts.claim_draw):
            return Analysis(board.fen(), value_to_cp(self.mcts._terminal_value(board)),
                            0.0, None, None, 0, None, [])

        result = self.mcts.run(board, simulations=simulations, add_noise=False)
        order = np.argsort(-result.visits)  # most-visited first

        lines: list[Line] = []
        for rank in order[:multipv]:
            move = result.moves[rank]
            q = float(result.q_values[rank])
            pv = self._pv_for_move(board, result, move)
            lines.append(Line(move.uci(), value_to_cp(q), (q + 1) / 2, [m.uci() for m in pv]))

        best_q = float(result.q_values.max())
        best_move = result.best_move()

        beautiful_move, breakdown = select_beautiful_move(board, result, self.cfg.beauty)
        # cost = eval difference between best and beautiful choice
        beautiful_idx = result.moves.index(beautiful_move)
        beautiful_q = float(result.q_values[beautiful_idx])
        beauty_cost = value_to_cp(best_q) - value_to_cp(beautiful_q)

        return Analysis(
            fen=board.fen(),
            eval_cp=value_to_cp(best_q),
            win_prob=(best_q + 1) / 2,
            best_move=best_move.uci(),
            beautiful_move=beautiful_move.uci(),
            beauty_cost_cp=max(0, beauty_cost),
            beauty=asdict(breakdown) if breakdown else None,
            lines=[asdict(line) for line in lines],
        )

    def _pv_for_move(self, board: chess.Board, result: SearchResult,
                     move: chess.Move, max_len: int = 8) -> list[chess.Move]:
        # The first move is fixed; the rest follows the search's principal line.
        full_pv = result.principal_variation(max_len)
        if full_pv and full_pv[0] == move:
            return full_pv
        return [move]
