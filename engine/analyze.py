"""High-level analysis API used by the UCI wrapper and the web server."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import chess
import numpy as np
import torch

from .config import Config, NetConfig
from .encoding import ENCODING_VERSION, legal_move_indices
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
        ckpt_encoding_version = 1
        if isinstance(state, dict):
            ckpt_encoding_version = int(state.get("encoding_version", 1))
        if ckpt_encoding_version != ENCODING_VERSION:
            raise ValueError(
                f"checkpoint encoding version {ckpt_encoding_version} does not match "
                f"current encoding version {ENCODING_VERSION}; use a checkpoint "
                f"trained with the current encoding"
            )
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
    visits: int
    visit_pct: float


@dataclass
class Analysis:
    fen: str
    eval_cp: int               # from side-to-move perspective
    win_prob: float            # side-to-move perspective, [0, 1]
    best_move: str | None
    lines: list[dict]          # top-N candidate lines (MultiPV)


class Analyzer:
    def __init__(self, checkpoint_path: str | None = None, cfg: Config | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or Config()
        self.evaluator = load_evaluator(checkpoint_path, self.cfg, device)
        self.mcts = MCTS(self.evaluator, self.cfg.mcts)

    def analyze(self, board: chess.Board, multipv: int = 5,
                simulations: int | None = None, pv_len: int = 8) -> Analysis:
        if board.is_game_over(claim_draw=self.cfg.mcts.claim_draw):
            return Analysis(board.fen(), value_to_cp(self.mcts._terminal_value(board)),
                            0.0, None, [])

        result = self.mcts.run(board, simulations=simulations, add_noise=False)
        order = np.argsort(-result.visits)  # most-visited first
        total_visits = float(result.visits.sum())

        lines: list[Line] = []
        for rank in order[:multipv]:
            move = result.moves[rank]
            q = float(result.q_values[rank])
            n = int(result.visits[rank])
            visit_pct = (n / total_visits * 100.0) if total_visits > 0 else 0.0
            pv = self._pv_for_move(board, result, move, max_len=pv_len)
            lines.append(Line(
                move.uci(), value_to_cp(q), (q + 1) / 2,
                [m.uci() for m in pv], n, visit_pct,
            ))

        best_q = float(result.q_values.max())
        best_move = result.best_move()

        return Analysis(
            fen=board.fen(),
            eval_cp=value_to_cp(best_q),
            win_prob=(best_q + 1) / 2,
            best_move=best_move.uci(),
            lines=[asdict(line) for line in lines],
        )

    def _pv_for_move(self, board: chess.Board, result: SearchResult,
                     move: chess.Move, max_len: int = 8) -> list[chess.Move]:
        pv = result.principal_variation_from(move, max_len)
        if len(pv) >= max_len:
            return pv
        # Tree may be shallow for side lines; extend with greedy net policy.
        b = board.copy()
        for m in pv:
            b.push(m)
        claim_draw = self.cfg.mcts.claim_draw
        while len(pv) < max_len:
            if b.is_game_over(claim_draw=claim_draw):
                break
            mapping = legal_move_indices(b)
            if not mapping:
                break
            logits, _ = self.evaluator.evaluate(b)
            best_idx = max(mapping, key=lambda i: float(logits[i]))
            next_move = mapping[best_idx]
            pv.append(next_move)
            b.push(next_move)
        return pv
