"""Monte Carlo Tree Search.

Search uses classic PUCT with negamax value backup. For move selection and as
the training target we use the Gumbel AlphaZero "completed-Q" improved policy
(Danihelka et al., 2022), which is markedly more sample-efficient at the low
simulation counts we can afford on free hardware.

Full sequential-halving budget allocation is a worthwhile future refinement;
here we keep PUCT for the search itself and apply the Gumbel improvement at the
root, which captures much of the benefit with far less complexity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from .config import MCTSConfig
from .encoding import index_to_move, legal_move_indices, move_to_index
from .network import NetEvaluator


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class _Node:
    __slots__ = ("prior", "children", "N", "W")

    def __init__(self, prior: float):
        self.prior = prior
        self.children: dict[int, _Node] = {}
        self.N = 0
        self.W = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


@dataclass
class SearchResult:
    moves: list[chess.Move]
    indices: list[int]
    visits: np.ndarray
    q_values: np.ndarray          # from side-to-move perspective
    priors: np.ndarray
    root_value: float
    _root: _Node
    _board: chess.Board
    _cfg: MCTSConfig

    def visit_policy(self) -> np.ndarray:
        total = self.visits.sum()
        return self.visits / total if total > 0 else self.priors

    def improved_policy(self) -> np.ndarray:
        """Gumbel completed-Q improved policy over the considered moves."""
        max_n = self.visits.max() if self.visits.size else 0.0
        sigma = (self._cfg.gumbel_c_visit + max_n) * self._cfg.gumbel_c_scale
        logits = np.log(np.clip(self.priors, 1e-9, None)) + sigma * self.q_values
        return _softmax(logits)

    def best_move(self) -> chess.Move:
        return self.moves[int(np.argmax(self.visits))]

    def principal_variation(self, max_len: int = 8) -> list[chess.Move]:
        pv: list[chess.Move] = []
        board = self._board.copy()
        node = self._root
        for _ in range(max_len):
            if not node.children:
                break
            idx = max(node.children, key=lambda i: node.children[i].N)
            move = index_to_move(idx, board)
            if move is None:
                break
            pv.append(move)
            board.push(move)
            node = node.children[idx]
        return pv


class MCTS:
    def __init__(self, evaluator: NetEvaluator, cfg: MCTSConfig | None = None):
        self.evaluator = evaluator
        self.cfg = cfg or MCTSConfig()

    def _expand(self, node: _Node, board: chess.Board) -> float:
        """Evaluate a leaf, attach children, return value (side-to-move perspective)."""
        logits, value = self.evaluator.evaluate(board)
        mapping = legal_move_indices(board)
        if not mapping:
            return value
        idxs = list(mapping.keys())
        priors = _softmax(np.array([logits[i] for i in idxs], dtype=np.float32))
        for i, p in zip(idxs, priors):
            node.children[i] = _Node(float(p))
        return value

    def run(self, board: chess.Board, simulations: int | None = None,
            add_noise: bool = False) -> SearchResult:
        sims = simulations if simulations is not None else self.cfg.simulations
        root = _Node(0.0)
        root_value = self._expand(root, board)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(sims):
            node = root
            sim_board = board.copy()
            path = [root]

            while node.expanded and not sim_board.is_game_over():
                idx, child = self._select_child(node)
                move = index_to_move(idx, sim_board)
                if move is None:  # safety: should not happen post round-trip test
                    break
                sim_board.push(move)
                node = child
                path.append(node)

            if sim_board.is_game_over():
                value = self._terminal_value(sim_board)
            else:
                value = self._expand(node, sim_board)

            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value

        return self._collect(root, board, root_value)

    def _select_child(self, node: _Node) -> tuple[int, _Node]:
        c_puct = self.cfg.c_puct
        sqrt_n = math.sqrt(node.N)
        best_score = -float("inf")
        best = None
        for idx, child in node.children.items():
            q = -child.Q  # child value is from opponent's perspective
            u = c_puct * child.prior * sqrt_n / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best = (idx, child)
        return best

    def _add_dirichlet_noise(self, root: _Node) -> None:
        idxs = list(root.children.keys())
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(idxs))
        eps = self.cfg.dirichlet_epsilon
        for i, n in zip(idxs, noise):
            child = root.children[i]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        return 0.0       # stalemate / draw

    def _collect(self, root: _Node, board: chess.Board, root_value: float) -> SearchResult:
        moves, indices, visits, qs, priors = [], [], [], [], []
        for idx, child in root.children.items():
            move = index_to_move(idx, board)
            if move is None:
                continue
            moves.append(move)
            indices.append(idx)
            visits.append(child.N)
            qs.append(-child.Q)  # value of the move from root side-to-move perspective
            priors.append(child.prior)
        return SearchResult(
            moves=moves,
            indices=indices,
            visits=np.array(visits, dtype=np.float64),
            q_values=np.array(qs, dtype=np.float64),
            priors=np.array(priors, dtype=np.float64),
            root_value=root_value,
            _root=root,
            _board=board,
            _cfg=self.cfg,
        )
