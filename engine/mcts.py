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
from typing import Generator

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
    __slots__ = ("prior", "children", "N", "W", "terminal_checked", "is_terminal", "terminal_value", "move")

    def __init__(self, prior: float, move: chess.Move | None = None):
        self.prior = prior
        self.move = move
        self.children: dict[int, _Node] = {}
        self.N = 0
        self.W = 0.0
        self.terminal_checked = False
        self.is_terminal = False
        self.terminal_value = 0.0

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
    clean_priors: np.ndarray
    root_value: float
    _root: _Node
    _board: chess.Board
    _cfg: MCTSConfig

    def visit_policy(self) -> np.ndarray:
        total = self.visits.sum()
        return self.visits / total if total > 0 else self.clean_priors

    @property
    def searched_root_q(self) -> float:
        total = float(self.visits.sum())
        if total > 0:
            return float(np.dot(self.visits, self.q_values) / total)
        if self._root.N > 0:
            return float(self._root.Q)
        return float(self.root_value)

    def improved_policy(self) -> np.ndarray:
        """Gumbel completed-Q improved policy over the considered moves."""
        max_n = self.visits.max() if self.visits.size else 0.0
        sigma = (self._cfg.gumbel_c_visit + max_n) * self._cfg.gumbel_c_scale
        q = self.q_values
        q_span = float(q.max() - q.min())
        q_norm = (q - q.min()) / q_span if q_span > 0 else np.zeros_like(q)
        logits = np.log(np.clip(self.clean_priors, 1e-9, None)) + sigma * q_norm
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

    def principal_variation_from(self, first: chess.Move, max_len: int = 8) -> list[chess.Move]:
        """PV starting with ``first``, then following max-N children (like ``principal_variation``)."""
        if max_len < 1:
            return []
        board = self._board.copy()
        idx = move_to_index(first, board)
        if idx not in self._root.children:
            return [first]
        pv: list[chess.Move] = [first]
        board.push(first)
        node = self._root.children[idx]
        for _ in range(max_len - 1):
            if not node.children:
                break
            child_idx = max(node.children, key=lambda i: node.children[i].N)
            move = index_to_move(child_idx, board)
            if move is None:
                break
            pv.append(move)
            board.push(move)
            node = node.children[child_idx]
        return pv


class MCTS:
    def __init__(self, evaluator: NetEvaluator | None, cfg: MCTSConfig | None = None):
        self.evaluator = evaluator
        self.cfg = cfg or MCTSConfig()

    def _expand_from_eval(self, node: _Node, board: chess.Board,
                          logits: np.ndarray, value: float) -> float:
        """Attach children from evaluator outputs; return side-to-move value."""
        mapping = legal_move_indices(board)
        if not mapping:
            return float(value)
        idxs = list(mapping.keys())
        priors = _softmax(np.take(logits, idxs).astype(np.float32, copy=False))
        for i, p in zip(idxs, priors):
            node.children[i] = _Node(float(p), move=mapping[i])
        return float(value)

    def search_gen(self, board: chess.Board, simulations: int | None = None,
                   add_noise: bool = False
                   ) -> Generator[chess.Board, tuple[np.ndarray, float], SearchResult]:
        sims = simulations if simulations is not None else self.cfg.simulations
        root_turn = board.turn
        root = _Node(0.0)
        root_terminal, root_terminal_value = self._terminal_eval(root, board, root_turn)
        if root_terminal:
            return self._collect(root, board, root_terminal_value, {})

        logits, value = yield board
        root_value = self._expand_from_eval(root, board, logits, value)
        root_clean_priors = {idx: child.prior for idx, child in root.children.items()}

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(sims):
            node = root
            path = [root]
            depth = 0

            # Root is already known non-terminal (checked before the sims loop).
            # Keep terminal_value from the leaf check so we do not re-call
            # _terminal_eval on the same (node, board) after the while.
            is_terminal = False
            terminal_value = 0.0
            while node.expanded and not is_terminal:
                idx, child = self._select_child(node)
                move = child.move
                if move is None:  # safety: should not happen post round-trip test
                    move = index_to_move(idx, board)
                if move is None:
                    break
                board.push(move)
                depth += 1
                node = child
                path.append(node)
                is_terminal, terminal_value = self._terminal_eval(node, board, root_turn)

            if is_terminal:
                value = terminal_value
            else:
                logits, value = yield board
                value = self._expand_from_eval(node, board, logits, value)

            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value

            for _ in range(depth):
                board.pop()

        return self._collect(root, board, root_value, root_clean_priors)

    def run(self, board: chess.Board, simulations: int | None = None,
            add_noise: bool = False) -> SearchResult:
        if self.evaluator is None:
            raise ValueError("MCTS.run requires an evaluator")

        gen = self.search_gen(board, simulations=simulations, add_noise=add_noise)
        try:
            req = next(gen)
        except StopIteration as stop:
            return stop.value

        while True:
            logits, value = self.evaluator.evaluate(req)
            try:
                req = gen.send((logits, value))
            except StopIteration as stop:
                return stop.value

    def _select_child(self, node: _Node) -> tuple[int, _Node]:
        c_puct = self.cfg.c_puct
        sqrt_n = math.sqrt(node.N)
        best_score = -float("inf")
        best_idx = None
        best_child = None
        for idx, child in node.children.items():
            child_n = child.N
            q = -child.Q  # child value is from opponent's perspective
            u = c_puct * child.prior * sqrt_n / (1 + child_n)
            score = q + u
            if score > best_score:
                best_score = score
                best_idx = idx
                best_child = child
        assert best_idx is not None and best_child is not None
        return best_idx, best_child

    def _add_dirichlet_noise(self, root: _Node) -> None:
        idxs = list(root.children.keys())
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(idxs))
        eps = self.cfg.dirichlet_epsilon
        for i, n in zip(idxs, noise):
            child = root.children[i]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _is_terminal(self, board: chess.Board) -> bool:
        # claim_draw=True is costlier per node, but keeps search aligned with
        # self-play and avoids overvaluing claimable repetition/50-move draws.
        return board.is_game_over(claim_draw=self.cfg.claim_draw)

    def _value_from_outcome(
        self,
        outcome: chess.Outcome,
        board: chess.Board,
        root_turn: chess.Color,
    ) -> float:
        if outcome.termination == chess.Termination.CHECKMATE:
            return -1.0  # side to move has been mated
        contempt = float(self.cfg.draw_contempt)
        # Root-relative contempt: regardless of simulation depth parity, a draw
        # backs up as a small negative value for the root side to move.
        return -contempt if board.turn == root_turn else contempt

    def _terminal_eval(self, node: _Node, board: chess.Board, root_turn: chess.Color
                       ) -> tuple[bool, float]:
        if node.terminal_checked:
            return node.is_terminal, node.terminal_value
        # Single board.outcome() call — outcome() is None iff not terminal.
        outcome = board.outcome(claim_draw=self.cfg.claim_draw)
        node.is_terminal = outcome is not None
        if outcome is not None:
            node.terminal_value = self._value_from_outcome(outcome, board, root_turn)
        else:
            node.terminal_value = 0.0
        node.terminal_checked = True
        return node.is_terminal, node.terminal_value

    def _terminal_value(self, board: chess.Board,
                        root_turn: chess.Color | None = None) -> float:
        if root_turn is None:
            root_turn = board.turn
        outcome = board.outcome(claim_draw=self.cfg.claim_draw)
        if outcome is None:
            return 0.0
        return self._value_from_outcome(outcome, board, root_turn)

    def _collect(self, root: _Node, board: chess.Board, root_value: float,
                 clean_priors: dict[int, float]) -> SearchResult:
        moves, indices, visits, qs, priors, clean = [], [], [], [], [], []
        for idx, child in root.children.items():
            move = child.move
            if move is None:
                continue
            moves.append(move)
            indices.append(idx)
            visits.append(child.N)
            # Completed-Q: for unvisited moves, fall back to the root value estimate
            # instead of treating them as exact draws (0.0).
            q = -child.Q if child.N > 0 else root_value
            qs.append(q)
            priors.append(child.prior)
            clean.append(clean_priors.get(idx, child.prior))
        return SearchResult(
            moves=moves,
            indices=indices,
            visits=np.array(visits, dtype=np.float64),
            q_values=np.array(qs, dtype=np.float64),
            priors=np.array(priors, dtype=np.float64),
            clean_priors=np.array(clean, dtype=np.float64),
            root_value=root_value,
            _root=root,
            _board=board,
            _cfg=self.cfg,
        )
