"""Smoke test: MCTS picks legal moves, self-play runs, analysis produces output.

Uses a tiny untrained net so it runs fast on CPU. Verifies plumbing, not skill.
"""

import json
from dataclasses import asdict

import chess

from engine.analyze import Analyzer
from engine.config import Config
from engine.mcts import MCTS
from engine.network import ChessNet, NetEvaluator
from engine.selfplay import play_game


def main() -> None:
    cfg = Config()
    cfg.net.blocks = 2
    cfg.net.filters = 16
    cfg.mcts.simulations = 24

    evaluator = NetEvaluator(ChessNet(cfg.net))

    # 1. MCTS returns a legal move from the opening position.
    mcts = MCTS(evaluator, cfg.mcts)
    board = chess.Board()
    result = mcts.run(board, add_noise=True)
    assert result.best_move() in board.legal_moves
    print("MCTS best move:", result.best_move(), "| considered:", len(result.moves))

    # 2. A short self-play game completes and produces value-labelled samples.
    cfg.train.max_game_moves = 12
    samples = play_game(evaluator, cfg, simulations=16)
    assert len(samples) > 0
    assert all(s.value in (-1.0, 1.0) or abs(s.value) <= 1.0 for s in samples)
    print(f"self-play produced {len(samples)} samples; final value {samples[-1].value}")

    # 3. Analyzer yields eval, best vs beautiful, and candidate lines.
    analyzer = Analyzer(None, cfg)
    analysis = analyzer.analyze(board, multipv=3, simulations=24)
    out = asdict(analysis)
    assert out["best_move"] in [m.uci() for m in board.legal_moves]
    assert len(out["lines"]) == 3
    print("analysis sample:")
    print(json.dumps({k: out[k] for k in
                      ["eval_cp", "best_move", "beautiful_move", "beauty_cost_cp", "beauty"]},
                     indent=2))
    print("top line:", out["lines"][0])
    print("\nALL PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
