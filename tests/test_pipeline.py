"""Smoke test: MCTS picks legal moves, self-play runs, analysis produces output.

Uses a tiny untrained net so it runs fast on CPU. Verifies plumbing, not skill.
"""

import json
from dataclasses import asdict

import chess
import numpy as np

from engine.analyze import Analyzer
from engine.config import Config
from engine.mcts import MCTS
from engine.network import ChessNet, NetEvaluator
from engine.selfplay import play_game, play_games_batched


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

    # 1b. Generator-driven search matches run() for fixed RNG state.
    np.random.seed(7)
    run_result = mcts.run(board, simulations=24, add_noise=True)
    np.random.seed(7)
    gen = mcts.search_gen(board, simulations=24, add_noise=True)
    req = next(gen)
    while True:
        logits, value = evaluator.evaluate(req)
        try:
            req = gen.send((logits, value))
        except StopIteration as stop:
            gen_result = stop.value
            break
    assert np.array_equal(run_result.visits, gen_result.visits)
    assert np.allclose(run_result.q_values, gen_result.q_values)

    # 2. A short self-play game completes and produces value-labelled samples.
    cfg.train.max_game_moves = 12
    game = play_game(evaluator, cfg, simulations=16)
    assert len(game.samples) > 0
    assert all(s.value in (-1.0, 1.0) or abs(s.value) <= 1.0 for s in game.samples)
    print(f"self-play produced {len(game.samples)} samples; "
          f"termination={game.termination}; final value {game.samples[-1].value}")

    # 2b. Batched self-play (concurrency=1) matches sequential self-play exactly.
    np.random.seed(11)
    seq_games = [play_game(evaluator, cfg, simulations=16) for _ in range(2)]
    np.random.seed(11)
    batched_games = play_games_batched(
        evaluator, cfg, simulations=16, num_games=2, concurrency=1
    )
    assert len(seq_games) == len(batched_games)
    for seq, batched in zip(seq_games, batched_games):
        assert seq.termination == batched.termination
        assert len(seq.samples) == len(batched.samples)
        for s1, s2 in zip(seq.samples, batched.samples):
            assert s1.player == s2.player
            assert np.allclose(s1.planes, s2.planes)
            assert np.allclose(s1.policy, s2.policy)
            assert s1.value == s2.value

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
