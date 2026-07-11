#!/usr/bin/env python3
"""Manual strength gate for Lightning AI (notebook cell 6 as a script).

Edit CHECKPOINT_A / CHECKPOINT_B below, then run from the terminal:
  cd immortalite-zero
  python lightning-ai/run_gate.py
"""

from __future__ import annotations

import os
import sys

# --- edit checkpoints and match settings here (matches run_train.py gate_* defaults) ---
# Use an int (iteration number) or the string "latest".
CHECKPOINT_A: int | str = 20
CHECKPOINT_B: int | str = 0

GATE_GAMES = 128
GATE_SIMS = 100
GATE_WORKERS = 4
GATE_CONCURRENCY = 128
GATE_EXPLORATION_MOVES = 20
DRAW_PENALTY = 1 / 3


def _resolve_checkpoint(ckpt_dir: str, ref: int | str) -> tuple[str, str]:
    if ref == "latest":
        return os.path.join(ckpt_dir, "latest.pt"), "Latest"
    iteration = int(ref)
    return os.path.join(ckpt_dir, f"ckpt_iter_{iteration:04d}.pt"), f"Iter {iteration}"


def _checkpoint_iteration(checkpoint_ref: int | str, state: dict) -> int:
    if checkpoint_ref == "latest":
        return int(state.get("iteration", -1)) if isinstance(state, dict) else -1
    return int(checkpoint_ref)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from paths import resolve_paths, validate_syzygy

    paths = resolve_paths()
    validate_syzygy(paths.tb_dir)
    os.chdir(paths.repo_dir)
    if paths.repo_dir not in sys.path:
        sys.path.insert(0, paths.repo_dir)

    import chess.syzygy
    import torch
    from engine.config import Config, NetConfig
    from engine.encoding import ENCODING_VERSION
    from engine.network import ChessNet
    from engine.sprt import ALPHA, BETA, ELO0, ELO1
    from engine.train import _elo_ci_verdict, _load_matching_state_dict, _log_gate_metrics, play_match

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    cfg.train.draw_penalty = DRAW_PENALTY
    cfg.train.syzygy_path = paths.tb_dir
    cfg.train.checkpoint_dir = paths.ckpt_dir
    tablebase = chess.syzygy.open_tablebase(paths.tb_dir)

    path_a, label_a = _resolve_checkpoint(paths.ckpt_dir, CHECKPOINT_A)
    path_b, label_b = _resolve_checkpoint(paths.ckpt_dir, CHECKPOINT_B)
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise FileNotFoundError(
            f"Missing checkpoint(s):\n  A ({label_a}): {path_a}\n  B ({label_b}): {path_b}"
        )

    def load_gate_net(path: str) -> tuple[ChessNet, dict]:
        state = torch.load(path, map_location=device)
        enc = int(state.get("encoding_version", 1)) if isinstance(state, dict) else 1
        if enc != ENCODING_VERSION:
            raise ValueError(f"{os.path.basename(path)}: encoding {enc} != {ENCODING_VERSION}")
        net_cfg = Config().net
        if isinstance(state, dict) and "net" in state:
            net_cfg = NetConfig(**state["net"])
        net = ChessNet(net_cfg).to(device)
        model = state["model"] if isinstance(state, dict) and "model" in state else state
        _load_matching_state_dict(net, model, label=f"gate load {os.path.basename(path)}", verbose=False)
        net.eval()
        return net, state

    print(f"Loading A ({label_a}): {path_a}")
    net_a, state_a = load_gate_net(path_a)
    print(f"Loading B ({label_b}): {path_b}")
    net_b, state_b = load_gate_net(path_b)
    print(
        f"\nMatch: {label_a} vs {label_b} "
        f"(SPRT cap {GATE_GAMES} games, {GATE_SIMS} sims, workers={GATE_WORKERS}, "
        f"elo0={ELO0}, elo1={ELO1})..."
    )

    metrics = play_match(
        net_a, net_b, cfg,
        n_games=GATE_GAMES,
        sims=GATE_SIMS,
        device=device,
        exploration_moves=GATE_EXPLORATION_MOVES,
        tablebase=tablebase,
        sprt=True,
        sprt_elo0=ELO0,
        sprt_elo1=ELO1,
        sprt_alpha=ALPHA,
        sprt_beta=BETA,
        workers=GATE_WORKERS,
        concurrency=GATE_CONCURRENCY,
    )
    tablebase.close()

    iter_a = _checkpoint_iteration(CHECKPOINT_A, state_a)
    iter_b = _checkpoint_iteration(CHECKPOINT_B, state_b)
    _log_gate_metrics(paths.ckpt_dir, iter_a, iter_b, metrics, GATE_GAMES)

    winrate = metrics["winrate"]
    wins = metrics["wins_as_white"] + metrics["wins_as_black"]
    losses = metrics["losses_as_white"] + metrics["losses_as_black"]
    draws = metrics["draws_as_white"] + metrics["draws_as_black"]
    wdl = f"+{wins} ={draws} -{losses}"
    ci_verdict = _elo_ci_verdict(float(metrics["elo_lower"]), float(metrics["elo_upper"]))
    games_played = int(metrics["games_played"])

    print("\n" + "=" * 40)
    print("MATCH COMPLETED")
    print(f"{label_a} score vs {label_b}: {winrate:.3f} [{wdl}] ({games_played} games)")
    print(f"  As White: W {metrics['wins_as_white']} L {metrics['losses_as_white']} D {metrics['draws_as_white']}")
    print(f"  As Black: W {metrics['wins_as_black']} L {metrics['losses_as_black']} D {metrics['draws_as_black']}")
    print(f"  Mean game length: {metrics['mean_game_len']:.1f} plies")
    print(f"  Terminations: {metrics['terminations']}")
    print(
        f"  Verdict: {ci_verdict} "
        f"Elo {metrics['elo']:+.1f} "
        f"[95% CI {metrics['elo_lower']:+.1f}, {metrics['elo_upper']:+.1f}] "
        f"LOS {metrics['los'] * 100:.1f}%"
    )
    print(f"  Logged to: {os.path.join(paths.ckpt_dir, 'metrics_gates.csv')}")
    print(f"Result: {label_a} {ci_verdict} vs {label_b}")
    print("=" * 40)


if __name__ == "__main__":
    main()
