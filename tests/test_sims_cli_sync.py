"""--sims must sync both train.sims_per_move and mcts.simulations."""

from engine.config import Config


def test_sims_assignment_keeps_train_and_mcts_aligned() -> None:
    """Mirrors engine.train CLI handling after --sims is provided."""
    cfg = Config()
    assert cfg.mcts.simulations == 100
    assert cfg.train.sims_per_move == 100

    args_sims = 150
    cfg.train.sims_per_move = args_sims
    cfg.mcts.simulations = args_sims

    assert cfg.train.sims_per_move == 150
    assert cfg.mcts.simulations == 150
