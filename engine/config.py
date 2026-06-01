"""Central configuration. Tweak here rather than scattering magic numbers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetConfig:
    # "Light" by default: small enough to run on CPU, big enough to be sound.
    blocks: int = 6
    filters: int = 64


@dataclass
class MCTSConfig:
    simulations: int = 100          # raise for stronger/slower analysis
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3    # root exploration noise (self-play only)
    dirichlet_epsilon: float = 0.25
    # Gumbel root selection (sample-efficient; great for low-sim self-play).
    use_gumbel: bool = True
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_considered: int = 16     # top-k root actions considered each move
    # claim_draw=True catches threefold/50-move draws inside search, but is
    # a bit more expensive than plain terminal checks.
    claim_draw: bool = True


@dataclass
class BeautyConfig:
    """Beauty-bias selector. 'balanced': prefer beauty unless it clearly loses."""
    enabled: bool = True
    # Soundness window in win-probability units [0,1]. Moves within this of the
    # best move are considered "sound enough" to be eligible for beauty bias.
    soundness_window: float = 0.05
    # Relative weights of the beauty ingredients.
    w_sacrifice: float = 1.0
    w_attack: float = 0.7
    w_tactical: float = 0.5
    w_surprise: float = 0.6


@dataclass
class TrainConfig:
    games_per_iteration: int = 25
    train_steps_per_iteration: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50_000
    max_game_moves: int = 200
    # Reward shaping for "beautiful" play.
    draw_penalty: float = 0.10      # discourage dull draws -> more decisive play
    fast_mate_bonus: float = 0.0    # >0 rewards quicker checkmates
    # Progressive simulation: start cheap, ramp up (MiniZero trick).
    sims_start: int = 30
    sims_end: int = 100
    sims_ramp_iterations: int = 20
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    beauty: BeautyConfig = field(default_factory=BeautyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT = Config()
