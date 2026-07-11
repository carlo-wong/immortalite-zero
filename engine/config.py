"""Central configuration. Tweak here rather than scattering magic numbers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetConfig:
    # "Light" by default: small enough to run on CPU, big enough to be sound.
    blocks: int = 6
    filters: int = 64
    value_bins: int = 51


@dataclass
class MCTSConfig:
    simulations: int = 100          # raise for stronger/slower analysis
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3    # root exploration noise (self-play only)
    dirichlet_epsilon: float = 0.25
    # Gumbel completed-Q policy-target scaling.
    # gumbel_c_scale matches mctx value_scale=0.1: sigma = 0.1 * (c_visit + max_N).
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 0.1
    # Search contempt during self-play/analysis: draw terminals penalized in search.
    # Gates override this to 0 for normal W/D/L strength metrics.
    draw_contempt: float = 1 / 3
    # claim_draw=True catches threefold/50-move draws inside search, but is
    # a bit more expensive than plain terminal checks.
    claim_draw: bool = True


@dataclass
class BeautyConfig:
    """Beauty-bias selector. 'balanced': prefer beauty unless it clearly loses."""
    enabled: bool = False
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
    selfplay_concurrency: int = 32
    train_steps_per_iteration: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    lr_min: float = 2e-4
    lr_warmup_iters: int = 2
    lr_total_iters: int = 100
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50_000
    replay_window: int = 50_000
    max_game_moves: int = 200
    syzygy_path: str | None = None
    tb_max_pieces: int = 5
    # Football 3-1-0 shaping: draw target = -1/3 (draw worth ~one-third of a win).
    draw_penalty: float = 1 / 3
    # Value label source for self-play samples:
    #   "outcome" — classic AZ: terminal z (±1 / -draw_penalty) for every ply
    #   "root_q"  — per-ply MCTS searched_root_q (side-to-move POV)
    value_target: str = "outcome"
    # Optional self-play resignation. Disabled by default.
    resign_threshold: float = -1.1  # enable with value in [-1, 1]
    resign_plies: int = 0            # consecutive plies below threshold before resign
    resign_min_moves: int = 0        # do not allow resignation before this many plies
    fast_mate_bonus: float = 0.0    # >0 rewards quicker checkmates
    sims_per_move: int = 100        # fixed MCTS sims/move (no ramp)
    checkpoint_dir: str = "checkpoints"
    grad_clip_norm: float = 10.0


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    beauty: BeautyConfig = field(default_factory=BeautyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT = Config()
