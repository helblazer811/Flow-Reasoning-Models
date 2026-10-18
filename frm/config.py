"""Training options preserved from the public pipeline."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    method: str = "flm"
    seq_len: int = 81
    vocab_size: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    ema_decay: float = 0.9999
    fused_adamw: bool = True
    t_min: float = 0.0
    t_max: float = 1.0
    max_steps: int = 100000
    local_batch_size: int = 128
    global_batch_size: int = 0
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    lr_min_ratio: float = 0.1
    log_every: int = 100
    val_every: int = 5000
    save_best_metric: Optional[str] = "val/solved"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5000
    resume: Optional[str] = None
    keep_last_only: bool = False
    reset_lr_on_resume: bool = False
    no_wandb: bool = True
    wandb_project: str = "flowmap"
    wandb_run_name: Optional[str] = None


@dataclass
class DataConfig:
    dataset: str = "sudoku"
    root: Optional[str] = None
    train_max_samples: Optional[int] = None
    balanced_by_rating: bool = True
    symmetry_probability: float = 0.0
    sizes: Tuple[str, ...] = ("5x5",)
    repack: bool = True
    pin_answer: bool = False


@dataclass
class LossConfig:
    self_cond_prob: float = 0.5
    self_forcing_rollout: float = 0.5
    sfr_t_min: float = 0.0
    sfr_fixed_depth: int = 16
    sfr_deep_sup_n: int = 4
    sfr_noise_scale: float = 0.0
    # Physical time for held-t FPF in Stage B; null trains over the full flow schedule.
    fixed_level: Optional[float] = None


@dataclass
class EvalConfig:
    split: str = "test"
    num_samples: Optional[int] = 256
    subset_seed: int = 0
    seed: int = 0
    batch_size: int = 64
    sampler: str = "euler"
    steps: int = 128
    noise_scale: float = 0.0
    held_level: float = 0.4
    held_iters: int = 1023
    renoise: float = 1.0
    self_cond: bool = True


@dataclass
class PipelineConfig:
    seed: int = 0
    num_workers: int = 4
    stage_a_steps: int = 100000
    stage_b_steps: int = 100000
    warm_start: str = "final"
    # Optional Stage B overrides of train.lr / train.warmup_steps.
    stage_b_weight_decay: Optional[float] = None
    stage_b_lr: Optional[float] = None
    stage_b_warmup_steps: Optional[int] = None


def load_config(path):
    """Load a single explicit YAML; reject unknown sections and keys."""
    from dataclasses import fields
    import yaml
    from .model import DitConfig

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    sections = dict(
        data=DataConfig,
        model=DitConfig,
        train=TrainConfig,
        loss=LossConfig,
        evaluation=EvalConfig,
        validation=EvalConfig,
        pipeline=PipelineConfig,
    )
    unknown = set(raw) - sections.keys()
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    result = {}
    for name, cls in sections.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        result[name] = cls(**values)
    if result["data"].dataset not in ("zebra", "maze_unique", "sudoku"):
        raise ValueError("data.dataset must be zebra, maze, maze_unique, or sudoku")
    if result["train"].method != "flm":
        raise ValueError("Only the FLM objective is supported")
    for name in ("evaluation", "validation"):
        c = result[name]
        if c.sampler not in ("euler", "sde", "held_t"):
            raise ValueError(f"{name}.sampler must be euler, sde, or held_t")
        if c.batch_size < 1 or c.steps < 1 or c.held_iters < 1:
            raise ValueError(
                f"{name} batch size, steps, and iterations must be positive"
            )
        if c.num_samples is not None and c.num_samples < 1:
            raise ValueError(f"{name}.num_samples must be positive or null")
        if (
            not 0 <= c.noise_scale <= 1
            or not 0 < c.held_level < 1
            or not 0 <= c.renoise <= 1
        ):
            raise ValueError(f"Invalid {name} noise/held-level parameters")
    p = result["pipeline"]
    if (
        p.warm_start not in ("best", "final")
        or min(p.stage_a_steps, p.stage_b_steps) < 1
    ):
        raise ValueError("Invalid pipeline stages or warm_start checkpoint")
    if p.stage_b_weight_decay is not None and p.stage_b_weight_decay < 0:
        raise ValueError("Stage B weight decay must be nonnegative")
    if (p.stage_b_lr is not None and p.stage_b_lr <= 0) or (
        p.stage_b_warmup_steps is not None and p.stage_b_warmup_steps < 0
    ):
        raise ValueError("Invalid Stage B learning-rate overrides")
    loss = result["loss"]
    if not 0 <= loss.sfr_noise_scale <= 1 or (
        loss.fixed_level is not None and not 0 < loss.fixed_level < 1
    ):
        raise ValueError("Invalid loss noise scale or fixed level")
    t = result["train"]
    if min(t.local_batch_size, t.log_every) < 1 or t.val_every < 0 or t.save_every < 0:
        raise ValueError("Invalid training batch size or cadence")
    return result
