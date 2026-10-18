#!/usr/bin/env python3
"""Shared residual-guided held-t inference primitives.

Ported from the public branch residual-guided inference implementation.  A trajectory uses the same held-t recurrence as
``conditional_generate_samples``: raw-logit self-conditioning, fresh Gaussian
re-noising, and clue re-pinning after every update.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]

CAP = 30.0
LUT_GRID = 4097


@dataclass(frozen=True)
class PolicyConfig:
    max_nfe: int = 4096
    max_breadth: int = 128
    held_t: float = 0.4
    renoise: float = 1.0
    check_every: int = 16
    residual_window: int = 64
    residual_metric: str = "euclidean"
    confirmation_checks: int = 8
    require_prediction_stability: bool = True
    min_nfe: int = 64
    batch_size: int = 128
    seed: int = 0
    restart_seed_offset: int = 1_000_003
    flops_per_forward: int = 896_039_936

    def validate(self) -> None:
        if self.max_nfe < 2:
            raise ValueError("max_nfe must be at least 2")
        if self.max_breadth < 1:
            raise ValueError("max_breadth must be positive")
        if not 0.0 <= self.held_t <= 1.0:
            raise ValueError("held_t must lie in [0, 1]")
        if not 0.0 <= self.renoise <= 1.0:
            raise ValueError("renoise must lie in [0, 1]")
        if self.check_every < 1 or self.residual_window < 1:
            raise ValueError("check_every and residual_window must be positive")
        if self.residual_metric not in {"euclidean", "symmetric_kl"}:
            raise ValueError("residual_metric must be 'euclidean' or 'symmetric_kl'")
        if self.confirmation_checks < 1:
            raise ValueError("confirmation_checks must be positive")
        if self.min_nfe < 2 or self.min_nfe > self.max_nfe:
            raise ValueError("min_nfe must lie in [2, max_nfe]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_benchmark(
    path: Path, max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load any fixed-length conditional benchmark in ``(solution, clue_mask)`` form.

    Despite the experiment directory's historical Sudoku name, the held-time recurrence is
    dataset-agnostic.  Zebra and Maze-Unique use the same representation with different
    vocabularies and sequence lengths, both of which are recovered from the checkpoint.
    """
    data = np.load(path)
    solutions = np.asarray(data["solutions"], dtype=np.int64)
    masks = np.asarray(data["clue_mask"], dtype=np.int64)
    if max_samples is not None:
        solutions = solutions[:max_samples]
        masks = masks[:max_samples]
    if solutions.ndim != 2 or masks.shape != solutions.shape:
        raise ValueError(
            "benchmark must contain matching [N,L] solutions and clue_mask arrays"
        )
    if not len(solutions):
        raise ValueError("benchmark is empty")
    if solutions.min(initial=0) < 0:
        raise ValueError("solution token ids must be nonnegative")
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("clue_mask must be binary")
    return solutions, masks


def load_model(checkpoint: Path, device: torch.device):
    from .evaluate import load_checkpoint

    model, lut, checkpoint_data = load_checkpoint(checkpoint, device, use_ema=True)
    meta = dict(checkpoint_data["config"])
    meta.update(
        step=checkpoint_data.get("step", -1),
        lut_a2g=lut,
        vocab_size=model.vocab_size,
        method="flm",
    )
    return model, meta


def _held_level(lut_a2g, held_t: float, device: torch.device) -> tuple[float, float]:
    grid = torch.linspace(0.0, 1.0, LUT_GRID, device=device)
    physical = lut_a2g(grid)
    index = int((physical - held_t).abs().argmin().item())
    return float(grid[index].item()), float(physical[index].item())


def _forward(model, z, tau, self_cond):
    raw = model(z, tau, self_cond=self_cond)
    capped = CAP * torch.tanh(raw / CAP)
    return capped - capped.logsumexp(dim=-1, keepdim=True), capped


def _symmetric_kl(probs, previous, free, denominator):
    eps = 1e-8
    p = probs.clamp_min(eps)
    q = previous.clamp_min(eps)
    sym = 0.5 * ((p * (p.log() - q.log())).sum(-1) + (q * (q.log() - p.log())).sum(-1))
    return (sym * free).sum(-1) / denominator


def _euclidean(probs, previous, free, denominator):
    distance = torch.linalg.vector_norm(probs - previous, ord=2, dim=-1)
    return (distance * free).sum(-1) / denominator


def _residual(probs, previous, free, denominator, metric: str):
    if metric == "euclidean":
        return _euclidean(probs, previous, free, denominator)
    if metric == "symmetric_kl":
        return _symmetric_kl(probs, previous, free, denominator)
    raise ValueError(f"unknown residual metric: {metric}")


def check_nfes(config: PolicyConfig) -> np.ndarray:
    values = list(range(config.check_every, config.max_nfe + 1, config.check_every))
    values = [value for value in values if value >= config.min_nfe]
    if not values or values[-1] != config.max_nfe:
        values.append(config.max_nfe)
    return np.asarray(sorted(set(values)), dtype=np.int32)


@torch.no_grad()
def run_full_trajectory_bank(
    model,
    meta: dict,
    solutions: np.ndarray,
    masks: np.ndarray,
    config: PolicyConfig,
    *,
    seed: int,
    device: torch.device,
    self_conditioning: bool = True,
) -> dict[str, np.ndarray]:
    """Run one complete trajectory per puzzle and cache every stopping checkpoint."""
    config.validate()
    # Match the production evaluator exactly: seed once per draw and consume one
    # continuous CUDA RNG stream across the ordered batches.
    seed_all(seed)
    n, length = solutions.shape
    checks = check_nfes(config)
    metric_names = ("euclidean", "symmetric_kl")
    residual_grids = {
        metric: np.full((len(checks), n), np.nan, dtype=np.float32)
        for metric in metric_names
    }
    ids_grid = np.zeros((len(checks), n, length), dtype=np.uint8)
    tau_hold, t_phys = _held_level(meta["lut_a2g"], config.held_t, device)
    check_to_row = {int(value): row for row, value in enumerate(checks)}

    for lo in range(0, n, config.batch_size):
        hi = min(lo + config.batch_size, n)
        sol = torch.as_tensor(solutions[lo:hi], dtype=torch.long, device=device)
        mask = torch.as_tensor(masks[lo:hi], dtype=torch.bool, device=device)
        mask3 = mask.unsqueeze(-1)
        clue_onehot = F.one_hot(sol, int(meta["vocab_size"])).float()
        free = (~mask).float()
        denominator = free.sum(-1).clamp_min(1.0)
        z = torch.randn((hi - lo, length, int(meta["vocab_size"])), device=device)
        z = torch.where(mask3, clue_onehot, z)
        tau = torch.full((hi - lo,), tau_hold, dtype=torch.float32, device=device)
        self_cond = None
        previous = None
        windows = {metric: deque() for metric in metric_names}
        window_sums = {
            metric: torch.zeros(hi - lo, device=device) for metric in metric_names
        }

        for nfe in range(1, config.max_nfe + 1):
            logprob, raw = _forward(model, z, tau, self_cond)
            self_cond = raw if self_conditioning else None
            probs = logprob.exp()
            if previous is not None:
                for metric in metric_names:
                    residual = _residual(probs, previous, free, denominator, metric)
                    windows[metric].append(residual)
                    window_sums[metric] = window_sums[metric] + residual
                    if len(windows[metric]) > config.residual_window:
                        window_sums[metric] = (
                            window_sums[metric] - windows[metric].popleft()
                        )
            previous = probs

            row = check_to_row.get(nfe)
            if row is not None:
                if not windows[config.residual_metric]:
                    raise RuntimeError(
                        "a stopping checkpoint was requested before residual exists"
                    )
                for metric in metric_names:
                    residual_grids[metric][row, lo:hi] = (
                        (window_sums[metric] / len(windows[metric]))
                        .float()
                        .cpu()
                        .numpy()
                    )
                ids = probs.argmax(-1)
                ids = torch.where(mask, sol, ids)
                ids_grid[row, lo:hi] = ids.to(torch.uint8).cpu().numpy()

            if nfe < config.max_nfe:
                eps = torch.randn_like(z)
                z_sde = t_phys * probs + (1.0 - t_phys) * eps
                z = (1.0 - config.renoise) * z + config.renoise * z_sde
                z = torch.where(mask3, clue_onehot, z)
        print(f"full trajectory seed={seed} batch={lo}:{hi}", flush=True)

    return {
        "check_nfe": checks,
        "residual_grid": residual_grids[config.residual_metric],
        "residual_grid_euclidean": residual_grids["euclidean"],
        "residual_grid_symmetric_kl": residual_grids["symmetric_kl"],
        "ids_grid": ids_grid,
    }


def correctness_grid(
    ids_grid: np.ndarray, solutions: np.ndarray, masks: np.ndarray | None = None
) -> np.ndarray:
    """Return exact free-cell correctness for every checkpoint and puzzle."""
    correct = ids_grid == solutions[None, :, :]
    if masks is not None:
        correct = correct | masks[None, :, :].astype(bool)
    return correct.all(axis=2)


def replay_threshold(
    residual_grid: np.ndarray,
    correct_grid: np.ndarray,
    checks: np.ndarray,
    threshold: float,
    *,
    ids_grid: np.ndarray | None = None,
    confirmation_checks: int = 1,
    require_prediction_stability: bool = False,
) -> dict:
    crossed = acceptance_grid(
        residual_grid,
        ids_grid,
        threshold,
        confirmation_checks=confirmation_checks,
        require_prediction_stability=require_prediction_stability,
    )
    any_crossed = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0)
    selected_row = np.where(any_crossed, first, len(checks) - 1)
    puzzle = np.arange(correct_grid.shape[1])
    selected_correct = correct_grid[selected_row, puzzle]
    used_nfe = np.where(any_crossed, checks[first], checks[-1])
    return {
        "accuracy": float(selected_correct.mean()),
        "solved": int(selected_correct.sum()),
        "mean_nfe": float(used_nfe.mean()),
        "accepted_fraction": float(any_crossed.mean()),
        "selected_row": selected_row,
        "used_nfe": used_nfe.astype(np.int32),
        "accepted": any_crossed,
    }


def acceptance_grid(
    residual_grid: np.ndarray,
    ids_grid: np.ndarray | None,
    threshold: float,
    *,
    confirmation_checks: int,
    require_prediction_stability: bool,
    stability_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Return checkpoints at which a fully internal acceptance rule is satisfied."""
    if confirmation_checks < 1:
        raise ValueError("confirmation_checks must be positive")
    if require_prediction_stability and ids_grid is None:
        raise ValueError("ids_grid is required for prediction-stability confirmation")
    below = residual_grid <= threshold
    accepted = np.zeros_like(below, dtype=bool)
    width = confirmation_checks
    below_streak = np.zeros(residual_grid.shape[1], dtype=np.int32)
    if require_prediction_stability and stability_eligible is None:
        stability_eligible = prediction_stability_grid(ids_grid, width)
    for row in range(len(residual_grid)):
        below_streak = np.where(below[row], below_streak + 1, 0)
        valid = below_streak >= width
        if require_prediction_stability:
            valid &= stability_eligible[row]
        accepted[row] = valid
    return accepted


def prediction_stability_grid(
    ids_grid: np.ndarray, confirmation_checks: int
) -> np.ndarray:
    """Mark checkpoints whose decoded grid is unchanged for the requested run length."""
    stable = np.zeros(ids_grid.shape[:2], dtype=bool)
    streak = np.ones(ids_grid.shape[1], dtype=np.int32)
    stable[0] = confirmation_checks <= 1
    for row in range(1, len(ids_grid)):
        unchanged = (ids_grid[row] == ids_grid[row - 1]).all(axis=1)
        streak = np.where(unchanged, streak + 1, 1)
        stable[row] = streak >= confirmation_checks
    return stable


def calibrate_threshold(
    residual_grid: np.ndarray,
    correct_grid: np.ndarray,
    checks: np.ndarray,
    *,
    max_accuracy_drop: float,
    quantiles: int = 501,
    ids_grid: np.ndarray | None = None,
    confirmation_checks: int = 1,
    require_prediction_stability: bool = False,
) -> tuple[dict, list[dict]]:
    if max_accuracy_drop < 0:
        raise ValueError("max_accuracy_drop must be nonnegative")
    fixed_accuracy = float(correct_grid[-1].mean())
    finite = residual_grid[np.isfinite(residual_grid)]
    never_stop = float(np.nextafter(finite.min(), -np.inf))
    candidates = np.unique(
        np.concatenate(
            ([never_stop], np.quantile(finite, np.linspace(0.0, 1.0, quantiles)))
        )
    )
    rows = []
    stability_eligible = None
    if require_prediction_stability:
        stability_eligible = prediction_stability_grid(ids_grid, confirmation_checks)
    for threshold in candidates:
        crossed = acceptance_grid(
            residual_grid,
            ids_grid,
            float(threshold),
            confirmation_checks=confirmation_checks,
            require_prediction_stability=require_prediction_stability,
            stability_eligible=stability_eligible,
        )
        any_crossed = crossed.any(axis=0)
        first = np.argmax(crossed, axis=0)
        selected_row = np.where(any_crossed, first, len(checks) - 1)
        puzzle = np.arange(correct_grid.shape[1])
        selected_correct = correct_grid[selected_row, puzzle]
        used_nfe = np.where(any_crossed, checks[first], checks[-1])
        replay = {
            "accuracy": float(selected_correct.mean()),
            "solved": int(selected_correct.sum()),
            "mean_nfe": float(used_nfe.mean()),
            "accepted_fraction": float(any_crossed.mean()),
        }
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": replay["accuracy"],
                "solved": replay["solved"],
                "mean_nfe": replay["mean_nfe"],
                "accepted_fraction": replay["accepted_fraction"],
            }
        )
    eligible = [
        row
        for row in rows
        if row["accuracy"] + 1e-12 >= fixed_accuracy - max_accuracy_drop
    ]
    if not eligible:
        raise RuntimeError("no threshold satisfies the calibration accuracy constraint")
    selected = min(
        eligible, key=lambda row: (row["mean_nfe"], -row["accuracy"], row["threshold"])
    )
    selected = {
        **selected,
        "fixed_accuracy": fixed_accuracy,
        "max_accuracy_drop": max_accuracy_drop,
        "selection_objective": (
            "minimize mean NFE subject to accuracy >= fixed-depth accuracy - max_accuracy_drop; "
            "tie-break by higher accuracy then lower threshold"
        ),
    }
    return selected, rows


def select_ids_from_bank(bank: dict[str, np.ndarray], threshold: float) -> dict:
    residual = bank["residual_grid"]
    ids = bank["ids_grid"]
    checks = bank["check_nfe"]
    crossed = acceptance_grid(
        residual,
        ids,
        threshold,
        confirmation_checks=int(bank.get("confirmation_checks", 1)),
        require_prediction_stability=bool(
            bank.get("require_prediction_stability", False)
        ),
    )
    accepted = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0)
    row = np.where(accepted, first, len(checks) - 1)
    puzzle = np.arange(ids.shape[1])
    return {
        "ids": ids[row, puzzle],
        "used_nfe": np.where(accepted, checks[first], checks[-1]).astype(np.int32),
        "accepted": accepted,
        "selected_row": row.astype(np.int32),
        "terminal_ids": ids[-1],
        "terminal_residual": residual[-1],
    }


@torch.no_grad()
def run_adaptive_trajectory(
    model,
    meta: dict,
    solutions: np.ndarray,
    masks: np.ndarray,
    config: PolicyConfig,
    *,
    threshold: float,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run one trajectory and physically remove puzzles as soon as they are accepted."""
    config.validate()
    seed_all(seed)
    n, length = solutions.shape
    output_ids = np.zeros((n, length), dtype=np.uint8)
    used_nfe = np.full(n, config.max_nfe, dtype=np.int32)
    accepted = np.zeros(n, dtype=bool)
    terminal_residual = np.full(n, np.inf, dtype=np.float32)
    tau_hold, t_phys = _held_level(meta["lut_a2g"], config.held_t, device)

    for lo in range(0, n, config.batch_size):
        hi = min(lo + config.batch_size, n)
        original = torch.arange(lo, hi, device=device)
        sol = torch.as_tensor(solutions[lo:hi], dtype=torch.long, device=device)
        mask = torch.as_tensor(masks[lo:hi], dtype=torch.bool, device=device)
        mask3 = mask.unsqueeze(-1)
        clue_onehot = F.one_hot(sol, int(meta["vocab_size"])).float()
        free = (~mask).float()
        denominator = free.sum(-1).clamp_min(1.0)
        z = torch.randn((hi - lo, length, int(meta["vocab_size"])), device=device)
        z = torch.where(mask3, clue_onehot, z)
        self_cond = None
        previous = None
        window: deque[torch.Tensor] = deque()
        window_sum = torch.zeros(hi - lo, device=device)
        consecutive_below = torch.zeros(hi - lo, dtype=torch.int16, device=device)
        stable_checks = torch.zeros(hi - lo, dtype=torch.int16, device=device)
        previous_ids = None

        for nfe in range(1, config.max_nfe + 1):
            tau = torch.full((len(sol),), tau_hold, dtype=torch.float32, device=device)
            logprob, raw = _forward(model, z, tau, self_cond)
            self_cond = raw
            probs = logprob.exp()
            if previous is not None:
                residual = _residual(
                    probs, previous, free, denominator, config.residual_metric
                )
                window.append(residual)
                window_sum = window_sum + residual
                if len(window) > config.residual_window:
                    window_sum = window_sum - window.popleft()
            previous = probs

            should_check = nfe >= config.min_nfe and (
                nfe % config.check_every == 0 or nfe == config.max_nfe
            )
            if should_check:
                statistic = window_sum / len(window)
                ids = torch.where(mask, sol, probs.argmax(-1))
                below = statistic <= threshold
                consecutive_below = torch.where(
                    below, consecutive_below + 1, torch.zeros_like(consecutive_below)
                )
                if previous_ids is None:
                    stable_checks.fill_(1)
                else:
                    unchanged = (ids == previous_ids).all(dim=1)
                    stable_checks = torch.where(
                        unchanged, stable_checks + 1, torch.ones_like(stable_checks)
                    )
                previous_ids = ids
                crosses = consecutive_below >= config.confirmation_checks
                if config.require_prediction_stability:
                    crosses &= stable_checks >= config.confirmation_checks
                if crosses.any():
                    destination = original[crosses].cpu().numpy()
                    output_ids[destination] = ids[crosses].to(torch.uint8).cpu().numpy()
                    used_nfe[destination] = nfe
                    accepted[destination] = True
                    terminal_residual[destination] = (
                        statistic[crosses].float().cpu().numpy()
                    )
                if nfe == config.max_nfe:
                    remains = ~crosses
                    if remains.any():
                        destination = original[remains].cpu().numpy()
                        output_ids[destination] = (
                            ids[remains].to(torch.uint8).cpu().numpy()
                        )
                        terminal_residual[destination] = (
                            statistic[remains].float().cpu().numpy()
                        )
                    break
                if crosses.any():
                    keep = ~crosses
                    if not keep.any():
                        break
                    original = original[keep]
                    sol = sol[keep]
                    mask = mask[keep]
                    mask3 = mask3[keep]
                    clue_onehot = clue_onehot[keep]
                    free = free[keep]
                    denominator = denominator[keep]
                    z = z[keep]
                    self_cond = self_cond[keep]
                    previous = previous[keep]
                    window = deque(value[keep] for value in window)
                    window_sum = window_sum[keep]
                    consecutive_below = consecutive_below[keep]
                    stable_checks = stable_checks[keep]
                    previous_ids = previous_ids[keep]
                    probs = probs[keep]

            if nfe < config.max_nfe:
                eps = torch.randn_like(z)
                z_sde = t_phys * probs + (1.0 - t_phys) * eps
                z = (1.0 - config.renoise) * z + config.renoise * z_sde
                z = torch.where(mask3, clue_onehot, z)
        print(
            f"adaptive trajectory seed={seed} batch={lo}:{hi} accepted={accepted[lo:hi].sum()}",
            flush=True,
        )

    return {
        "ids": output_ids,
        "used_nfe": used_nfe,
        "accepted": accepted,
        "terminal_residual": terminal_residual,
    }


def manifest_base(checkpoint: Path, benchmark: Path, config: PolicyConfig) -> dict:
    return {
        "schema_version": 1,
        "git_commit": git_head(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": sha256(benchmark),
        "policy_config": asdict(config),
        "residual": (
            f"trailing-window mean of per-puzzle {config.residual_metric} distance between "
            "consecutive denoiser distributions, averaged over non-clue cells"
        ),
        "oracle_used_for_test_selection": False,
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
