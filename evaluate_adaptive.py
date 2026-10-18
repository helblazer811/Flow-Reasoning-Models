#!/usr/bin/env python3
"""Evaluate fixed-depth, residual-stopped, and sequential-restart FRM policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from frm.adaptive import (
    PolicyConfig,
    load_benchmark,
    load_model,
    manifest_base,
    run_adaptive_trajectory,
    run_full_trajectory_bank,
    correctness_grid,
    seed_all,
    select_ids_from_bank,
    sha256,
    write_json,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="smoke-test prefix only"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    return parser.parse_args(argv)


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def policy_row(
    name: str, correct: np.ndarray, nfe: np.ndarray, config: PolicyConfig
) -> dict:
    mean_nfe = float(nfe.mean())
    return {
        "policy": name,
        "solved": int(correct.sum()),
        "num_puzzles": int(len(correct)),
        "solve_rate": float(correct.mean()),
        "mean_nfe": mean_nfe,
        "mean_tflops": (
            mean_nfe * config.flops_per_forward / 1e12
            if config.flops_per_forward > 0
            else None
        ),
    }


def restart_curve_row(
    attempt: int,
    selected_ids: np.ndarray,
    total_nfe: np.ndarray,
    accepted: np.ndarray,
    solutions: np.ndarray,
    masks: np.ndarray,
    config: PolicyConfig,
) -> dict:
    """Score a frozen breadth prefix without using labels for any runtime decision."""
    correct = correctness_grid(selected_ids[None], solutions, masks)[0]
    mean_nfe = float(total_nfe.mean())
    return {
        "breadth_cap": int(attempt),
        "solved": int(correct.sum()),
        "num_puzzles": int(len(correct)),
        "solve_rate": float(correct.mean()),
        "mean_nfe": mean_nfe,
        "mean_tflops": (
            mean_nfe * config.flops_per_forward / 1e12
            if config.flops_per_forward > 0
            else None
        ),
        "accepted": int(accepted.sum()),
        "unresolved": int((~accepted).sum()),
    }


def calibration_benchmark_path(calibration_path, benchmark):
    path = Path(benchmark)
    # Anchor to the snapshot folder, not the target of the threshold file's symlink.
    return path if path.is_absolute() else Path(calibration_path).absolute().parent / path


def main(argv=None) -> None:
    args = parse_args(argv)
    calibration = json.loads(args.calibration.read_text())
    config = PolicyConfig(**calibration["policy_config"])
    config.validate()
    threshold = float(calibration["selected"]["threshold"])
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != calibration["checkpoint_sha256"]:
        raise ValueError(
            "evaluation checkpoint does not match the calibrated checkpoint"
        )
    benchmark_hash = sha256(args.benchmark)
    if benchmark_hash == calibration["benchmark_sha256"]:
        raise ValueError(
            "evaluation benchmark must be disjoint from the calibration benchmark"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solutions, masks = load_benchmark(args.benchmark, args.max_samples)
    calibration_benchmark = calibration_benchmark_path(args.calibration, calibration["benchmark"])
    if sha256(calibration_benchmark) != calibration["benchmark_sha256"]:
        raise ValueError("calibration benchmark changed after threshold selection")
    calibration_solutions, calibration_masks = load_benchmark(calibration_benchmark)
    calibration_rows = {
        np.stack((s, m)).astype("<i8").tobytes()
        for s, m in zip(calibration_solutions, calibration_masks)
    }
    if any(
        np.stack((s, m)).astype("<i8").tobytes() in calibration_rows
        for s, m in zip(solutions, masks)
    ):
        raise ValueError("calibration and evaluation puzzle sets overlap")
    setup = {
        "checkpoint_sha256": checkpoint_hash,
        "benchmark_sha256": benchmark_hash,
        "calibration_sha256": sha256(args.calibration),
        "policy_config": calibration["policy_config"],
        "threshold": threshold,
        "num_puzzles": len(solutions),
        "max_samples_override": args.max_samples,
    }
    run_fingerprint = fingerprint(setup)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_all(config.seed)
    model, meta = load_model(args.checkpoint, device)

    first_path = args.output_dir / "first_trajectory.npz"
    if args.resume and first_path.exists():
        cached = np.load(first_path)
        if str(cached["run_fingerprint"].item()) != run_fingerprint:
            raise ValueError(
                "first-trajectory cache belongs to a different evaluation setup"
            )
        first = {
            "check_nfe": cached["check_nfe"],
            "residual_grid": cached["residual_grid"],
            "ids_grid": cached["ids_grid"],
        }
        for metric in ("euclidean", "symmetric_kl"):
            key = f"residual_grid_{metric}"
            if key in cached.files:
                first[key] = cached[key]
        print(f"resumed {first_path}", flush=True)
    else:
        first = run_full_trajectory_bank(
            model, meta, solutions, masks, config, seed=config.seed, device=device
        )
        save_npz(first_path, run_fingerprint=run_fingerprint, **first)

    replay_bank = {
        **first,
        "confirmation_checks": config.confirmation_checks,
        "require_prediction_stability": config.require_prediction_stability,
    }
    early = select_ids_from_bank(replay_bank, threshold)
    fixed_ids = first["ids_grid"][-1]
    fixed_nfe = np.full(len(solutions), config.max_nfe, dtype=np.int32)
    early_ids = early["ids"]
    early_nfe = early["used_nfe"]

    progress_path = args.output_dir / "restart_progress.npz"
    if args.resume and progress_path.exists():
        progress = np.load(progress_path)
        if str(progress["run_fingerprint"].item()) != run_fingerprint:
            raise ValueError("restart cache belongs to a different evaluation setup")
        selected_ids = progress["selected_ids"]
        total_nfe = progress["total_nfe"]
        accepted = progress["accepted"]
        accepted_attempt = progress["accepted_attempt"]
        best_ids = progress["best_ids"]
        best_residual = progress["best_residual"]
        completed_attempt = int(progress["completed_attempt"])
        attempt_rows = json.loads(str(progress["attempt_rows_json"].item()))
        print(f"resumed through attempt {completed_attempt}", flush=True)
    else:
        selected_ids = early_ids.copy()
        total_nfe = early_nfe.astype(np.int64)
        accepted = early["accepted"].copy()
        accepted_attempt = np.where(accepted, 1, 0).astype(np.int16)
        best_ids = early["terminal_ids"].copy()
        best_residual = early["terminal_residual"].astype(np.float64)
        completed_attempt = 1
        attempt_rows = [
            {
                "attempt": 1,
                "attempted": len(solutions),
                "accepted": int(accepted.sum()),
                "remaining": int((~accepted).sum()),
                "mean_nfe_among_attempted": float(early_nfe.mean()),
            }
        ]
        restart_curve = [
            restart_curve_row(
                1, selected_ids, total_nfe, accepted, solutions, masks, config
            )
        ]

    if args.resume and progress_path.exists():
        cached_curve = np.load(progress_path)
        restart_curve = (
            json.loads(str(cached_curve["restart_curve_json"].item()))
            if "restart_curve_json" in cached_curve.files
            else []
        )

    for attempt in range(completed_attempt + 1, config.max_breadth + 1):
        unresolved = np.flatnonzero(~accepted)
        if not len(unresolved):
            break
        attempt_seed = config.seed + (attempt - 1) * config.restart_seed_offset
        result = run_adaptive_trajectory(
            model,
            meta,
            solutions[unresolved],
            masks[unresolved],
            config,
            threshold=threshold,
            seed=attempt_seed,
            device=device,
        )
        total_nfe[unresolved] += result["used_nfe"]
        crossed_local = np.flatnonzero(result["accepted"])
        crossed_global = unresolved[crossed_local]
        selected_ids[crossed_global] = result["ids"][crossed_local]
        accepted[crossed_global] = True
        accepted_attempt[crossed_global] = attempt

        failed_local = np.flatnonzero(~result["accepted"])
        failed_global = unresolved[failed_local]
        improved = (
            result["terminal_residual"][failed_local] < best_residual[failed_global]
        )
        improved_global = failed_global[improved]
        improved_local = failed_local[improved]
        best_residual[improved_global] = result["terminal_residual"][improved_local]
        best_ids[improved_global] = result["ids"][improved_local]
        selected_ids[failed_global] = best_ids[failed_global]

        attempt_rows.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "attempted": int(len(unresolved)),
                "accepted": int(len(crossed_global)),
                "remaining": int((~accepted).sum()),
                "mean_nfe_among_attempted": float(result["used_nfe"].mean()),
            }
        )
        restart_curve.append(
            restart_curve_row(
                attempt, selected_ids, total_nfe, accepted, solutions, masks, config
            )
        )
        save_npz(
            progress_path,
            run_fingerprint=run_fingerprint,
            selected_ids=selected_ids,
            total_nfe=total_nfe,
            accepted=accepted,
            accepted_attempt=accepted_attempt,
            best_ids=best_ids,
            best_residual=best_residual,
            completed_attempt=np.asarray(attempt, dtype=np.int16),
            attempt_rows_json=json.dumps(attempt_rows),
            restart_curve_json=json.dumps(restart_curve),
        )
        print(json.dumps(attempt_rows[-1]), flush=True)

    fixed_correct = correctness_grid(fixed_ids[None], solutions, masks)[0]
    early_correct = correctness_grid(early_ids[None], solutions, masks)[0]
    restart_correct = correctness_grid(selected_ids[None], solutions, masks)[0]
    rows = [
        policy_row("Fixed-depth FRM", fixed_correct, fixed_nfe, config),
        policy_row("+ Residual early stopping", early_correct, early_nfe, config),
        policy_row("+ Sequential restarts", restart_correct, total_nfe, config),
    ]
    payload = {
        **manifest_base(args.checkpoint, args.benchmark, config),
        "kind": "residual_guided_inference_evaluation",
        "run_fingerprint": run_fingerprint,
        "num_puzzles": len(solutions),
        "max_samples_override": args.max_samples,
        "model_step": int(meta.get("step", -1)),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256(args.calibration),
        "calibration_benchmark_sha256": calibration["benchmark_sha256"],
        "threshold": threshold,
        "threshold_selection": calibration["selected"],
        "policies": rows,
        "accepted_by_threshold": int(accepted.sum()),
        "unresolved_at_breadth_cap": int((~accepted).sum()),
        "attempts": attempt_rows,
        "restart_frontier": restart_curve,
        "selection_rule": (
            "stop each trajectory once the trailing-mean residual remains <= threshold for "
            f"{config.confirmation_checks} checks"
            + (
                " and the decoded grid remains unchanged; "
                if config.require_prediction_stability
                else "; "
            )
            + "restart only trajectories that reach max_nfe without acceptance; if max_breadth is "
            "exhausted, return the attempted terminal prediction with minimum residual"
        ),
        "labels_used_for": "final scoring only; never stopping, restart, or candidate selection",
    }
    write_json(args.output_dir / "evaluation.json", payload)
    save_npz(
        args.output_dir / "per_puzzle.npz",
        fixed_correct=fixed_correct,
        fixed_nfe=fixed_nfe,
        early_correct=early_correct,
        early_nfe=early_nfe,
        restart_correct=restart_correct,
        restart_nfe=total_nfe,
        accepted_attempt=accepted_attempt,
        selected_ids=selected_ids,
        best_residual=best_residual,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
