#!/usr/bin/env python3
"""Calibrate the online residual stopping threshold on a held-out development set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frm.adaptive import (
    PolicyConfig,
    calibrate_threshold,
    correctness_grid,
    load_benchmark,
    load_model,
    manifest_base,
    run_full_trajectory_bank,
    seed_all,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-nfe", type=int, default=4096)
    parser.add_argument("--max-breadth", type=int, default=128)
    parser.add_argument("--held-t", type=float, default=0.4)
    parser.add_argument("--renoise", type=float, default=1.0)
    parser.add_argument("--check-every", type=int, default=16)
    parser.add_argument("--residual-window", type=int, default=64)
    parser.add_argument(
        "--residual-metric",
        choices=("euclidean", "symmetric_kl"),
        default="euclidean",
    )
    parser.add_argument("--confirmation-checks", type=int, default=8)
    parser.add_argument(
        "--no-prediction-stability",
        action="store_true",
        help="do not require the decoded grid to remain unchanged during confirmation",
    )
    parser.add_argument("--min-nfe", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restart-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--flops-per-forward", type=int, default=896_039_936)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.001)
    parser.add_argument("--threshold-quantiles", type=int, default=501)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="smoke-test prefix only"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyConfig(
        max_nfe=args.max_nfe,
        max_breadth=args.max_breadth,
        held_t=args.held_t,
        renoise=args.renoise,
        check_every=args.check_every,
        residual_window=args.residual_window,
        residual_metric=args.residual_metric,
        confirmation_checks=args.confirmation_checks,
        require_prediction_stability=not args.no_prediction_stability,
        min_nfe=args.min_nfe,
        batch_size=args.batch_size,
        seed=args.seed,
        restart_seed_offset=args.restart_seed_offset,
        flops_per_forward=args.flops_per_forward,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solutions, masks = load_benchmark(args.benchmark, args.max_samples)
    model, meta = load_model(args.checkpoint, device)
    bank = run_full_trajectory_bank(
        model, meta, solutions, masks, config, seed=config.seed, device=device
    )
    correct = correctness_grid(bank["ids_grid"], solutions, masks)
    selected, sweep = calibrate_threshold(
        bank["residual_grid"],
        correct,
        bank["check_nfe"],
        max_accuracy_drop=args.max_accuracy_drop,
        quantiles=args.threshold_quantiles,
        ids_grid=bank["ids_grid"],
        confirmation_checks=config.confirmation_checks,
        require_prediction_stability=config.require_prediction_stability,
    )

    np.savez_compressed(
        args.output_dir / "calibration_bank.npz",
        check_nfe=bank["check_nfe"],
        residual_grid=bank["residual_grid"],
        residual_grid_euclidean=bank["residual_grid_euclidean"],
        residual_grid_symmetric_kl=bank["residual_grid_symmetric_kl"],
        ids_grid=bank["ids_grid"],
        correct_grid=correct,
    )
    payload = {
        **manifest_base(args.checkpoint, args.benchmark, config),
        "kind": "residual_stopping_calibration",
        "num_puzzles": len(solutions),
        "max_samples_override": args.max_samples,
        "model_step": int(meta.get("step", -1)),
        "labels_used_for": "development-set threshold calibration only",
        "selected": selected,
        "candidate_thresholds": args.threshold_quantiles,
        "calibration_bank": "calibration_bank.npz",
        "threshold_sweep": "threshold_sweep.json",
    }
    write_json(args.output_dir / "threshold.json", payload)
    write_json(args.output_dir / "threshold_sweep.json", sweep)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
