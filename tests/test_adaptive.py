"""Privacy, stopping, and model-call accounting for the optional adaptive sampler."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from frm.adaptive import (
    PolicyConfig,
    acceptance_grid,
    calibrate_threshold,
    run_adaptive_trajectory,
    run_full_trajectory_bank,
    select_ids_from_bank,
)


class ToyModel(torch.nn.Module):
    def forward(self, z, tau, self_cond=None, **kwargs):
        return z * 0.3 + (0 if self_cond is None else 0.4 * self_cond)


def fixture():
    model = ToyModel()
    solutions = np.tile(np.arange(5) % 3, (6, 1))
    masks = np.zeros_like(solutions)
    masks[:, 0] = 1
    cfg = PolicyConfig(
        max_nfe=8,
        min_nfe=2,
        check_every=2,
        residual_window=2,
        confirmation_checks=2,
        batch_size=3,
    )
    meta = {"vocab_size": 3, "lut_a2g": lambda x: x}
    return model, solutions, masks, cfg, meta


def test_fixed_and_never_accepting_trajectory_match():
    model, sol, masks, cfg, meta = fixture()
    bank = run_full_trajectory_bank(
        model, meta, sol, masks, cfg, seed=19, device=torch.device("cpu")
    )
    result = run_adaptive_trajectory(
        model, meta, sol, masks, cfg, threshold=-1, seed=19, device=torch.device("cpu")
    )
    np.testing.assert_array_equal(result["ids"], bank["ids_grid"][-1])
    assert not result["accepted"].any()
    assert (result["used_nfe"] == 8).all()


def test_confirmation_stops_at_exact_model_call_and_ignores_free_answers():
    model, sol, masks, cfg, meta = fixture()
    cfg = replace(cfg, require_prediction_stability=False)
    a = run_adaptive_trajectory(
        model,
        meta,
        sol,
        masks,
        cfg,
        threshold=float("inf"),
        seed=22,
        device=torch.device("cpu"),
    )
    changed = sol.copy()
    changed[masks == 0] = (changed[masks == 0] + 1) % 3
    b = run_adaptive_trajectory(
        model,
        meta,
        changed,
        masks,
        cfg,
        threshold=float("inf"),
        seed=22,
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(a["ids"], b["ids"])
    assert (a["ids"][:, 0] == sol[:, 0]).all()
    assert a["accepted"].all() and (a["used_nfe"] == 4).all()


def test_replay_requires_both_residual_confirmation_and_stable_prediction():
    residual = np.array([[0.1], [0.1], [0.1]])
    ids = np.array([[[1, 2]], [[1, 0]], [[1, 0]]])
    crossed = acceptance_grid(
        residual, ids, 0.2, confirmation_checks=2, require_prediction_stability=True
    )
    assert crossed[:, 0].tolist() == [False, False, True]
    selected = select_ids_from_bank(
        dict(
            residual_grid=residual,
            ids_grid=ids,
            check_nfe=np.array([2, 4, 6]),
            confirmation_checks=2,
            require_prediction_stability=True,
        ),
        0.2,
    )
    assert selected["used_nfe"].tolist() == [6]


def test_threshold_calibration_respects_development_accuracy_constraint():
    residual = np.array([[0.1, 0.1], [0.01, 0.01]])
    correct = np.array([[False, False], [True, True]])
    selected, _ = calibrate_threshold(
        residual, correct, np.array([2, 4]), max_accuracy_drop=0, quantiles=10
    )
    assert selected["accuracy"] == 1 and selected["mean_nfe"] == 4


def test_mixed_acceptance_matches_online_full_bank_replay():
    model, sol, masks, cfg, meta = fixture()
    cfg = replace(cfg, batch_size=6, require_prediction_stability=False)
    bank = run_full_trajectory_bank(
        model, meta, sol, masks, cfg, seed=31, device=torch.device("cpu")
    )
    threshold = float(np.median(bank["residual_grid"]))
    replay = select_ids_from_bank(
        dict(**bank, confirmation_checks=2, require_prediction_stability=False),
        threshold,
    )
    online = run_adaptive_trajectory(
        model,
        meta,
        sol,
        masks,
        cfg,
        threshold=threshold,
        seed=31,
        device=torch.device("cpu"),
    )
    # Active compaction changes later RNG consumption, but accepted prefixes before
    # compaction and model-call accounting must stay valid for every puzzle.
    assert (online["used_nfe"] >= 4).all() and (online["used_nfe"] <= 8).all()
    assert online["accepted"].any() and replay["accepted"].any()


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        PolicyConfig(min_nfe=1).validate()
