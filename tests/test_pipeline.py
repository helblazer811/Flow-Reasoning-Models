"""CPU regression checks for the release's loss, samplers, and checkpoint contract."""

from dataclasses import asdict
import numpy as np
import pytest
import torch
from frm.model import DIT, DitConfig
from frm.config import load_config
from frm.objective import (
    conditional_generate_samples,
    held_level_tau,
    make_conditional_flm_loss,
)
from frm.ema import ExponentialMovingAverage
from frm.evaluate import load_checkpoint, metrics
from frm.noise_schedule import TorchLUT, build_luts
from frm.datasets.sudoku import apply_sudoku_symmetry, valid_mask


@pytest.fixture
def model():
    torch.manual_seed(7)
    model = DIT(
        DitConfig(
            hidden_size=16, cond_dim=8, length=16, n_blocks=1, n_heads=2, dropout=0.0
        ),
        10,
    )
    with torch.no_grad():
        model.output_layer.linear.weight.normal_(0, 0.1)
    return model


@pytest.fixture(scope="module")
def lut():
    return TorchLUT(build_luts(K=10)[0], device=torch.device("cpu"))


@pytest.mark.parametrize("sampler", ("euler", "sde", "held_t"))
def test_sampling_preserves_clues_and_does_not_read_free_answers(model, lut, sampler):
    sol = torch.arange(16).remainder(9).add(1).view(1, -1)
    mask = (torch.arange(16) % 3 == 0).view(1, -1).long()
    different = sol.clone()
    different[~mask.bool()] = different[~mask.bool()].remainder(9) + 1
    kw = dict(
        lut_a2g=lut,
        num_steps=2,
        vocab_size=10,
        self_cond=True,
        device=torch.device("cpu"),
    )
    if sampler == "sde":
        kw["noise_scale"] = 0.2
    if sampler == "held_t":
        kw.update(held_tau=0.4, held_iters=2, renoise=1.0)
    torch.manual_seed(20)
    first = conditional_generate_samples(model, sol, mask, **kw)
    torch.manual_seed(20)
    second = conditional_generate_samples(model, different, mask, **kw)
    assert torch.equal(first, second)
    assert torch.equal(first[mask.bool()], sol[mask.bool()])


@pytest.mark.parametrize("rollout", (0.0, 0.5, 1.0))
def test_loss_has_finite_gradients(model, lut, rollout):
    sol = torch.arange(64).remainder(9).add(1).view(4, 16)
    mask = (sol % 3 == 0).long()
    fn = make_conditional_flm_loss(
        lut,
        0.0,
        1.0,
        10,
        self_cond_prob=0.5,
        self_forcing_rollout=rollout,
        sfr_fixed_depth=2,
        sfr_deep_sup_n=2,
    )
    torch.manual_seed(21)
    loss = fn(model, torch.stack((sol, mask), dim=1))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_held_level_matches_headline_tau():
    lut = TorchLUT(build_luts(K=10)[0], device=torch.device("cpu"))
    tau, t = held_level_tau(lut, 0.4)
    assert tau == 647 / 4096 == 0.157958984375
    assert abs(t - 0.4) < 1e-3


def test_held_t_fpf_loss_has_finite_gradients(model, lut):
    sol = torch.arange(64).remainder(9).add(1).view(4, 16)
    mask = (sol % 3 == 0).long()
    tau, _ = held_level_tau(lut, 0.4)
    fn = make_conditional_flm_loss(
        lut,
        0.0,
        1.0,
        10,
        self_cond_prob=0.5,
        self_forcing_rollout=1.0,
        sfr_fixed_depth=2,
        sfr_deep_sup_n=2,
        sfr_noise_scale=1.0,
        fixed_tau=tau,
    )
    torch.manual_seed(21)
    loss = fn(model, torch.stack((sol, mask), dim=1))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("task", ("sudoku", "maze_unique", "zebra"))
def test_release_configs_load(task):
    cfg = load_config(f"configs/{task}.yaml")
    assert cfg["data"].dataset == task


def test_sudoku_config_is_fixed_time_variant_recipe():
    cfg = load_config("configs/sudoku.yaml")
    p, loss = cfg["pipeline"], cfg["loss"]
    assert (p.stage_a_steps, p.stage_b_steps) == (100000, 175000)
    assert (p.stage_b_lr, p.stage_b_warmup_steps) == (1.0e-4, 1000)
    assert loss.fixed_level == 0.4 and loss.sfr_noise_scale == 1.0
    assert loss.self_forcing_rollout == 1.0


def test_legacy_checkpoint_prefixes_and_ema(model, tmp_path):
    cfg = asdict(model.config)
    cfg.update(seq_len=16, vocab_size=10, method="flm")
    ema = ExponentialMovingAverage(model.parameters(), 0.9999)
    ck = dict(
        config=cfg,
        model={"module._orig_mod." + k: v for k, v in model.state_dict().items()},
        ema=ema.state_dict(),
    )
    path = tmp_path / "legacy.pt"
    torch.save(ck, path)
    loaded, _, _ = load_checkpoint(path, torch.device("cpu"))
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_metrics_reject_invalid_sudoku_and_wrong_answers():
    solution = (
        (np.arange(9)[None, :] + (np.arange(9) * 3 + np.arange(9) // 3)[:, None]) % 9
        + 1
    ).reshape(1, -1)
    mask = np.zeros_like(solution)
    mask[:, 0] = 1
    assert metrics(solution, solution, mask, "sudoku")["solved"] == 1.0
    wrong = solution.copy()
    wrong[:, 1] = wrong[:, 0]
    result = metrics(wrong, solution, mask, "sudoku")
    assert result["solved"] == result["exact_match"] == result["validity"] == 0.0
    assert result["clue_consistency"] == 1.0
    assert metrics(wrong, solution, mask, "maze_unique")["solved"] == 0.0
    assert metrics(wrong, solution, mask, "zebra")["solved"] == 0.0


def test_sudoku_augmentation_preserves_validity_and_clue_count():
    solution = (
        (np.arange(9)[None, :] + (np.arange(9) * 3 + np.arange(9) // 3)[:, None]) % 9
        + 1
    ).reshape(-1)
    mask = (np.arange(81) % 3 == 0).astype(np.int64)
    changed, changed_mask = apply_sudoku_symmetry(
        solution, mask, np.random.default_rng(3)
    )
    assert valid_mask(changed[None])[0]
    assert changed_mask.sum() == mask.sum()
