"""Evaluate a local model bundle or a checkpoint with an explicit configuration."""

import argparse
import json
from pathlib import Path
from dataclasses import asdict, fields
import random
import numpy as np
import torch
from .model import DIT, DitConfig
from .ema import ExponentialMovingAverage
from .noise_schedule import TorchLUT, build_luts
from .objective import conditional_generate_samples
from .datasets.sudoku import valid_mask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_state_dict(state):
    result = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def load_checkpoint(checkpoint, device, use_ema=True):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    if cfg.get("method", "flm") != "flm":
        raise ValueError("Only FLM checkpoints are supported")
    unsupported = (
        "double_temb",
        "causal",
        "self_cond_cont",
        "channelwise_neg",
        "mask_flag_embed",
        "puzzle_emb_ndim",
        "num_seg_types",
        "two_grid_pos",
        "input_cond_embed",
        "learnable_loss_weighting",
        "is_di4c",
        "tie_word_embeddings",
        "anchor_embed_norm",
        "tie_unembed",
    )
    if (
        any(cfg.get(k, False) for k in unsupported)
        or cfg.get("flow_space", "onehot") != "onehot"
        or cfg.get("cond_inject", "none") != "none"
        or cfg.get("pos_mode", "1d") != "1d"
    ):
        raise ValueError("Checkpoint uses an architecture outside this minimal release")
    required = (
        "hidden_size",
        "cond_dim",
        "n_blocks",
        "n_heads",
        "vocab_size",
        "seq_len",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Checkpoint is missing architecture fields: {missing}")
    values = {f.name: cfg[f.name] for f in fields(DitConfig) if f.name in cfg}
    values["length"] = cfg["seq_len"]
    values.setdefault("self_cond", False)
    model = DIT(DitConfig(**values), cfg["vocab_size"]).to(device)
    model.load_state_dict(clean_state_dict(ck["model"]), strict=True)
    if use_ema:
        if "ema" not in ck:
            raise ValueError(
                "Checkpoint has no EMA weights; use --no-ema to load raw weights"
            )
        ema = ExponentialMovingAverage(model.parameters(), cfg.get("ema_decay", 0.9999))
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model.parameters())
    model.eval()
    lut = TorchLUT(
        build_luts(K=cfg.get("lut_k", 0) or cfg["vocab_size"])[0], device=device
    )
    return model, lut, ck


def metrics(samples, solutions, masks, dataset):
    correct = samples == solutions
    free = ~masks.astype(bool)
    solved = (correct | ~free).all(axis=1)
    counts = free.sum(axis=1)
    token_acc = np.where(
        counts > 0, (correct & free).sum(axis=1) / np.maximum(counts, 1), 1.0
    )
    out = dict(
        n_samples=len(samples),
        exact_match=float(correct.all(axis=1).mean()),
        solved=float(solved.mean()),
        token_accuracy=float(token_acc.mean()),
        clue_consistency=float((correct | free).all(axis=1).mean()),
    )
    if dataset == "sudoku":
        valid = valid_mask(samples)
        out["validity"] = float(valid.mean())
        out["solved"] = float((valid & (correct | free).all(axis=1)).mean())
    return out


@torch.no_grad()
def evaluate_dataset(model, ds, config, lut, dataset, device):
    if config.self_cond and model.self_embed is None:
        raise ValueError(
            "Evaluation requests self-conditioning but checkpoint lacks it"
        )
    # Fixed evaluation RNG without perturbing the ongoing training stream.
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    predictions, solutions, masks = [], [], []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        for lo in range(0, len(ds), config.batch_size):
            batch = torch.stack(
                [ds[i] for i in range(lo, min(lo + config.batch_size, len(ds)))]
            )
            sol, mask = batch[:, 0].to(device), batch[:, 1].to(device)
            ids = conditional_generate_samples(
                model,
                sol,
                mask,
                lut_a2g=lut,
                num_steps=config.steps,
                vocab_size=model.vocab_size,
                device=device,
                self_cond=config.self_cond,
                noise_scale=config.noise_scale if config.sampler == "sde" else 0.0,
                held_tau=config.held_level if config.sampler == "held_t" else None,
                held_iters=config.held_iters,
                renoise=config.renoise,
            )
            predictions.append(ids.cpu().numpy())
            solutions.append(sol.cpu().numpy())
            masks.append(mask.cpu().numpy())
    pred, sol, mask = map(np.concatenate, (predictions, solutions, masks))
    return metrics(pred, sol, mask, dataset)


def evaluate(model_dir, *, output_dir=None, device=None, max_samples=None):
    from .adaptive import sha256

    bundle = Path(model_dir)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle contains an invalid file path")
        if sha256(bundle / relative) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    suffix = f"_smoke_{max_samples}" if max_samples is not None else ""
    output = Path(output_dir or f"outputs/{bundle.name}{suffix}")
    args = [
        "--checkpoint",
        str(bundle / "model.pt"),
        "--benchmark",
        str(bundle / "test.npz"),
        "--calibration",
        str(bundle / "threshold.json"),
        "--output-dir",
        str(output),
        "--resume",
    ]
    if device:
        args += ["--device", str(device)]
    if max_samples is not None:
        args += ["--max-samples", str(max_samples)]
    from evaluate_adaptive import main as run

    run(args)
    return json.loads((output / "evaluation.json").read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Required for --sampler fixed_depth")
    p.add_argument(
        "--sampler",
        choices=("sequential_restart", "fixed_depth"),
        default="sequential_restart",
    )
    p.add_argument("--benchmark", help="Test NPZ for calibrated sequential restarts")
    p.add_argument("--calibration", help="Checkpoint-specific threshold.json")
    p.add_argument("--resume", action="store_true")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--model-dir", type=Path)
    p.add_argument("--output-dir")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output", default="outputs/evaluation.json")
    p.add_argument("--device", choices=("cuda", "cpu"))
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args(argv)
    if args.model_dir is not None:
        if (
            args.sampler != "sequential_restart"
            or args.no_ema
            or args.config
            or args.benchmark
            or args.calibration
        ):
            p.error("--model-dir uses the bundled calibrated sequential-restart recipe")
        return evaluate(
            args.model_dir,
            output_dir=args.output_dir,
            device=args.device,
            max_samples=args.max_samples,
        )
    if args.output_dir is not None or args.max_samples is not None:
        p.error("--output-dir and --max-samples require --model-dir")
    if args.sampler == "sequential_restart":
        if not args.benchmark or not args.calibration:
            p.error(
                "sequential_restart requires --benchmark and --calibration; run calibrate.py on a separate development set first, or explicitly select --sampler fixed_depth"
            )
        if args.no_ema:
            p.error("calibrated sequential restarts require EMA weights")
        from evaluate_adaptive import main as adaptive_main

        adaptive_args = [
            "--checkpoint",
            args.checkpoint,
            "--benchmark",
            args.benchmark,
            "--calibration",
            args.calibration,
            "--output-dir",
            str(Path(args.output).parent / Path(args.output).stem),
        ]
        if args.resume:
            adaptive_args.append("--resume")
        if args.device:
            adaptive_args.extend(["--device", args.device])
        return adaptive_main(adaptive_args)
    if not args.config:
        p.error("--config is required for --sampler fixed_depth")
    import torch
    from frm.config import load_config
    from frm.datasets.loading import load_dataset

    cfg = load_config(args.config)
    data, ev = cfg["data"], cfg["evaluation"]
    if args.data_root:
        data.root = args.data_root
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, lut, ck = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    provenance = ck.get("provenance", {})
    stored_dataset = provenance.get("dataset")
    if stored_dataset and stored_dataset != data.dataset:
        raise ValueError(
            f"Checkpoint dataset {stored_dataset!r} differs from {data.dataset!r}"
        )
    ds = load_dataset(
        data,
        split=ev.split,
        max_samples=ev.num_samples,
        seed=ev.subset_seed,
        seq_len=model.config.length,
    )
    if ds.seq_len != model.config.length or ds.vocab_size != model.vocab_size:
        raise ValueError(
            "Dataset dimensions differ from checkpoint; check data version and preprocessing"
        )
    result = dict(
        dataset=data.dataset,
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_step=ck.get("step"),
        use_ema=not args.no_ema,
        protocol=asdict(ev),
        metrics=evaluate_dataset(model, ds, ev, lut, data.dataset, device),
        checkpoint_provenance=provenance,
    )
    result["nfe_per_puzzle"] = (
        ev.held_iters if ev.sampler == "held_t" else ev.steps
    ) + 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
