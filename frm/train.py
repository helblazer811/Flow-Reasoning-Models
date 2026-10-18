"""Train self-conditioning and fixed-point forcing with EMA and resumable checkpoints."""

import argparse
import copy
import json
from dataclasses import asdict
import contextlib
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .ema import ExponentialMovingAverage


def _config_to_dict(cfg):
    return cfg if isinstance(cfg, dict) else vars(cfg)


def _save_checkpoint(
    path,
    step,
    model,
    optimizer,
    ema,
    saved_cfg,
    scheduler=None,
    best_metric_val=None,
    provenance=None,
):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "config": saved_cfg,
        "provenance": dict(provenance or {}),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if best_metric_val is not None:
        payload["best_metric_val"] = best_metric_val
    torch.save(payload, path)


def train(
    model,
    train_loader,
    cfg,
    *,
    device=None,
    tokenizer=None,
    model_config=None,
    loss_fn=None,
    val_fn=None,
    dist_ctx=None,
    train_sampler=None,
    checkpoint_provenance=None,
):
    """Train ``model`` on ``train_loader`` under ``cfg`` (a ``TrainConfig``).

    Args:
        model:        DiT mapping (x_t, tau) -> logits. May be a DDP-wrapped module.
        train_loader: yields (B, L) LongTensors of token ids.
        cfg:          TrainConfig (or dict-like with the same fields).
        device:       defaults to the model's device.
        tokenizer:    optional; enables in-training sample decoding to W&B.
        model_config: optional DitConfig; merged into the saved checkpoint config
                      so evaluation can rebuild the architecture.
        loss_fn:      optional ``loss_fn(model, x0_on_device) -> scalar``. When given
                      (e.g. for flow-map distillation), it replaces the built-in FLM
                      loss; the rest of the loop (optim/EMA/ckpt/logging) is reused.
        val_fn:       optional ``val_fn(model, step) -> dict[str, float] | None``.
                      When given and ``cfg.val_every > 0`` it is called every
                      ``val_every`` steps under the EMA weights; the returned scalars
                      are printed and logged to W&B (keys used verbatim). Like
                      ``loss_fn``, the validation logic itself stays dataset-specific.
        dist_ctx:     optional ``frm.distributed.DistContext``. When ``None`` (default)
                      or ``world_size == 1`` the loop runs exactly as the single-GPU path
                      (no distributed branches taken — bit-for-bit unchanged). When it
                      describes a real >1 world, EMA/checkpoint act on the unwrapped
                      module and logging / W&B / ckpt / val_fn run on rank 0 only.
        train_sampler: optional ``DistributedSampler`` whose ``set_epoch`` is advanced each
                      time the loader is re-iterated (correct cross-epoch reshuffling under DDP).

    Returns:
        Path to the final checkpoint (rank 0). Non-main ranks return the same path.
    """
    # Distributed context (None / world_size==1 => single-GPU path, no behaviour change).
    is_dist = dist_ctx is not None and getattr(dist_ctx, "is_dist", False)
    is_main = (dist_ctx is None) or dist_ctx.is_main
    # Unwrap for EMA/state_dict/checkpoint: peel BOTH the DDP "module." and the torch.compile
    # "_orig_mod." wrappers (in any order) so neither prefix leaks into checkpoints — eval/resume
    # load with a strict state_dict and would fail on a prefixed key.
    core = model
    while hasattr(core, "module") or hasattr(core, "_orig_mod"):
        core = core.module if hasattr(core, "module") else core._orig_mod

    if device is None:
        device = next(model.parameters()).device
    c = cfg

    # Saved config = training knobs + model architecture (for eval reconstruction)
    saved_cfg = dict(_config_to_dict(c))
    if model_config is not None:
        saved_cfg.update(_config_to_dict(model_config))

    # Gradient accumulation
    accum_steps = 1
    if c.global_batch_size and c.global_batch_size > c.local_batch_size:
        assert c.global_batch_size % c.local_batch_size == 0, (
            "global_batch_size must be a multiple of local_batch_size"
        )
        accum_steps = c.global_batch_size // c.local_batch_size

    # Fused AdamW when on CUDA + requested (one kernel for the step; fewer launches at 200M-600M).
    _use_fused = (
        bool(getattr(c, "fused_adamw", True))
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    )
    _adamw_kw = dict(
        lr=c.lr, betas=tuple(c.betas), eps=c.eps, weight_decay=c.weight_decay
    )
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=_use_fused, **_adamw_kw)
    except (RuntimeError, ValueError):  # fused unsupported on this build -> fall back
        optimizer = torch.optim.AdamW(model.parameters(), **_adamw_kw)

    # EMA tracks the *core* (unwrapped) params so DDP's "module." prefix never leaks into
    # checkpoints and EMA store/copy/restore operate on the actual weights.
    ema = ExponentialMovingAverage(core.parameters(), decay=c.ema_decay)

    scheduler = None
    if c.warmup_steps and c.warmup_steps > 0:
        lr_sched = getattr(c, "lr_schedule", "constant")
        if lr_sched == "cosine":
            # Warmup -> cosine decay to lr*lr_min_ratio over max_steps (PUMA/MDLM recipe).
            # transformers' cosine helper decays to 0 at num_training_steps; add a floor by
            # blending its factor with lr_min_ratio via a lambda so the LR bottoms out at the
            # ratio rather than exactly 0.
            from transformers import get_cosine_schedule_with_warmup

            min_ratio = float(getattr(c, "lr_min_ratio", 0.0) or 0.0)
            if min_ratio <= 0.0:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=c.warmup_steps,
                    num_training_steps=c.max_steps,
                )
            else:
                from torch.optim.lr_scheduler import LambdaLR

                warmup = int(c.warmup_steps)
                total = int(c.max_steps)

                def _cosine_with_floor(step):
                    if step < warmup:
                        return step / max(1, warmup)
                    progress = (step - warmup) / max(1, total - warmup)
                    progress = min(1.0, progress)
                    cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
                    return min_ratio + (1.0 - min_ratio) * cos  # 1 -> min_ratio

                scheduler = LambdaLR(optimizer, _cosine_with_floor)
        else:
            from transformers import get_constant_schedule_with_warmup

            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=c.warmup_steps
            )

    ckpt_dir = Path(c.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    if c.resume:
        print(f"Resuming from {c.resume} ...")
        ckpt = torch.load(c.resume, map_location=device, weights_only=False)
        core.load_state_dict(
            ckpt["model"]
        )  # checkpoints store the unwrapped state_dict
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        if getattr(c, "reset_lr_on_resume", False):
            # Overwrite the restored (possibly cosine-floored ~0) LR with the config LR, so a
            # constant-LR continuation of a finished/decayed run actually trains. With a scheduler
            # present this would be immediately re-overridden on scheduler.step(), so it is only a
            # clean flat continuation when warmup_steps == 0 (scheduler is None); we still reset the
            # scheduler's base_lrs to keep any downstream scheduler.step() consistent.
            for g in optimizer.param_groups:
                g["lr"] = c.lr
            if scheduler is not None:
                scheduler.base_lrs = [c.lr for _ in scheduler.base_lrs]
            if is_main:
                print(f"  reset_lr_on_resume: forced optimizer LR -> {c.lr:g}")
        print(f"  Resumed at step {global_step}")

    # Peak-selection tracker (see cfg.save_best_metric). Persisted in checkpoints so a preempted
    # + resumed job keeps its running peak rather than resetting and overwriting best.pt downward.
    best_metric_val = float("-inf")
    if c.resume and "best_metric_val" in ckpt:
        best_metric_val = ckpt["best_metric_val"]
        if is_main and c.save_best_metric:
            print(f"  Resumed best {c.save_best_metric} = {best_metric_val:.4f}")

    # W&B + tqdm + console output live on rank 0 only (other ranks stay silent).
    use_wandb = (not c.no_wandb) and is_main
    if use_wandb:
        import wandb

        wandb.init(project=c.wandb_project, name=c.wandb_run_name, config=saved_cfg)

    if loss_fn is None and is_main:
        print(
            f"Expected step-0 loss (random init) ~ log(V) = {math.log(c.vocab_size):.3f}"
        )

    model.train()
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    pbar = tqdm(
        total=c.max_steps - global_step,
        desc="Training",
        unit="step",
        disable=not is_main,
    )
    t0 = time.time()
    # accumulate optional loss_fn aux scalars (e.g. ELF's separate MSE/CE) over the log_every window,
    # so W&B shows two CLEAN per-scale lines instead of one train/loss that jumps between the MSE and
    # CE regimes (they're on different scales -> the mixed single line is uninterpretable).
    aux_accum: dict = {}
    aux_count: dict = {}

    while global_step < c.max_steps:
        optimizer.zero_grad()
        # Accumulate the step loss as a DETACHED ON-DEVICE tensor; only sync it (.item()) at log
        # cadence below. Calling .item() every micro-step forced a device->host sync per step that
        # serialized the GPU against the async H2D copy + backward (no value off the log path).
        step_loss_t = None
        for micro in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(
                        epoch
                    )  # reshuffle the shard each pass under DDP
                data_iter = iter(train_loader)
                batch = next(data_iter)
            # Batch is normally a single (B,2,L) tensor. Conditional datasets that also carry a
            # per-example puzzle id (ARC) yield a (tensor, tensor) tuple/list; move each element to
            # device and hand the whole tuple to loss_fn. Separated-stream datasets (e.g. the xattn
            # arm's SeparatedConditionalTokenDataset) yield a DICT of named tensors — move each tensor
            # value to device and hand the whole dict to loss_fn. Existing single-tensor datasets
            # unaffected.
            if isinstance(batch, dict):
                x0 = {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
            elif isinstance(batch, (tuple, list)):
                x0 = tuple(b.to(device, non_blocking=True) for b in batch)
            else:
                x0 = batch.to(device, non_blocking=True)
            # Under DDP, each backward() triggers a gradient all-reduce. During gradient
            # accumulation we only need the reduced gradient on the FINAL micro-step, so wrap
            # the earlier micro-steps in model.no_sync() to skip the (accum_steps-1) redundant
            # all-reduces. No-op when not distributed or accum_steps==1.
            is_last_micro = micro == accum_steps - 1
            sync_cm = (
                model.no_sync()
                if (is_dist and not is_last_micro and hasattr(model, "no_sync"))
                else contextlib.nullcontext()
            )
            with sync_cm:
                if loss_fn is not None:
                    loss = loss_fn(model, x0)
                else:
                    raise ValueError("A conditional loss_fn is required")
                # loss_fn may optionally return (loss, aux_dict) — split off the aux scalars to log
                # (e.g. the ELF loss's separate MSE / CE components). Back-compat: scalar-only is fine.
                aux = None
                if isinstance(loss, tuple):
                    loss, aux = loss
                    if aux:
                        for _k, _v in aux.items():
                            aux_accum[_k] = aux_accum.get(_k, 0.0) + float(_v)
                            aux_count[_k] = aux_count.get(_k, 0) + 1
                (loss / accum_steps).backward()
            ld = loss.detach() / accum_steps
            step_loss_t = ld if step_loss_t is None else step_loss_t + ld

        if c.grad_clip and c.grad_clip > 0:
            gnorm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        else:
            gnorm_t = None
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        ema.update(core.parameters())
        global_step += 1
        pbar.update(1)

        if global_step % c.log_every == 0:
            step_loss = (
                step_loss_t.item() if step_loss_t is not None else 0.0
            )  # single sync / log
            elapsed = time.time() - t0
            sps = c.log_every / elapsed if elapsed > 0 else 0.0
            t0 = time.time()
            # Throughput + diagnostics, at parity with the ELF trainer (apples-to-apples + post-hoc).
            world = (
                dist_ctx.world_size
                if (dist_ctx is not None and hasattr(dist_ctx, "world_size"))
                else 1
            )
            toks_per_step = c.local_batch_size * accum_steps * world * c.seq_len
            tok_s = sps * toks_per_step
            tokens_seen = global_step * toks_per_step
            gnorm = float(gnorm_t) if gnorm_t is not None else 0.0
            vram = (
                (torch.cuda.max_memory_allocated() / 1e9)
                if torch.cuda.is_available()
                else 0.0
            )
            if is_main:
                pbar.set_postfix(
                    loss=f"{step_loss:.4f}",
                    sps=f"{sps:.1f}",
                    **{"tok/s": f"{tok_s / 1e3:.0f}k", "gn": f"{gnorm:.2f}"},
                )
                print(
                    f"[step {global_step}] loss={step_loss:.4f} sps={sps:.1f} "
                    f"tok/s={tok_s / 1e3:.0f}k gnorm={gnorm:.3f} vram={vram:.1f}GB "
                    f"tok_seen={tokens_seen / 1e9:.3f}B",
                    flush=True,
                )
            if use_wandb:
                log_d = {
                    "train/loss": step_loss,
                    "train/steps_per_sec": sps,
                    "train/tokens_per_sec": tok_s,
                    "train/tokens_seen": tokens_seen,
                    "train/grad_norm": gnorm,
                    "train/vram_alloc_gb": vram,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                # window-averaged aux scalars (e.g. elf/mse, elf/ce) — each on its own clean scale.
                for _k, _tot in aux_accum.items():
                    log_d[f"train/{_k}"] = _tot / max(1, aux_count.get(_k, 1))
                wandb.log(log_d, step=global_step)
            aux_accum.clear()
            aux_count.clear()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        if (
            val_fn is not None
            and getattr(c, "val_every", 0)
            and global_step % c.val_every == 0
        ):
            if is_main:
                ema.store(core.parameters())
                ema.copy_to(core.parameters())
                # Free the training step's cached activations before the val hook and run it under
                # no_grad: at large model size the val CE/sampling forward would otherwise build an
                # autograd graph on top of the resident training memory and OOM (observed on the 200M
                # FineWeb arm, step 2000). no_grad + empty_cache keeps the gate memory-safe.
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                try:
                    with torch.no_grad():
                        metrics = val_fn(core, global_step)
                    # Peak-select while EMA weights are still copied into core: if the tracked
                    # metric hit a new high, snapshot core (== EMA weights) as best.pt. Done inside
                    # the ema.store/restore window so best.pt holds the EMA weights that were scored.
                    if (
                        metrics
                        and c.save_best_metric
                        and c.save_best_metric in metrics
                        and metrics[c.save_best_metric] > best_metric_val
                    ):
                        best_metric_val = metrics[c.save_best_metric]
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        _save_checkpoint(
                            ckpt_dir / "best.pt",
                            global_step,
                            core,
                            optimizer,
                            ema,
                            saved_cfg,
                            scheduler,
                            best_metric_val=best_metric_val,
                            provenance=checkpoint_provenance,
                        )
                        print(
                            f"\n[best step {global_step}] {c.save_best_metric}"
                            f"={best_metric_val:.4f} -> best.pt"
                        )
                finally:
                    ema.restore(core.parameters())
                    model.train()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if metrics:
                    summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"\n[val step {global_step}] {summary}")
                    if use_wandb:
                        wandb.log(dict(metrics), step=global_step)
            if is_dist:
                import torch.distributed as _dist

                _dist.barrier()

        if c.save_every and global_step % c.save_every == 0 and is_main:
            # Re-ensure the parent dir exists at save time: on shared GPFS it can transiently
            # vanish (eventual consistency / cleanup races) between when it was created and the
            # checkpoint write, crashing torch.save / symlink_to with "parent dir does not exist".
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{global_step:08d}.pt"
            _save_checkpoint(
                ckpt_path,
                global_step,
                core,
                optimizer,
                ema,
                saved_cfg,
                scheduler,
                best_metric_val=best_metric_val if c.save_best_metric else None,
                provenance=checkpoint_provenance,
            )
            last = ckpt_dir / "last.pt"
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(ckpt_path.name)
            print(f"\nCheckpoint saved: {ckpt_path}")

    pbar.close()
    final_path = ckpt_dir / "final.pt"
    if is_main:
        ckpt_dir.mkdir(
            parents=True, exist_ok=True
        )  # GPFS dir may have transiently vanished
        _save_checkpoint(
            final_path,
            global_step,
            core,
            optimizer,
            ema,
            saved_cfg,
            scheduler,
            best_metric_val=best_metric_val if c.save_best_metric else None,
            provenance=checkpoint_provenance,
        )
        print(f"\nTraining complete. Final checkpoint: {final_path}")
    if use_wandb:
        wandb.finish()
    if is_dist:
        import torch.distributed as _dist

        _dist.barrier()
    return final_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--max-steps", type=int, help="Override BOTH stage budgets for smoke tests"
    )
    p.add_argument("--stage", choices=("all", "a", "b"), default="all")
    p.add_argument("--init-from", help="Stage B weights-only warm start")
    p.add_argument(
        "--resume",
        help="Resume one explicitly selected stage, including optimizer and EMA",
    )
    p.add_argument("--print-config", action="store_true")
    args = p.parse_args(argv)
    import torch
    from frm.config import load_config
    from frm.model import DIT
    from frm.objective import held_level_tau, make_conditional_flm_loss
    from frm.noise_schedule import TorchLUT, build_luts
    from frm.datasets.loading import load_dataset
    from frm.evaluate import seed_everything, clean_state_dict, evaluate_dataset
    from frm.distributed import (
        init_distributed,
        device_for,
        build_ddp_dataloader,
        wrap_ddp,
        barrier,
        cleanup,
    )

    cfg = load_config(args.config)
    data, pipeline = cfg["data"], cfg["pipeline"]
    if args.seed is not None:
        pipeline.seed = args.seed
    if args.data_root:
        data.root = args.data_root
    if args.max_steps is not None:
        if args.max_steps < 1:
            p.error("--max-steps must be positive")
        pipeline.stage_a_steps = pipeline.stage_b_steps = args.max_steps
    if args.resume and (args.stage == "all" or args.init_from):
        p.error("--resume requires --stage a or b and cannot accompany --init-from")
    if args.init_from and args.stage != "b":
        p.error("--init-from requires --stage b")
    if args.print_config:
        print(json.dumps({k: asdict(v) for k, v in cfg.items()}, indent=2))
        return
    ctx = init_distributed()
    device = device_for(ctx)
    output = Path(args.output_dir or f"checkpoints/{data.dataset}/seed_{pipeline.seed}")
    stages = ("a", "b") if args.stage == "all" else (args.stage,)
    if not args.resume:
        for stage in stages:
            target = output / f"stage_{stage}"
            if target.exists() and any(target.glob("*.pt")):
                raise FileExistsError(
                    f"{target} already contains checkpoints; use --resume "
                    "for one stage or choose a new --output-dir"
                )
    try:
        for stage in ("a", "b") if args.stage == "all" else (args.stage,):
            seed_everything(pipeline.seed)
            # Serialize first cache construction to avoid concurrent writers under DDP.
            if ctx.is_main:
                load_dataset(
                    data,
                    split="train",
                    max_samples=data.train_max_samples,
                    seed=pipeline.seed,
                    augment=False,
                )
            barrier(ctx)
            ds = load_dataset(
                data,
                split="train",
                max_samples=data.train_max_samples,
                seed=pipeline.seed,
                augment=True,
            )
            model_cfg = copy.deepcopy(cfg["model"])
            model_cfg.length = ds.seq_len
            model = DIT(model_cfg, ds.vocab_size).to(device)
            train_cfg = copy.deepcopy(cfg["train"])
            train_cfg.seq_len, train_cfg.vocab_size = ds.seq_len, ds.vocab_size
            train_cfg.max_steps = (
                pipeline.stage_a_steps if stage == "a" else pipeline.stage_b_steps
            )
            if stage == "b" and pipeline.stage_b_weight_decay is not None:
                train_cfg.weight_decay = pipeline.stage_b_weight_decay
            if stage == "b" and pipeline.stage_b_lr is not None:
                train_cfg.lr = pipeline.stage_b_lr
            if stage == "b" and pipeline.stage_b_warmup_steps is not None:
                train_cfg.warmup_steps = pipeline.stage_b_warmup_steps
            train_cfg.checkpoint_dir = str(output / f"stage_{stage}")
            train_cfg.resume = args.resume
            stage_cfg = {k: asdict(v) for k, v in cfg.items()}
            if ctx.is_main:
                output.mkdir(parents=True, exist_ok=True)
                (output / "config.json").write_text(
                    json.dumps(stage_cfg, indent=2) + "\n"
                )
                print(
                    f"Stage {stage.upper()}: {train_cfg.max_steps} steps on {device}, world_size={ctx.world_size}"
                )
            if stage == "b" and not args.resume:
                source = (
                    Path(args.init_from)
                    if args.init_from
                    else output / "stage_a" / f"{pipeline.warm_start}.pt"
                )
                if not source.exists() and pipeline.warm_start == "best":
                    source = output / "stage_a/final.pt"
                ck = torch.load(source, map_location=device, weights_only=False)
                if ck.get("config", {}).get("method", "flm") != "flm":
                    raise ValueError("Warm start must be FLM")
                model.load_state_dict(clean_state_dict(ck["model"]), strict=True)
                if ctx.is_main:
                    print(
                        f"Warm start from raw weights: {source}; fresh optimizer and EMA"
                    )
            loader, sampler = build_ddp_dataloader(
                ds,
                ctx=ctx,
                batch_size=train_cfg.local_batch_size,
                shuffle=True,
                num_workers=pipeline.num_workers,
                drop_last=True,
            )
            if not len(loader):
                raise ValueError("Training dataset is smaller than local_batch_size")
            lut = TorchLUT(build_luts(K=ds.vocab_size)[0], device=device)
            loss = asdict(cfg["loss"])
            fixed_level = loss.pop("fixed_level")
            if stage == "a":
                # Stage A is always plain self-conditioning over the full flow schedule.
                loss["self_forcing_rollout"] = 0.0
                loss["sfr_noise_scale"] = 0.0
            elif fixed_level is not None:
                loss["fixed_tau"] = held_level_tau(lut, fixed_level, device)[0]
                if ctx.is_main:
                    print(
                        f"Held-t FPF at t={fixed_level} (tau={loss['fixed_tau']:.12g})"
                    )
            if not model_cfg.self_cond and (
                loss["self_cond_prob"] or loss["self_forcing_rollout"]
            ):
                raise ValueError("Training loss requires model.self_cond=true")
            loss_fn = make_conditional_flm_loss(
                lut, train_cfg.t_min, train_cfg.t_max, ds.vocab_size, **loss
            )
            # Every rank must enter the trainer's validation barrier, even though
            # only rank zero evaluates the dataset.
            val_fn = (lambda current_model, step: {}) if train_cfg.val_every else None
            if train_cfg.val_every and ctx.is_main:
                val_ds = load_dataset(
                    data,
                    split=cfg["validation"].split,
                    max_samples=cfg["validation"].num_samples,
                    seed=cfg["validation"].subset_seed,
                    seq_len=ds.seq_len,
                )

                def val_fn(current_model, step):
                    m = evaluate_dataset(
                        current_model,
                        val_ds,
                        cfg["validation"],
                        lut,
                        data.dataset,
                        device,
                    )
                    return {
                        "val/solved": m["solved"],
                        "val/token_accuracy": m["token_accuracy"],
                    }

            wrapped = wrap_ddp(model, ctx, find_unused_parameters=True)
            train(
                wrapped,
                loader,
                train_cfg,
                device=device,
                model_config=model_cfg,
                loss_fn=loss_fn,
                val_fn=val_fn,
                dist_ctx=ctx,
                train_sampler=sampler,
                checkpoint_provenance={
                    "dataset": data.dataset,
                    "stage": stage,
                    "seed": pipeline.seed,
                    "recipe": stage_cfg,
                },
            )
            barrier(ctx)
    finally:
        cleanup(ctx)


if __name__ == "__main__":
    main()
