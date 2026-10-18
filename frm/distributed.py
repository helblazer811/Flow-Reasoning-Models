"""Task-agnostic single-node DDP helpers for the shared FLM training loop.

This is a thin, **opt-in** layer over ``frm.trainer.train``. The design contract:

  * **Single-GPU is bit-for-bit unchanged.** ``trainer.train`` takes an optional
    ``dist_ctx`` (a :class:`DistContext`); when it is ``None`` (the default) or
    ``world_size == 1`` every distributed branch is skipped and the loop runs exactly
    as it always has. Experiments that never touch this module are unaffected.
  * **Homogeneous DDP.** Every rank is a full peer: it runs the same loop, owns a
    replica of the model (wrapped in :class:`~torch.nn.parallel.DistributedDataParallel`),
    and pulls a disjoint shard of each batch via a :class:`~torch.utils.data.distributed.DistributedSampler`.
    DDP all-reduces gradients on every ``backward``; logging / W&B / checkpoint / ``val_fn``
    run on rank 0 only.
  * The ``loss_fn(model, x0)`` contract is unchanged: each rank evaluates the loss on its
    own shard. A loss that internally samples (tilt / bootstrap) or forwards a frozen
    verifier just does so per-rank; the verifier must be replicated on each rank with
    ``requires_grad_(False)``.

Launch with ``torchrun --standalone --nproc_per_node=N script.py``. ``init_distributed()``
reads the torchrun env (``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``); with no such env (plain
``python script.py``) it returns a degenerate single-process context and nothing distributed
is initialised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistContext:
    """Resolved distributed context for one process. ``is_dist=False`` == single process."""

    rank: int
    local_rank: int
    world_size: int
    is_dist: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistContext:
    """Initialise the process group from the torchrun env, or return a single-proc context.

    Reads ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``. When ``WORLD_SIZE`` is unset or 1 we do
    **not** init a process group (``is_dist=False``) — the caller then runs the ordinary
    single-GPU path. Otherwise we init NCCL and pin this process to ``cuda:LOCAL_RANK``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistContext(rank=0, local_rank=0, world_size=1, is_dist=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not dist.is_initialized():
        # Generous collective timeout so the periodic-val barrier (trainer runs val on
        # rank 0 only while the other ranks wait) never trips the NCCL watchdog: a full
        # ARC/Sudoku val (256 puzzles x 128 sampling steps under EMA) can exceed the
        # 10-min NCCL default, which killed the 4-GPU ARC FPF runs with an ALLREDUCE
        # timeout at the barrier. Override via DIST_TIMEOUT_MIN (minutes).
        from datetime import timedelta

        _timeout_min = int(os.environ.get("DIST_TIMEOUT_MIN", "120"))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=_timeout_min))
    torch.cuda.set_device(local_rank)
    return DistContext(
        rank=rank, local_rank=local_rank, world_size=world_size, is_dist=True
    )


def device_for(ctx: DistContext) -> torch.device:
    """The CUDA device this rank owns (``cuda:local_rank`` when distributed)."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{ctx.local_rank}" if ctx.is_dist else "cuda")
    return torch.device("cpu")


def wrap_ddp(
    model,
    ctx: DistContext,
    *,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
):
    """Wrap ``model`` in DDP when distributed; otherwise return it unchanged.

    Returns ``model`` as-is for single-process runs so ``model.parameters()`` /
    ``model.state_dict()`` keep their non-prefixed form (bit-for-bit single-GPU behaviour).

    ``gradient_as_bucket_view=True`` lets DDP's reduction buckets alias the grad tensors
    instead of copying into a separate flat buffer — a few % off every backward's all-reduce
    and lower peak memory, with no behavioural change. ``find_unused_parameters`` defaults
    False (the standard FLM/FMLM forward uses every parameter every step); pass True only for
    a branching objective where some params are skipped on a given step (else DDP would hang on
    the gradient all-reduce), at the cost of an extra autograd-graph traversal per backward.
    """
    if not ctx.is_dist:
        return model
    # device_ids/output_device are valid only for GPU modules; a CPU module (e.g. a gloo-backed
    # test run) must omit them.
    on_cuda = next(model.parameters()).is_cuda
    if on_cuda:
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers,
        )
    return DistributedDataParallel(
        model,
        gradient_as_bucket_view=True,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def unwrap(model):
    """Underlying module (strip the DDP wrapper) — used for state_dict / EMA / sampling."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_ddp_dataloader(
    dataset,
    *,
    ctx: DistContext,
    batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    collate_fn=None,
    pin_memory=True,
):
    """A loader that shards across ranks under DDP, else a plain shuffled loader.

    Returns ``(loader, sampler)``. ``sampler`` is a :class:`DistributedSampler` when
    distributed (call ``sampler.set_epoch(e)`` each pass for correct reshuffling) or ``None``
    for the single-process case (in which the loader handles shuffling itself).
    """
    # Keep workers alive across epochs + read ahead so shard/collate cost hides behind the GPU step
    # (pin_memory makes the trainer's non_blocking H2D copy actually async).
    extra = {}
    if num_workers and num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 4
    if ctx.is_dist:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **extra,
        )
        return loader, sampler
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **extra,
    )
    return loader, None


def all_gather_objects(obj, ctx: DistContext) -> list:
    """Gather an arbitrary picklable ``obj`` from every rank into a length-``world_size`` list.

    Single-process: returns ``[obj]``. Used to pool the ragged per-rank propose→verify
    positives into one shared set before the tilt step.
    """
    if not ctx.is_dist:
        return [obj]
    gathered = [None] * ctx.world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def barrier(ctx: DistContext) -> None:
    if ctx.is_dist:
        dist.barrier()


def cleanup(ctx: DistContext) -> None:
    if ctx.is_dist and dist.is_initialized():
        dist.destroy_process_group()
