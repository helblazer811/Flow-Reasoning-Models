"""EqR's official 30x30 Maze-Unique benchmark.

The released arrays use the HRM puzzle layout and token ids ``1..5`` for
``# SGo`` (0 is padding).  This loader converts them to FlowMapReasoning's
shared maze states and conditional ``(solution, clue_mask)`` representation.
Large artifacts are cached under :func:`default_dataset_root`, never in git.
"""

from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from .sudoku import default_dataset_root

S_WALL, S_EMPTY, S_PATH, S_START, S_GOAL = 1, 2, 3, 4, 5
GRID_VOCAB_SIZE = 6


def _grid_route_len(tokens, size=None):
    """Shortest route length (in cells) ``s->g`` through ``{s,g} u PATH``, or None.

    The predicted PATH cells (plus start/goal) define the labeled-path subgraph; this
    BFS finds the shortest walk within it, which exists iff the labeling connects s->g.
    """
    size = _infer_size(tokens, size)
    _, s, g, path_cells = decode_grid(tokens, size)
    if s is None or g is None:
        return None
    allowed = set(path_cells) | {s, g}
    if s == g:
        return 1
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            n = 0
            while cur is not None:
                n += 1
                cur = prev[cur]
            return n
        for nb in _neighbors(cur[0], cur[1], size):
            if nb in allowed and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return None


def valid_path_grid(tokens, size=None):
    """True iff the labeled PATH cells connect START to GOAL (4-connected route)."""
    return _grid_route_len(tokens, size) is not None


def _neighbors(r, c, size):
    if r > 0:
        yield r - 1, c
    if r + 1 < size:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < size:
        yield r, c + 1


def _infer_size(tokens, size=None):
    """Grid side length: explicit ``size`` or ``round(sqrt(len(tokens)))``.

    Lets the grid helpers work for any square grid (41x41, 61x61, ...) without the
    caller threading the size -- it is recovered from the flat token length.
    """
    return (
        size
        if size is not None
        else int(round(len(np.asarray(tokens).reshape(-1)) ** 0.5))
    )


def decode_grid(tokens, size=None):
    """``tokens`` -> ``(free_mask[size,size] bool, s, g, path_cells list[(r,c)])``.

    ``free_mask`` is True wherever the token is not S_WALL; ``s`` / ``g`` are the
    START / GOAL cells (None if absent); ``path_cells`` are the S_PATH cells. ``size``
    defaults to ``sqrt(len(tokens))`` so any square grid works.
    """
    size = _infer_size(tokens, size)
    grid = np.asarray(tokens).reshape(size, size)
    free_mask = grid != S_WALL
    sloc = np.argwhere(grid == S_START)
    gloc = np.argwhere(grid == S_GOAL)
    s = tuple(sloc[0]) if len(sloc) else None
    g = tuple(gloc[0]) if len(gloc) else None
    path_cells = [tuple(p) for p in np.argwhere(grid == S_PATH)]
    return free_mask, s, g, path_cells


HF_REPO = "locuslab/EqR-data"
HF_REVISION = "bcb7d45c157c06a054b715933557d266f5bd9d7f"
DATASET_ID = "maze_unique"
GRID = 30
L = GRID * GRID
VOCAB_SIZE = GRID_VOCAB_SIZE
HF_SUBDIR = "maze-30x30-unique-1k"
SPLITS = ("train", "test")

# EqR: 0 pad, 1 '#', 2 ' ', 3 'S', 4 'G', 5 'o'.
# FMR: 0 reserved, 1 wall, 2 empty, 3 path, 4 start, 5 goal.
_EQR_TO_FMR = np.asarray([0, 1, 2, 4, 5, 3], dtype=np.uint8)


def download_maze_unique(root=None, split="test", force=False):
    """Download and normalize one official EqR Maze-Unique split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root) if root is not None else default_dataset_root()
    out_dir = root / DATASET_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}.npz"
    if out.exists() and not force:
        return out

    from huggingface_hub import hf_hub_download

    prefix = f"{HF_SUBDIR}/{split}/all__"
    inputs_path = hf_hub_download(
        HF_REPO, prefix + "inputs.npy", repo_type="dataset", revision=HF_REVISION
    )
    labels_path = hf_hub_download(
        HF_REPO, prefix + "labels.npy", repo_type="dataset", revision=HF_REVISION
    )
    inputs = np.load(inputs_path).astype(np.int64)
    labels = np.load(labels_path).astype(np.int64)
    if inputs.shape != (1000, L) or labels.shape != inputs.shape:
        raise RuntimeError(
            f"unexpected Maze-Unique arrays: inputs={inputs.shape}, labels={labels.shape}"
        )
    if inputs.min() < 1 or labels.min() < 1 or inputs.max() > 5 or labels.max() > 5:
        raise RuntimeError(
            "Maze-Unique token ids are outside the expected EqR range 1..5"
        )

    solutions = _EQR_TO_FMR[labels]
    # Walls, start, and goal are immutable clues. Open cells (EqR space/path) are predicted.
    clue_mask = (inputs != 2).astype(np.uint8)
    path_lengths = (labels == 5).sum(axis=1) + 1  # edge length: interior 'o' cells + 1
    if path_lengths.min() < 100 or path_lengths.max() > 140:
        raise RuntimeError(
            f"unexpected Maze-Unique path range {path_lengths.min()}..{path_lengths.max()}"
        )
    np.savez_compressed(
        out,
        solutions=solutions,
        clue_mask=clue_mask,
        path_lengths=path_lengths,
        source_revision=np.asarray(HF_REVISION),
    )
    print(f"[maze_unique:{split}] cached {len(solutions)} puzzles -> {out}")
    return out


class MazeUniqueDataset(Dataset):
    dataset_id = DATASET_ID
    source_repo = HF_REPO
    source_revision = HF_REVISION

    def __init__(
        self,
        root=None,
        split="test",
        download=True,
        max_samples=None,
        subsample_seed=None,
    ):
        base = Path(root) if root is not None else default_dataset_root()
        path = (
            download_maze_unique(base, split)
            if download
            else base / DATASET_ID / f"{split}.npz"
        )
        if not Path(path).exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        solutions = data["solutions"].astype(np.int64)
        masks = data["clue_mask"].astype(np.int64)
        idx = self._select(len(solutions), max_samples, subsample_seed)
        self.solutions = solutions[idx]
        self.clue_masks = masks[idx]
        self.buckets = np.zeros(len(idx), dtype=np.int64)
        self.grid, self.seq_len, self.vocab_size = GRID, L, VOCAB_SIZE
        self.split = split

    @staticmethod
    def _select(n, max_samples, seed):
        if max_samples is None or max_samples >= n:
            return np.arange(n)
        if seed is None:
            return np.arange(max_samples)
        return np.random.default_rng(seed).permutation(n)[:max_samples]

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        return torch.stack(
            [
                torch.from_numpy(self.solutions[idx]).long(),
                torch.from_numpy(self.clue_masks[idx]).long(),
            ]
        )
