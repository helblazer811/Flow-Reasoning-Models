"""The **Sudoku-Extreme** hard-puzzle benchmark (Sapient HRM).

Wraps ``sapientinc/sudoku-extreme`` (the hard-puzzle set used by the Hierarchical
Reasoning Model work): real 9x9 puzzles paired with their unique solutions, tagged
with a difficulty ``rating`` (0 = trivial, 385 = hardest in the test split). Unlike
the procedurally-generated grids in ``synthetic_sudoku.py`` (``SyntheticSudokuDataset``) these
are *curated* puzzles requiring long deductive / backtracking chains.

The CSV columns are ``source, question, answer, rating`` where ``question`` /
``answer`` are 81-char row-major strings (``question`` uses ``.`` for a blank cell,
digits ``1..9`` for the given clues; ``answer`` is the fully-solved grid). We parse
each puzzle into the **conditional / infilling** representation already used by the
Sudoku FLM solver:

  - ``solution``  — length-81 int array, digits ``1..9`` (the full answer);
  - ``clue_mask`` — length-81 int8, ``1`` where the puzzle gives a clue, else ``0``.

A clue cell holds the same digit in the puzzle and the answer, so feeding the
solution + clue_mask through the clue-injecting sampler (clue cells clamped clean,
the rest generated) reproduces the original puzzle exactly. ``__getitem__`` returns
the packed ``stack([solution, clue_mask])`` ``(2, 81)`` LongTensor — drop-in for the
conditional loss / sampler in ``paper_experiments/train_sudoku/conditional.py``.

Data is downloaded once and cached as a compressed ``.npz`` under ``root`` (defaults
to a local-scratch mirror; see the repo CLAUDE.md scratch convention). Subsequent
constructions reuse the cache; nothing lands in the git repo.
"""

import os
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HF_REPO = "sapientinc/sudoku-extreme"
SPLITS = ("train", "test")
L = 81

# Difficulty buckets keyed on the puzzle ``rating`` (inclusive lower, exclusive
# upper). Used to stratify subsampling and to break metrics down by hardness.
DIFFICULTY_EDGES = (0, 5, 20, 50, 10**9)
DIFFICULTY_NAMES = ("easy", "medium", "hard", "extreme")


def default_dataset_root():
    """Portable dataset cache, overridable via FMR_SCRATCH_ROOT."""
    base = os.environ.get("FMR_SCRATCH_ROOT")
    return (
        Path(base) / "datasets"
        if base
        else Path.home() / ".cache" / "flowmap" / "datasets"
    )


# ---------------------------------------------------------------------------
# Download + parse + cache
# ---------------------------------------------------------------------------
def _parse_csv(csv_path):
    """Parse a sudoku-extreme CSV into numpy arrays (vectorised over rows).

    Returns ``(solutions [N,81] uint8 1..9, clue_mask [N,81] uint8, ratings [N]
    int32, sources [N] object)``.
    """
    import csv as _csv

    sources, questions, answers, ratings = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = _csv.reader(f)
        next(reader)  # header: source, question, answer, rating
        for src, q, a, rt in reader:
            sources.append(src)
            questions.append(q)
            answers.append(a)
            ratings.append(int(rt))

    n = len(answers)
    # ASCII char codes, [N, 81]. "".join is far cheaper than per-row parsing.
    q_arr = np.frombuffer("".join(questions).encode("ascii"), dtype=np.uint8).reshape(
        n, L
    )
    a_arr = np.frombuffer("".join(answers).encode("ascii"), dtype=np.uint8).reshape(
        n, L
    )

    solutions = (a_arr - ord("0")).astype(np.uint8)  # 1..9
    # A clue is any cell the puzzle fills in (not '.' and not '0').
    clue_mask = ((q_arr != ord(".")) & (q_arr != ord("0"))).astype(np.uint8)
    return (
        solutions,
        clue_mask,
        np.asarray(ratings, dtype=np.int32),
        np.asarray(sources),
    )


def download_sudoku_extreme(root=None, split="test", force=False):
    """Ensure the parsed sudoku-extreme ``split`` is cached; return the ``.npz`` path.

    Downloads ``<split>.csv`` from the HuggingFace hub (only if the parsed cache is
    missing or ``force``), parses it, and writes ``<root>/sudoku_extreme/<split>.npz``.
    Idempotent: a second call with the cache present does no network I/O.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root) if root is not None else default_dataset_root()
    out_dir = root / "sudoku_extreme"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}.npz"

    if npz_path.exists() and not force:
        return npz_path

    from huggingface_hub import hf_hub_download

    print(f"[sudoku-extreme] downloading {HF_REPO}:{split}.csv ...")
    csv_path = hf_hub_download(HF_REPO, f"{split}.csv", repo_type="dataset")
    print(f"[sudoku-extreme] parsing {csv_path} ...")
    solutions, clue_mask, ratings, sources = _parse_csv(csv_path)
    np.savez_compressed(
        npz_path,
        solutions=solutions,
        clue_mask=clue_mask,
        ratings=ratings,
        sources=sources,
    )
    print(
        f"[sudoku-extreme] cached {len(solutions):,} puzzles -> {npz_path} "
        f"(clues {clue_mask.sum(1).min()}-{clue_mask.sum(1).max()}, "
        f"rating {ratings.min()}-{ratings.max()})"
    )
    return npz_path


def difficulty_bucket(ratings, edges=DIFFICULTY_EDGES):
    """Map an array of ratings to bucket indices ``0..len(edges)-2`` (right-open)."""
    ratings = np.asarray(ratings)
    # np.digitize with the interior edges; edges[0] is the global lower bound.
    return np.clip(np.digitize(ratings, edges[1:-1], right=False), 0, len(edges) - 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SudokuExtremeDataset(Dataset):
    """Torch dataset over the cached sudoku-extreme split (conditional format).

    Auto-downloads + caches on first use. ``__getitem__`` returns the packed
    ``(2, 81)`` LongTensor ``stack([solution, clue_mask])`` so it flows through the
    standard dataloader and the conditional loss / sampler unchanged.

    Args:
        root: cache root (defaults to the scratch mirror, see ``default_dataset_root``).
        split: ``"test"`` (default) or ``"train"``.
        download: auto-download if the cache is missing (default True).
        max_samples: keep at most this many puzzles (after optional subsampling).
        subsample_seed: if set, shuffle deterministically before applying
            ``max_samples`` (so a capped subset is a representative draw, not just
            the head of the file which is ordered by source).
        balanced_by_rating: when subsampling, draw ~equally from each difficulty
            bucket (``DIFFICULTY_EDGES``) instead of uniformly — useful so a small
            val subset still contains hard puzzles.
        difficulty_edges: rating bucket edges for ``balanced_by_rating`` / ``.buckets``.

    Exposes numpy views for evaluation: ``.solutions``, ``.clue_masks``,
    ``.ratings``, ``.sources``, and ``.buckets`` (difficulty bucket index per puzzle).
    """

    def __init__(
        self,
        root=None,
        split="test",
        download=True,
        max_samples=None,
        subsample_seed=None,
        balanced_by_rating=False,
        difficulty_edges=DIFFICULTY_EDGES,
    ):
        npz_path = (
            download_sudoku_extreme(root, split)
            if download
            else (
                Path(root or default_dataset_root()) / "sudoku_extreme" / f"{split}.npz"
            )
        )
        if not Path(npz_path).exists():
            raise FileNotFoundError(
                f"{npz_path} not found; run download_sudoku_extreme(split={split!r}) "
                f"or pass download=True."
            )
        data = np.load(npz_path, allow_pickle=True)
        sols = data["solutions"].astype(np.int64)
        masks = data["clue_mask"].astype(np.int64)
        ratings = data["ratings"].astype(np.int64)
        sources = data["sources"]
        self.difficulty_edges = tuple(difficulty_edges)

        idx = self._select(
            len(sols), ratings, max_samples, subsample_seed, balanced_by_rating
        )
        self.solutions = sols[idx]
        self.clue_masks = masks[idx]
        self.ratings = ratings[idx]
        self.sources = sources[idx]
        self.buckets = difficulty_bucket(self.ratings, self.difficulty_edges)

    def _select(self, n, ratings, max_samples, seed, balanced):
        """Return the row indices to keep (subsample / cap / stratify)."""
        if max_samples is None or max_samples >= n:
            return np.arange(n)
        if seed is None:
            return np.arange(max_samples)  # head of file (caller opted out of shuffle)
        rng = np.random.default_rng(seed)
        if not balanced:
            return rng.permutation(n)[:max_samples]
        # Balanced-by-rating: take ~max_samples/num_buckets from each non-empty
        # bucket, topping up from the rest if some buckets are too small.
        buckets = difficulty_bucket(ratings, self.difficulty_edges)
        n_buckets = len(self.difficulty_edges) - 1
        per = max(1, max_samples // n_buckets)
        chosen = []
        for b in range(n_buckets):
            members = np.where(buckets == b)[0]
            if len(members) == 0:
                continue
            take = min(per, len(members))
            chosen.append(rng.choice(members, size=take, replace=False))
        chosen = np.concatenate(chosen) if chosen else np.arange(0)
        if len(chosen) < max_samples:  # top up to the cap from the remainder
            remaining = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
            extra = rng.choice(
                remaining,
                size=min(max_samples - len(chosen), len(remaining)),
                replace=False,
            )
            chosen = np.concatenate([chosen, extra])
        rng.shuffle(chosen)
        return chosen[:max_samples]

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        sol = torch.from_numpy(self.solutions[idx]).long()
        mask = torch.from_numpy(self.clue_masks[idx]).long()
        return torch.stack([sol, mask], dim=0)  # (2, 81)

    def bucket_names(self):
        """Difficulty bucket name per puzzle (parallel to ``.buckets``)."""
        names = (
            DIFFICULTY_NAMES
            if self.difficulty_edges == DIFFICULTY_EDGES
            else tuple(
                f"r{self.difficulty_edges[i]}-{self.difficulty_edges[i + 1]}"
                for i in range(len(self.difficulty_edges) - 1)
            )
        )
        return [names[b] for b in self.buckets]


BOX = 3


def _dims(box):
    """``(grid_side, L)`` for a Sudoku with ``box x box`` subgrids.

    ``box=3`` -> ``(9, 81)`` (standard); ``box=2`` -> ``(4, 16)`` (the small 4x4
    variant). All grid-size-dependent primitives below derive their geometry from
    ``box`` so the same code drives any order while defaulting to 9x9.
    """
    grid = box * box
    return grid, grid * grid


def is_valid_sudoku(grid_flat, box=BOX):
    """True if a length-L array (values 1..grid) is a valid complete Sudoku."""
    grid, _ = _dims(box)
    g = np.asarray(grid_flat).reshape(grid, grid)
    target = set(range(1, grid + 1))
    for i in range(grid):
        if set(g[i, :].tolist()) != target:
            return False
        if set(g[:, i].tolist()) != target:
            return False
    for br in range(0, grid, box):
        for bc in range(0, grid, box):
            block = g[br : br + box, bc : bc + box].reshape(-1)
            if set(block.tolist()) != target:
                return False
    return True


def valid_mask(grids, box=BOX):
    """Vectorized validity over a batch [B, L] (values expected in 1..grid) -> bool[B]."""
    grid, _ = _dims(box)
    g = np.asarray(grids)
    B = g.shape[0]
    g = g.reshape(B, grid, grid)

    def _all_perms(blocks):
        # blocks: [B, grid, grid]; each inner row must be a permutation of 1..grid.
        s = np.sort(blocks, axis=-1)
        ref = np.arange(1, grid + 1)
        return (s == ref).all(axis=(-1, -2))

    rows_ok = _all_perms(g)
    cols_ok = _all_perms(g.transpose(0, 2, 1))
    boxes = (
        g.reshape(B, box, box, box, box).transpose(0, 1, 3, 2, 4).reshape(B, grid, grid)
    )
    boxes_ok = _all_perms(boxes)
    return rows_ok & cols_ok & boxes_ok


def _axis_permutation(rng, box=3):
    """Length-``side`` index permutation (``side = box*box``): permute the ``box``
    blocks, then the ``box`` lines within each block. This is exactly the row (or
    column) part of the Sudoku symmetry group — it never mixes lines across block
    boundaries, so box constraints are preserved.

    Args:
        rng: a ``np.random.Generator``.
        box: Sudoku order (3 for the standard 9x9 grid).
    """
    blocks = rng.permutation(box)  # order of the box blocks
    perm = []
    for b in blocks:
        lines = box * b + rng.permutation(box)  # box lines within block b, shuffled
        perm.extend(lines.tolist())
    return np.asarray(perm, dtype=np.intp)  # (side,)


class SudokuSymmetryAugmentedDataset(Dataset):
    """Apply a random validity-preserving Sudoku symmetry, on the fly.

    Wraps a conditional Sudoku dataset whose ``__getitem__`` returns a ``(2, L)``
    LongTensor ``stack([solution, clue_mask])`` (L = ``box**4``; the default 9x9
    Sudoku has L=81, box=3). With probability ``aug_prob`` a sample is transformed by
    a random element of the Sudoku symmetry group, applied **jointly** to the 9x9
    ``solution`` and ``clue_mask`` so clue positions track their cells. The group:

      - **transpose** the grid (rows<->cols),
      - permute the 3 **bands** (row-blocks) and, within each band, its 3 **rows**,
      - permute the 3 **stacks** (col-blocks) and, within each stack, its 3 **cols**,
      - **relabel** the digits ``1..9`` by a random permutation (solution values only;
        the positional ``clue_mask`` is untouched by relabelling).

    Every such map sends a valid grid to a valid grid and keeps the (solution,
    clue_mask) pair self-consistent, so the conditional FLM / MDLM loss and sampler
    are unchanged. Unlike :class:`ExtraClueAugmentedDataset` (which only adds clues),
    this *expands the effective training set* with fresh rule-abiding puzzles — useful
    in the small-data regime where puzzle count, not clue count, is the bottleneck.

    Args:
        dataset: wrapped conditional Sudoku dataset (returns ``(2, L)``).
        aug_prob: per-sample probability of applying a symmetry (``0`` = passthrough).
        seed: base RNG seed; each DataLoader worker derives an independent stream
            from ``seed + worker_id`` (fork-safe, matching ExtraClueAugmentedDataset).
        relabel_digits: if True, also apply a random digit permutation (the largest
            multiplier in the group). The spatial symmetries are always applied.
        box: Sudoku order (3 for the standard 9x9 grid; ``box**2`` cells per side).
    """

    def __init__(self, dataset, aug_prob=0.0, seed=0, relabel_digits=True, box=3):
        self.dataset = dataset
        self.aug_prob = float(aug_prob)
        self._seed = int(seed)
        self.relabel_digits = bool(relabel_digits)
        self.box = int(box)
        self.side = self.box * self.box  # 9
        self._rng = None  # lazily created per-worker (see _get_rng)

    def __len__(self):
        return len(self.dataset)

    def _get_rng(self):
        """Per-worker NumPy RNG, seeded once off the worker id (fork-safe)."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            self._rng = np.random.default_rng(self._seed + worker_id)
        return self._rng

    def __getitem__(self, idx):
        x0 = self.dataset[idx]  # (2, L) LongTensor: [solution, clue_mask]
        sol, mask = x0[0], x0[1]
        if self.aug_prob <= 0.0:
            return torch.stack([sol, mask], dim=0)

        rng = self._get_rng()
        if rng.random() >= self.aug_prob:
            return torch.stack([sol, mask], dim=0)

        # Delegate to the shared pure transform (single source of truth, also used by
        # the fixed-dataset materializer). Operates on (L,) numpy, always applies.
        s, m = apply_sudoku_symmetry(
            sol.numpy(),
            mask.numpy(),
            rng,
            box=self.box,
            relabel_digits=self.relabel_digits,
        )
        sol = torch.from_numpy(s).long()
        mask = torch.from_numpy(m).long()
        return torch.stack([sol, mask], dim=0)


def apply_sudoku_symmetry(sol, mask, rng, *, box=3, relabel_digits=True):
    """Apply ONE random validity-preserving Sudoku symmetry to ``(sol, mask)``.

    The full Sudoku symmetry group element: optional transpose (p=0.5), a row
    permutation (band + within-band line shuffle), a column permutation (stack +
    within-stack line shuffle), and — when ``relabel_digits`` — a random relabelling
    of the digits ``1..box**2`` (0 reserved/blank is fixed). Applied **jointly** to
    the solution and clue mask so clue positions track their cells; every such map
    sends a valid grid to a valid grid and keeps the (solution, clue_mask) pair
    self-consistent. This is the same transform as HRM's ``shuffle_sudoku``.

    Unlike :class:`SudokuSymmetryAugmentedDataset` (which gates on a per-sample
    ``aug_prob``), this ALWAYS applies a transform — the caller decides how many
    augmentations to draw (e.g. the fixed-dataset materializer in
    ``experiments/sudoku_extreme_1k_selfcond/1_generate_data.py``).

    Args:
        sol:  ``(L,)`` int numpy array of digits ``1..box**2`` (``L = (box*box)**2``).
        mask: ``(L,)`` int numpy array, ``1`` where a clue is given.
        rng:  a ``np.random.Generator`` (the source of all randomness for this draw).
        box:  Sudoku order (3 for the standard 9x9 grid).
        relabel_digits: also apply a random digit permutation (the largest multiplier
            in the group). The spatial symmetries are always applied.

    Returns:
        ``(sol_out, mask_out)`` as ``(L,)`` int numpy arrays (same dtype as inputs).
    """
    side = box * box
    s = np.asarray(sol).reshape(side, side)
    m = np.asarray(mask).reshape(side, side)
    if rng.random() < 0.5:  # transpose
        s, m = s.T.copy(), m.T.copy()
    rp = _axis_permutation(rng, box)  # rows: band + within-band
    cp = _axis_permutation(rng, box)  # cols: stack + within-stack
    s = s[rp][:, cp]
    m = m[rp][:, cp]
    if relabel_digits:
        # 0 is reserved (never a real digit); permute 1..side among themselves.
        sigma = np.empty(side + 1, dtype=s.dtype)
        sigma[0] = 0
        sigma[1:] = 1 + rng.permutation(side)
        s = sigma[s]
    return (np.ascontiguousarray(s).reshape(-1), np.ascontiguousarray(m).reshape(-1))


REPO_ID = "locuslab/EqR-data"
SOURCE_DATASET = "sudoku-extreme-1k-aug-1000"
SOURCE_REVISION = "bcb7d45c157c06a054b715933557d266f5bd9d7f"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sudoku_lite(
    output_dir, *, cache_dir=None, num_samples=2048, revision=SOURCE_REVISION
):
    """Build the released EqR Sudoku-Lite benchmark and its provenance manifest."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    output_dir = Path(output_dir)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit(
            "install huggingface_hub to download the released EqR data"
        ) from error

    revision = HfApi().dataset_info(REPO_ID, revision=revision).sha
    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=[f"{SOURCE_DATASET}/test/*"],
        )
    )
    source = snapshot / SOURCE_DATASET / "test"
    inputs = np.load(source / "all__inputs.npy", mmap_mode="r")
    labels = np.load(source / "all__labels.npy", mmap_mode="r")
    puzzle_indices = np.load(source / "all__puzzle_indices.npy", mmap_mode="r")
    group_indices = np.load(source / "all__group_indices.npy", mmap_mode="r")
    n = num_samples
    if min(len(inputs), len(labels)) < n:
        raise RuntimeError(f"released test split contains fewer than {n} examples")
    expected = np.arange(n + 1)
    if not np.array_equal(puzzle_indices[: n + 1], expected):
        raise RuntimeError("released rows are not one-example-per-puzzle as expected")
    if not np.array_equal(group_indices[: n + 1], expected):
        raise RuntimeError("released rows are not one-puzzle-per-group as expected")

    solutions = np.asarray(labels[:n], dtype=np.int64) - 1
    clue_mask = np.asarray(inputs[:n]) != 1
    if not np.array_equal(np.asarray(inputs[:n])[clue_mask] - 1, solutions[clue_mask]):
        raise RuntimeError("released clue tokens disagree with the solution labels")

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark = output / "benchmark.npz"
    np.savez_compressed(
        benchmark,
        solutions=solutions.astype(np.uint8),
        clue_mask=clue_mask.astype(np.uint8),
        source_indices=np.arange(n, dtype=np.int64),
    )
    manifest = {
        "name": f"EqR Sudoku-Lite first-{n}",
        "source_repo": REPO_ID,
        "source_revision": revision,
        "source_dataset": SOURCE_DATASET,
        "selection": {"method": "first_n_in_released_test_order", "n": n, "seed": None},
        "note": "The evaluation seed changes stochastic trajectories, not puzzle membership.",
        "source_sha256": {
            path.name: _file_sha256(path)
            for path in sorted(source.glob("*"))
            if path.is_file()
        },
        "benchmark": {
            "path": str(benchmark),
            "sha256": _file_sha256(benchmark),
            "representation": "solutions[uint8,N,81] + clue_mask[uint8,N,81]",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main(argv=None):
    """Download Sudoku-Extreme or prepare the optional Sudoku-Lite benchmark."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument(
        "--lite", action="store_true", help="Prepare EqR Sudoku-Lite instead"
    )
    ap.add_argument("--root", default=None, help="Dataset cache root")
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--revision", default=SOURCE_REVISION)
    args = ap.parse_args(argv)
    if args.lite:
        if args.output_dir is None:
            ap.error("--lite requires --output-dir")
        prepare_sudoku_lite(
            args.output_dir,
            cache_dir=args.cache_dir,
            num_samples=args.num_samples,
            revision=args.revision,
        )
    else:
        if args.output_dir is not None or args.cache_dir is not None:
            ap.error("--output-dir and --cache-dir require --lite")
        download_sudoku_extreme(args.root, args.split, force=args.force)


if __name__ == "__main__":
    main()
