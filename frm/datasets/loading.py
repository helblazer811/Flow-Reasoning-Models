"""Shared dataset construction for training and evaluation."""

from .zebra import ZebraDataset, parse_sizes
from .maze_unique import MazeUniqueDataset
from .sudoku import SudokuExtremeDataset, SudokuSymmetryAugmentedDataset


def load_dataset(
    data, *, split, max_samples=None, seed=None, seq_len=None, augment=False
):
    if data.dataset == "zebra":
        ds = ZebraDataset(
            root=data.root,
            split=split,
            sizes=parse_sizes(data.sizes),
            max_samples=max_samples,
            subsample_seed=seed,
            repack=data.repack and not data.pin_answer,
            pin_answer=data.pin_answer,
            force_seq_len=seq_len,
        )
    elif data.dataset == "maze_unique":
        ds = MazeUniqueDataset(
            root=data.root, split=split, max_samples=max_samples, subsample_seed=seed
        )
    elif data.dataset == "sudoku":
        ds = SudokuExtremeDataset(
            root=data.root,
            split=split,
            max_samples=max_samples,
            subsample_seed=seed,
            balanced_by_rating=data.balanced_by_rating,
        )
        ds.seq_len, ds.vocab_size = 81, 10
        if augment and data.symmetry_probability > 0:
            ds = SudokuSymmetryAugmentedDataset(
                ds, aug_prob=data.symmetry_probability, seed=seed
            )
            ds.seq_len, ds.vocab_size = 81, 10
    else:
        raise ValueError(f"Unsupported dataset: {data.dataset!r}")
    if not len(ds):
        raise ValueError(f"No {data.dataset} puzzles in split {split!r}")
    return ds
