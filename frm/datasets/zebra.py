"""The **Shah et al.** Zebra / Einstein logic-puzzle dataset (Kulin Shah et al., 2024).

Wraps the Zebra data from "Causal Language Modeling Can Elicit Search and Reasoning
Capabilities on Logic Puzzles" (https://arxiv.org/abs/2409.10502,
code: https://github.com/kulinshah98/llm-reasoning-logic-puzzles). ~1.5M train and
~0.1M test puzzles, sizes uniformly mixed over 3..6 houses x 3..6 attributes. Hosted in
the *same* Google Drive folder as our Shah Sudoku (``zebra-{train,test}-data.pkl``).

**Raw format** (pickled ``list`` of puzzles; each puzzle is ``[token_seq, answer_table,
solving_order]``)::

    token_seq    : list[str] — the full LM input: serialized clue tokens, then a single
                   "ANSWER" token, then a (partial, solver-ordered) answer trace. The
                   relation vocabulary is {=, !=, immediate-left, left-of, inbetween,
                   ends, nbr} with LHS/RHS/c/n/CLUE_END structural tokens and int tokens
                   "0".."5". (We use the clue prefix verbatim; see below.)
    answer_table : int [n_attrs+1, n_houses] — row 0 = house numbers 0..n-1 (trivial);
                   rows 1..n_attrs = the assignment (value index 0..n-1) of each attribute
                   in each house. This is the always-complete, fixed-shape solution.
    solving_order: int [*, 2] — the solver's (attr, house) decision order (unused here).

**Our representation** (parsed + cached as ``.npz``). We mimic Shah's tokenization
*where it defines the puzzle* — the clue prefix and vocabulary are byte-identical — but
replace the Shah-AR-specific solver *trace* after ``ANSWER`` with the **canonical
solution grid** drawn from ``answer_table``. The trace is variable-length, partial, and
order-dependent (a poor target for a non-causal flow model); ``answer_table`` is a clean
fixed-shape grid that plugs straight into the conditional FLM/MDLM/FMLM solver + MCTS.

Each puzzle is encoded as a length-``L`` token sequence::

    [ clue tokens ... CLUE_END ] ANSWER [ a(0,0) a(0,1) ... a(A-1,H-1) | EXTRA pad ] EXTRA ...
    └────────── clue_mask = 1 (clamped prefix + ANSWER) ──────────┘└ =0 on real cells ┘└ =1 ┘

- The answer block is a fixed ``MAX_ATTRS x MAX_HOUSES`` (=6x6) grid placed immediately
  after ``ANSWER``; cell ``(a, h)`` for ``a < n_attrs, h < n_houses`` holds the value
  token and is *free* (``clue_mask = 0``); padding cells and the trailing tail are
  ``EXTRA`` (``clue_mask = 1``). Row 0 of ``answer_table`` (house numbers) is dropped —
  it carries no information.
- Vocabulary mirrors Shah exactly: ``{tok: i+1}`` over ``sorted(set(all clue/seq
  tokens))`` (ids ``1..V``), ``EXTRA = V+1``, id ``0`` reserved/blank. ``vocab_size =
  V + 2``. Built on **train** and frozen to ``vocab.json``; both splits share it.

``__getitem__`` returns the packed ``stack([tokens, clue_mask])`` ``(2, L)`` LongTensor —
a drop-in for the conditional loss / sampler (same contract as ``ShahSudokuDataset``).

Raw data is downloaded automatically from Google Drive (``gdown``) on first use (reusing
the Shah Sudoku folder if already present) and parsed into a compressed ``.npz`` cache;
all artifacts land in the localscratch mirror — nothing written to the git repo.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sudoku import default_dataset_root

GDRIVE_FOLDER_ID = "1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW"
FILES = {
    "train": "zebra-train-data.pkl",
    "test": "zebra-test-data.pkl",
}
SPLITS = ("train", "test")
ANSWER_TOKEN = "ANSWER"
PAD_TOKEN = "EXTRA"
MAX_ATTRS = 6
MAX_HOUSES = 6
ANSWER_CELLS = MAX_ATTRS * MAX_HOUSES  # 36 fixed answer-grid slots after ANSWER


# ---------------------------------------------------------------------------
# Raw-file location (reuse the already-downloaded Shah folder if present)
# ---------------------------------------------------------------------------
def _find_or_download_raw(root, split, force=False):
    """Return a path to the raw ``zebra-<split>-data.pkl``, downloading if needed.

    Searches the zebra cache and the sibling ``shah_sudoku/raw`` dir (the Shah Google
    Drive folder holds both Sudoku and Zebra, so a prior Sudoku download already has the
    pkl). Only calls ``gdown`` when no copy exists anywhere.
    """
    name = FILES[split]
    zebra_raw = root / "zebra" / "raw"
    candidates = [zebra_raw / name, root / "shah_sudoku" / "raw" / name]
    if not force:
        for c in candidates:
            if c.exists():
                return c

    zebra_raw.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required to download the Shah Zebra dataset. "
            "Install it with: pip install gdown"
        ) from e
    print(f"[zebra] downloading Google Drive folder -> {zebra_raw} ...")
    gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(zebra_raw), quiet=False)
    target = zebra_raw / name
    if not target.exists():
        raise FileNotFoundError(
            f"Expected {target} after download — the Google Drive folder may have "
            f"changed. Files present: {list(zebra_raw.iterdir())}"
        )
    return target


def _load_pickle(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Shared (train-derived) vocab + sequence length
# ---------------------------------------------------------------------------
def _build_shared(root, force=False):
    """Ensure ``zebra/vocab.json`` + ``zebra/meta.json`` exist (built from train).

    Mirrors Shah's vocab construction: ``{tok: i+1}`` over the sorted token set, plus an
    ``EXTRA`` pad token at ``V+1`` (id 0 reserved). ``meta`` records ``seq_len`` (the max
    clue-prefix-through-ANSWER length over train, ``+ ANSWER_CELLS``), ``vocab_size``, the
    ``answer_token_id``, and the fixed answer-grid shape.
    """
    cache = root / "zebra"
    cache.mkdir(parents=True, exist_ok=True)
    vocab_path, meta_path = cache / "vocab.json", cache / "meta.json"
    if vocab_path.exists() and meta_path.exists() and not force:
        return json.loads(vocab_path.read_text()), json.loads(meta_path.read_text())

    raw = _find_or_download_raw(root, "train", force=force)
    print(f"[zebra] building vocab + meta from {raw} ...")
    data = _load_pickle(raw)

    vocab = sorted({tok for p in data for tok in p[0]})
    vocab_dict = {tok: i + 1 for i, tok in enumerate(vocab)}  # ids 1..V (Shah scheme)
    vocab_dict[PAD_TOKEN] = len(vocab) + 1  # EXTRA = V+1
    if ANSWER_TOKEN not in vocab_dict:
        raise ValueError(f"'{ANSWER_TOKEN}' token missing from Zebra vocab: {vocab}")
    answer_token_id = vocab_dict[ANSWER_TOKEN]

    max_prefix = 0
    for p in data:
        ai = p[0].index(ANSWER_TOKEN)
        max_prefix = max(max_prefix, ai + 1)  # clue tokens + ANSWER
    seq_len = max_prefix + ANSWER_CELLS

    meta = {
        "seq_len": int(seq_len),
        "vocab_size": int(len(vocab) + 2),  # ids 0..V+1
        "answer_token_id": int(answer_token_id),
        "pad_id": int(vocab_dict[PAD_TOKEN]),
        "max_attrs": MAX_ATTRS,
        "max_houses": MAX_HOUSES,
        "answer_cells": ANSWER_CELLS,
        "max_prefix": int(max_prefix),
    }
    vocab_path.write_text(json.dumps(vocab_dict, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[zebra] vocab={len(vocab)} tokens (+0,EXTRA -> vocab_size={meta['vocab_size']}), "
        f"seq_len={seq_len} (max clue prefix {max_prefix} + {ANSWER_CELLS} answer cells)"
    )
    return vocab_dict, meta


# ---------------------------------------------------------------------------
# Parse + cache one split
# ---------------------------------------------------------------------------
def _parse_split(data, vocab_dict, meta):
    """Encode a list of raw puzzles into fixed-shape arrays (see module docstring)."""
    L = meta["seq_len"]
    pad_id = meta["pad_id"]
    n = len(data)

    tokens = np.full((n, L), pad_id, dtype=np.int16)
    clue_mask = np.ones((n, L), dtype=np.int8)  # 1 = clamped; flip answer cells
    answer_start = np.zeros(n, dtype=np.int16)
    n_attrs = np.zeros(n, dtype=np.int8)
    n_houses = np.zeros(n, dtype=np.int8)
    # Raw value grid (0..H-1) for plotting; -1 marks unused cells.
    answer_grid = np.full((n, MAX_ATTRS, MAX_HOUSES), -1, dtype=np.int8)

    for i, p in enumerate(data):
        seq, table = p[0], np.asarray(p[1])
        ai = seq.index(ANSWER_TOKEN)
        prefix = seq[: ai + 1]  # clue tokens + ANSWER
        plen = len(prefix)
        if plen + ANSWER_CELLS > L:
            raise ValueError(
                f"puzzle {i}: prefix {plen} + {ANSWER_CELLS} answer cells exceeds "
                f"seq_len {L}; rebuild the vocab/meta with force=True."
            )
        tokens[i, :plen] = [vocab_dict[t] for t in prefix]

        na, nh = table.shape[0] - 1, table.shape[1]  # drop the house-number row
        n_attrs[i], n_houses[i] = na, nh
        answer_start[i] = plen
        # Canonical (attr-major, house-minor) answer block, padded to MAX_ATTRS x MAX_HOUSES.
        block = np.full(ANSWER_CELLS, pad_id, dtype=np.int16)
        block_free = np.zeros(ANSWER_CELLS, dtype=np.int8)  # 1 where a real value cell
        for a in range(na):
            for h in range(nh):
                v = int(table[a + 1, h])
                block[a * MAX_HOUSES + h] = vocab_dict[str(v)]
                block_free[a * MAX_HOUSES + h] = 1
                answer_grid[i, a, h] = v
        tokens[i, plen : plen + ANSWER_CELLS] = block
        clue_mask[i, plen : plen + ANSWER_CELLS] = 1 - block_free  # 0 on real cells

    return tokens, clue_mask, answer_start, n_attrs, n_houses, answer_grid


def download_zebra(root=None, split="test", force=False):
    """Ensure the parsed Zebra ``split`` is cached; return the ``.npz`` path.

    Locates (or downloads) the raw pkl, builds the shared train vocab/meta if needed,
    parses the split, and writes ``<root>/zebra/<split>.npz``. Idempotent.

    Requires ``gdown`` (``pip install gdown``) only when no raw copy is on disk.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root) if root is not None else default_dataset_root()
    npz_path = root / "zebra" / f"{split}.npz"
    if npz_path.exists() and not force:
        return npz_path

    vocab_dict, meta = _build_shared(root, force=force)
    raw = _find_or_download_raw(root, split, force=force)
    print(f"[zebra] parsing {raw} ...")
    data = _load_pickle(raw)
    tokens, clue_mask, answer_start, n_attrs, n_houses, answer_grid = _parse_split(
        data, vocab_dict, meta
    )
    np.savez_compressed(
        npz_path,
        tokens=tokens,
        clue_mask=clue_mask,
        answer_start=answer_start,
        n_attrs=n_attrs,
        n_houses=n_houses,
        answer_grid=answer_grid,
    )
    free = (clue_mask == 0).sum(1)
    print(
        f"[zebra] cached {len(tokens):,} puzzles -> {npz_path} "
        f"(free cells {free.min()}-{free.max()}, sizes "
        f"{n_attrs.min()}-{n_attrs.max()} attrs x {n_houses.min()}-{n_houses.max()} houses)"
    )
    return npz_path


# ---------------------------------------------------------------------------
# Decode / grid helpers
# ---------------------------------------------------------------------------
def parse_sizes(sizes):
    """Parse ``['3x3', '4x4']`` -> ``[(3, 3), (4, 4)]``; ``None`` -> ``None`` (all sizes)."""
    if not sizes:
        return None
    out = []
    for s in sizes:
        a, h = str(s).lower().split("x")
        out.append((int(a), int(h)))
    return out


def load_vocab(root=None):
    """Return ``(vocab_dict, reverse_vocab, meta)`` for the cached Zebra dataset."""
    root = Path(root) if root is not None else default_dataset_root()
    cache = root / "zebra"
    vocab_dict = json.loads((cache / "vocab.json").read_text())
    meta = json.loads((cache / "meta.json").read_text())
    reverse = {v: k for k, v in vocab_dict.items()}
    reverse[0] = "<pad0>"
    return vocab_dict, reverse, meta


def decode(ids, reverse_vocab):
    """Map a sequence of token ids back to their string tokens (for debugging)."""
    return [reverse_vocab.get(int(i), f"<{int(i)}>") for i in np.asarray(ids).ravel()]


def tokens_to_grid(tokens, answer_start, n_attrs, n_houses, reverse_vocab):
    """Reconstruct the predicted ``(n_attrs, n_houses)`` value grid from a token seq.

    Reads the fixed ``MAX_ATTRS x MAX_HOUSES`` answer block at ``answer_start`` and
    decodes each real cell's token back to its integer value. Returns an int array with
    ``-1`` at out-of-range cells.
    """
    tokens = np.asarray(tokens).ravel()
    grid = np.full((MAX_ATTRS, MAX_HOUSES), -1, dtype=np.int64)
    for a in range(int(n_attrs)):
        for h in range(int(n_houses)):
            tok = reverse_vocab.get(int(tokens[answer_start + a * MAX_HOUSES + h]), "")
            grid[a, h] = int(tok) if tok.lstrip("-").isdigit() else -1
    return grid[: int(n_attrs), : int(n_houses)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ZebraDataset(Dataset):
    """Torch dataset over the cached Shah Zebra split (conditional / prefix format).

    Auto-downloads + caches on first use. ``__getitem__`` returns the packed ``(2, L)``
    LongTensor ``stack([tokens, clue_mask])`` — a drop-in for the conditional loss /
    sampler (same contract as ``ShahSudokuDataset``).

    Args:
        root: cache root (defaults to the scratch mirror, see ``default_dataset_root``).
        split: ``"test"`` (default) or ``"train"``.
        download: auto-download + parse if the cache is missing (default True).
        max_samples: keep at most this many puzzles (after the size filter + subsampling).
        subsample_seed: if set, shuffle deterministically before applying ``max_samples``.
        sizes: optional iterable of ``(n_attrs, n_houses)`` tuples; keep only puzzles of
            those sizes (e.g. ``[(3, 3)]`` for the small fixed-size study). ``None`` keeps
            all 3..6 x 3..6 sizes.
        repack: when a size filter is active, trim the trailing all-clamped columns so the
            DiT runs at a shorter ``seq_len`` (big speedup for small puzzles). The trimmed
            length is derived from ``meta`` (``max_prefix`` for the kept sizes is unknown,
            so we use the subset's own max) — pass the resulting ``.seq_len`` consistently
            between train and eval (it is stored in the checkpoint config).
        pin_answer: pin the answer grid to a **constant absolute position** for every puzzle
            instead of letting it float right after a variable-length clue prefix. The raw
            encoding places ``ANSWER`` + the 6x6 grid immediately after the clues, so
            ``answer_start`` ranges over ~[34, 315] across sizes — a non-causal DiT with
            absolute position embeddings then has to learn the answer-grid structure at every
            offset, which badly under-trains even trivial sizes (see results.md 2026-06-11).
            With ``pin_answer=True`` the clue tokens stay left-anchored at index 0, the gap
            between them and ``ANSWER`` is filled with clamped pad, and ``ANSWER`` + grid are
            placed at a fixed anchor (``meta['max_prefix'] - 1``), so ``answer_start`` is
            constant (``= max_prefix``) for all puzzles. Train and eval **must** use the same
            ``pin_answer`` setting. Incompatible with ``repack`` (pinning needs the full L).

    Attributes: ``.tokens``, ``.clue_masks`` ``[N, L]`` int64; ``.answer_start`` ``[N]``;
        ``.n_attrs``, ``.n_houses`` ``[N]``; ``.answer_grid`` ``[N, 6, 6]`` (raw values,
        -1 = unused); and ``.seq_len``, ``.vocab_size``, ``.meta`` read from the shared meta.
    """

    def __init__(
        self,
        root=None,
        split="test",
        download=True,
        max_samples=None,
        subsample_seed=None,
        sizes=None,
        repack=False,
        force_seq_len=None,
        pin_answer=False,
    ):
        if download:
            npz_path = download_zebra(root, split)
        else:
            root_p = Path(root) if root is not None else default_dataset_root()
            npz_path = root_p / "zebra" / f"{split}.npz"
        if not Path(npz_path).exists():
            raise FileNotFoundError(
                f"{npz_path} not found; run download_zebra(split={split!r}) or pass "
                "download=True."
            )

        _, self.reverse_vocab, self.meta = load_vocab(root)
        self.seq_len = self.meta["seq_len"]
        self.vocab_size = self.meta["vocab_size"]

        data = np.load(npz_path)
        n_attrs_all = data["n_attrs"]
        n_houses_all = data["n_houses"]
        keep = np.ones(len(n_attrs_all), dtype=bool)
        if sizes is not None:
            want = {(int(a), int(h)) for a, h in sizes}
            keep = np.array(
                [(int(a), int(h)) in want for a, h in zip(n_attrs_all, n_houses_all)]
            )
        kept = np.where(keep)[0]
        idx = kept[self._select(len(kept), max_samples, subsample_seed)]

        self.tokens = data["tokens"][idx].astype(np.int64)
        self.clue_masks = data["clue_mask"][idx].astype(np.int64)
        self.answer_start = data["answer_start"][idx].astype(np.int64)
        self.n_attrs = n_attrs_all[idx].astype(np.int64)
        self.n_houses = n_houses_all[idx].astype(np.int64)
        self.answer_grid = data["answer_grid"][idx].astype(np.int64)

        # Pin the answer grid to a constant absolute position (Option B: clues stay
        # left-anchored at index 0, pad fills the gap, ANSWER + grid sit at a fixed anchor).
        # This removes the floating-answer-position difficulty that under-trains the model.
        if pin_answer:
            if repack:
                raise ValueError(
                    "pin_answer is incompatible with repack (pinning needs full L)"
                )
            self._pin_answer()

        # Trim trailing all-clamped (pad) columns for a shorter DiT length. Eval must
        # reuse the training length: pass force_seq_len (= checkpoint's seq_len).
        if force_seq_len is not None:
            if (self.clue_masks[:, force_seq_len:] == 0).any():
                raise ValueError(
                    "Checkpoint sequence length truncates Zebra answer cells; "
                    "use the matching dataset split and preprocessing"
                )
            self.tokens = self.tokens[:, :force_seq_len]
            self.clue_masks = self.clue_masks[:, :force_seq_len]
            self.seq_len = int(force_seq_len)
        elif repack and len(self.tokens):
            L_sub = int((self.answer_start + self.meta["answer_cells"]).max())
            self.tokens = self.tokens[:, :L_sub]
            self.clue_masks = self.clue_masks[:, :L_sub]
            self.seq_len = L_sub

    def _pin_answer(self):
        """Rewrite tokens/clue_masks so the ANSWER+grid block sits at a constant position.

        Each raw row is ``[clue tokens ... ANSWER][grid][pad...]`` with the ANSWER+grid block
        starting at ``answer_start`` (variable). We move that whole ``1 + answer_cells`` block
        (ANSWER + the 6x6 grid) to a fixed anchor ``pin_start = max_prefix - 1`` so the grid
        always starts at ``max_prefix``. Clue tokens are left where they are (anchored at 0);
        the gap between the last clue and the anchor is filled with clamped pad. Idempotent
        only for a never-pinned array — call once.
        """
        pad_id = self.meta["pad_id"]
        cells = self.meta["answer_cells"]
        max_prefix = self.meta["max_prefix"]
        L = self.seq_len
        pin_start = (
            max_prefix - 1
        )  # index where ANSWER lands; grid at [max_prefix, ...]
        grid_start = max_prefix
        assert grid_start + cells <= L, f"pin overflow: {grid_start}+{cells} > {L}"

        n = len(self.tokens)
        new_tok = np.full((n, L), pad_id, dtype=np.int64)
        new_mask = np.ones((n, L), dtype=np.int64)  # 1 = clamped everywhere by default
        for i in range(n):
            st = int(
                self.answer_start[i]
            )  # old ANSWER index (= #clues incl ANSWER - 1 ... = prefix len - 1? see below)
            # old layout: clues occupy [0, st), ANSWER at index st? No: answer_start == prefix
            # length (clues + ANSWER), grid at [st, st+cells). The ANSWER token is the last
            # prefix token, at index st-1. So the movable block is [st-1, st+cells): ANSWER+grid.
            blk_lo = st - 1
            block_tok = self.tokens[
                i, blk_lo : st + cells
            ]  # ANSWER + grid, length 1+cells
            block_mask = self.clue_masks[i, blk_lo : st + cells]
            # clue tokens (everything before ANSWER) stay at [0, st-1)
            new_tok[i, :blk_lo] = self.tokens[i, :blk_lo]
            new_mask[i, :blk_lo] = self.clue_masks[
                i, :blk_lo
            ]  # clamped clues (already 1)
            # gap [st-1, pin_start) stays pad + clamped (defaults). Place block at the anchor.
            new_tok[i, pin_start : pin_start + 1 + cells] = block_tok
            new_mask[i, pin_start : pin_start + 1 + cells] = block_mask
            self.answer_start[i] = grid_start
        self.tokens = new_tok
        self.clue_masks = new_mask

    @staticmethod
    def _select(n, max_samples, seed):
        if max_samples is None or max_samples >= n:
            return np.arange(n)
        if seed is None:
            return np.arange(max_samples)
        rng = np.random.default_rng(seed)
        return rng.permutation(n)[:max_samples]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = torch.from_numpy(self.tokens[idx]).long()
        mask = torch.from_numpy(self.clue_masks[idx]).long()
        return torch.stack([tok, mask], dim=0)  # (2, L)


# ---------------------------------------------------------------------------
# Metrics (lifted from paper_experiments/train_zebra/4_evaluate.py so any
# experiment can reuse them without importlib-loading that training script).
# ---------------------------------------------------------------------------


def compute_zebra_metrics(samples, clue_mask, gt, n_attrs, n_houses):
    """Exact-match metrics for Zebra completions.

    Args:
        samples: int [N, L] predicted token sequences.
        clue_mask: [N, L] {0,1}; 1 marks clamped (clue/pad) positions, 0 the answer cells.
        gt: int [N, L] ground-truth token sequences.
        n_attrs, n_houses: [N] puzzle dimensions (for the per-size breakdown).

    Returns:
        (overall dict, by_size list of dicts). ``solved`` = fraction with *every* free
        cell correct; ``token_acc`` = fraction of free cells correct; ``cells_correct`` =
        per-puzzle mean free-cell accuracy averaged over puzzles.
    """
    samples = np.asarray(samples)
    clue = np.asarray(clue_mask).astype(bool)
    gt = np.asarray(gt)
    n_attrs = np.asarray(n_attrs)
    n_houses = np.asarray(n_houses)
    free = ~clue
    n = len(samples)

    correct_cell = samples == gt
    solved = (correct_cell | clue).all(axis=1)  # all free cells correct
    free_counts = free.sum(1)
    correct_free = (correct_cell & free).sum(1)
    per_cells = correct_free / np.maximum(free_counts, 1)

    overall = {
        "n_samples": int(n),
        "solved": float(solved.mean()) if n else 0.0,
        "n_solved": int(solved.sum()),
        "token_acc": float((correct_cell & free).sum() / max(int(free.sum()), 1)),
        "cells_correct": float(per_cells.mean()) if n else 0.0,
        "mean_free": float(free_counts.mean()) if n else 0.0,
    }

    by_size = []
    for a, h in sorted(set(zip(n_attrs.tolist(), n_houses.tolist()))):
        sel = (n_attrs == a) & (n_houses == h)
        cnt = int(sel.sum())
        by_size.append(
            {
                "n_attrs": int(a),
                "n_houses": int(h),
                "n": cnt,
                "solved": float(solved[sel].mean()) if cnt else 0.0,
                "token_acc": float(
                    (correct_cell[sel] & free[sel]).sum() / max(int(free[sel].sum()), 1)
                ),
            }
        )
    return overall, by_size


if __name__ == "__main__":  # `python -m frm.datasets.zebra`
    import argparse

    ap = argparse.ArgumentParser(description="Download + cache Shah Zebra dataset")
    ap.add_argument("--root", default=None, help="cache root (default: scratch mirror)")
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--force", action="store_true", help="re-parse even if cached")
    a = ap.parse_args()
    download_zebra(a.root, a.split, force=a.force)
