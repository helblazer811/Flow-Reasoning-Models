"""Procedurally generate symbolic Zebra/Einstein puzzles.

This is a clean-room reconstruction of the procedure documented by Shah et al.
(arXiv:2409.10502), not a byte-for-byte recreation of their unavailable pickle files.
It emits the raw three-item contract consumed by :mod:`frm.datasets.zebra`:

    [token_sequence, answer_table, solving_order]

The hidden solution contains one permutation per attribute.  True, nonduplicate clues are
sampled from the seven relation types described in the paper until the puzzle has a unique
solution under a finite-domain constraint solver.  A final deletion pass removes redundant
clues.  Generation is deterministic for a given seed and records a provenance manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RELATIONS = ("=", "!=", "immediate-left", "nbr", "ends", "left-of", "inbetween")
Operand = tuple[str, int, int]  # ("c", attribute, value) or ("n", 0, house)


@dataclass(frozen=True)
class Clue:
    relation: str
    operands: tuple[Operand, ...]

    def tokens(self) -> list[str]:
        # Binary clues exactly follow the public README grammar. Unary/ternary relations
        # extend it naturally by repeating RHS before each additional operand.
        out = [self.relation, "LHS", *map(str, self.operands[0])]
        for operand in self.operands[1:]:
            out.extend(("RHS", *map(str, operand)))
        out.append("CLUE_END")
        return out


def _loc(solution: np.ndarray, operand: Operand) -> int:
    kind, attr, value = operand
    if kind == "n":
        return value
    return int(np.flatnonzero(solution[attr] == value)[0])


def _holds(relation: str, locations: tuple[int, ...], n_houses: int) -> bool:
    if relation == "=":
        return locations[0] == locations[1]
    if relation == "!=":
        return locations[0] != locations[1]
    if relation == "immediate-left":
        return locations[0] + 1 == locations[1]
    if relation == "nbr":
        return abs(locations[0] - locations[1]) == 1
    if relation == "ends":
        return locations[0] in (0, n_houses - 1)
    if relation == "left-of":
        return locations[0] < locations[1]
    if relation == "inbetween":
        middle, left, right = locations
        return (left < middle < right) or (right < middle < left)
    raise ValueError(relation)


def _variables(n_attrs: int, n_houses: int) -> list[Operand]:
    return [("c", a, v) for a in range(n_attrs) for v in range(n_houses)]


def count_solutions(
    clues: list[Clue],
    n_attrs: int,
    n_houses: int,
    limit: int = 2,
    return_first: bool = False,
):
    """Count satisfying assignments up to ``limit`` using MRV backtracking."""
    variables = _variables(n_attrs, n_houses)
    assignment: dict[Operand, int] = {}
    used = [set() for _ in range(n_attrs)]
    incident = {v: [] for v in variables}
    for clue in clues:
        for op in clue.operands:
            if op[0] == "c":
                incident[op].append(clue)

    def location(op: Operand):
        return op[2] if op[0] == "n" else assignment.get(op)

    def compatible(clue: Clue) -> bool:
        locs = tuple(location(op) for op in clue.operands)
        if any(x is None for x in locs):
            return True
        return _holds(clue.relation, locs, n_houses)

    def domain(var: Operand):
        a = var[1]
        vals = []
        for h in range(n_houses):
            if h in used[a]:
                continue
            assignment[var] = h
            if all(compatible(c) for c in incident[var]):
                vals.append(h)
            del assignment[var]
        return vals

    count = 0
    first = None

    def search():
        nonlocal count, first
        if count >= limit:
            return
        if len(assignment) == len(variables):
            count += 1
            if first is None:
                first = dict(assignment)
            return
        best_var, best_dom = None, None
        for var in variables:
            if var in assignment:
                continue
            dom = domain(var)
            if not dom:
                return
            if best_dom is None or len(dom) < len(best_dom):
                best_var, best_dom = var, dom
                if len(dom) == 1:
                    break
        a = best_var[1]
        for h in best_dom:
            assignment[best_var] = h
            used[a].add(h)
            search()
            used[a].remove(h)
            del assignment[best_var]
            if count >= limit:
                return

    search()
    return (count, first) if return_first else count


def _candidate_pool(
    solution: np.ndarray, rng: np.random.Generator
) -> tuple[list[Clue], list[Clue]]:
    n_attrs, n_houses = solution.shape
    entities = _variables(n_attrs, n_houses)
    houses = [("n", 0, h) for h in range(n_houses)]
    anchors = [Clue("=", (op, ("n", 0, _loc(solution, op)))) for op in entities]
    pool: set[Clue] = set()

    # Sample a generous pool of true relational clues. Equality anchors are kept separate
    # and added only when the relational clues do not yet uniquely identify the solution.
    all_ops = entities + houses
    target = max(200, 20 * n_attrs * n_houses)
    for _ in range(target * 20):
        relation = rng.choice(RELATIONS[1:])
        if relation == "ends":
            ops = (entities[int(rng.integers(len(entities)))],)
        elif relation == "inbetween":
            ids = rng.choice(len(entities), size=3, replace=False)
            ops = tuple(entities[int(i)] for i in ids)
        else:
            ids = rng.choice(len(all_ops), size=2, replace=False)
            ops = tuple(all_ops[int(i)] for i in ids)
        locs = tuple(_loc(solution, op) for op in ops)
        if _holds(str(relation), locs, n_houses):
            pool.add(Clue(str(relation), ops))
        if len(pool) >= target:
            break
    pool = list(pool)
    rng.shuffle(pool)
    rng.shuffle(anchors)
    return pool, anchors


def generate_puzzle(
    n_attrs: int,
    n_houses: int,
    rng: np.random.Generator,
    minimize: bool = True,
    required_relation: str | None = None,
):
    solution = np.stack([rng.permutation(n_houses) for _ in range(n_attrs)])
    pool, anchors = _candidate_pool(solution, rng)
    required = None
    if required_relation == "=":
        required = anchors[0]
    elif required_relation is not None:
        required = next((c for c in pool if c.relation == required_relation), None)
        if required is None:
            raise RuntimeError(
                f"could not sample required relation {required_relation!r}"
            )
    clues: list[Clue] = [] if required is None else [required]

    # Mimic the documented add-until-solvable procedure. Check in small batches to avoid
    # spending most generation time proving high-multiplicity underconstrained states.
    for i, clue in enumerate(pool):
        if clue not in clues:
            clues.append(clue)
        if len(clues) >= n_attrs and (i + 1) % 4 == 0:
            if count_solutions(clues, n_attrs, n_houses, limit=2) == 1:
                break
    if count_solutions(clues, n_attrs, n_houses, limit=2) != 1:
        for clue in anchors:
            clues.append(clue)
            if count_solutions(clues, n_attrs, n_houses, limit=2) == 1:
                break
    if count_solutions(clues, n_attrs, n_houses, limit=2) != 1:
        raise RuntimeError("failed to construct a unique puzzle")

    if minimize:
        order = rng.permutation(len(clues)).tolist()
        for idx in sorted(order, reverse=True):
            if required is not None and clues[idx] == required:
                continue
            candidate = clues[:idx] + clues[idx + 1 :]
            if (
                candidate
                and count_solutions(candidate, n_attrs, n_houses, limit=2) == 1
            ):
                clues = candidate

    rng.shuffle(clues)
    tokens = [tok for clue in clues for tok in clue.tokens()] + ["ANSWER"]
    table = np.vstack((np.arange(n_houses, dtype=np.int16), solution.astype(np.int16)))
    # The flow-model parser uses answer_table rather than the autoregressive solving trace.
    solving_order = np.asarray(
        [(a, h) for a in range(n_attrs) for h in range(n_houses)], dtype=np.int16
    )
    return [tokens, table, solving_order]


def _fingerprint(puzzle) -> str:
    payload = json.dumps(
        [puzzle[0], np.asarray(puzzle[1]).tolist()], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_split(
    count: int,
    sizes: list[tuple[int, int]],
    seed: int,
    exclude: set[str] | None = None,
    minimize: bool = True,
):
    rng = np.random.default_rng(seed)
    exclude = set() if exclude is None else set(exclude)
    rows, hashes = [], set()
    while len(rows) < count:
        a, h = sizes[len(rows) % len(sizes)]
        # The parser freezes vocabulary from train. Preserve one example of every paper
        # relation in the first seven rows so even small deterministic splits have the
        # canonical relation vocabulary.
        required = (
            RELATIONS[len(rows)] if len(rows) < min(len(RELATIONS), count) else None
        )
        puzzle = generate_puzzle(
            a, h, rng, minimize=minimize, required_relation=required
        )
        digest = _fingerprint(puzzle)
        if digest in hashes or digest in exclude:
            continue
        rows.append(puzzle)
        hashes.add(digest)
    return rows, hashes


def _generate_shard(args):
    """Process-pool entry point (must remain top-level for multiprocessing)."""
    shard_id, count, sizes, seed, minimize = args
    rows, _ = generate_split(
        count, sizes, seed + 1_000_003 * shard_id, minimize=minimize
    )
    return shard_id, rows


def generate_split_parallel(
    count: int,
    sizes: list[tuple[int, int]],
    seed: int,
    workers: int,
    exclude: set[str] | None = None,
    minimize: bool = True,
):
    """Generate independent deterministic shards and merge/deduplicate in shard order."""
    if workers <= 1 or count < 2:
        return generate_split(count, sizes, seed, exclude=exclude, minimize=minimize)
    workers = min(workers, count)
    base, remainder = divmod(count, workers)
    specs = [(i, base + (i < remainder), sizes, seed, minimize) for i in range(workers)]
    # Linux/PACE uses fork: the already-imported NumPy state is copy-on-write and startup is
    # much cheaper than importing the whole training package in 32 spawned interpreters.
    context = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context()
    )
    shards = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as pool:
        for shard_id, rows in pool.map(_generate_shard, specs):
            print(
                f"[zebra-gen] shard {shard_id + 1}/{workers} complete: {len(rows)} puzzles",
                flush=True,
            )
            shards.append((shard_id, rows))

    blocked = set() if exclude is None else set(exclude)
    merged, hashes = [], set()
    for _, rows in sorted(shards):
        for puzzle in rows:
            digest = _fingerprint(puzzle)
            if digest in blocked or digest in hashes:
                continue
            merged.append(puzzle)
            hashes.add(digest)
            if len(merged) == count:
                return merged, hashes

    # Cross-shard collisions are extraordinarily unlikely, but fill any deficit so the
    # requested cardinality and disjointness are hard guarantees rather than assumptions.
    fill_seed = seed + 1_000_003 * workers + 97
    while len(merged) < count:
        extra, _ = generate_split(
            count - len(merged),
            sizes,
            fill_seed,
            exclude=blocked | hashes,
            minimize=minimize,
        )
        for puzzle in extra:
            digest = _fingerprint(puzzle)
            if digest not in blocked and digest not in hashes:
                merged.append(puzzle)
                hashes.add(digest)
        fill_seed += 1_000_003
    return merged, hashes


def _parse_sizes(values: list[str]) -> list[tuple[int, int]]:
    out = []
    for value in values:
        a, h = value.lower().split("x")
        a, h = int(a), int(h)
        if not (3 <= a <= 6 and 3 <= h <= 6):
            raise ValueError("Zebra sizes must be within 3x3..6x6")
        out.append((a, h))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Generate deterministic Zebra puzzle pickles"
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sizes", nargs="+", default=["5x5"])
    ap.add_argument("--train-count", type=int, default=20_000)
    ap.add_argument("--test-count", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=240910502)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent CPU generation processes (PACE: use 32)",
    )
    ap.add_argument(
        "--no-minimize",
        action="store_true",
        help="skip redundant-clue deletion (much faster, less paper-faithful)",
    )
    args = ap.parse_args()
    sizes = _parse_sizes(args.sizes)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, train_hashes = generate_split_parallel(
        args.train_count, sizes, args.seed, args.workers, minimize=not args.no_minimize
    )
    test, test_hashes = generate_split_parallel(
        args.test_count,
        sizes,
        args.seed + 1,
        args.workers,
        exclude=train_hashes,
        minimize=not args.no_minimize,
    )
    paths = {"train": out / "zebra-train-data.pkl", "test": out / "zebra-test-data.pkl"}
    for split, rows in (("train", train), ("test", test)):
        with paths[split].open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "generator": "frm.datasets.generate_zebra",
        "provenance": "clean-room procedural reconstruction; not the original Shah split",
        "paper": "arXiv:2409.10502",
        "seed": args.seed,
        "workers": args.workers,
        "sizes": [f"{a}x{h}" for a, h in sizes],
        "train_count": len(train),
        "test_count": len(test),
        "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(paths["test"].read_bytes()).hexdigest(),
        "overlap": len(train_hashes & test_hashes),
        "relations": list(RELATIONS),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
