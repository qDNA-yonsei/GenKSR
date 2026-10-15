#!/usr/bin/env python3
"""Generate and freeze the paired Jxy manifest for the XXZ sparsity sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REGIME_RANGES = {
    "sparse": (0.25, 0.35),
    "middle": (0.55, 0.65),
    "dense": (0.90, 1.10),
}
SPLIT_COUNTS_PER_REGIME = {
    "train": 15,
    "validation": 5,
    "test": 10,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the frozen pure-XXZ Jxy ratio manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--jzz", type=float, default=1.0)
    parser.add_argument(
        "--output", default="data/xxz_sparsity_1d/ratio_manifest.csv")
    return parser.parse_args(argv)


def _coordinate_seed(seed, *coordinates):
    return int(np.random.SeedSequence(
        [int(seed), *(int(value) for value in coordinates)]
    ).generate_state(1, dtype=np.uint32)[0])


def generate_manifest_rows(seed=20260715, jzz=1.0):
    """Return deterministic, split-stratified ratios shared by every N."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(jzz) or jzz <= 0:
        raise ValueError("jzz must be positive and finite")

    rows = []
    split_ids = Counter()
    master_id = 0
    for split_index, (split, count) in enumerate(
            SPLIT_COUNTS_PER_REGIME.items()):
        for regime_index, (regime, bounds) in enumerate(REGIME_RANGES.items()):
            low, high = bounds
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, split_index, regime_index]))
            # One random point per equal-width stratum avoids accidental
            # clumping while preserving a uniform-within-range design.
            unit = (np.arange(count, dtype=float) + rng.random(count)) / count
            rng.shuffle(unit)
            ratios = low + (high - low) * unit
            for regime_id, j_xy in enumerate(ratios):
                rows.append({
                    "master_id": int(master_id),
                    "split": split,
                    "split_id": int(split_ids[split]),
                    "regime": regime,
                    "regime_id": int(regime_id),
                    "paired_key": f"{split}_{regime}_{regime_id:02d}",
                    "J_xy": float(j_xy),
                    "J_zz": float(jzz),
                    "range_low": float(low),
                    "range_high": float(high),
                    "instance_seed": _coordinate_seed(seed, master_id),
                })
                master_id += 1
                split_ids[split] += 1
    validate_manifest_rows(rows)
    return rows


def validate_manifest_rows(rows):
    expected_total = sum(SPLIT_COUNTS_PER_REGIME.values()) * len(REGIME_RANGES)
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, got {len(rows)}")
    if [int(row["master_id"]) for row in rows] != list(range(expected_total)):
        raise ValueError("master_id must be contiguous and unique")
    if len({row["paired_key"] for row in rows}) != expected_total:
        raise ValueError("paired_key values must be unique")
    ratios = [float(row["J_xy"]) for row in rows]
    if len(set(ratios)) != expected_total:
        raise ValueError("J_xy values must be distinct across all splits")

    for split, expected_count in SPLIT_COUNTS_PER_REGIME.items():
        split_rows = [row for row in rows if row["split"] == split]
        expected_split_total = expected_count * len(REGIME_RANGES)
        if len(split_rows) != expected_split_total:
            raise ValueError(
                f"Split {split} expected {expected_split_total} rows, "
                f"got {len(split_rows)}")
        if sorted(int(row["split_id"]) for row in split_rows) != list(
                range(expected_split_total)):
            raise ValueError(f"Split {split} has invalid split_id values")
        for regime, (low, high) in REGIME_RANGES.items():
            selected = [row for row in split_rows if row["regime"] == regime]
            if len(selected) != expected_count:
                raise ValueError(
                    f"{split}/{regime} expected {expected_count}, got {len(selected)}")
            if not all(low <= float(row["J_xy"]) < high for row in selected):
                raise ValueError(f"{split}/{regime} contains an out-of-range J_xy")
            if any(not np.isclose(float(row["J_zz"]), 1.0) for row in selected):
                raise ValueError("The frozen experiment requires J_zz=1")
    return True


def _csv_text(rows):
    if not rows:
        raise ValueError("Cannot serialize an empty manifest")
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_frozen_manifest(rows, output, seed):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_text = _csv_text(rows)
    if output.exists():
        if output.read_bytes() != csv_text.encode("utf-8"):
            raise FileExistsError(
                f"Frozen manifest differs from requested design: {output}")
        print(f"Verified existing frozen manifest: {output}")
        return output

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(csv_text)
    os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "pure_xxz_sparsity_scaling",
        "seed": int(seed),
        "sampling": "independent random point in each equal-width stratum",
        "paired_across_system_sizes": True,
        "system_sizes": [12, 16, 20],
        "jzz": 1.0,
        "boundary_pin": 0.0,
        "regime_ranges": {
            name: list(bounds) for name, bounds in REGIME_RANGES.items()},
        "split_counts_per_regime": SPLIT_COUNTS_PER_REGIME,
        "rows": len(rows),
        "csv_sha256": digest,
    }
    metadata_path = output.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(metadata_temporary, metadata_path)
    print(f"Wrote frozen manifest: {output}")
    print(f"SHA-256: {digest}")
    return output


def main(argv=None):
    args = parse_args(argv)
    rows = generate_manifest_rows(seed=args.seed, jzz=args.jzz)
    write_frozen_manifest(rows, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
