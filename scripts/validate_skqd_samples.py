"""Validate SKQD model samples and summarize symmetry/coverage statistics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SAMPLE_RE = re.compile(r"model_samples_D(?P<dimension>\d+)_(?P<shots>\d+)_id(?P<hamiltonian>\d+)\.npy$")


def validate_directory(
    sample_dir: Path,
    *,
    n_qubit: int,
    max_krylov_dim: int,
    num_hamiltonians: int,
    snapshots: int,
) -> dict:
    expected_pairs = {
        (dimension, hamiltonian)
        for dimension in range(max_krylov_dim)
        for hamiltonian in range(num_hamiltonians)
    }
    observed_pairs: set[tuple[int, int]] = set()
    rates = {dimension: [] for dimension in range(max_krylov_dim)}
    unique_sz0_counts = {dimension: [] for dimension in range(max_krylov_dim)}
    all_unique_counts = {dimension: [] for dimension in range(max_krylov_dim)}
    invalid_files: list[dict] = []

    files = sorted(sample_dir.glob(f"model_samples_D*_{snapshots}_id*.npy"))
    for path in files:
        match = SAMPLE_RE.fullmatch(path.name)
        if match is None:
            continue
        dimension = int(match.group("dimension"))
        hamiltonian = int(match.group("hamiltonian"))
        observed_pairs.add((dimension, hamiltonian))

        samples = np.load(path, allow_pickle=False)
        issues = []
        if samples.shape != (snapshots, n_qubit):
            issues.append(f"shape={samples.shape}")
        if not np.all((samples == 0) | (samples == 1)):
            issues.append("non-binary values")
        if issues:
            invalid_files.append({"file": path.name, "issues": issues})
            continue

        selected_mask = samples.sum(axis=1) == n_qubit // 2
        rates[dimension].append(float(np.mean(selected_mask)))
        unique_sz0_counts[dimension].append(
            int(np.unique(samples[selected_mask], axis=0).shape[0])
        )
        all_unique_counts[dimension].append(int(np.unique(samples, axis=0).shape[0]))

    per_dimension = {}
    all_rates = []
    for dimension in range(max_krylov_dim):
        dimension_rates = rates[dimension]
        dimension_unique_sz0 = unique_sz0_counts[dimension]
        dimension_all_unique = all_unique_counts[dimension]
        all_rates.extend(dimension_rates)
        per_dimension[str(dimension)] = {
            "sz0_rate_mean": float(np.mean(dimension_rates)) if dimension_rates else None,
            "sz0_rate_min": float(np.min(dimension_rates)) if dimension_rates else None,
            "unique_sz0_count_mean": (
                float(np.mean(dimension_unique_sz0)) if dimension_unique_sz0 else None
            ),
            "unique_sz0_count_min": (
                int(np.min(dimension_unique_sz0)) if dimension_unique_sz0 else None
            ),
            "all_unique_count_mean": (
                float(np.mean(dimension_all_unique)) if dimension_all_unique else None
            ),
            "all_unique_count_min": (
                int(np.min(dimension_all_unique)) if dimension_all_unique else None
            ),
        }

    return {
        "sample_directory": str(sample_dir.resolve()),
        "sample_file_count": len(files),
        "expected_sample_file_count": len(expected_pairs),
        "complete_grid": observed_pairs == expected_pairs,
        "missing_pairs": sorted(expected_pairs - observed_pairs),
        "unexpected_pairs": sorted(observed_pairs - expected_pairs),
        "invalid_files": invalid_files,
        "overall_sz0_rate": float(np.mean(all_rates)) if all_rates else None,
        "per_dimension": per_dimension,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--n-qubit", type=int, default=16)
    parser.add_argument("--max-krylov-dim", type=int, default=30)
    parser.add_argument("--num-hamiltonians", type=int, default=20)
    parser.add_argument("--snapshots", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.parent.parent.parent.name for path in args.sample_dirs]
    if len(labels) != len(args.sample_dirs):
        raise ValueError("--labels must have one entry per sample directory")

    report = {
        label: validate_directory(
            sample_dir,
            n_qubit=args.n_qubit,
            max_krylov_dim=args.max_krylov_dim,
            num_hamiltonians=args.num_hamiltonians,
            snapshots=args.snapshots,
        )
        for label, sample_dir in zip(labels, args.sample_dirs, strict=True)
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
