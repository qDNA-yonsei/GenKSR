"""Aggregate SKQD energy errors and effective sample statistics into JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


METHODS = ("device", "transformer", "mamba")


def _parse_run(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("run must have the form LABEL:PATH")
    label, path = value.split(":", 1)
    return label, Path(path)


def _load(path: Path) -> dict:
    if path.is_dir():
        path = path / "all_results.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def _statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: None for key in ("count", "mean", "std", "median", "q25", "q75", "min", "max")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize(results: dict, split: str, dimensions: list[int]) -> list[dict]:
    split_results = results[split]
    exact = split_results["exact_energy"]
    rows = []
    for method in METHODS:
        for dimension in dimensions:
            index = dimension - 1
            errors = []
            input_counts = []
            postselected_counts = []
            cumulative_basis_counts = []
            for hamiltonian, entry in split_results.get(method, {}).items():
                energies = np.asarray(entry["energies"], dtype=float)
                if index >= len(energies) or hamiltonian not in exact:
                    continue
                errors.append(abs(float(energies[index]) - float(exact[hamiltonian])))
                if "num_input_samples_per_dim" in entry:
                    input_counts.append(entry["num_input_samples_per_dim"][index])
                    postselected_counts.append(entry["num_postselected_samples_per_dim"][index])
                    cumulative_basis_counts.append(
                        entry["num_cumulative_unique_postselected_per_dim"][index]
                    )

            row = {"method": method, "dimension": dimension, **_statistics(errors)}
            if input_counts:
                total_input = int(np.sum(input_counts))
                row.update({
                    "postselection_rate": (
                        float(np.sum(postselected_counts) / total_input)
                        if total_input else None
                    ),
                    "postselected_count_mean": float(np.mean(postselected_counts)),
                    "cumulative_unique_basis_mean": float(np.mean(cumulative_basis_counts)),
                })
            else:
                row.update({
                    "postselection_rate": None,
                    "postselected_count_mean": None,
                    "cumulative_unique_basis_mean": None,
                })
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=_parse_run, required=True,
                        help="One or more LABEL:PATH entries.")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[15, 30])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        label: summarize(_load(path), args.split, args.dimensions)
        for label, path in args.runs
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rows = [dict(run=label, **row) for label, run_rows in report.items() for row in run_rows]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
