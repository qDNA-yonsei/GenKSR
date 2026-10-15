"""Validate structural and variational invariants of SKQD evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
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


def validate_run(results: dict, split: str, n_qubit: int, tolerance: float) -> dict:
    split_results = results[split]
    exact = split_results["exact_energy"]
    max_basis = math.comb(n_qubit, n_qubit // 2)
    report = {
        "hamiltonian_count": len(exact),
        "max_sz0_basis_size": max_basis,
        "methods": {},
    }

    for method in METHODS:
        entries = split_results.get(method, {})
        variational_violations = []
        monotonic_violations = []
        statistics_violations = []
        energy_lengths = set()
        finite_dimensions = set()

        for hamiltonian, entry in entries.items():
            energies = np.asarray(entry["energies"], dtype=float)
            energy_lengths.add(int(len(energies)))
            finite = np.flatnonzero(np.isfinite(energies))
            finite_dimensions.update((finite + 1).tolist())
            exact_energy = float(exact[hamiltonian])

            for index in finite:
                if energies[index] < exact_energy - tolerance:
                    variational_violations.append({
                        "hamiltonian": hamiltonian,
                        "dimension": int(index + 1),
                        "energy": float(energies[index]),
                        "exact_energy": exact_energy,
                    })
            finite_energies = energies[finite]
            for left, right, difference in zip(
                finite[:-1], finite[1:], np.diff(finite_energies), strict=True
            ):
                if difference > tolerance:
                    monotonic_violations.append({
                        "hamiltonian": hamiltonian,
                        "left_dimension": int(left + 1),
                        "right_dimension": int(right + 1),
                        "increase": float(difference),
                    })

            required_stats = (
                "num_input_samples_per_dim",
                "num_postselected_samples_per_dim",
                "num_cumulative_unique_postselected_per_dim",
            )
            if not all(key in entry for key in required_stats):
                statistics_violations.append({
                    "hamiltonian": hamiltonian,
                    "issue": "missing sample statistics",
                })
                continue
            inputs = np.asarray(entry[required_stats[0]], dtype=int)
            selected = np.asarray(entry[required_stats[1]], dtype=int)
            cumulative_basis = np.asarray(entry[required_stats[2]], dtype=int)
            if not (len(inputs) == len(selected) == len(cumulative_basis) == len(energies)):
                statistics_violations.append({
                    "hamiltonian": hamiltonian,
                    "issue": "sample-statistic length mismatch",
                })
            if np.any(selected < 0) or np.any(selected > inputs):
                statistics_violations.append({
                    "hamiltonian": hamiltonian,
                    "issue": "postselected count outside [0, input]",
                })
            if np.any(np.diff(cumulative_basis) < 0) or np.any(cumulative_basis > max_basis):
                statistics_violations.append({
                    "hamiltonian": hamiltonian,
                    "issue": "invalid cumulative basis size",
                })

        report["methods"][method] = {
            "entry_count": len(entries),
            "energy_lengths": sorted(energy_lengths),
            "finite_dimensions": sorted(finite_dimensions),
            "variational_violation_count": len(variational_violations),
            "monotonic_violation_count": len(monotonic_violations),
            "sample_statistics_violation_count": len(statistics_violations),
            "variational_violations": variational_violations,
            "monotonic_violations": monotonic_violations,
            "sample_statistics_violations": statistics_violations,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=_parse_run, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-qubit", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        label: validate_run(_load(path), args.split, args.n_qubit, args.tolerance)
        for label, path in args.runs
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
