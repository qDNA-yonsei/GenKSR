#!/usr/bin/env python3
"""Aggregate independent empirical-bootstrap repeats of train-reuse baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiments.skqd_train_reuse_baselines import (
    _load_existing_sweep,
    _raw_comparison_rows,
    _write_csv,
    aggregate_raw,
    paired_summary,
    plot_heatmap,
    plot_results,
)


RAW_NUMERIC_FIELDS = (
    "raw_draws",
    "accepted_draws",
    "acceptance_rate",
    "basis_size",
    "coverage",
    "energy",
    "exact_energy",
    "delta_e",
    "captured_ground_mass",
    "missing_ground_mass",
)
PAIR_NUMERIC_FIELDS = (
    "common_basis_size",
    "coverage",
    "model_raw_basis_size",
    "baseline_raw_basis_size",
    "model_energy",
    "baseline_energy",
    "exact_energy",
    "model_delta_e",
    "baseline_delta_e",
    "delta_e_difference",
    "model_captured_ground_mass",
    "baseline_captured_ground_mass",
    "captured_mass_difference",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate five SKQD train-reuse bootstrap seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        default="artifacts/v3_skqd_train_reuse_baselines_d30/repeats",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v3_skqd_train_reuse_baselines_d30/aggregate_5seeds",
    )
    parser.add_argument(
        "--existing-sweep-root",
        default="artifacts/v3_skqd_support_m_sweep_d30",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717, 20260718],
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument("--high-j2-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args(argv)


def _read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def _parse_raw(row, repeat_seed):
    parsed = {
        "repeat_seed": int(repeat_seed),
        "hamiltonian_id": int(row["hamiltonian_id"]),
        "J2": float(row["J2"]),
        "M": int(row["M"]),
        "D": int(row["D"]),
        "method": row["method"],
    }
    for field in RAW_NUMERIC_FIELDS:
        parsed[field] = float(row[field])
    return parsed


def _parse_pair(row, repeat_seed):
    parsed = {
        "repeat_seed": int(repeat_seed),
        "hamiltonian_id": int(row["hamiltonian_id"]),
        "J2": float(row["J2"]),
        "M": int(row["M"]),
        "D": int(row["D"]),
        "model": row["model"],
        "baseline": row["baseline"],
    }
    for field in PAIR_NUMERIC_FIELDS:
        parsed[field] = float(row[field])
    return parsed


def load_repeats(args):
    root = Path(args.input_root)
    raw_rows = []
    pair_rows = []
    configs = []
    for repeat_seed in args.seeds:
        run_dir = root / f"seed{repeat_seed}"
        with (run_dir / "config.json").open() as handle:
            config = json.load(handle)
        if int(config["seed"]) != repeat_seed:
            raise ValueError(f"Seed mismatch in {run_dir}")
        configs.append(config)
        raw_rows.extend(
            _parse_raw(row, repeat_seed)
            for row in _read_csv(run_dir / "baseline_raw_results.csv")
        )
        pair_rows.extend(
            _parse_pair(row, repeat_seed)
            for row in _read_csv(run_dir / "pairwise_unique_results.csv")
        )
    expected_raw = len(args.seeds) * 20 * len(args.m_values) * 2
    expected_pair = len(args.seeds) * 20 * len(args.m_values) * 2 * 2
    if len(raw_rows) != expected_raw or len(pair_rows) != expected_pair:
        raise ValueError(
            f"Unexpected repeat rows: raw={len(raw_rows)}, pair={len(pair_rows)}"
        )
    return raw_rows, pair_rows, configs


def aggregate_raw_repeats(rows, seeds):
    keys = sorted({
        (row["hamiltonian_id"], row["M"], row["method"])
        for row in rows
    })
    aggregated = []
    for hid, m, method in keys:
        subset = [
            row for row in rows
            if row["hamiltonian_id"] == hid and row["M"] == m
            and row["method"] == method
        ]
        if len(subset) != len(seeds):
            raise ValueError(f"Incomplete raw repeats for {(hid, m, method)}")
        record = {
            "hamiltonian_id": hid,
            "J2": float(subset[0]["J2"]),
            "M": m,
            "D": int(subset[0]["D"]),
            "method": method,
            "n_repeats": len(subset),
        }
        for field in RAW_NUMERIC_FIELDS:
            record[field] = float(np.median([row[field] for row in subset]))
        aggregated.append(record)
    return aggregated


def aggregate_pair_repeats(rows, seeds):
    keys = sorted({
        (row["hamiltonian_id"], row["M"], row["model"], row["baseline"])
        for row in rows
    })
    aggregated = []
    for hid, m, model, baseline in keys:
        subset = [
            row for row in rows
            if row["hamiltonian_id"] == hid and row["M"] == m
            and row["model"] == model and row["baseline"] == baseline
        ]
        if len(subset) != len(seeds):
            raise ValueError(
                f"Incomplete pair repeats for {(hid, m, model, baseline)}"
            )
        record = {
            "hamiltonian_id": hid,
            "J2": float(subset[0]["J2"]),
            "M": m,
            "D": int(subset[0]["D"]),
            "model": model,
            "baseline": baseline,
            "n_repeats": len(subset),
        }
        for field in PAIR_NUMERIC_FIELDS:
            record[field] = float(np.median([row[field] for row in subset]))
        aggregated.append(record)
    return aggregated


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_raw, repeat_pairs, configs = load_repeats(args)
    baseline_raw = aggregate_raw_repeats(repeat_raw, args.seeds)
    pair_rows = aggregate_pair_repeats(repeat_pairs, args.seeds)

    existing_rows = _load_existing_sweep(args)
    raw_rows = _raw_comparison_rows(existing_rows, baseline_raw, args.m_values)
    raw_summary = aggregate_raw(raw_rows, args.m_values)
    comparisons = paired_summary(raw_rows, pair_rows, args)

    protocol = dict(configs[0])
    protocol["task"] = "sqkd_train_reuse_baselines_5seed_aggregate"
    protocol["bootstrap_sampling_seeds"] = args.seeds
    protocol["baseline_repeats"] = len(args.seeds)
    protocol["aggregation"] = (
        "per-Hamiltonian median over empirical-bootstrap seeds, then paired "
        "statistics over Hamiltonians"
    )

    _write_csv(output_dir / "repeat_raw_results.csv", repeat_raw)
    _write_csv(output_dir / "repeat_pairwise_unique_results.csv", repeat_pairs)
    _write_csv(output_dir / "baseline_raw_per_hamiltonian.csv", baseline_raw)
    _write_csv(output_dir / "pairwise_unique_per_hamiltonian.csv", pair_rows)
    _write_csv(output_dir / "raw_comparison_per_hamiltonian.csv", raw_rows)
    _write_csv(output_dir / "raw_summary.csv", raw_summary)
    _write_csv(output_dir / "paired_summary.csv", comparisons)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(protocol, handle, indent=2)
    plot_results(raw_summary, comparisons, output_dir, protocol)
    plot_heatmap(pair_rows, output_dir, protocol)
    print(f"Wrote five-seed aggregate to {output_dir}")


if __name__ == "__main__":
    main()
