#!/usr/bin/env python3
"""Disentangle Krylov depth D from per-state samples M at fixed D*M.

The canonical D-sweep increases both the number of distinct Krylov states and
the total number of raw samples.  This experiment fixes the total raw budget
``B = D*M`` and varies D, answering whether deeper Krylov coverage helps after
removing the trivial increase in total draws.

Methods:

* exact-state device samples;
* saved Transformer and Mamba samples;
* equal-resource train-only kNN-5 empirical mixture;
* uniform sampling from the full Hilbert space followed by Sz=0 postselection;
* uniform sampling directly in the Sz=0 sector as a symmetry-aware control.

The neural/device samples are one saved realization.  kNN resampling and the
uniform controls use independent repeats, aggregated per Hamiltonian before
statistics over Hamiltonians.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from genksr.evaluation.skqd import projected_xxz_hamiltonian
from scripts.experiments.skqd_support_controlled import (
    _basis_metrics,
    _lowest_eigenpair,
    _unique_in_arrival_order,
)
from scripts.experiments.skqd_train_reuse_baselines import (
    TrainingSampleCache,
    _build_condition_library,
    _generate_baseline_streams,
    _infer_j2_dt,
    _load_protocol,
)


FIXED_METHODS = ("device", "transformer", "mamba")
REPEATED_METHODS = ("knn5_uniform", "uniform_full", "uniform_sector")
METHODS = FIXED_METHODS + REPEATED_METHODS
DISPLAY_NAMES = {
    "device": "Exact-state",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "knn5_uniform": "kNN-5 mixture",
    "uniform_full": "Uniform full",
    "uniform_sector": "Uniform Sz=0",
}
COLORS = {
    "device": "tab:blue",
    "transformer": "tab:orange",
    "mamba": "tab:red",
    "knn5_uniform": "tab:cyan",
    "uniform_full": "tab:green",
    "uniform_sector": "tab:purple",
}
MARKERS = {
    "device": "o",
    "transformer": "o",
    "mamba": "o",
    "knn5_uniform": "o",
    "uniform_full": "x",
    "uniform_sector": "s",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SKQD fixed-total-budget D-vs-M experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation-dir",
        default="artifacts/v3_skqd_formal/evaluation_random_j2_10000",
    )
    parser.add_argument("--data-root", default="data/heisenberg_SQKD_2d")
    parser.add_argument(
        "--output-dir",
        default="artifacts/v3_skqd_fixed_total_budget",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[3000, 15000, 60000, 150000],
    )
    parser.add_argument(
        "--d-values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
    )
    parser.add_argument("--train-krylov-dim", type=int, default=15)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--baseline-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--hamiltonian-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional held-out subset for a smoke test",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _build_grid(args, max_snapshots):
    budgets = sorted(set(args.budgets))
    d_values = sorted(set(args.d_values))
    if not budgets or not d_values or min(budgets) <= 0 or min(d_values) <= 0:
        raise ValueError("budgets and d-values must be positive")
    combinations = []
    for budget in budgets:
        for dimension in d_values:
            if budget % dimension != 0:
                continue
            samples = budget // dimension
            if samples <= max_snapshots:
                combinations.append({
                    "budget": int(budget),
                    "D": int(dimension),
                    "M": int(samples),
                })
    for budget in budgets:
        count = sum(row["budget"] == budget for row in combinations)
        if count < 2:
            raise ValueError(
                f"Budget {budget} has fewer than two feasible D values")
    return budgets, d_values, combinations


def _validate_args(args):
    if args.train_krylov_dim <= 0 or args.train_samples <= 0:
        raise ValueError("Training limits must be positive")
    if args.knn_k <= 0 or args.baseline_repeats <= 0:
        raise ValueError("k and repeat counts must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_path(method, hid, dimension, snapshots, test_dir, model_dirs):
    if method == "device":
        return test_dir / f"data_D{dimension}_id{hid}.npy"
    return (
        model_dirs[method]
        / "samples"
        / "test"
        / "model"
        / f"model_samples_D{dimension}_{snapshots}_id{hid}.npy"
    )


def _load_fixed_raw_rows(
        method, hid, max_dimension, max_samples, snapshots,
        test_dir, model_dirs, n_qubit):
    """Return one integer label per raw row, with -1 for rejected samples."""
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    rows_by_dimension = {}
    for dimension in range(max_dimension):
        path = _sample_path(
            method, hid, dimension, snapshots, test_dir, model_dirs)
        samples = np.asarray(np.load(path, mmap_mode="r")[:max_samples])
        if samples.ndim != 2 or samples.shape[1] != n_qubit:
            raise ValueError(f"Unexpected sample shape in {path}: {samples.shape}")
        if not np.all((samples == 0) | (samples == 1)):
            raise ValueError(f"Non-binary sample in {path}")
        states = samples.astype(np.int64) @ powers
        selected = np.sum(samples, axis=1) == n_qubit // 2
        states[~selected] = -1
        rows_by_dimension[dimension] = states
    return rows_by_dimension


def _fixed_stream(rows_by_dimension, dimension, samples_per_dimension):
    pieces = []
    for d in range(dimension):
        raw = rows_by_dimension[d][:samples_per_dimension]
        pieces.append(raw[raw >= 0])
    return np.concatenate(pieces)


def _accepted_stream(streams_by_dimension, dimension, samples_per_dimension):
    return np.concatenate([
        streams_by_dimension[d][:samples_per_dimension]
        for d in range(dimension)
    ])


def _uniform_control_streams(
        seed, hid, repeat, max_budget, hilbert_size, sector_states):
    """Generate nested raw-budget streams for both random controls.

    ``uniform_full_raw`` contains one integer in ``[0, 2**N)`` per raw draw;
    callers must postselect its prefix.  ``uniform_sector`` samples directly
    from the known symmetry sector, so every draw is accepted.  Seed
    coordinates 0 and 1 intentionally match the support-controlled benchmark.
    """
    full_rng = np.random.default_rng(
        np.random.SeedSequence([seed, hid, repeat, 0]))
    sector_rng = np.random.default_rng(
        np.random.SeedSequence([seed, hid, repeat, 1]))
    return {
        "uniform_full_raw": full_rng.integers(
            0, hilbert_size, size=max_budget, dtype=np.int64),
        "uniform_sector": sector_rng.choice(
            sector_states, size=max_budget, replace=True),
    }


def _record(
        hid, j2, method, repeat, budget, dimension, samples,
        accepted, basis, sector_size, energy, exact_energy,
        delta_e, captured_mass):
    return {
        "hamiltonian_id": int(hid),
        "J2": float(j2),
        "method": method,
        "repeat": int(repeat),
        "budget": int(budget),
        "D": int(dimension),
        "M": int(samples),
        "raw_draws": int(budget),
        "accepted_draws": int(accepted),
        "acceptance_rate": float(accepted / budget),
        "basis_size": int(len(basis)),
        "coverage": float(len(basis) / sector_size),
        "energy": float(energy),
        "exact_energy": float(exact_energy),
        "delta_e": float(delta_e),
        "captured_ground_mass": float(captured_mass),
        "missing_ground_mass": float(max(1.0 - captured_mass, 0.0)),
    }


def run_experiment(args):
    # _load_protocol expects these fields because it is shared with the kNN run.
    args.krylov_dim = max(args.d_values)
    args.m_values = [1]
    evaluation_args, model_dirs = _load_protocol(args)
    snapshots = int(evaluation_args["snapshots"])
    budgets, d_values, combinations = _build_grid(args, snapshots)
    args.m_values = sorted({row["M"] for row in combinations})

    n_qubit = int(evaluation_args["n_qubit"])
    num_ones = n_qubit // 2
    hilbert_size = 1 << n_qubit
    sector_states = np.fromiter(
        (state for state in range(hilbert_size)
         if int(state).bit_count() == num_ones),
        dtype=np.int64,
        count=math.comb(n_qubit, num_ones),
    )
    sector_size = len(sector_states)
    sector_index = np.full(hilbert_size, -1, dtype=np.int64)
    sector_index[sector_states] = np.arange(sector_size)

    data_root = Path(args.data_root) / "16qubits"
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    conditions, train_hamiltonians, time_scale = _build_condition_library(
        train_dir, args)
    sample_cache = TrainingSampleCache(
        train_dir, args.train_samples, n_qubit)
    hamiltonian_ids = (
        list(range(20)) if args.hamiltonian_ids is None
        else sorted(set(args.hamiltonian_ids))
    )
    if any(hid < 0 or hid >= 20 for hid in hamiltonian_ids):
        raise ValueError("hamiltonian-ids must be between 0 and 19")

    max_dimension = max(row["D"] for row in combinations)
    max_samples = max(row["M"] for row in combinations)
    max_budget = max(budgets)
    raw_results = []
    source_mapping = []

    for position, hid in enumerate(hamiltonian_ids, start=1):
        print(
            f"[{position}/{len(hamiltonian_ids)}] fixed-budget id={hid}",
            flush=True,
        )
        coupling = np.load(test_dir / f"coupling_matrix_id{hid}.npy")
        test_j2, test_dt = _infer_j2_dt(coupling)
        sector_hamiltonian = projected_xxz_hamiltonian(
            sector_states,
            coupling_matrix=coupling,
            n_qubit=n_qubit,
            delta=float(evaluation_args.get("delta", 1.0)),
            h_z=float(evaluation_args.get("h_z", 0.0)),
        )
        if not sparse.isspmatrix_csr(sector_hamiltonian):
            sector_hamiltonian = sector_hamiltonian.tocsr()
        exact_energy, ground_state = _lowest_eigenpair(sector_hamiltonian)
        ground_probabilities = np.abs(ground_state) ** 2
        ground_probabilities /= ground_probabilities.sum()

        fixed_rows = {
            method: _load_fixed_raw_rows(
                method,
                hid,
                max_dimension,
                max_samples,
                snapshots,
                test_dir,
                model_dirs,
                n_qubit,
            )
            for method in FIXED_METHODS
        }

        knn_by_repeat = []
        uniform_full_by_repeat = []
        uniform_sector_by_repeat = []
        for repeat in range(args.baseline_repeats):
            repeat_args = argparse.Namespace(**vars(args))
            repeat_args.seed = args.seed + repeat
            generated, mapping = _generate_baseline_streams(
                hid,
                test_j2,
                test_dt,
                conditions,
                train_hamiltonians,
                time_scale,
                sample_cache,
                repeat_args,
            )
            knn_by_repeat.append(generated["knn5_uniform"])
            if repeat == 0:
                source_mapping.extend(mapping)
            uniform_controls = _uniform_control_streams(
                args.seed,
                hid,
                repeat,
                max_budget,
                hilbert_size,
                sector_states,
            )
            uniform_full_by_repeat.append(
                uniform_controls["uniform_full_raw"])
            uniform_sector_by_repeat.append(
                uniform_controls["uniform_sector"])

        uniform_full_metric_cache = {}
        uniform_sector_metric_cache = {}
        for combination in combinations:
            budget = combination["budget"]
            dimension = combination["D"]
            samples = combination["M"]

            for method in FIXED_METHODS:
                stream = _fixed_stream(
                    fixed_rows[method], dimension, samples)
                basis = _unique_in_arrival_order(stream)
                energy, delta_e, captured_mass = _basis_metrics(
                    basis,
                    sector_hamiltonian,
                    sector_index,
                    ground_probabilities,
                    exact_energy,
                )
                raw_results.append(_record(
                    hid,
                    test_j2,
                    method,
                    0,
                    budget,
                    dimension,
                    samples,
                    len(stream),
                    basis,
                    sector_size,
                    energy,
                    exact_energy,
                    delta_e,
                    captured_mass,
                ))

            for repeat, streams_by_dimension in enumerate(knn_by_repeat):
                stream = _accepted_stream(
                    streams_by_dimension, dimension, samples)
                basis = _unique_in_arrival_order(stream)
                energy, delta_e, captured_mass = _basis_metrics(
                    basis,
                    sector_hamiltonian,
                    sector_index,
                    ground_probabilities,
                    exact_energy,
                )
                raw_results.append(_record(
                    hid,
                    test_j2,
                    "knn5_uniform",
                    repeat,
                    budget,
                    dimension,
                    samples,
                    len(stream),
                    basis,
                    sector_size,
                    energy,
                    exact_energy,
                    delta_e,
                    captured_mass,
                ))

            for repeat, uniform_raw_stream in enumerate(uniform_full_by_repeat):
                cache_key = (repeat, budget)
                if cache_key not in uniform_full_metric_cache:
                    raw_prefix = uniform_raw_stream[:budget]
                    stream = raw_prefix[sector_index[raw_prefix] >= 0]
                    basis = _unique_in_arrival_order(stream)
                    metrics = _basis_metrics(
                        basis,
                        sector_hamiltonian,
                        sector_index,
                        ground_probabilities,
                        exact_energy,
                    )
                    uniform_full_metric_cache[cache_key] = (
                        len(stream), basis, metrics[0], metrics[1], metrics[2])
                accepted, basis, energy, delta_e, captured_mass = (
                    uniform_full_metric_cache[cache_key])
                raw_results.append(_record(
                    hid,
                    test_j2,
                    "uniform_full",
                    repeat,
                    budget,
                    dimension,
                    samples,
                    accepted,
                    basis,
                    sector_size,
                    energy,
                    exact_energy,
                    delta_e,
                    captured_mass,
                ))

            for repeat, uniform_stream in enumerate(uniform_sector_by_repeat):
                cache_key = (repeat, budget)
                if cache_key not in uniform_sector_metric_cache:
                    stream = uniform_stream[:budget]
                    basis = _unique_in_arrival_order(stream)
                    metrics = _basis_metrics(
                        basis,
                        sector_hamiltonian,
                        sector_index,
                        ground_probabilities,
                        exact_energy,
                    )
                    uniform_sector_metric_cache[cache_key] = (
                        basis, metrics[0], metrics[1], metrics[2])
                basis, energy, delta_e, captured_mass = (
                    uniform_sector_metric_cache[cache_key])
                raw_results.append(_record(
                    hid,
                    test_j2,
                    "uniform_sector",
                    repeat,
                    budget,
                    dimension,
                    samples,
                    budget,
                    basis,
                    sector_size,
                    energy,
                    exact_energy,
                    delta_e,
                    captured_mass,
                ))
        print(
            f"[{position}/{len(hamiltonian_ids)}] id={hid}: complete",
            flush=True,
        )

    protocol = {
        "task": "sqkd_fixed_total_budget",
        "n_qubit": n_qubit,
        "budgets": budgets,
        "d_values": d_values,
        "combinations": combinations,
        "hamiltonian_ids": hamiltonian_ids,
        "methods": list(METHODS),
        "fixed_methods_note": "one saved sampling realization",
        "baseline_repeats": args.baseline_repeats,
        "baseline_seeds": [
            args.seed + repeat for repeat in range(args.baseline_repeats)
        ],
        "train_krylov_dim": args.train_krylov_dim,
        "train_samples_per_condition": args.train_samples,
        "knn_k": args.knn_k,
        "max_train_physical_time": time_scale,
        "budget_definition": "raw draws B=D*M before model postselection",
        "uniform_controls": {
            "uniform_full": (
                "B draws from all 2**N bitstrings, then Sz=0 postselection; "
                "same seeded raw prefix at fixed B for every D"
            ),
            "uniform_sector": (
                "B direct draws from the known Sz=0 sector; same seeded prefix "
                "at fixed B for every D"
            ),
            "seed_coordinates": {
                "uniform_full": "[seed, hamiltonian_id, repeat, 0]",
                "uniform_sector": "[seed, hamiltonian_id, repeat, 1]",
            },
        },
        "seed": args.seed,
    }
    return protocol, raw_results, source_mapping


def per_hamiltonian_medians(raw_results):
    keys = sorted({
        (row["hamiltonian_id"], row["method"], row["budget"], row["D"], row["M"])
        for row in raw_results
    })
    rows = []
    for hid, method, budget, dimension, samples in keys:
        subset = [
            row for row in raw_results
            if row["hamiltonian_id"] == hid and row["method"] == method
            and row["budget"] == budget and row["D"] == dimension
            and row["M"] == samples
        ]
        record = {
            "hamiltonian_id": int(hid),
            "J2": float(subset[0]["J2"]),
            "method": method,
            "budget": int(budget),
            "D": int(dimension),
            "M": int(samples),
            "n_repeats": len(subset),
        }
        for field in (
            "delta_e",
            "basis_size",
            "coverage",
            "acceptance_rate",
            "captured_ground_mass",
            "missing_ground_mass",
        ):
            record[field] = float(np.median([row[field] for row in subset]))
        rows.append(record)
    return rows


def aggregate_summary(per_hamiltonian):
    keys = sorted({
        (row["method"], row["budget"], row["D"], row["M"])
        for row in per_hamiltonian
    })
    summary = []
    for method, budget, dimension, samples in keys:
        subset = [
            row for row in per_hamiltonian
            if row["method"] == method and row["budget"] == budget
            and row["D"] == dimension and row["M"] == samples
        ]
        record = {
            "method": method,
            "budget": int(budget),
            "D": int(dimension),
            "M": int(samples),
            "n_hamiltonians": len(subset),
        }
        for field in (
            "delta_e",
            "basis_size",
            "coverage",
            "acceptance_rate",
            "captured_ground_mass",
            "missing_ground_mass",
        ):
            values = np.asarray([row[field] for row in subset], dtype=float)
            record[f"{field}_median"] = float(np.median(values))
            record[f"{field}_q25"] = float(np.quantile(values, 0.25))
            record[f"{field}_q75"] = float(np.quantile(values, 0.75))
            record[f"{field}_mean"] = float(np.mean(values))
        summary.append(record)
    return summary


def _bootstrap_ci(values, repeats, seed_sequence):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_sequence)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    return (
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def compare_d15_d30(per_hamiltonian, args):
    lookup = {
        (row["method"], row["budget"], row["D"], row["hamiltonian_id"]): row
        for row in per_hamiltonian
    }
    ids = sorted({row["hamiltonian_id"] for row in per_hamiltonian})
    budgets = sorted({row["budget"] for row in per_hamiltonian})
    rows = []
    comparison_index = 0
    for method in METHODS:
        for budget in budgets:
            if not all(
                (method, budget, dimension, ids[0]) in lookup
                for dimension in (15, 30)
            ):
                continue
            d15 = np.asarray([
                lookup[(method, budget, 15, hid)]["delta_e"] for hid in ids
            ])
            d30 = np.asarray([
                lookup[(method, budget, 30, hid)]["delta_e"] for hid in ids
            ])
            basis15 = np.asarray([
                lookup[(method, budget, 15, hid)]["basis_size"] for hid in ids
            ])
            basis30 = np.asarray([
                lookup[(method, budget, 30, hid)]["basis_size"] for hid in ids
            ])
            mass15 = np.asarray([
                lookup[(method, budget, 15, hid)]["captured_ground_mass"]
                for hid in ids
            ])
            mass30 = np.asarray([
                lookup[(method, budget, 30, hid)]["captured_ground_mass"]
                for hid in ids
            ])
            differences = d30 - d15
            ci_low, ci_high = _bootstrap_ci(
                differences,
                args.bootstrap_repeats,
                np.random.SeedSequence([args.seed, comparison_index]),
            )
            nonzero = differences[
                ~np.isclose(differences, 0.0, atol=1e-14)]
            p_value = 1.0 if len(nonzero) == 0 else float(
                wilcoxon(nonzero, alternative="two-sided").pvalue)
            rows.append({
                "method": method,
                "budget": int(budget),
                "D15_M": int(budget // 15),
                "D30_M": int(budget // 30),
                "median_D15_delta_e": float(np.median(d15)),
                "median_D30_delta_e": float(np.median(d30)),
                "median_D30_minus_D15": float(np.median(differences)),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "D30_energy_win_rate": float(np.mean(differences < 0.0)),
                "wilcoxon_p_two_sided": p_value,
                "median_D30_minus_D15_basis": float(
                    np.median(basis30 - basis15)),
                "median_D30_minus_D15_captured_mass": float(
                    np.median(mass30 - mass15)),
            })
            comparison_index += 1
    return rows


def optimal_dimensions(summary):
    rows = []
    for method in METHODS:
        budgets = sorted({
            row["budget"] for row in summary if row["method"] == method
        })
        for budget in budgets:
            subset = [
                row for row in summary
                if row["method"] == method and row["budget"] == budget
            ]
            best = min(subset, key=lambda row: (row["delta_e_median"], row["D"]))
            rows.append({
                "method": method,
                "budget": int(budget),
                "best_D": int(best["D"]),
                "best_M": int(best["M"]),
                "best_median_delta_e": float(best["delta_e_median"]),
                "best_median_basis_size": float(best["basis_size_median"]),
                "best_median_captured_ground_mass": float(
                    best["captured_ground_mass_median"]),
            })
    return rows


def _summary_lookup(summary):
    return {
        (row["method"], row["budget"], row["D"]): row
        for row in summary
    }


def plot_heatmaps(summary, protocol, output_dir):
    output_dir = Path(output_dir)
    budgets = protocol["budgets"]
    d_values = protocol["d_values"]
    lookup = _summary_lookup(summary)
    nonuniform_values = []
    matrices = {}
    annotations = {}
    for method in METHODS:
        matrix = np.full((len(budgets), len(d_values)), np.nan)
        annotation = np.full((len(budgets), len(d_values)), np.nan)
        for row_index, budget in enumerate(budgets):
            for column_index, dimension in enumerate(d_values):
                row = lookup.get((method, budget, dimension))
                if row is not None:
                    value = max(row["delta_e_median"], 1e-8)
                    matrix[row_index, column_index] = np.log10(value)
                    annotation[row_index, column_index] = row["delta_e_median"]
                    if method != "uniform_sector":
                        nonuniform_values.append(np.log10(value))
        matrices[method] = matrix
        annotations[method] = annotation
    vmin = float(np.min(nonuniform_values))
    vmax = float(np.max(nonuniform_values))

    figure, axes = plt.subplots(
        1, len(METHODS), figsize=(25, 5.2), sharex=True, sharey=True,
        constrained_layout=True)
    image = None
    for axis, method in zip(axes, METHODS):
        display_matrix = np.clip(matrices[method], vmin, vmax)
        image = axis.imshow(
            display_matrix, aspect="auto", origin="lower",
            cmap="viridis_r", vmin=vmin, vmax=vmax)
        axis.set_title(DISPLAY_NAMES[method])
        axis.set_xticks(range(len(d_values)), [str(value) for value in d_values])
        axis.set_yticks(
            range(len(budgets)), [f"{value:,}" for value in budgets])
        for row_index in range(len(budgets)):
            for column_index in range(len(d_values)):
                if np.isfinite(annotations[method][row_index, column_index]):
                    value = annotations[method][row_index, column_index]
                    label = "0" if np.isclose(value, 0.0, atol=1e-12) else f"{value:.2g}"
                    axis.text(
                        column_index,
                        row_index,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
    for axis in axes:
        axis.set_xlabel("D")
    axes[0].set_ylabel("Fixed raw budget B")
    colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.82)
    colorbar.set_label(r"$\log_{10}$ median energy error")
    figure.suptitle("SKQD fixed-total-budget D-vs-M")
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"fixed_budget_heatmaps.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_d15_d30(comparisons, optimal, protocol, output_dir):
    output_dir = Path(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method in METHODS:
        subset = sorted(
            [row for row in comparisons if row["method"] == method],
            key=lambda row: row["budget"],
        )
        axes[0].plot(
            [row["budget"] for row in subset],
            [row["median_D30_minus_D15"] for row in subset],
            marker=MARKERS[method],
            color=COLORS[method],
            label=DISPLAY_NAMES[method],
            zorder=4 if method == "uniform_full" else 3,
        )
        if not method.startswith("uniform_"):
            method_optimal = sorted(
                [row for row in optimal if row["method"] == method],
                key=lambda row: row["budget"],
            )
            axes[1].plot(
                [row["budget"] for row in method_optimal],
                [row["best_D"] for row in method_optimal],
                marker="o", color=COLORS[method], label=DISPLAY_NAMES[method])
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Fixed raw budget B")
    axes[0].set_ylabel(r"Median $\Delta E_{D=30}-\Delta E_{D=15}$")
    axes[0].set_title("(a) Does extrapolated D help at fixed B?")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_yticks(protocol["d_values"])
    axes[1].set_xlabel("Fixed raw budget B")
    axes[1].set_ylabel("D with lowest median energy error")
    axes[1].set_title("(b) Budget-dependent optimal D")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"fixed_budget_d15_d30.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def save_results(args, protocol, raw_results, source_mapping):
    output_dir = Path(args.output_dir)
    per_hamiltonian = per_hamiltonian_medians(raw_results)
    summary = aggregate_summary(per_hamiltonian)
    comparisons = compare_d15_d30(per_hamiltonian, args)
    optimal = optimal_dimensions(summary)

    _write_csv(output_dir / "raw_results.csv", raw_results)
    _write_csv(output_dir / "per_hamiltonian.csv", per_hamiltonian)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "d15_vs_d30.csv", comparisons)
    _write_csv(output_dir / "optimal_dimensions.csv", optimal)
    with (output_dir / "source_mapping.json").open("w") as handle:
        json.dump(source_mapping, handle, indent=2)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(protocol, handle, indent=2)
    plot_heatmaps(summary, protocol, output_dir)
    plot_d15_d30(comparisons, optimal, protocol, output_dir)
    print(f"Wrote fixed-total-budget artifacts to {output_dir}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    protocol, raw_results, source_mapping = run_experiment(args)
    save_results(args, protocol, raw_results, source_mapping)


if __name__ == "__main__":
    main()
