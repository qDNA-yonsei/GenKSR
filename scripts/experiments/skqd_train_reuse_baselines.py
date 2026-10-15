#!/usr/bin/env python3
"""Evaluate equal-resource train-reuse baselines for canonical SKQD.

This is Gate 1-B after the uniform support-controlled benchmark.  It asks
whether a conditional neural generator is needed, or whether one can simply
reuse nearby training measurement records.

The baselines are deliberately restricted to exactly the information available
to the trained model:

* training Hamiltonians only;
* training dimensions ``D0..D14`` only;
* the first 1,000 measurement records per training condition only.

``nearest_train`` first chooses the training Hamiltonian with closest ``J2``
and then the training dimension with closest physical time.  ``knn5_uniform``
selects five nearest conditions in normalized ``(J2, t)`` space and samples an
equal-weight empirical mixture.  Empirical records are bootstrapped with
replacement to permit an inference budget larger than the original 1,000
records; this does not create new support outside the selected training data.

Both raw-draw comparisons and pairwise equal-unique-basis comparisons are
reported across the existing D=30 M sweep.
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from genksr.evaluation.skqd import projected_xxz_hamiltonian
from scripts.experiments.skqd_support_controlled import (
    _basis_metrics,
    _bits_to_integer_stream,
    _load_saved_stream,
    _lowest_eigenpair,
    _unique_in_arrival_order,
)


MODELS = ("transformer", "mamba")
BASELINES = ("nearest_train", "knn5_uniform")
REFERENCE_METHODS = ("device", "transformer", "mamba", "uniform_sector")
DISPLAY_NAMES = {
    "device": "Exact-state",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "nearest_train": "Nearest train",
    "knn5_uniform": "kNN-5 mixture",
    "uniform_sector": "Uniform Sz=0",
}
COLORS = {
    "device": "tab:blue",
    "transformer": "tab:orange",
    "mamba": "tab:red",
    "nearest_train": "tab:brown",
    "knn5_uniform": "tab:cyan",
    "uniform_sector": "tab:purple",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train-reuse baselines for the D=30 SKQD support sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation-dir",
        default="artifacts/v3_skqd_formal/evaluation_random_j2_10000",
    )
    parser.add_argument("--data-root", default="data/heisenberg_SQKD_2d")
    parser.add_argument(
        "--existing-sweep-root",
        default="artifacts/v3_skqd_support_m_sweep_d30",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v3_skqd_train_reuse_baselines_d30",
    )
    parser.add_argument("--krylov-dim", type=int, default=30)
    parser.add_argument("--train-krylov-dim", type=int, default=15)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument("--high-j2-threshold", type=float, default=0.8)
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


def _validate_args(args):
    if args.krylov_dim <= 0 or args.train_krylov_dim <= 0:
        raise ValueError("Krylov dimensions must be positive")
    if args.train_samples <= 0 or args.knn_k <= 0:
        raise ValueError("train-samples and knn-k must be positive")
    if not args.m_values or any(m <= 0 for m in args.m_values):
        raise ValueError("m-values must be positive")
    args.m_values = sorted(set(args.m_values))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _infer_j2_dt(coupling):
    coupling = np.asarray(coupling, dtype=float)
    n_qubit = coupling.shape[0]
    upper = coupling[np.triu_indices(n_qubit, k=1)]
    nonzero = upper[~np.isclose(upper, 0.0)]
    non_j1 = nonzero[~np.isclose(nonzero, 1.0)]
    if len(non_j1) == 0 or not np.allclose(non_j1, non_j1[0]):
        raise ValueError("Expected a single non-J1 coupling value")
    j2 = float(non_j1[0])
    # build_2d_xxz_hamiltonian contributes XX, YY, and ZZ for every edge.
    dt = float(np.pi / (3.0 * np.sum(np.abs(nonzero))))
    return j2, dt


def _load_protocol(args):
    evaluation_dir = Path(args.evaluation_dir)
    with (evaluation_dir / "evaluation_args.json").open() as handle:
        evaluation_args = json.load(handle)
    if evaluation_args.get("task") != "sqkd":
        raise ValueError("Expected a canonical SQKD evaluation")
    if int(evaluation_args["n_qubit"]) != 16:
        raise ValueError("This experiment currently supports the 16-qubit run")
    if args.krylov_dim > int(evaluation_args["max_krylov_dim"]):
        raise ValueError("Requested D exceeds saved model samples")
    if max(args.m_values) > int(evaluation_args["snapshots"]):
        raise ValueError("Requested M exceeds saved model samples")

    model_dirs = {
        "transformer": Path(evaluation_args["transformer_results"]),
        "mamba": Path(evaluation_args["mamba_results"]),
    }
    for method, path in model_dirs.items():
        with (path / "config.json").open() as handle:
            model_config = json.load(handle)
        data_config = model_config["data"]
        if int(data_config["krylov_dim"]) != args.train_krylov_dim:
            raise ValueError(f"{method} train-D does not match requested baseline")
        if int(data_config["train_samples"]) != args.train_samples:
            raise ValueError(f"{method} train sample budget does not match baseline")
    return evaluation_args, model_dirs


def _load_existing_sweep(args):
    path = Path(args.existing_sweep_root) / "combined_per_hamiltonian.csv"
    rows = _read_csv(path)
    parsed = []
    for row in rows:
        if row["matching"] != "raw":
            continue
        record = {
            "M": int(row["M"]),
            "matching": "raw",
            "method": row["method"],
            "hamiltonian_id": int(row["hamiltonian_id"]),
            "J2": float(row["J2"]),
        }
        for field in (
            "delta_e",
            "basis_size",
            "coverage",
            "acceptance_rate",
            "captured_ground_mass",
            "missing_ground_mass",
        ):
            record[field] = float(row[field])
        parsed.append(record)
    return parsed


def _build_condition_library(train_dir, args):
    conditions = []
    hamiltonians = {}
    for hid in range(30):
        path = train_dir / f"coupling_matrix_id{hid}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        coupling = np.load(path)
        j2, dt = _infer_j2_dt(coupling)
        hamiltonians[hid] = {"J2": j2, "dt": dt}
        for dimension in range(args.train_krylov_dim):
            conditions.append({
                "hid": hid,
                "dimension": dimension,
                "J2": j2,
                "time": float(dimension * dt),
            })
    time_scale = max(condition["time"] for condition in conditions)
    if time_scale <= 0:
        raise ValueError("Training physical-time scale must be positive")
    for condition in conditions:
        condition["features"] = np.asarray([
            condition["J2"], condition["time"] / time_scale], dtype=float)
    return conditions, hamiltonians, time_scale


class TrainingSampleCache:
    def __init__(self, train_dir, train_samples, n_qubit):
        self.train_dir = Path(train_dir)
        self.train_samples = int(train_samples)
        self.n_qubit = int(n_qubit)
        self.num_ones = n_qubit // 2
        self.cache = {}

    def get(self, hid, dimension):
        key = (int(hid), int(dimension))
        if key not in self.cache:
            path = self.train_dir / f"data_D{dimension}_id{hid}.npy"
            samples = np.load(path, mmap_mode="r")
            if len(samples) < self.train_samples:
                raise ValueError(f"Not enough training records in {path}")
            states = _bits_to_integer_stream(
                samples[:self.train_samples],
                n_qubit=self.n_qubit,
                num_ones=self.num_ones,
            )
            if len(states) != self.train_samples:
                raise ValueError(f"Training source left the Sz=0 sector: {path}")
            self.cache[key] = states
        return self.cache[key]


def _condition_distance(condition, target_features):
    return float(np.linalg.norm(condition["features"] - target_features))


def _select_sources(
        test_j2, target_time, conditions, train_hamiltonians,
        time_scale, knn_k):
    target_features = np.asarray([test_j2, target_time / time_scale], dtype=float)
    nearest_hid = min(
        train_hamiltonians,
        key=lambda hid: (abs(train_hamiltonians[hid]["J2"] - test_j2), hid),
    )
    same_h_conditions = [
        condition for condition in conditions if condition["hid"] == nearest_hid
    ]
    nearest_condition = min(
        same_h_conditions,
        key=lambda condition: (
            abs(condition["time"] - target_time), condition["dimension"]),
    )
    ranked = sorted(
        conditions,
        key=lambda condition: (
            _condition_distance(condition, target_features),
            condition["hid"],
            condition["dimension"],
        ),
    )
    knn_conditions = ranked[:knn_k]
    return target_features, nearest_condition, knn_conditions


def _bootstrap_source(source_states, size, rng):
    indices = rng.integers(0, len(source_states), size=size)
    return source_states[indices]


def _generate_baseline_streams(
        hid, test_j2, test_dt, conditions, train_hamiltonians, time_scale,
        sample_cache, args):
    max_m = max(args.m_values)
    by_method = {method: {} for method in BASELINES}
    mapping = []
    for dimension in range(args.krylov_dim):
        target_time = float(dimension * test_dt)
        target_features, nearest, neighbors = _select_sources(
            test_j2,
            target_time,
            conditions,
            train_hamiltonians,
            time_scale,
            args.knn_k,
        )

        nearest_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, hid, dimension, 0]))
        nearest_states = sample_cache.get(nearest["hid"], nearest["dimension"])
        by_method["nearest_train"][dimension] = _bootstrap_source(
            nearest_states, max_m, nearest_rng)

        knn_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, hid, dimension, 1]))
        choices = knn_rng.integers(0, len(neighbors), size=max_m)
        mixture = np.empty(max_m, dtype=np.int64)
        for neighbor_index, neighbor in enumerate(neighbors):
            positions = np.flatnonzero(choices == neighbor_index)
            if len(positions) == 0:
                continue
            source = sample_cache.get(neighbor["hid"], neighbor["dimension"])
            mixture[positions] = _bootstrap_source(source, len(positions), knn_rng)
        by_method["knn5_uniform"][dimension] = mixture

        mapping.append({
            "target_hamiltonian_id": int(hid),
            "target_dimension": int(dimension),
            "target_J2": float(test_j2),
            "target_time": target_time,
            "target_features": target_features.tolist(),
            "nearest_train": {
                "train_hamiltonian_id": int(nearest["hid"]),
                "train_dimension": int(nearest["dimension"]),
                "train_J2": float(nearest["J2"]),
                "train_time": float(nearest["time"]),
                "feature_distance": _condition_distance(nearest, target_features),
            },
            "knn5_uniform": [
                {
                    "train_hamiltonian_id": int(condition["hid"]),
                    "train_dimension": int(condition["dimension"]),
                    "train_J2": float(condition["J2"]),
                    "train_time": float(condition["time"]),
                    "feature_distance": _condition_distance(
                        condition, target_features),
                    "weight": float(1.0 / len(neighbors)),
                }
                for condition in neighbors
            ],
        })
    return by_method, mapping


def _stream_at_m(streams_by_dimension, m, krylov_dim):
    return np.concatenate([
        streams_by_dimension[dimension][:m]
        for dimension in range(krylov_dim)
    ])


def _existing_lookup(existing_rows):
    return {
        (row["M"], row["method"], row["hamiltonian_id"]): row
        for row in existing_rows
    }


def run_experiment(args):
    evaluation_args, model_dirs = _load_protocol(args)
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
    existing_rows = _load_existing_sweep(args)
    hamiltonian_ids = (
        list(range(20)) if args.hamiltonian_ids is None
        else sorted(set(args.hamiltonian_ids))
    )
    if any(hid < 0 or hid >= 20 for hid in hamiltonian_ids):
        raise ValueError("hamiltonian-ids must be between 0 and 19")
    existing_rows = [
        row for row in existing_rows if row["hamiltonian_id"] in hamiltonian_ids
    ]
    existing = _existing_lookup(existing_rows)
    snapshots = int(evaluation_args["snapshots"])

    baseline_raw_rows = []
    pair_rows = []
    source_mapping = []
    test_parameter_summary = []

    for position, hid in enumerate(hamiltonian_ids, start=1):
        print(
            f"[{position}/{len(hamiltonian_ids)}] test Hamiltonian id={hid}",
            flush=True,
        )
        coupling = np.load(test_dir / f"coupling_matrix_id{hid}.npy")
        test_j2, test_dt = _infer_j2_dt(coupling)
        nearest_hid = min(
            train_hamiltonians,
            key=lambda train_hid: (
                abs(train_hamiltonians[train_hid]["J2"] - test_j2), train_hid),
        )
        test_parameter_summary.append({
            "test_hamiltonian_id": int(hid),
            "test_J2": float(test_j2),
            "test_dt": float(test_dt),
            "nearest_train_hamiltonian_id": int(nearest_hid),
            "nearest_train_J2": float(train_hamiltonians[nearest_hid]["J2"]),
            "nearest_J2_gap": float(abs(
                train_hamiltonians[nearest_hid]["J2"] - test_j2)),
        })

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

        baseline_by_dimension, mapping = _generate_baseline_streams(
            hid,
            test_j2,
            test_dt,
            conditions,
            train_hamiltonians,
            time_scale,
            sample_cache,
            args,
        )
        source_mapping.extend(mapping)

        for m in args.m_values:
            raw_draws = args.krylov_dim * m
            model_bases = {}
            for model in MODELS:
                model_stream = _load_saved_stream(
                    model,
                    hid,
                    args.krylov_dim,
                    m,
                    snapshots,
                    test_dir,
                    model_dirs,
                    n_qubit,
                    num_ones,
                )
                model_bases[model] = _unique_in_arrival_order(model_stream)
                expected_size = int(existing[(m, model, hid)]["basis_size"])
                if len(model_bases[model]) != expected_size:
                    raise RuntimeError(
                        f"Saved model basis mismatch: id={hid}, M={m}, {model}"
                    )

            baseline_bases = {}
            for baseline in BASELINES:
                baseline_stream = _stream_at_m(
                    baseline_by_dimension[baseline], m, args.krylov_dim)
                basis = _unique_in_arrival_order(baseline_stream)
                baseline_bases[baseline] = basis
                energy, delta_e, captured_mass = _basis_metrics(
                    basis,
                    sector_hamiltonian,
                    sector_index,
                    ground_probabilities,
                    exact_energy,
                )
                baseline_raw_rows.append({
                    "hamiltonian_id": int(hid),
                    "J2": float(test_j2),
                    "M": int(m),
                    "D": int(args.krylov_dim),
                    "method": baseline,
                    "raw_draws": int(raw_draws),
                    "accepted_draws": int(raw_draws),
                    "acceptance_rate": 1.0,
                    "basis_size": int(len(basis)),
                    "coverage": float(len(basis) / sector_size),
                    "energy": float(energy),
                    "exact_energy": float(exact_energy),
                    "delta_e": float(delta_e),
                    "captured_ground_mass": float(captured_mass),
                    "missing_ground_mass": float(max(1.0 - captured_mass, 0.0)),
                })

            metric_cache = {}
            for model in MODELS:
                for baseline in BASELINES:
                    model_basis = model_bases[model]
                    baseline_basis = baseline_bases[baseline]
                    common_k = min(len(model_basis), len(baseline_basis))

                    model_key = (model, common_k)
                    if model_key not in metric_cache:
                        metric_cache[model_key] = _basis_metrics(
                            model_basis[:common_k],
                            sector_hamiltonian,
                            sector_index,
                            ground_probabilities,
                            exact_energy,
                        )
                    baseline_key = (baseline, common_k)
                    if baseline_key not in metric_cache:
                        metric_cache[baseline_key] = _basis_metrics(
                            baseline_basis[:common_k],
                            sector_hamiltonian,
                            sector_index,
                            ground_probabilities,
                            exact_energy,
                        )
                    model_energy, model_delta, model_mass = metric_cache[model_key]
                    baseline_energy, baseline_delta, baseline_mass = metric_cache[
                        baseline_key]
                    pair_rows.append({
                        "hamiltonian_id": int(hid),
                        "J2": float(test_j2),
                        "M": int(m),
                        "D": int(args.krylov_dim),
                        "model": model,
                        "baseline": baseline,
                        "common_basis_size": int(common_k),
                        "coverage": float(common_k / sector_size),
                        "model_raw_basis_size": int(len(model_basis)),
                        "baseline_raw_basis_size": int(len(baseline_basis)),
                        "model_energy": float(model_energy),
                        "baseline_energy": float(baseline_energy),
                        "exact_energy": float(exact_energy),
                        "model_delta_e": float(model_delta),
                        "baseline_delta_e": float(baseline_delta),
                        "delta_e_difference": float(model_delta - baseline_delta),
                        "model_captured_ground_mass": float(model_mass),
                        "baseline_captured_ground_mass": float(baseline_mass),
                        "captured_mass_difference": float(model_mass - baseline_mass),
                    })
        print(
            f"[{position}/{len(hamiltonian_ids)}] id={hid}: complete",
            flush=True,
        )

    protocol = {
        "task": "sqkd_train_reuse_baselines",
        "n_qubit": n_qubit,
        "krylov_dim": args.krylov_dim,
        "m_values": args.m_values,
        "train_hamiltonians": 30,
        "test_hamiltonians": len(hamiltonian_ids),
        "test_hamiltonian_ids": hamiltonian_ids,
        "train_krylov_dimensions": list(range(args.train_krylov_dim)),
        "train_samples_per_condition": args.train_samples,
        "inference_sampling": "bootstrap with replacement from allowed train records",
        "feature_vector": "(J2, physical_time / max_train_physical_time)",
        "max_train_physical_time": time_scale,
        "nearest_train": (
            "nearest J2 training Hamiltonian, then nearest physical-time train dimension"
        ),
        "knn5_uniform": (
            f"uniform empirical mixture of k={args.knn_k} nearest train conditions"
        ),
        "hyperparameter_note": (
            "k and feature scaling fixed before inspecting downstream baseline energy; "
            "no test-label tuning"
        ),
        "high_j2_threshold": args.high_j2_threshold,
        "high_j2_split_note": "exploratory/post-pilot",
        "seed": args.seed,
    }
    return (
        protocol,
        existing_rows,
        baseline_raw_rows,
        pair_rows,
        source_mapping,
        test_parameter_summary,
    )


def _bootstrap_ci(values, repeats, seed_sequence):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_sequence)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    return (
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def _raw_comparison_rows(existing_rows, baseline_raw_rows, m_values):
    rows = [
        row for row in existing_rows
        if row["M"] in m_values and row["method"] in REFERENCE_METHODS
    ]
    rows.extend({
        "M": row["M"],
        "matching": "raw",
        "method": row["method"],
        "hamiltonian_id": row["hamiltonian_id"],
        "J2": row["J2"],
        "delta_e": row["delta_e"],
        "basis_size": row["basis_size"],
        "coverage": row["coverage"],
        "acceptance_rate": row["acceptance_rate"],
        "captured_ground_mass": row["captured_ground_mass"],
        "missing_ground_mass": row["missing_ground_mass"],
    } for row in baseline_raw_rows)
    return rows


def aggregate_raw(raw_rows, m_values):
    summary = []
    methods = REFERENCE_METHODS + BASELINES
    for m in m_values:
        for method in methods:
            subset = [
                row for row in raw_rows
                if row["M"] == m and row["method"] == method
            ]
            if not subset:
                continue
            record = {"M": int(m), "method": method, "n_hamiltonians": len(subset)}
            for field in (
                "delta_e",
                "basis_size",
                "coverage",
                "captured_ground_mass",
                "missing_ground_mass",
            ):
                values = np.asarray([row[field] for row in subset], dtype=float)
                record[f"{field}_median"] = float(np.median(values))
                record[f"{field}_q25"] = float(np.quantile(values, 0.25))
                record[f"{field}_q75"] = float(np.quantile(values, 0.75))
            summary.append(record)
    return summary


def paired_summary(raw_rows, pair_rows, args):
    raw_lookup = {
        (row["M"], row["method"], row["hamiltonian_id"]): row
        for row in raw_rows
    }
    unique_lookup = {
        (row["M"], row["model"], row["baseline"], row["hamiltonian_id"]): row
        for row in pair_rows
    }
    ids = sorted({row["hamiltonian_id"] for row in pair_rows})
    j2_by_id = {
        hid: unique_lookup[(args.m_values[0], MODELS[0], BASELINES[0], hid)]["J2"]
        for hid in ids
    }
    regimes = {
        "all": ids,
        "J2_lt_threshold": [
            hid for hid in ids if j2_by_id[hid] < args.high_j2_threshold
        ],
        "J2_ge_threshold": [
            hid for hid in ids if j2_by_id[hid] >= args.high_j2_threshold
        ],
    }

    records = []
    comparison_index = 0
    for matching in ("raw", "unique"):
        for m in args.m_values:
            for model in MODELS:
                for baseline in BASELINES:
                    for regime, regime_ids in regimes.items():
                        if not regime_ids:
                            continue
                        if matching == "raw":
                            model_errors = np.asarray([
                                raw_lookup[(m, model, hid)]["delta_e"]
                                for hid in regime_ids
                            ])
                            baseline_errors = np.asarray([
                                raw_lookup[(m, baseline, hid)]["delta_e"]
                                for hid in regime_ids
                            ])
                            model_masses = np.asarray([
                                raw_lookup[(m, model, hid)]["captured_ground_mass"]
                                for hid in regime_ids
                            ])
                            baseline_masses = np.asarray([
                                raw_lookup[(m, baseline, hid)]["captured_ground_mass"]
                                for hid in regime_ids
                            ])
                            common_k = np.full(len(regime_ids), np.nan)
                        else:
                            subset = [
                                unique_lookup[(m, model, baseline, hid)]
                                for hid in regime_ids
                            ]
                            model_errors = np.asarray([
                                row["model_delta_e"] for row in subset])
                            baseline_errors = np.asarray([
                                row["baseline_delta_e"] for row in subset])
                            model_masses = np.asarray([
                                row["model_captured_ground_mass"] for row in subset])
                            baseline_masses = np.asarray([
                                row["baseline_captured_ground_mass"] for row in subset])
                            common_k = np.asarray([
                                row["common_basis_size"] for row in subset])

                        energy_diffs = model_errors - baseline_errors
                        mass_diffs = model_masses - baseline_masses
                        ci_low, ci_high = _bootstrap_ci(
                            energy_diffs,
                            args.bootstrap_repeats,
                            np.random.SeedSequence(
                                [args.seed, comparison_index, 0]),
                        )
                        mass_low, mass_high = _bootstrap_ci(
                            mass_diffs,
                            args.bootstrap_repeats,
                            np.random.SeedSequence(
                                [args.seed, comparison_index, 1]),
                        )
                        nonzero = energy_diffs[
                            ~np.isclose(energy_diffs, 0.0, atol=1e-14)]
                        p_value = 1.0 if len(nonzero) == 0 else float(
                            wilcoxon(nonzero, alternative="two-sided").pvalue)
                        records.append({
                            "matching": matching,
                            "M": int(m),
                            "model": model,
                            "baseline": baseline,
                            "regime": regime,
                            "n_hamiltonians": len(regime_ids),
                            "median_common_basis_size": (
                                float(np.nanmedian(common_k))
                                if matching == "unique" else ""
                            ),
                            "median_model_delta_e": float(np.median(model_errors)),
                            "median_baseline_delta_e": float(
                                np.median(baseline_errors)),
                            "median_delta_e_difference": float(
                                np.median(energy_diffs)),
                            "delta_e_difference_ci95_low": ci_low,
                            "delta_e_difference_ci95_high": ci_high,
                            "energy_win_rate": float(np.mean(energy_diffs < 0.0)),
                            "wilcoxon_p_two_sided": p_value,
                            "median_captured_mass_difference": float(
                                np.median(mass_diffs)),
                            "captured_mass_difference_ci95_low": mass_low,
                            "captured_mass_difference_ci95_high": mass_high,
                            "captured_mass_win_rate": float(
                                np.mean(mass_diffs > 0.0)),
                        })
                        comparison_index += 1
    return records


def _summary_lookup(summary):
    return {(row["M"], row["method"]): row for row in summary}


def plot_results(raw_summary, comparisons, output_dir, protocol):
    output_dir = Path(output_dir)
    lookup = _summary_lookup(raw_summary)
    m_values = protocol["m_values"]
    plot_methods = (
        "device", "transformer", "mamba", "nearest_train",
        "knn5_uniform", "uniform_sector",
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.reshape(-1)

    for method in plot_methods:
        medians = np.asarray([
            max(lookup[(m, method)]["delta_e_median"], 1e-8)
            for m in m_values
        ])
        axes[0].plot(
            m_values, medians, marker="o", color=COLORS[method],
            label=DISPLAY_NAMES[method])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Samples per Krylov dimension M")
    axes[0].set_ylabel(r"Median raw-budget energy error $E_{\mathrm{proj}}-E_0$")
    axes[0].set_title("(a) Raw-draw comparison")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    for method in plot_methods:
        axes[1].plot(
            m_values,
            [lookup[(m, method)]["basis_size_median"] for m in m_values],
            marker="o", color=COLORS[method], label=DISPLAY_NAMES[method])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Samples per Krylov dimension M")
    axes[1].set_ylabel("Median raw unique basis size")
    axes[1].set_title("(b) Support growth")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    line_styles = {"nearest_train": "--", "knn5_uniform": "-"}
    for axis, matching, title in (
        (axes[2], "raw", "(c) Raw-budget model advantage"),
        (axes[3], "unique", "(d) Pairwise equal-K model advantage"),
    ):
        for model in MODELS:
            for baseline in BASELINES:
                subset = sorted([
                    row for row in comparisons
                    if row["matching"] == matching and row["model"] == model
                    and row["baseline"] == baseline and row["regime"] == "all"
                ], key=lambda row: row["M"])
                axis.plot(
                    [row["M"] for row in subset],
                    [row["median_delta_e_difference"] for row in subset],
                    marker="o",
                    linestyle=line_styles[baseline],
                    color=COLORS[model],
                    label=(
                        f"{DISPLAY_NAMES[model]} - {DISPLAY_NAMES[baseline]}"
                    ),
                )
        axis.axhline(0.0, color="black", linewidth=1, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel("Samples per Krylov dimension M")
        axis.set_ylabel(
            r"Median $\Delta E_{\mathrm{model}}-\Delta E_{\mathrm{baseline}}$"
        )
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Equal-resource train-reuse baselines (D=30, 20 held-out Hamiltonians)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"train_reuse_baselines.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_heatmap(pair_rows, output_dir, protocol):
    output_dir = Path(output_dir)
    m_values = protocol["m_values"]
    lookup = {
        (row["M"], row["model"], row["baseline"], row["hamiltonian_id"]): row
        for row in pair_rows
    }
    ids = sorted(
        {row["hamiltonian_id"] for row in pair_rows},
        key=lambda hid: lookup[(m_values[0], MODELS[0], "knn5_uniform", hid)]["J2"],
    )
    matrices = {
        model: np.asarray([
            [
                lookup[(m, model, "knn5_uniform", hid)]["delta_e_difference"]
                for m in m_values
            ]
            for hid in ids
        ])
        for model in MODELS
    }
    minimum = min(float(np.min(matrix)) for matrix in matrices.values())
    maximum = max(float(np.max(matrix)) for matrix in matrices.values())
    if not (minimum < 0.0 < maximum):
        maximum = max(maximum, 1e-8)
        minimum = min(minimum, -1e-8)
    norm = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)

    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    image = None
    for axis, model in zip(axes, MODELS):
        image = axis.imshow(
            matrices[model], aspect="auto", cmap="coolwarm", norm=norm)
        axis.set_xticks(range(len(m_values)), [str(m) for m in m_values], rotation=40)
        axis.set_xlabel("M")
        axis.set_title(DISPLAY_NAMES[model])
    axes[0].set_yticks(
        range(len(ids)),
        [f"{lookup[(m_values[0], MODELS[0], 'knn5_uniform', hid)]['J2']:.3f}"
         for hid in ids],
    )
    axes[0].set_ylabel("Held-out J2")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label(
        r"Equal-K $\Delta E_{\mathrm{model}}-\Delta E_{\mathrm{kNN5}}$"
        " (negative is better)"
    )
    figure.suptitle("Per-Hamiltonian neural advantage over train-data kNN")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.08)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"train_reuse_advantage_j2_m.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def save_results(args, outputs):
    (
        protocol,
        existing_rows,
        baseline_raw_rows,
        pair_rows,
        source_mapping,
        test_parameter_summary,
    ) = outputs
    output_dir = Path(args.output_dir)
    raw_rows = _raw_comparison_rows(existing_rows, baseline_raw_rows, args.m_values)
    raw_summary = aggregate_raw(raw_rows, args.m_values)
    comparisons = paired_summary(raw_rows, pair_rows, args)

    _write_csv(output_dir / "baseline_raw_results.csv", baseline_raw_rows)
    _write_csv(output_dir / "raw_comparison_per_hamiltonian.csv", raw_rows)
    _write_csv(output_dir / "raw_summary.csv", raw_summary)
    _write_csv(output_dir / "pairwise_unique_results.csv", pair_rows)
    _write_csv(output_dir / "paired_summary.csv", comparisons)
    _write_csv(output_dir / "test_parameter_summary.csv", test_parameter_summary)
    with (output_dir / "source_mapping.json").open("w") as handle:
        json.dump(source_mapping, handle, indent=2)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(protocol, handle, indent=2)
    plot_results(raw_summary, comparisons, output_dir, protocol)
    plot_heatmap(pair_rows, output_dir, protocol)
    print(f"Wrote train-reuse baseline artifacts to {output_dir}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    outputs = run_experiment(args)
    save_results(args, outputs)


if __name__ == "__main__":
    main()
