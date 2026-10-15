#!/usr/bin/env python3
"""Pilot support-controlled benchmark for the canonical 16-qubit SKQD run.

The formal GenKSR result compares methods at the same raw draw budget.  At
``D=30, M=10_000`` all methods cover most of the 12,870-state ``Sz=0`` sector,
so that comparison cannot tell whether a low energy comes from targeted sample
quality or merely from a very large sampled subspace.

This script reuses the saved held-out samples and adds two deliberately simple
baselines:

* ``uniform_full``: uniform draws from all ``2**N`` bitstrings followed by the
  same ``Sz=0`` postselection as GenKSR;
* ``uniform_sector``: uniform draws directly from the ``Sz=0`` sector.

For every held-out Hamiltonian it reports both:

* ``raw``: each method uses all samples available at the common raw budget;
* ``unique``: every method is truncated to the same number of first-seen unique
  postselected basis states.

The second comparison is the pilot's decision statistic.  It removes the size
of the projected eigensolver subspace as an explanation and asks whether the
states proposed by a method are better than a uniform subset of equal size.

Saved model/device samples provide one fixed sampling realization.  Uniform
baseline repeats are independently seeded; they do not turn the fixed model
sample into a multi-seed model experiment.  This is a falsification pilot, not
the final uncertainty analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as sparse_linalg  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from genksr.evaluation.skqd import projected_xxz_hamiltonian


FIXED_METHODS = ("device", "transformer", "mamba")
BASELINE_METHODS = ("uniform_full", "uniform_sector")
METHODS = FIXED_METHODS + BASELINE_METHODS
DISPLAY_NAMES = {
    "device": "Exact-state",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "uniform_full": "Uniform full",
    "uniform_sector": "Uniform Sz=0",
}
COLORS = {
    "device": "tab:blue",
    "transformer": "tab:orange",
    "mamba": "tab:red",
    "uniform_full": "tab:green",
    "uniform_sector": "tab:purple",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Support-controlled pilot for the canonical v3 SKQD samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation-dir",
        default="artifacts/v3_skqd_formal/evaluation_random_j2_10000",
        help="Canonical evaluation containing all_results.pkl and evaluation_args.json",
    )
    parser.add_argument(
        "--data-root",
        default="data/heisenberg_SQKD_2d",
        help="Root containing 16qubits/test device samples",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v3_skqd_support_controlled_pilot_d30_m10000",
    )
    parser.add_argument("--krylov-dim", type=int, default=30)
    parser.add_argument("--samples-per-dim", type=int, default=10_000)
    parser.add_argument("--baseline-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--hamiltonian-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset for a smoke test",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=5000,
        help="Paired bootstrap repeats over held-out Hamiltonians",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing output directory",
    )
    return parser.parse_args(argv)


def _validate_args(args):
    if args.krylov_dim <= 0 or args.samples_per_dim <= 0:
        raise ValueError("krylov-dim and samples-per-dim must be positive")
    if args.baseline_repeats <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("repeat counts must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_canonical_inputs(args):
    evaluation_dir = Path(args.evaluation_dir)
    args_path = evaluation_dir / "evaluation_args.json"
    results_path = evaluation_dir / "all_results.pkl"
    if not args_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            f"Expected evaluation_args.json and all_results.pkl in {evaluation_dir}"
        )

    with args_path.open() as handle:
        evaluation_args = json.load(handle)
    with results_path.open("rb") as handle:
        canonical_results = pickle.load(handle)

    if evaluation_args.get("task") != "sqkd" or evaluation_args.get("n_qubit") != 16:
        raise ValueError("This pilot currently supports only canonical 16-qubit SQKD")
    if args.krylov_dim > int(evaluation_args["max_krylov_dim"]):
        raise ValueError("Requested D exceeds the canonical saved evaluation")
    if args.samples_per_dim > int(evaluation_args["snapshots"]):
        raise ValueError("Requested M exceeds the saved sample-file snapshot count")

    data_dir = Path(args.data_root) / "16qubits" / "test"
    model_dirs = {
        "transformer": Path(evaluation_args["transformer_results"]),
        "mamba": Path(evaluation_args["mamba_results"]),
    }
    return evaluation_args, canonical_results, data_dir, model_dirs


def _available_hamiltonian_ids(data_dir):
    ids = []
    pattern = re.compile(r"coupling_matrix_id(\d+)\.npy$")
    for path in data_dir.glob("coupling_matrix_id*.npy"):
        match = pattern.search(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _sample_path(method, hid, dimension, snapshots, data_dir, model_dirs):
    if method == "device":
        return data_dir / f"data_D{dimension}_id{hid}.npy"
    return (
        model_dirs[method]
        / "samples"
        / "test"
        / "model"
        / f"model_samples_D{dimension}_{snapshots}_id{hid}.npy"
    )


def _bits_to_integer_stream(samples, n_qubit, num_ones):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != n_qubit:
        raise ValueError(f"Unexpected sample shape: {samples.shape}")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("SQKD sample file contains a non-binary outcome")
    selected = samples[np.sum(samples, axis=1) == num_ones]
    if len(selected) == 0:
        return np.empty(0, dtype=np.int64)
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    return selected.astype(np.int64) @ powers


def _load_saved_stream(
        method, hid, krylov_dim, samples_per_dim, snapshots,
        data_dir, model_dirs, n_qubit, num_ones):
    accepted_parts = []
    for dimension in range(krylov_dim):
        path = _sample_path(
            method, hid, dimension, snapshots, data_dir, model_dirs)
        if not path.exists():
            raise FileNotFoundError(path)
        samples = np.load(path, mmap_mode="r")
        if len(samples) < samples_per_dim:
            raise ValueError(
                f"{path} contains {len(samples)} rows; need {samples_per_dim}"
            )
        accepted_parts.append(
            _bits_to_integer_stream(
                samples[:samples_per_dim], n_qubit=n_qubit, num_ones=num_ones)
        )
    return np.concatenate(accepted_parts)


def _unique_in_arrival_order(states):
    states = np.asarray(states, dtype=np.int64)
    if len(states) == 0:
        return states
    _, first_indices = np.unique(states, return_index=True)
    return states[np.sort(first_indices)]


def _uniform_streams(
        hid, repeat, total_draws, hilbert_size, sector_states, sector_mask, seed):
    full_rng = np.random.default_rng(
        np.random.SeedSequence([seed, hid, repeat, 0]))
    sector_rng = np.random.default_rng(
        np.random.SeedSequence([seed, hid, repeat, 1]))
    full_draws = full_rng.integers(
        0, hilbert_size, size=total_draws, dtype=np.int64)
    uniform_full = full_draws[sector_mask[full_draws]]
    uniform_sector = sector_rng.choice(
        sector_states, size=total_draws, replace=True)
    return {
        "uniform_full": uniform_full,
        "uniform_sector": uniform_sector,
    }


def _lowest_eigenpair(matrix, tol=1e-12):
    dim = matrix.shape[0]
    if dim == 0:
        return np.nan, np.empty(0, dtype=float)
    if dim <= 2:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return float(values[0]), np.asarray(vectors[:, 0])
    v0 = np.linspace(1.0, 2.0, dim, dtype=float)
    v0 /= np.linalg.norm(v0)
    values, vectors = sparse_linalg.eigsh(
        matrix, which="SA", k=1, v0=v0, tol=tol)
    return float(values.real[0]), np.asarray(vectors[:, 0]).reshape(-1)


def _basis_metrics(
        basis, sector_hamiltonian, sector_index, ground_probabilities,
        exact_energy, cached_energy=None):
    basis = np.asarray(basis, dtype=np.int64)
    indices = sector_index[basis]
    if np.any(indices < 0):
        raise ValueError("A postselected basis state is outside the symmetry sector")
    captured_mass = float(np.sum(ground_probabilities[indices]))

    if cached_energy is None or not np.isfinite(cached_energy):
        if len(indices) == sector_hamiltonian.shape[0]:
            energy = exact_energy
        else:
            projected = sector_hamiltonian[indices][:, indices]
            energy, _ = _lowest_eigenpair(projected, tol=1e-10)
    else:
        energy = float(cached_energy)

    delta_e = float(energy - exact_energy)
    if delta_e < -1e-7:
        raise RuntimeError(
            f"Projected energy {energy} is below exact energy {exact_energy}"
        )
    return float(energy), max(delta_e, 0.0), captured_mass


def _canonical_energy(
        canonical_results, method, hid, krylov_dim, samples_per_dim):
    entry = canonical_results["test"][method][hid]
    final_index = krylov_dim - 1
    input_counts = entry.get("num_input_samples_per_dim", [])
    if final_index >= len(input_counts) or input_counts[final_index] != samples_per_dim:
        return None
    energy = float(entry["energies"][final_index])
    return energy if np.isfinite(energy) else None


def _record(
        hid, j2, method, repeat, matching, krylov_dim, samples_per_dim,
        raw_draws, accepted_draws, basis, sector_size, energy, exact_energy,
        delta_e, captured_mass, common_basis_size):
    return {
        "hamiltonian_id": int(hid),
        "J2": float(j2),
        "method": method,
        "repeat": int(repeat),
        "matching": matching,
        "D": int(krylov_dim),
        "M": int(samples_per_dim),
        "raw_draws": int(raw_draws),
        "accepted_draws": int(accepted_draws),
        "acceptance_rate": float(accepted_draws / raw_draws),
        "basis_size": int(len(basis)),
        "coverage": float(len(basis) / sector_size),
        "common_basis_size": int(common_basis_size),
        "energy": float(energy),
        "exact_energy": float(exact_energy),
        "delta_e": float(delta_e),
        "captured_ground_mass": float(captured_mass),
        "missing_ground_mass": float(max(1.0 - captured_mass, 0.0)),
    }


def run_pilot(args):
    (
        evaluation_args,
        canonical_results,
        data_dir,
        model_dirs,
    ) = _load_canonical_inputs(args)
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
    sector_mask = np.zeros(hilbert_size, dtype=bool)
    sector_mask[sector_states] = True
    sector_index = np.full(hilbert_size, -1, dtype=np.int64)
    sector_index[sector_states] = np.arange(sector_size)

    available_ids = _available_hamiltonian_ids(data_dir)
    hamiltonian_ids = available_ids if args.hamiltonian_ids is None \
        else sorted(set(args.hamiltonian_ids))
    missing_ids = sorted(set(hamiltonian_ids) - set(available_ids))
    if missing_ids:
        raise ValueError(f"Unknown held-out Hamiltonian ids: {missing_ids}")

    snapshots = int(evaluation_args["snapshots"])
    total_draws = args.krylov_dim * args.samples_per_dim
    rows = []
    common_support = {}

    for position, hid in enumerate(hamiltonian_ids, start=1):
        print(
            f"[{position}/{len(hamiltonian_ids)}] Hamiltonian id={hid}: loading streams",
            flush=True,
        )
        coupling = np.load(data_dir / f"coupling_matrix_id{hid}.npy")
        nonzero_upper = coupling[np.triu_indices(n_qubit, k=1)]
        nonzero_upper = nonzero_upper[~np.isclose(nonzero_upper, 0.0)]
        non_j1 = nonzero_upper[~np.isclose(nonzero_upper, 1.0)]
        if len(non_j1) == 0 or not np.allclose(non_j1, non_j1[0]):
            raise ValueError(f"Could not infer a single J2 value for id={hid}")
        j2 = float(non_j1[0])
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

        saved_exact = float(canonical_results["test"]["exact_energy"][hid])
        if not np.isclose(exact_energy, saved_exact, atol=1e-8):
            raise RuntimeError(
                f"Exact-energy mismatch for id={hid}: {exact_energy} vs {saved_exact}"
            )

        streams = {}
        for method in FIXED_METHODS:
            stream = _load_saved_stream(
                method,
                hid,
                args.krylov_dim,
                args.samples_per_dim,
                snapshots,
                data_dir,
                model_dirs,
                n_qubit,
                num_ones,
            )
            streams[(method, 0)] = stream

        for repeat in range(args.baseline_repeats):
            generated = _uniform_streams(
                hid,
                repeat,
                total_draws,
                hilbert_size,
                sector_states,
                sector_mask,
                args.seed,
            )
            for method, stream in generated.items():
                streams[(method, repeat)] = stream

        unique_streams = {
            key: _unique_in_arrival_order(stream)
            for key, stream in streams.items()
        }
        common_basis_size = min(len(basis) for basis in unique_streams.values())
        common_support[hid] = {
            "common_basis_size": int(common_basis_size),
            "raw_basis_sizes": {
                f"{method}:r{repeat}": int(len(basis))
                for (method, repeat), basis in unique_streams.items()
            },
        }
        print(
            f"[{position}/{len(hamiltonian_ids)}] id={hid}: "
            f"common unique basis K={common_basis_size}",
            flush=True,
        )

        for (method, repeat), stream in streams.items():
            raw_basis = unique_streams[(method, repeat)]
            cached_energy = None
            if method in FIXED_METHODS:
                cached_energy = _canonical_energy(
                    canonical_results,
                    method,
                    hid,
                    args.krylov_dim,
                    args.samples_per_dim,
                )
            raw_energy, raw_delta, raw_mass = _basis_metrics(
                raw_basis,
                sector_hamiltonian,
                sector_index,
                ground_probabilities,
                exact_energy,
                cached_energy=cached_energy,
            )
            rows.append(_record(
                hid,
                j2,
                method,
                repeat,
                "raw",
                args.krylov_dim,
                args.samples_per_dim,
                total_draws,
                len(stream),
                raw_basis,
                sector_size,
                raw_energy,
                exact_energy,
                raw_delta,
                raw_mass,
                common_basis_size,
            ))

            matched_basis = raw_basis[:common_basis_size]
            matched_energy, matched_delta, matched_mass = _basis_metrics(
                matched_basis,
                sector_hamiltonian,
                sector_index,
                ground_probabilities,
                exact_energy,
            )
            rows.append(_record(
                hid,
                j2,
                method,
                repeat,
                "unique",
                args.krylov_dim,
                args.samples_per_dim,
                total_draws,
                len(stream),
                matched_basis,
                sector_size,
                matched_energy,
                exact_energy,
                matched_delta,
                matched_mass,
                common_basis_size,
            ))
        print(
            f"[{position}/{len(hamiltonian_ids)}] id={hid}: energies complete",
            flush=True,
        )

    config = {
        "task": "sqkd_support_controlled_pilot",
        "source_evaluation": str(Path(args.evaluation_dir).resolve()),
        "source_evaluation_args": evaluation_args,
        "data_root": str(Path(args.data_root).resolve()),
        "n_qubit": n_qubit,
        "sector": f"Hamming weight {num_ones}",
        "sector_size": sector_size,
        "hamiltonian_ids": hamiltonian_ids,
        "krylov_dim": args.krylov_dim,
        "samples_per_dim": args.samples_per_dim,
        "total_raw_draws": total_draws,
        "baseline_repeats": args.baseline_repeats,
        "seed": args.seed,
        "matching": {
            "raw": "all D*M raw draws per method",
            "unique": "first-seen postselected states truncated to per-Hamiltonian common K",
        },
        "model_uncertainty_note": (
            "Saved device/Transformer/Mamba samples are one fixed realization; "
            "only uniform baselines are independently repeated."
        ),
    }
    return config, rows, common_support


def _per_hamiltonian_medians(rows):
    records = []
    for matching in ("raw", "unique"):
        for method in METHODS:
            method_rows = [
                row for row in rows
                if row["matching"] == matching and row["method"] == method
            ]
            ids = sorted({row["hamiltonian_id"] for row in method_rows})
            for hid in ids:
                subset = [
                    row for row in method_rows if row["hamiltonian_id"] == hid
                ]
                record = {
                    "matching": matching,
                    "method": method,
                    "hamiltonian_id": hid,
                    "J2": float(subset[0]["J2"]),
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
                records.append(record)
    return records


def aggregate_results(per_hamiltonian):
    summary = []
    for matching in ("raw", "unique"):
        for method in METHODS:
            subset = [
                row for row in per_hamiltonian
                if row["matching"] == matching and row["method"] == method
            ]
            if not subset:
                continue
            record = {
                "matching": matching,
                "method": method,
                "n_hamiltonians": len(subset),
                "baseline_repeats": int(max(row["n_repeats"] for row in subset)),
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


def _paired_bootstrap_ci(values, repeats, seed):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def paired_comparisons(per_hamiltonian, bootstrap_repeats, seed):
    lookup = {
        (row["matching"], row["method"], row["hamiltonian_id"]): row
        for row in per_hamiltonian
    }
    comparisons = []
    for matching in ("raw", "unique"):
        for model in FIXED_METHODS:
            for baseline in BASELINE_METHODS:
                ids = sorted({
                    hid for match, method, hid in lookup
                    if match == matching and method == model
                    and (matching, baseline, hid) in lookup
                })
                energy_diffs = np.asarray([
                    lookup[(matching, model, hid)]["delta_e"]
                    - lookup[(matching, baseline, hid)]["delta_e"]
                    for hid in ids
                ])
                mass_diffs = np.asarray([
                    lookup[(matching, model, hid)]["captured_ground_mass"]
                    - lookup[(matching, baseline, hid)]["captured_ground_mass"]
                    for hid in ids
                ])
                energy_ci = _paired_bootstrap_ci(
                    energy_diffs,
                    bootstrap_repeats,
                    np.random.SeedSequence([seed, len(comparisons), 0]),
                )
                mass_ci = _paired_bootstrap_ci(
                    mass_diffs,
                    bootstrap_repeats,
                    np.random.SeedSequence([seed, len(comparisons), 1]),
                )
                nonzero = energy_diffs[~np.isclose(energy_diffs, 0.0, atol=1e-14)]
                p_value = 1.0 if len(nonzero) == 0 else float(
                    wilcoxon(nonzero, alternative="two-sided").pvalue)
                comparisons.append({
                    "matching": matching,
                    "model": model,
                    "baseline": baseline,
                    "n_hamiltonians": len(ids),
                    "median_delta_e_difference": float(np.median(energy_diffs)),
                    "delta_e_difference_ci95_low": energy_ci[0],
                    "delta_e_difference_ci95_high": energy_ci[1],
                    "energy_win_rate": float(np.mean(energy_diffs < 0.0)),
                    "wilcoxon_p_two_sided": p_value,
                    "median_captured_mass_difference": float(np.median(mass_diffs)),
                    "captured_mass_difference_ci95_low": mass_ci[0],
                    "captured_mass_difference_ci95_high": mass_ci[1],
                    "captured_mass_win_rate": float(np.mean(mass_diffs > 0.0)),
                })
    return comparisons


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(per_hamiltonian, output_dir, config):
    output_dir = Path(output_dir)
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.reshape(-1)

    for axis, matching in zip(axes[:2], ("raw", "unique")):
        values = []
        for method in METHODS:
            method_values = [
                max(row["delta_e"], 1e-12)
                for row in per_hamiltonian
                if row["matching"] == matching and row["method"] == method
            ]
            values.append(method_values)
        box = axis.boxplot(values, patch_artist=True, showfliers=True)
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.6)
        axis.set_yscale("log")
        axis.set_xticks(
            range(1, len(METHODS) + 1),
            [DISPLAY_NAMES[method] for method in METHODS],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(r"Energy error $E_{\mathrm{proj}}-E_0$")
        axis.set_title(
            "(a) Raw-draw matched" if matching == "raw"
            else "(b) Unique-basis matched"
        )
        axis.grid(True, axis="y", which="both", alpha=0.25)

    raw_rows = [row for row in per_hamiltonian if row["matching"] == "raw"]
    for method in METHODS:
        subset = [row for row in raw_rows if row["method"] == method]
        axes[2].scatter(
            [100.0 * row["coverage"] for row in subset],
            [max(row["delta_e"], 1e-12) for row in subset],
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
            alpha=0.8,
            s=38,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Raw sampled Sz=0 sector coverage (%)")
    axes[2].set_ylabel(r"Energy error $E_{\mathrm{proj}}-E_0$")
    axes[2].set_title("(c) Coverage and energy")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(fontsize=9)

    lookup = {
        (row["matching"], row["method"], row["hamiltonian_id"]): row
        for row in per_hamiltonian
    }
    ids = sorted({
        row["hamiltonian_id"] for row in per_hamiltonian
        if row["matching"] == "unique"
    })
    for method in FIXED_METHODS:
        axes[3].plot(
            [lookup[("unique", method, hid)]["J2"] for hid in ids],
            [
                lookup[("unique", method, hid)]["delta_e"]
                - lookup[("unique", "uniform_sector", hid)]["delta_e"]
                for hid in ids
            ],
            linestyle="none",
            marker="o",
            color=COLORS[method],
            label=DISPLAY_NAMES[method],
            alpha=0.85,
        )
    axes[3].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[3].set_xlabel("Held-out $J_2$")
    axes[3].set_ylabel(
        r"Unique-matched $\Delta E_{\mathrm{method}}"
        r"-\Delta E_{\mathrm{uniform}\ S_z=0}$"
    )
    axes[3].set_title("(d) Advantage over uniform sector")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(fontsize=9)

    figure.suptitle(
        f"Support-controlled SKQD pilot: D={config['krylov_dim']}, "
        f"M={config['samples_per_dim']}, "
        f"{len(config['hamiltonian_ids'])} held-out Hamiltonians"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"support_controlled_pilot.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def save_results(args, config, rows, common_support):
    output_dir = Path(args.output_dir)
    per_hamiltonian = _per_hamiltonian_medians(rows)
    summary = aggregate_results(per_hamiltonian)
    comparisons = paired_comparisons(
        per_hamiltonian,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    _write_csv(output_dir / "raw_results.csv", rows)
    _write_csv(output_dir / "per_hamiltonian.csv", per_hamiltonian)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "paired_comparisons.csv", comparisons)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    with (output_dir / "common_support.json").open("w") as handle:
        json.dump(common_support, handle, indent=2)
    with (output_dir / "results.json").open("w") as handle:
        json.dump({
            "config": config,
            "summary": summary,
            "paired_comparisons": comparisons,
        }, handle, indent=2)
    _plot(per_hamiltonian, output_dir, config)
    print(f"Wrote support-controlled pilot to {output_dir}", flush=True)
    return summary, comparisons


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    config, rows, common_support = run_pilot(args)
    save_results(args, config, rows, common_support)


if __name__ == "__main__":
    main()
