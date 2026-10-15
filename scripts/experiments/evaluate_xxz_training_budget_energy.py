#!/usr/bin/env python3
"""Aggregate NLL and matched direct-SKQD energy results for the budget grid."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import linalg as sparse_linalg


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genksr.evaluation.sparsity import (  # noqa: E402
    fixed_hamming_weight_basis,
    lowest_eigenpairs,
    xxz_open_chain_sector_components,
)
from genksr.protocol import xxz_chain_pauli_l1_bound  # noqa: E402


REGIME_ORDER = {"sparse": 0, "middle": 1, "dense": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run-dir", type=Path,
        default=Path("artifacts/v3_xxz_training_budget_20260801"))
    parser.add_argument(
        "--evaluation-dir", type=Path,
        default=Path(
            "artifacts/v3_xxz_training_budget_20260801/"
            "formal_evaluation_seed0"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/xxz_sparsity_1d"))
    parser.add_argument("--eig-tol", type=float, default=1e-10)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_instances(split_dir):
    rows = read_csv(Path(split_dir) / "instances.csv")
    indexed = {int(row["split_id"]): row for row in rows}
    require(len(rows) == len(indexed), f"duplicate IDs in {split_dir}")
    return indexed


def samples_to_states(samples, n_qubit):
    samples = np.asarray(samples)
    require(samples.ndim == 2 and samples.shape[1] == n_qubit,
            f"unexpected sample shape {samples.shape} for N={n_qubit}")
    require(np.all((samples == 0) | (samples == 1)),
            "samples contain a non-binary outcome")
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    states = samples.astype(np.int64) @ powers
    accepted = samples.sum(axis=1) == n_qubit // 2
    states[~accepted] = -1
    return states


def load_basis(sample_paths, n_qubit, shots):
    streams = []
    raw_draws = 0
    for path in sample_paths:
        samples = np.load(path, allow_pickle=False, mmap_mode="r")[:shots]
        require(len(samples) == shots, f"insufficient samples in {path}")
        states = samples_to_states(samples, n_qubit)
        raw_draws += len(states)
        streams.append(states[states >= 0])
    accepted = np.concatenate(streams)
    basis = np.unique(accepted)
    return basis, raw_draws, len(accepted)


def lowest_energy(matrix, tol):
    dimension = int(matrix.shape[0])
    if dimension == 0:
        return np.nan
    if dimension <= 4:
        return float(np.linalg.eigvalsh(matrix.toarray())[0])
    v0 = np.linspace(1.0, 2.0, dimension, dtype=float)
    v0 /= np.linalg.norm(v0)
    return float(sparse_linalg.eigsh(
        matrix, k=1, which="SA", return_eigenvectors=False,
        v0=v0, tol=tol)[0])


def projected_metrics(basis, sector_hamiltonian, sector_index,
                      ground_probabilities, exact_energy, coefficient_l1,
                      eig_tol):
    if len(basis) == 0:
        return {
            "energy": np.nan,
            "delta_e": np.nan,
            "normalized_delta_e": np.nan,
            "captured_ground_mass": 0.0,
        }
    positions = sector_index[np.asarray(basis, dtype=np.int64)]
    require(np.all(positions >= 0), "basis contains a state outside Sz=0")
    projected = sector_hamiltonian[positions][:, positions].tocsr()
    energy = lowest_energy(projected, eig_tol)
    delta_e = float(energy - exact_energy)
    require(delta_e >= -1e-7 * coefficient_l1,
            f"variational-bound violation: {delta_e}")
    delta_e = max(delta_e, 0.0)
    return {
        "energy": float(energy),
        "delta_e": float(delta_e),
        "normalized_delta_e": float(delta_e / coefficient_l1),
        "captured_ground_mass": float(
            np.sum(ground_probabilities[positions])),
    }


def mean_std_sem(values):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0, "cannot summarize an empty value list")
    require(np.all(np.isfinite(values)),
            "cannot summarize non-finite evaluation values")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
    return mean, std, float(std / math.sqrt(len(values)))


def aggregate_nll(context_rows, jobs):
    by_job_hamiltonian = defaultdict(list)
    by_job_regime_hamiltonian = defaultdict(list)
    for row in context_rows:
        value = float(row["nll_nats_per_qubit"])
        job_name = row["job_name"]
        hid = int(row["hamiltonian_id"])
        regime = row["regime"]
        by_job_hamiltonian[(job_name, hid)].append(value)
        by_job_regime_hamiltonian[(job_name, regime, hid)].append(value)

    hamiltonian_means = {
        key: float(np.mean(values))
        for key, values in by_job_hamiltonian.items()
    }
    summary = {}
    regime_summary = {}
    for job in jobs:
        job_name = job["name"]
        h_values = [
            value
            for (name, _hid), value in hamiltonian_means.items()
            if name == job_name
        ]
        require(len(h_values) == 30,
                f"{job_name}: expected 30 held-out Hamiltonian NLL values")
        mean, std, sem = mean_std_sem(h_values)
        summary[job_name] = {
            "nll_nats_per_qubit_mean": mean,
            "nll_hamiltonian_std": std,
            "nll_hamiltonian_sem": sem,
            "nll_hamiltonian_min": float(np.min(h_values)),
            "nll_hamiltonian_max": float(np.max(h_values)),
        }
        for regime in REGIME_ORDER:
            values = [
                float(np.mean(items))
                for (name, group, _hid), items
                in by_job_regime_hamiltonian.items()
                if name == job_name and group == regime
            ]
            require(len(values) == 10,
                    f"{job_name}/{regime}: expected 10 held-out Hamiltonians")
            rmean, rstd, rsem = mean_std_sem(values)
            regime_summary[(job_name, regime)] = {
                "nll_nats_per_qubit_mean": rmean,
                "nll_hamiltonian_std": rstd,
                "nll_hamiltonian_sem": rsem,
            }
    return summary, regime_summary, hamiltonian_means


def aggregate_energy(energy_rows, jobs):
    grouped = defaultdict(list)
    regime_grouped = defaultdict(list)
    for row in energy_rows:
        grouped[row["job_name"]].append(row)
        regime_grouped[(row["job_name"], row["regime"])].append(row)

    summary = {}
    regime_summary = {}
    metrics = (
        "model_normalized_delta_e",
        "direct_normalized_delta_e",
        "signed_energy_difference",
        "absolute_energy_difference",
        "signed_normalized_energy_difference",
        "absolute_normalized_energy_difference",
        "model_acceptance_rate",
        "model_basis_size",
        "direct_basis_size",
        "model_captured_ground_mass",
        "direct_captured_ground_mass",
    )
    for job in jobs:
        job_name = job["name"]
        rows = grouped[job_name]
        require(len(rows) == 30,
                f"{job_name}: expected 30 held-out energy rows")
        output = {}
        for metric in metrics:
            mean, std, sem = mean_std_sem(
                [float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_hamiltonian_std"] = std
            output[f"{metric}_hamiltonian_sem"] = sem
        summary[job_name] = output

        for regime in REGIME_ORDER:
            subset = regime_grouped[(job_name, regime)]
            require(len(subset) == 10,
                    f"{job_name}/{regime}: expected 10 energy rows")
            routput = {}
            for metric in metrics:
                mean, std, sem = mean_std_sem(
                    [float(row[metric]) for row in subset])
                routput[f"{metric}_mean"] = mean
                routput[f"{metric}_hamiltonian_std"] = std
                routput[f"{metric}_hamiltonian_sem"] = sem
            regime_summary[(job_name, regime)] = routput
    return summary, regime_summary


def format_float(value):
    return f"{float(value):.6g}"


def write_results_markdown(path, cell_rows, minimum_rows, protocol):
    lines = [
        "# XXZ training-budget study",
        "",
        "This is single-training-seed finite-size evidence. Error bars in the "
        "CSV summaries describe variation across the 30 held-out Hamiltonians; "
        "they are not training-seed uncertainty.",
        "The XXZ data are noiseless simulator samples, so the reported fit "
        "count is a QPU-shot-equivalent budget rather than measured QPU use.",
        "",
        "The primary criterion was registered before held-out evaluation: NLL "
        "degradation relative to the same-size full-budget anchor no larger "
        f"than {protocol['primary_accuracy_criterion']['nll_gap_to_same_N_full_budget_anchor_nats_per_qubit_max']} "
        "nat/qubit and mean absolute normalized energy difference from matched "
        "direct finite-sampling SKQD no larger than "
        f"{protocol['primary_accuracy_criterion']['mean_absolute_normalized_energy_difference_vs_direct_max']}.",
        "",
        "## Smallest passing tested budgets",
        "",
        "| N | Smallest tested fit budget | N_H | M_train | NLL gap (H-SEM) | Mean absolute energy gap (H-SEM) | Signed energy gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if minimum_rows:
        for row in minimum_rows:
            lines.append(
                f"| {row['N']} | {row['fit_samples']} | "
                f"{row['num_train_hamiltonians']} | "
                f"{row['train_shots_per_condition']} | "
                f"{format_float(row['nll_gap_to_full_budget_anchor'])} "
                f"({format_float(row['nll_gap_hamiltonian_sem'])}) | "
                f"{format_float(row['absolute_normalized_energy_difference_mean'])} "
                f"({format_float(row['absolute_normalized_energy_difference_hamiltonian_sem'])}) | "
                f"{format_float(row['signed_normalized_energy_difference_mean'])} |")
    else:
        lines.append(
            "| — | No grid cell met both registered criteria | — | — | — | — | — |")
    lines.extend([
        "",
        "All 48 cells are recorded in `cell_summary.csv`. These three system "
        "sizes do not support an asymptotic exponent or a polynomial-versus-"
        "exponential conclusion. If the lowest tested cell passes, its value is "
        "an upper bound on the grid-resolved threshold, not an identified "
        "minimum. Cells with equal products but different allocations between "
        "training-Hamiltonian coverage and shots per condition remain separate; "
        "the product alone need not determine accuracy.",
        "",
        "## Full-budget anchors",
        "",
        "| N | Held-out NLL (nat/qubit) | GenKSR normalized energy error | Direct normalized energy error | Mean absolute paired gap |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in cell_rows:
        if (int(row["num_train_hamiltonians"]) == 45
                and int(row["train_shots_per_condition"]) == 1000):
            lines.append(
                f"| {row['N']} | {format_float(row['nll_nats_per_qubit_mean'])} | "
                f"{format_float(row['model_normalized_delta_e_mean'])} | "
                f"{format_float(row['direct_normalized_delta_e_mean'])} | "
                f"{format_float(row['absolute_normalized_energy_difference_mean'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    require(args.eig_tol > 0, "eig-tol must be positive")
    jobs = json.loads((args.training_run_dir / "jobs.json").read_text())
    jobs = sorted(
        jobs, key=lambda job: (
            job["n_qubit"], job["fit_samples"],
            job["num_hamiltonians"], job["train_shots"], job["name"]))
    require(len(jobs) == 48 and {job["seed"] for job in jobs} == {0},
            "expected the formal 48-cell, single-seed grid")
    protocol_path = args.evaluation_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    eval_shots = int(protocol["downstream_sample_shots_per_condition"])
    eval_sample_seed = int(protocol["downstream_sample_seed"])
    depth = int(protocol["D_eval"])
    require(depth == 15, "formal evaluator expects D_eval=15")

    context_rows = []
    for job in jobs:
        model_dir = args.evaluation_dir / "models" / job["name"]
        model_protocol = json.loads((model_dir / "protocol.json").read_text())
        require(model_protocol.get("status") == "complete",
                f"incomplete model evaluation: {job['name']}")
        require(int(model_protocol["sample_shots_per_condition"]) == eval_shots,
                f"sample budget mismatch: {job['name']}")
        require(int(model_protocol["sample_seed"]) == eval_sample_seed,
                f"sample seed mismatch: {job['name']}")
        require(int(model_protocol["N"]) == int(job["n_qubit"]),
                f"system-size mismatch: {job['name']}")
        require(
            int(model_protocol["num_train_hamiltonians"])
            == int(job["num_hamiltonians"]),
            f"training-Hamiltonian mismatch: {job['name']}")
        require(
            int(model_protocol["train_shots_per_condition"])
            == int(job["train_shots"]),
            f"training-shot mismatch: {job['name']}")
        require(int(model_protocol["fit_samples"]) == int(job["fit_samples"]),
                f"fit-budget mismatch: {job['name']}")
        checkpoint_path = Path(model_protocol["checkpoint_path"])
        require(sha256_file(checkpoint_path) == model_protocol["checkpoint_sha256"],
                f"checkpoint changed after sampling: {job['name']}")
        rows = read_csv(model_dir / "heldout_nll_context.csv")
        require(len(rows) == 30 * depth,
                f"likelihood context count mismatch: {job['name']}")
        require({row["job_name"] for row in rows} == {job["name"]},
                f"likelihood job-name mismatch: {job['name']}")
        context_rows.extend(rows)
    write_csv_atomic(args.evaluation_dir / "heldout_nll_context.csv", context_rows)
    nll_summary, nll_regime_summary, nll_hamiltonian_means = aggregate_nll(
        context_rows, jobs)

    started = time.perf_counter()
    energy_rows = []
    ground_rows = []
    jobs_by_size = defaultdict(list)
    for job in jobs:
        jobs_by_size[int(job["n_qubit"])].append(job)

    for n_qubit in sorted(jobs_by_size):
        print(f"N={n_qubit}: constructing fixed-magnetization sector", flush=True)
        sector_basis = fixed_hamming_weight_basis(n_qubit, n_qubit // 2)
        sector_index = np.full(1 << n_qubit, -1, dtype=np.int32)
        sector_index[sector_basis] = np.arange(len(sector_basis), dtype=np.int32)
        h_xy, h_zz = xxz_open_chain_sector_components(n_qubit, sector_basis)
        test_dir = args.data_root / f"{n_qubit}qubits" / "test"
        instances = load_instances(test_dir)
        require(len(instances) == 30,
                f"N={n_qubit}: expected 30 held-out Hamiltonians")

        for position, hamiltonian_id in enumerate(sorted(instances), start=1):
            row = instances[hamiltonian_id]
            jxy = float(row["J_xy"])
            jzz = float(row["J_zz"])
            coefficient_l1 = xxz_chain_pauli_l1_bound(
                n_qubit, jxy, jzz)
            sector_hamiltonian = (jxy * h_xy + jzz * h_zz).tocsr()
            eigenvalues, eigenvectors = lowest_eigenpairs(
                sector_hamiltonian, k=2, tol=args.eig_tol)
            exact_energy = float(eigenvalues[0])
            probabilities = np.abs(eigenvectors[:, 0]) ** 2
            probabilities /= probabilities.sum()
            ground_rows.append({
                "N": int(n_qubit),
                "hamiltonian_id": int(hamiltonian_id),
                "paired_key": row["paired_key"],
                "regime": row["regime"],
                "J_xy": jxy,
                "J_zz": jzz,
                "coefficient_l1": float(coefficient_l1),
                "exact_energy": exact_energy,
                "gap": float(eigenvalues[1] - eigenvalues[0]),
                "sector_size": int(len(sector_basis)),
            })

            direct_paths = [
                test_dir / f"data_D{dimension}_id{hamiltonian_id}.npy"
                for dimension in range(depth)
            ]
            direct_basis, direct_raw, direct_accepted = load_basis(
                direct_paths, n_qubit, eval_shots)
            require(direct_accepted == direct_raw,
                    "direct reference left the fixed-magnetization sector")
            direct = projected_metrics(
                direct_basis, sector_hamiltonian, sector_index, probabilities,
                exact_energy, coefficient_l1, args.eig_tol)

            for job in jobs_by_size[n_qubit]:
                model_sample_dir = (
                    args.evaluation_dir / "models" / job["name"] /
                    "samples" / "test" / "model")
                model_paths = [
                    model_sample_dir /
                    f"model_samples_D{dimension}_{eval_shots}_id{hamiltonian_id}.npy"
                    for dimension in range(depth)
                ]
                model_basis, model_raw, model_accepted = load_basis(
                    model_paths, n_qubit, eval_shots)
                model = projected_metrics(
                    model_basis, sector_hamiltonian, sector_index,
                    probabilities, exact_energy, coefficient_l1, args.eig_tol)
                signed_difference = (
                    float(model["normalized_delta_e"])
                    - float(direct["normalized_delta_e"]))
                raw_difference = float(model["energy"] - direct["energy"])
                energy_rows.append({
                    "job_name": job["name"],
                    "N": int(n_qubit),
                    "num_train_hamiltonians": int(job["num_hamiltonians"]),
                    "train_shots_per_condition": int(job["train_shots"]),
                    "fit_samples": int(job["fit_samples"]),
                    "train_seed": int(job["seed"]),
                    "hamiltonian_id": int(hamiltonian_id),
                    "paired_key": row["paired_key"],
                    "regime": row["regime"],
                    "J_xy": jxy,
                    "D_eval": int(depth),
                    "M_eval": int(eval_shots),
                    "raw_draws": int(model_raw),
                    "model_accepted_draws": int(model_accepted),
                    "direct_accepted_draws": int(direct_accepted),
                    "model_acceptance_rate": float(model_accepted / model_raw),
                    "direct_acceptance_rate": float(
                        direct_accepted / direct_raw),
                    "model_basis_size": int(len(model_basis)),
                    "direct_basis_size": int(len(direct_basis)),
                    "exact_energy": exact_energy,
                    "model_energy": float(model["energy"]),
                    "direct_energy": float(direct["energy"]),
                    "model_normalized_delta_e": float(
                        model["normalized_delta_e"]),
                    "direct_normalized_delta_e": float(
                        direct["normalized_delta_e"]),
                    "signed_energy_difference": raw_difference,
                    "absolute_energy_difference": abs(raw_difference),
                    "signed_normalized_energy_difference": signed_difference,
                    "absolute_normalized_energy_difference": abs(
                        signed_difference),
                    "model_captured_ground_mass": float(
                        model["captured_ground_mass"]),
                    "direct_captured_ground_mass": float(
                        direct["captured_ground_mass"]),
                })
            print(
                f"N={n_qubit} held-out Hamiltonian "
                f"{position}/30 complete", flush=True)
        write_csv_atomic(args.evaluation_dir / "energy_context.csv", energy_rows)
        write_csv_atomic(args.evaluation_dir / "ground_metrics.csv", ground_rows)

    energy_summary, energy_regime_summary = aggregate_energy(energy_rows, jobs)
    cell_rows = []
    for job in jobs:
        job_name = job["name"]
        cell_rows.append({
            "job_name": job_name,
            "N": int(job["n_qubit"]),
            "num_train_hamiltonians": int(job["num_hamiltonians"]),
            "train_shots_per_condition": int(job["train_shots"]),
            "D_train": int(job["krylov_conditions"]),
            "fit_samples": int(job["fit_samples"]),
            "train_seed": int(job["seed"]),
            "num_heldout_hamiltonians": 30,
            "D_eval": int(depth),
            "M_eval": int(eval_shots),
            **nll_summary[job_name],
            **energy_summary[job_name],
        })

    anchor_nll = {}
    anchor_jobs = {}
    for row in cell_rows:
        if (int(row["num_train_hamiltonians"]) == 45
                and int(row["train_shots_per_condition"]) == 1000):
            anchor_nll[int(row["N"])] = float(
                row["nll_nats_per_qubit_mean"])
            anchor_jobs[int(row["N"])] = row["job_name"]
    require(set(anchor_nll) == {12, 16, 20},
            "missing a same-size full-budget NLL anchor")
    criterion = protocol["primary_accuracy_criterion"]
    nll_limit = float(
        criterion[
            "nll_gap_to_same_N_full_budget_anchor_nats_per_qubit_max"])
    energy_limit = float(
        criterion[
            "mean_absolute_normalized_energy_difference_vs_direct_max"])
    for row in cell_rows:
        anchor_job = anchor_jobs[int(row["N"])]
        heldout_ids = sorted(
            hid for (name, hid) in nll_hamiltonian_means
            if name == row["job_name"])
        require(len(heldout_ids) == 30,
                f"missing paired NLL Hamiltonians: {row['job_name']}")
        paired_gaps = [
            nll_hamiltonian_means[(row["job_name"], hid)]
            - nll_hamiltonian_means[(anchor_job, hid)]
            for hid in heldout_ids
        ]
        gap, gap_std, gap_sem = mean_std_sem(paired_gaps)
        require(np.isclose(
            gap,
            float(row["nll_nats_per_qubit_mean"])
            - anchor_nll[int(row["N"])],
            rtol=0.0, atol=1e-12),
            f"paired NLL gap aggregation mismatch: {row['job_name']}")
        row["nll_gap_to_full_budget_anchor"] = gap
        row["nll_gap_hamiltonian_std"] = gap_std
        row["nll_gap_hamiltonian_sem"] = gap_sem
        row["passes_primary_nll_criterion"] = bool(gap <= nll_limit)
        row["passes_primary_energy_criterion"] = bool(
            float(row["absolute_normalized_energy_difference_mean"])
            <= energy_limit)
        row["passes_primary_joint_criterion"] = bool(
            row["passes_primary_nll_criterion"]
            and row["passes_primary_energy_criterion"])

    regime_rows = []
    for job in jobs:
        for regime in sorted(REGIME_ORDER, key=REGIME_ORDER.get):
            regime_rows.append({
                "job_name": job["name"],
                "N": int(job["n_qubit"]),
                "num_train_hamiltonians": int(job["num_hamiltonians"]),
                "train_shots_per_condition": int(job["train_shots"]),
                "fit_samples": int(job["fit_samples"]),
                "train_seed": int(job["seed"]),
                "regime": regime,
                "num_heldout_hamiltonians": 10,
                **nll_regime_summary[(job["name"], regime)],
                **energy_regime_summary[(job["name"], regime)],
            })

    minimum_rows = []
    for n_qubit in (12, 16, 20):
        passing = [
            row for row in cell_rows
            if int(row["N"]) == n_qubit
            and row["passes_primary_joint_criterion"]
        ]
        if not passing:
            continue
        minimum_budget = min(int(row["fit_samples"]) for row in passing)
        for row in passing:
            if int(row["fit_samples"]) == minimum_budget:
                minimum_rows.append(dict(row))

    write_csv_atomic(args.evaluation_dir / "energy_context.csv", energy_rows)
    write_csv_atomic(args.evaluation_dir / "ground_metrics.csv", ground_rows)
    write_csv_atomic(args.evaluation_dir / "cell_summary.csv", cell_rows)
    write_csv_atomic(args.evaluation_dir / "cell_regime_summary.csv", regime_rows)
    write_csv_atomic(args.evaluation_dir / "minimum_budget.csv", minimum_rows)
    write_results_markdown(
        args.evaluation_dir / "RESULTS.md", cell_rows, minimum_rows, protocol)

    aggregate_protocol = {
        "status": "complete",
        "single_training_seed": True,
        "training_seed_repeats": 1,
        "heldout_hamiltonians_per_size": 30,
        "downstream_sampling_repeats": 1,
        "D_eval": depth,
        "M_eval": eval_shots,
        "direct_baseline": (
            "fixed noiseless-state sample streams, with the same D_eval and "
            "M_eval as every GenKSR cell"),
        "energy_normalization": (
            "energy errors and paired differences divided by the Hamiltonian "
            "Pauli-coefficient L1 bound for each held-out instance"),
        "primary_accuracy_criterion": criterion,
        "test_manifest_sha256": {
            str(n_qubit): sha256_file(
                args.data_root / f"{n_qubit}qubits" / "test" /
                "manifest.json")
            for n_qubit in (12, 16, 20)
        },
        "row_counts": {
            "heldout_nll_context": len(context_rows),
            "energy_context": len(energy_rows),
            "ground_metrics": len(ground_rows),
            "cell_summary": len(cell_rows),
            "cell_regime_summary": len(regime_rows),
            "minimum_budget": len(minimum_rows),
        },
        "energy_evaluator_sha256": sha256_file(Path(__file__)),
        "wall_seconds": float(time.perf_counter() - started),
        "interpretation_limit": (
            "Held-out-Hamiltonian variation is not training-seed uncertainty. "
            "Three sizes do not establish asymptotic scaling."),
    }
    write_json_atomic(
        args.evaluation_dir / "aggregate_protocol.json", aggregate_protocol)
    print(
        f"Complete: {len(cell_rows)} cells, {len(energy_rows)} paired energy "
        f"rows, {aggregate_protocol['wall_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
