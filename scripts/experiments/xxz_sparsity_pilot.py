#!/usr/bin/env python3
"""Run the exact 9-cell pilot for the 1D XXZ sparsity experiment.

This stage deliberately does not train GenKSR.  It checks whether the proposed
Jxy regimes create distinguishable computational-basis ground-state sparsity,
whether a nearly degenerate Neel doublet makes a single eigenvector unstable,
and whether exact-state SKQD responds to the two planned raw budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as sparse_linalg  # noqa: E402

from genksr.evaluation.skqd import (  # noqa: E402
    projected_xxz_hamiltonian_from_couplings,
)
from genksr.evaluation.sparsity import (  # noqa: E402
    basis_state_probability,
    exact_krylov_states as exact_sector_krylov_states,
    fixed_hamming_weight_basis,
    lowest_eigenpairs,
    sparsity_metrics,
    xxz_open_chain_sector_components,
)
from genksr.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    neel_reference_bitstring,
    xxz_chain_pauli_l1_bound,
    xxz_chain_timestep,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact 1D XXZ sparsity x system-size pilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument(
        "--jxy-values", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    parser.add_argument(
        "--regime-labels", nargs="+", default=["sparse", "middle", "dense"])
    parser.add_argument("--jzz", type=float, default=1.0)
    parser.add_argument(
        "--boundary-pin", type=float, default=0.0,
        help="Non-negative h_pin in the fixed term -h_pin*Z_0",
    )
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--budgets", type=int, nargs="+", default=[3000, 15000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--eig-tol", type=float, default=1e-10)
    parser.add_argument("--minimum-separation", type=float, default=5.0)
    parser.add_argument("--gap-warning-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--output-dir", default="artifacts/v3_xxz_sparsity_scaling/pilot")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing named output files in a non-empty directory",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not args.sizes or any(n < 2 or n % 2 for n in args.sizes):
        raise ValueError("sizes must be non-empty even integers >= 2")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("sizes must be unique")
    if not args.jxy_values or any(
            not np.isfinite(value) or value <= 0 for value in args.jxy_values):
        raise ValueError("jxy-values must be positive and finite")
    if len(args.jxy_values) != len(args.regime_labels):
        raise ValueError("regime-labels must have the same length as jxy-values")
    if len(set(args.regime_labels)) != len(args.regime_labels):
        raise ValueError("regime-labels must be unique")
    if not np.isfinite(args.jzz) or args.jzz <= 0:
        raise ValueError("jzz must be positive and finite")
    if not np.isfinite(args.boundary_pin) or args.boundary_pin < 0:
        raise ValueError("boundary-pin must be non-negative and finite")
    if args.depth <= 0 or args.repeats <= 0:
        raise ValueError("depth and repeats must be positive")
    if not args.budgets or any(
            budget <= 0 or budget % args.depth for budget in args.budgets):
        raise ValueError("every budget must be positive and divisible by depth")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets must be unique")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.eig_tol <= 0 or args.minimum_separation <= 1:
        raise ValueError("eig-tol must be positive and minimum-separation > 1")
    if args.gap_warning_threshold <= 0:
        raise ValueError("gap-warning-threshold must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)


def chain_coupling_matrix(n_qubit: int, weight: float):
    """Uniform nearest-neighbour open-chain coupling matrix."""
    matrix = np.zeros((n_qubit, n_qubit), dtype=float)
    sites = np.arange(n_qubit - 1)
    matrix[sites, sites + 1] = float(weight)
    matrix[sites + 1, sites] = float(weight)
    return matrix


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _lowest_energy(matrix, tol):
    dimension = int(matrix.shape[0])
    if dimension == 0:
        return np.nan
    if dimension <= 4:
        return float(np.linalg.eigvalsh(matrix.toarray())[0])
    v0 = np.linspace(1.0, 2.0, dimension, dtype=float)
    v0 /= np.linalg.norm(v0)
    return float(sparse_linalg.eigsh(
        matrix, k=1, which="SA", return_eigenvectors=False,
        v0=v0, tol=tol,
    )[0])


def exact_krylov_states(hamiltonian, initial_state, depth, dt):
    """Return exact states at times ``d*dt`` for ``d=0,...,depth-1``."""
    return exact_sector_krylov_states(
        hamiltonian, initial_state=initial_state, depth=depth, dt=dt)


def sample_nested_exact_streams(
        probabilities_by_dimension, max_samples_per_dimension,
        seed_coordinates):
    """Sample one stream per dimension; smaller budgets use stream prefixes."""
    streams = []
    seed_sequence = np.random.SeedSequence([int(v) for v in seed_coordinates])
    child_sequences = seed_sequence.spawn(len(probabilities_by_dimension))
    for probabilities, child_seed in zip(
            probabilities_by_dimension, child_sequences):
        probs = np.asarray(probabilities, dtype=float)
        probs = probs / probs.sum()
        cumulative = np.cumsum(probs)
        cumulative[-1] = 1.0
        rng = np.random.default_rng(child_seed)
        draws = np.searchsorted(
            cumulative,
            rng.random(max_samples_per_dimension),
            side="right",
        ).astype(np.int64)
        streams.append(draws)
    return streams


def _cell_ground_metrics(
        n_qubit, jxy, jzz, regime, sector_basis, hamiltonian,
        eig_tol, neel_integer, construction_seconds, z_fields):
    eig_start = time.perf_counter()
    eigenvalues, eigenvectors = lowest_eigenpairs(
        hamiltonian, k=2, tol=eig_tol)
    eig_seconds = time.perf_counter() - eig_start

    ground_probabilities = np.abs(eigenvectors[:, 0]) ** 2
    ground_probabilities /= ground_probabilities.sum()
    single = sparsity_metrics(
        ground_probabilities, sector_size=len(sector_basis))

    # The equally weighted density matrix of the two lowest states is invariant
    # to rotations within a numerically unresolved doublet.
    doublet_probabilities = np.sum(np.abs(eigenvectors[:, :2]) ** 2, axis=1)
    doublet_probabilities /= doublet_probabilities.sum()
    doublet = sparsity_metrics(
        doublet_probabilities, sector_size=len(sector_basis))

    coefficient_l1 = xxz_chain_pauli_l1_bound(
        n_qubit, jxy, jzz, z_fields=z_fields)
    gap = float(eigenvalues[1] - eigenvalues[0])
    row = {
        "N": int(n_qubit),
        "regime": regime,
        "J_xy": float(jxy),
        "J_zz": float(jzz),
        "boundary_pin": float(abs(z_fields[0])),
        "pin_site": 0,
        "pin_z_coefficient": float(z_fields[0]),
        "sector_size": int(len(sector_basis)),
        "coefficient_l1": float(coefficient_l1),
        "dt": float(xxz_chain_timestep(
            n_qubit, jxy, jzz, z_fields=z_fields)),
        "E0": float(eigenvalues[0]),
        "E1": float(eigenvalues[1]),
        "gap": gap,
        "normalized_gap": float(gap / coefficient_l1),
        "L95": int(single["l95"]),
        "f95": float(single["f95"]),
        "L99": int(single["l99"]),
        "f99": float(single["f99"]),
        "effective_support": float(single["effective_support"]),
        "effective_support_fraction": float(
            single["effective_support_fraction"]),
        "doublet_L95": int(doublet["l95"]),
        "doublet_f95": float(doublet["f95"]),
        "doublet_L99": int(doublet["l99"]),
        "doublet_f99": float(doublet["f99"]),
        "neel_probability": basis_state_probability(
            ground_probabilities, sector_basis, neel_integer),
        "neel_doublet_average_probability": basis_state_probability(
            doublet_probabilities, sector_basis, neel_integer),
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "construction_seconds": float(construction_seconds),
        "eigensolve_seconds": float(eig_seconds),
    }
    return row, ground_probabilities, eigenvectors[:, 0]


def _sampling_rows_for_cell(
        args, cell_index, ground_row, ground_probabilities,
        hamiltonian, krylov_states):
    probabilities_by_dimension = np.abs(krylov_states) ** 2
    norm_errors = np.abs(probabilities_by_dimension.sum(axis=1) - 1.0)
    if float(np.max(norm_errors)) > 1e-8:
        raise RuntimeError(
            f"Exact evolution norm drifted by {float(np.max(norm_errors))}")
    probabilities_by_dimension /= probabilities_by_dimension.sum(
        axis=1, keepdims=True)

    max_samples = max(args.budgets) // args.depth
    coefficient_l1 = float(ground_row["coefficient_l1"])
    exact_energy = float(ground_row["E0"])
    rows = []
    for repeat in range(args.repeats):
        streams = sample_nested_exact_streams(
            probabilities_by_dimension,
            max_samples_per_dimension=max_samples,
            seed_coordinates=[args.seed, cell_index, repeat],
        )
        for budget in sorted(args.budgets):
            samples_per_dimension = budget // args.depth
            sampled_rows = np.concatenate([
                stream[:samples_per_dimension] for stream in streams])
            basis_rows = np.unique(sampled_rows)
            projected = hamiltonian[basis_rows][:, basis_rows].tocsr()
            solve_start = time.perf_counter()
            energy = _lowest_energy(projected, tol=args.eig_tol)
            solve_seconds = time.perf_counter() - solve_start
            delta_e = float(energy - exact_energy)
            if delta_e < -1e-7 * coefficient_l1:
                raise RuntimeError(
                    f"Projected energy violated variational bound: {delta_e}")
            rows.append({
                "N": int(ground_row["N"]),
                "regime": ground_row["regime"],
                "J_xy": float(ground_row["J_xy"]),
                "J_zz": float(ground_row["J_zz"]),
                "repeat": int(repeat),
                "budget": int(budget),
                "D": int(args.depth),
                "M": int(samples_per_dimension),
                "raw_draws": int(budget),
                "accepted_draws": int(budget),
                "acceptance_rate": 1.0,
                "basis_size": int(len(basis_rows)),
                "sector_size": int(ground_row["sector_size"]),
                "coverage": float(
                    len(basis_rows) / int(ground_row["sector_size"])),
                "captured_ground_mass": float(
                    np.sum(ground_probabilities[basis_rows])),
                "energy": float(energy),
                "exact_energy": exact_energy,
                "delta_e": max(delta_e, 0.0),
                "normalized_delta_e": max(delta_e, 0.0) / coefficient_l1,
                "projected_solve_seconds": float(solve_seconds),
            })
    return rows, float(np.max(norm_errors))


def run_cell(args, n_qubit, jxy, regime, cell_index):
    print(
        f"[pilot {cell_index + 1}/{len(args.sizes) * len(args.jxy_values)}] "
        f"N={n_qubit}, regime={regime}, Jxy={jxy:g}",
        flush=True,
    )
    sector_basis = fixed_hamming_weight_basis(n_qubit, n_qubit // 2)
    z_fields = np.zeros(n_qubit, dtype=float)
    z_fields[0] = -float(args.boundary_pin)

    construction_start = time.perf_counter()
    h_xy, h_zz = xxz_open_chain_sector_components(n_qubit, sector_basis)
    hamiltonian = float(jxy) * h_xy + float(args.jzz) * h_zz
    if np.any(z_fields):
        field_diagonal = np.zeros(len(sector_basis), dtype=float)
        for qubit, field in enumerate(z_fields):
            field_diagonal += field * (
                1.0 - 2.0 * ((sector_basis >> qubit) & 1))
        hamiltonian = hamiltonian + sparse.diags(field_diagonal, format="csr")
    hamiltonian = hamiltonian.tocsr()
    construction_seconds = time.perf_counter() - construction_start
    if not sparse.isspmatrix_csr(hamiltonian):
        raise TypeError("Expected a CSR sector Hamiltonian")
    hermiticity_error = sparse_linalg.norm(hamiltonian - hamiltonian.T)
    if hermiticity_error > 1e-10:
        raise RuntimeError(f"Hamiltonian is not symmetric: {hermiticity_error}")

    neel_string = neel_reference_bitstring(n_qubit)
    neel_integer = int(neel_string, 2)
    neel_position = int(np.searchsorted(sector_basis, neel_integer))
    if (neel_position >= len(sector_basis)
            or int(sector_basis[neel_position]) != neel_integer):
        raise RuntimeError("Néel reference state is outside the half-filled sector")

    ground_row, ground_probabilities, _ = _cell_ground_metrics(
        n_qubit=n_qubit,
        jxy=jxy,
        jzz=args.jzz,
        regime=regime,
        sector_basis=sector_basis,
        hamiltonian=hamiltonian,
        eig_tol=args.eig_tol,
        neel_integer=neel_integer,
        construction_seconds=construction_seconds,
        z_fields=z_fields,
    )

    initial_state = np.zeros(len(sector_basis), dtype=complex)
    initial_state[neel_position] = 1.0
    evolution_start = time.perf_counter()
    krylov_states = exact_krylov_states(
        hamiltonian,
        initial_state=initial_state,
        depth=args.depth,
        dt=float(ground_row["dt"]),
    )
    ground_row["evolution_seconds"] = float(
        time.perf_counter() - evolution_start)

    sampling_rows, norm_error = _sampling_rows_for_cell(
        args=args,
        cell_index=cell_index,
        ground_row=ground_row,
        ground_probabilities=ground_probabilities,
        hamiltonian=hamiltonian,
        krylov_states=krylov_states,
    )
    ground_row["maximum_evolution_norm_error"] = norm_error
    print(
        f"  sector={len(sector_basis):,}, f95={ground_row['f95']:.3e}, "
        f"doublet_f95={ground_row['doublet_f95']:.3e}, "
        f"normalized_gap={ground_row['normalized_gap']:.3e}",
        flush=True,
    )
    return ground_row, sampling_rows


def summarize_sampling(rows):
    grouped = {}
    for row in rows:
        key = (row["N"], row["regime"], row["J_xy"], row["budget"])
        grouped.setdefault(key, []).append(row)

    summary = []
    metric_names = (
        "basis_size", "coverage", "captured_ground_mass", "delta_e",
        "normalized_delta_e", "projected_solve_seconds",
    )
    for key in sorted(grouped):
        group = grouped[key]
        first = group[0]
        output = {
            "N": int(first["N"]),
            "regime": first["regime"],
            "J_xy": float(first["J_xy"]),
            "J_zz": float(first["J_zz"]),
            "budget": int(first["budget"]),
            "D": int(first["D"]),
            "M": int(first["M"]),
            "repeats": int(len(group)),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_std"] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0)
        summary.append(output)
    return summary


def _make_sparsity_figure(ground_rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    regimes = list(dict.fromkeys(row["regime"] for row in ground_rows))
    for regime in regimes:
        selected = sorted(
            (row for row in ground_rows if row["regime"] == regime),
            key=lambda row: row["N"],
        )
        sizes = [row["N"] for row in selected]
        line, = axes[0].plot(
            sizes, [row["f95"] for row in selected], "o-", label=regime)
        axes[0].plot(
            sizes, [row["doublet_f95"] for row in selected], "x--",
            color=line.get_color(), alpha=0.7)
        axes[1].plot(
            sizes, [row["normalized_gap"] for row in selected],
            "o-", label=regime)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel(r"$f_{95}=L_{95}/\binom{N}{N/2}$")
    axes[0].set_title("Ground-state sparsity (solid) / doublet (dashed)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Normalized E1-E0 gap")
    axes[1].set_title("Near-degeneracy diagnostic")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"pilot_sparsity.{extension}", dpi=180)
    plt.close(fig)


def _make_skqd_figure(summary_rows, output_dir):
    budgets = sorted({row["budget"] for row in summary_rows})
    fig, axes = plt.subplots(
        1, len(budgets), figsize=(6 * len(budgets), 4.8), squeeze=False)
    regime_jxy = {}
    for row in summary_rows:
        regime_jxy[row["regime"]] = float(row["J_xy"])
    regimes = sorted(regime_jxy, key=regime_jxy.get)
    for axis, budget in zip(axes[0], budgets):
        for regime in regimes:
            selected = sorted(
                (row for row in summary_rows
                 if row["regime"] == regime and row["budget"] == budget),
                key=lambda row: row["N"],
            )
            sizes = [row["N"] for row in selected]
            means = np.maximum(
                [row["normalized_delta_e_mean"] for row in selected], 1e-16)
            stds = [row["normalized_delta_e_std"] for row in selected]
            axis.errorbar(sizes, means, yerr=stds, marker="o", label=regime)
        axis.set_yscale("log")
        axis.set_xlabel("N")
        axis.set_ylabel("Normalized exact-state SKQD error")
        axis.set_title(f"Raw budget B={budget:,}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"pilot_exact_skqd.{extension}", dpi=180)
    plt.close(fig)


def assess_gates(args, ground_rows, summary_rows):
    by_size = {}
    for row in ground_rows:
        by_size.setdefault(row["N"], {})[row["regime"]] = row

    expected_regimes = list(args.regime_labels)
    ordered = True
    separated = True
    size_checks = []
    for n_qubit in sorted(by_size):
        rows = by_size[n_qubit]
        values = [float(rows[name]["f95"]) for name in expected_regimes]
        is_ordered = all(left < right for left, right in zip(values, values[1:]))
        separation = values[-1] / values[0]
        is_separated = separation >= args.minimum_separation
        ordered &= is_ordered
        separated &= is_separated
        size_checks.append({
            "N": int(n_qubit),
            "f95_ordered": bool(is_ordered),
            "dense_to_sparse_f95_ratio": float(separation),
            "meets_minimum_separation": bool(is_separated),
        })

    near_degenerate = [
        {
            "N": int(row["N"]),
            "regime": row["regime"],
            "J_xy": float(row["J_xy"]),
            "normalized_gap": float(row["normalized_gap"]),
        }
        for row in ground_rows
        if float(row["normalized_gap"]) < args.gap_warning_threshold
    ]

    summary_by_cell = {}
    for row in summary_rows:
        key = (row["N"], row["regime"])
        summary_by_cell.setdefault(key, {})[row["budget"]] = row
    sorted_budgets = sorted(args.budgets)
    budget_nonworsening = True
    budget_checks = []
    for key in sorted(summary_by_cell):
        lower = summary_by_cell[key][sorted_budgets[0]][
            "normalized_delta_e_mean"]
        upper = summary_by_cell[key][sorted_budgets[-1]][
            "normalized_delta_e_mean"]
        nonworsening = upper <= lower + 1e-12
        budget_nonworsening &= nonworsening
        budget_checks.append({
            "N": int(key[0]),
            "regime": key[1],
            "low_budget_error": float(lower),
            "high_budget_error": float(upper),
            "high_budget_nonworsening": bool(nonworsening),
        })

    return {
        "f95_ordering_pass": bool(ordered),
        "minimum_separation_pass": bool(separated),
        "budget_nonworsening_pass": bool(budget_nonworsening),
        "near_degeneracy_detected": bool(near_degenerate),
        "near_degenerate_cells": near_degenerate,
        "size_checks": size_checks,
        "budget_checks": budget_checks,
        "automatic_candidate_range_pass": bool(ordered and separated),
        "manual_review_required_before_full_data": True,
    }


def _write_analysis(path, args, ground_rows, sampling_summary, gates, wall_seconds):
    lines = [
        "# 1D XXZ sparsity pilot 결과",
        "",
        "이 파일은 exact 9-cell pilot이 자동 생성한 요약이다. 모델 학습 결과가 아니다.",
        "",
        "## 실행 설정",
        "",
        f"- sizes: `{args.sizes}`",
        f"- J_xy: `{args.jxy_values}`",
        f"- J_zz: `{args.jzz}`",
        f"- boundary pin: `-{args.boundary_pin} Z_0`",
        f"- D: `{args.depth}`",
        f"- raw budgets: `{args.budgets}`",
        f"- sampling repeats: `{args.repeats}`",
        f"- wall time: `{wall_seconds:.1f} s`",
        "",
        "## Ground-state sparsity",
        "",
        "| N | regime | J_xy | L95 | f95 | doublet f95 | normalized gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ground_rows, key=lambda item: (item["N"], item["J_xy"])):
        lines.append(
            f"| {row['N']} | {row['regime']} | {row['J_xy']:.3g} | "
            f"{row['L95']} | {row['f95']:.3e} | "
            f"{row['doublet_f95']:.3e} | {row['normalized_gap']:.3e} |")

    lines.extend([
        "",
        "## Exact-state SKQD",
        "",
        "| N | regime | budget | basis size | captured mass | normalized error |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in sorted(
            sampling_summary,
            key=lambda item: (item["N"], item["J_xy"], item["budget"])):
        lines.append(
            f"| {row['N']} | {row['regime']} | {row['budget']} | "
            f"{row['basis_size_mean']:.1f} | "
            f"{row['captured_ground_mass_mean']:.4f} | "
            f"{row['normalized_delta_e_mean']:.3e} ± "
            f"{row['normalized_delta_e_std']:.1e} |")

    lines.extend([
        "",
        "## 자동 gate",
        "",
        f"- f95 ordering: **{'PASS' if gates['f95_ordering_pass'] else 'FAIL'}**",
        "- minimum sparse/dense separation: "
        f"**{'PASS' if gates['minimum_separation_pass'] else 'FAIL'}**",
        "- larger budget non-worsening: "
        f"**{'PASS' if gates['budget_nonworsening_pass'] else 'FAIL'}**",
        "- near-degeneracy detected: "
        f"**{'YES' if gates['near_degeneracy_detected'] else 'NO'}**",
        "",
        "자동 gate와 무관하게 전체 데이터 생성 전 사람의 검토가 필요하다. "
        "특히 near-degenerate cell에서는 단일 eigenvector의 f95와 "
        "doublet-invariant f95를 함께 해석해야 한다.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args):
    validate_args(args)
    output_dir = Path(args.output_dir)
    start = time.perf_counter()

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "xxz_sparsity_scaling_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hamiltonian": "sum_i J_xy(XX+YY)+J_zz*ZZ",
        "boundary_pin": {
            "site": 0,
            "coefficient": -float(args.boundary_pin),
            "term": "-h_pin*Z_0",
        },
        "boundary": "open",
        "simulation": "exact_Sz0_sector",
        "initial_state": "displayed-order 1010... Neel",
        "stored_index_time": "D=d corresponds to t=d*dt",
        "sizes": list(args.sizes),
        "jxy_values": list(args.jxy_values),
        "regime_labels": list(args.regime_labels),
        "jzz": float(args.jzz),
        "depth": int(args.depth),
        "budgets": list(args.budgets),
        "samples_per_dimension": [
            int(budget // args.depth) for budget in args.budgets],
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "eig_tol": float(args.eig_tol),
        "gap_warning_threshold": float(args.gap_warning_threshold),
        "minimum_separation": float(args.minimum_separation),
    }
    with (output_dir / "protocol.json").open("w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)

    ground_rows = []
    sampling_rows = []
    cells = [
        (n_qubit, jxy, regime)
        for n_qubit in sorted(args.sizes)
        for jxy, regime in zip(args.jxy_values, args.regime_labels)
    ]
    for cell_index, (n_qubit, jxy, regime) in enumerate(cells):
        ground_row, cell_sampling_rows = run_cell(
            args, n_qubit, jxy, regime, cell_index)
        ground_rows.append(ground_row)
        sampling_rows.extend(cell_sampling_rows)

    sampling_summary = summarize_sampling(sampling_rows)
    gates = assess_gates(args, ground_rows, sampling_summary)
    wall_seconds = time.perf_counter() - start
    gates["wall_seconds"] = float(wall_seconds)

    _write_csv(output_dir / "raw_metrics.csv", ground_rows)
    _write_csv(output_dir / "sampling_metrics.csv", sampling_rows)
    _write_csv(output_dir / "summary.csv", sampling_summary)
    with (output_dir / "gate_report.json").open("w") as handle:
        json.dump(gates, handle, indent=2, sort_keys=True)
    _make_sparsity_figure(ground_rows, output_dir)
    _make_skqd_figure(sampling_summary, output_dir)
    _write_analysis(
        output_dir / "ANALYSIS.md",
        args=args,
        ground_rows=ground_rows,
        sampling_summary=sampling_summary,
        gates=gates,
        wall_seconds=wall_seconds,
    )
    print(
        f"[done] wrote pilot outputs to {output_dir} in {wall_seconds:.1f}s",
        flush=True,
    )
    return ground_rows, sampling_rows, sampling_summary, gates


def main(argv=None):
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
