#!/usr/bin/env python3
"""Compare exact-evolution and uniform-random SKQD as samples per dimension vary.

This is the canonical v3 counterpart of Stage 4 in ``compar_skqd_v5.ipynb``.
For a fixed 4x4 J1-J2 Hamiltonian and Krylov depth D, it compares:

* exact simulation: M computational-basis draws from every Trotter-evolved state;
* random: D*M uniform draws from the full 2**N Hilbert space, followed by Sz=0
  post-selection (the notebook's raw-budget-matched random baseline).

The quantum probabilities are computed once. Independent repeat-level draws are
then made from those probabilities, using nested prefixes so the sampled basis
at a smaller M is a subset of the basis at every larger M.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from genksr.evaluation.skqd import exact_ground_energy, projected_xxz_energy
from genksr.protocol import (
    PROTOCOL_VERSION,
    build_sqkd_evolution_gate,
    evolution_timestep,
    neel_reference_bitstring,
)
from genksr.utils.quantum import (
    build_2d_xxz_hamiltonian,
    sample_coupling_matrix_2d_j1j2,
)


METHODS = ("exact_simulation", "random")
STYLE = {
    "exact_simulation": dict(color="tab:blue", marker="s", linestyle="-",
                             label="Exact simulation"),
    "random": dict(color="tab:green", marker="D", linestyle=":",
                   label="Random"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="v3 SKQD energy error vs per-dimension sample count M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--output-dir", default="artifacts/v3_skqd_m_sweep_j2_0.5")
    parser.add_argument("--results-json", default=None,
                        help="Input JSON for --mode plot; default: OUTPUT_DIR/results.json")
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, default=0.5)
    parser.add_argument("--krylov-dim", type=int, default=30)
    parser.add_argument("--m-values", type=int, nargs="+",
                        default=[50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--num-trotter-steps", type=int, default=6)
    parser.add_argument("--trotter-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sqkd-dt", type=float, default=None,
                        help="Override v3 pi/sum(abs(coeffs)) timestep")
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--h-z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _validate_args(args):
    if args.L <= 0 or args.krylov_dim <= 0 or args.repeats <= 0:
        raise ValueError("L, krylov_dim, and repeats must be positive")
    if args.num_trotter_steps <= 0:
        raise ValueError("num_trotter_steps must be positive")
    if not args.m_values or any(m <= 0 for m in args.m_values):
        raise ValueError("m-values must be a non-empty list of positive integers")
    if len(set(args.m_values)) != len(args.m_values):
        raise ValueError("m-values must not contain duplicates")
    args.m_values = sorted(args.m_values)


def _build_problem(args):
    n_qubit = args.L * args.L
    coupling = sample_coupling_matrix_2d_j1j2(
        L=args.L, J1=args.J1, J2_range=(args.J2, args.J2),
        rng=np.random.default_rng(args.seed),
    )
    hamiltonian = build_2d_xxz_hamiltonian(
        N=n_qubit, J=coupling, delta=args.delta, h_z=args.h_z)
    dt = evolution_timestep(hamiltonian, override=args.sqkd_dt)
    return n_qubit, coupling, hamiltonian, dt


def exact_evolution_probabilities(
        hamiltonian, n_qubit, lattice_side, krylov_dim, dt,
        num_trotter_steps, trotter_order):
    """Return probability rows for |psi_d>, d=0..krylov_dim-1."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    reference = neel_reference_bitstring(n_qubit, lattice_side=lattice_side)
    prepare = QuantumCircuit(n_qubit)
    for qubit, bit in enumerate(reversed(reference)):
        if bit == "1":
            prepare.x(qubit)

    evolution = QuantumCircuit(n_qubit)
    evolution.append(
        build_sqkd_evolution_gate(
            hamiltonian, dt, reps=num_trotter_steps, order=trotter_order),
        qargs=range(n_qubit),
    )
    evolution = evolution.decompose()

    state = Statevector.from_instruction(prepare)
    probabilities = []
    for d in tqdm(range(krylov_dim), desc="v3 exact statevectors"):
        probs = np.asarray(state.probabilities(), dtype=float)
        probabilities.append(probs / probs.sum())
        if d + 1 < krylov_dim:
            state = state.evolve(evolution)
    return np.asarray(probabilities)


def _postselect_unique(states, num_ones):
    unique = np.unique(np.asarray(states, dtype=np.int64))
    keep = np.fromiter(
        (int(state).bit_count() == num_ones for state in unique),
        dtype=bool, count=len(unique),
    )
    return unique[keep]


def _record(method, repeat, m, total_draws, basis, energy, exact_energy, sector_size):
    delta_e = float(energy - exact_energy)
    if delta_e < -1e-7:
        raise RuntimeError(
            f"Projected {method} energy {energy} is below exact energy {exact_energy}"
        )
    return {
        "repeat": int(repeat),
        "method": method,
        "M": int(m),
        "total_draws": int(total_draws),
        "basis_size": int(len(basis)),
        "coverage": float(len(basis) / sector_size),
        "energy": float(energy),
        "delta_e": max(delta_e, 0.0),
    }


def run_sweep(args):
    n_qubit, coupling, hamiltonian, dt = _build_problem(args)
    num_ones = n_qubit // 2
    sector_size = math.comb(n_qubit, num_ones)
    hilbert_size = 1 << n_qubit
    max_m = max(args.m_values)
    exact_energy = exact_ground_energy(hamiltonian)

    probabilities = exact_evolution_probabilities(
        hamiltonian, n_qubit, args.L, args.krylov_dim, dt,
        args.num_trotter_steps, args.trotter_order,
    )
    sector_indices = np.fromiter(
        (state for state in range(hilbert_size) if state.bit_count() == num_ones),
        dtype=np.int64, count=sector_size,
    )
    sector_mass = probabilities[:, sector_indices].sum(axis=1)
    if not np.allclose(sector_mass, 1.0, atol=1e-10):
        raise RuntimeError(
            f"Exact evolution left the Sz=0 sector; minimum mass={sector_mass.min()}"
        )

    raw = []
    repeat_seeds = np.random.SeedSequence(args.seed).spawn(args.repeats)
    for repeat, repeat_seed in enumerate(repeat_seeds):
        exact_seed, random_seed = repeat_seed.spawn(2)
        exact_rng = np.random.default_rng(exact_seed)
        random_rng = np.random.default_rng(random_seed)

        exact_draws = [
            exact_rng.choice(hilbert_size, size=max_m, replace=True, p=probs)
            for probs in probabilities
        ]
        random_draws = random_rng.integers(
            0, hilbert_size, size=args.krylov_dim * max_m, dtype=np.int64)

        repeat_rows = []
        for m in args.m_values:
            total = args.krylov_dim * m
            exact_basis = _postselect_unique(
                np.concatenate([draws[:m] for draws in exact_draws]), num_ones)
            random_basis = _postselect_unique(random_draws[:total], num_ones)

            for method, basis in (
                    ("exact_simulation", exact_basis), ("random", random_basis)):
                energy = projected_xxz_energy(
                    basis, coupling, n_qubit, delta=args.delta, h_z=args.h_z)
                if not np.isfinite(energy):
                    raise RuntimeError(
                        f"{method} produced no finite energy at repeat={repeat}, M={m}"
                    )
                row = _record(
                    method, repeat, m, total, basis, energy,
                    exact_energy, sector_size)
                raw.append(row)
                repeat_rows.append(row)
                print(
                    f"repeat={repeat + 1}/{args.repeats} M={m:5d} "
                    f"{method:16s} |B|={len(basis):5d} dE={row['delta_e']:.6f}",
                    flush=True,
                )

        for method in METHODS:
            rows = [row for row in repeat_rows if row["method"] == method]
            sizes = np.asarray([row["basis_size"] for row in rows])
            energies = np.asarray([row["energy"] for row in rows])
            if np.any(np.diff(sizes) < 0):
                raise RuntimeError(f"{method} basis size decreased as M increased")
            if np.any(np.diff(energies) > 1e-7):
                raise RuntimeError(f"{method} variational energy increased as M increased")

    config = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "sqkd_m_sweep",
        "n_qubit": n_qubit,
        "L": args.L,
        "J1": args.J1,
        "J2": args.J2,
        "delta": args.delta,
        "h_z": args.h_z,
        "krylov_dim": args.krylov_dim,
        "m_values": args.m_values,
        "repeats": args.repeats,
        "num_trotter_steps": args.num_trotter_steps,
        "trotter_order": args.trotter_order,
        "sqkd_dt": args.sqkd_dt,
        "resolved_dt": dt,
        "seed": args.seed,
        "reference_state": "2D checkerboard Neel",
        "time_index": "state d -> d*dt",
        "exact_energy": exact_energy,
        "sector_size": sector_size,
        "random_baseline": "D*M uniform full-Hilbert-space draws, then Sz=0 post-selection",
    }
    return config, raw


def aggregate_results(raw, m_values):
    aggregated = []
    for method in METHODS:
        for m in m_values:
            rows = [row for row in raw if row["method"] == method and row["M"] == m]
            if not rows:
                continue
            record = {"method": method, "M": int(m), "n_repeats": len(rows)}
            for field in ("energy", "delta_e", "basis_size", "coverage"):
                values = np.asarray([row[field] for row in rows], dtype=float)
                record[f"{field}_mean"] = float(np.mean(values))
                record[f"{field}_std"] = float(np.std(values))
                record[f"{field}_ci95"] = float(1.96 * np.std(values) / np.sqrt(len(values)))
            aggregated.append(record)
    return aggregated


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, config, raw, aggregated):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    _write_csv(os.path.join(output_dir, "raw_results.csv"), raw)
    _write_csv(os.path.join(output_dir, "aggregate_results.csv"), aggregated)
    result = {"config": config, "raw_results": raw, "aggregate_results": aggregated}
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote results: {path}")
    return result


def plot_results(result, output_dir):
    config = result["config"]
    aggregated = result["aggregate_results"]
    fig, (ax_error, ax_coverage) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method in METHODS:
        rows = sorted(
            (row for row in aggregated if row["method"] == method),
            key=lambda row: row["M"],
        )
        if not rows:
            continue
        m_values = np.asarray([row["M"] for row in rows])
        errors = np.maximum([row["delta_e_mean"] for row in rows], 1e-12)
        error_std = np.asarray([row["delta_e_std"] for row in rows])
        coverage = 100.0 * np.asarray([row["coverage_mean"] for row in rows])
        coverage_std = 100.0 * np.asarray([row["coverage_std"] for row in rows])
        style = STYLE[method]

        ax_error.errorbar(
            m_values, errors, yerr=error_std, color=style["color"],
            marker=style["marker"], linestyle=style["linestyle"],
            linewidth=2, markersize=7, capsize=4, label=style["label"],
        )
        ax_coverage.errorbar(
            m_values, coverage, yerr=coverage_std, color=style["color"],
            marker=style["marker"], linestyle=style["linestyle"],
            linewidth=2, markersize=7, capsize=4, label=style["label"],
        )

    ax_error.set_xscale("log")
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Per-dimension samples $M$", fontsize=15)
    ax_error.set_ylabel(r"Energy error $E_{\mathrm{SKQD}}-E_0$", fontsize=15)
    ax_error.set_title("(a) Ground-energy error", fontsize=15)
    ax_error.grid(True, which="both", alpha=0.3)
    ax_error.legend(fontsize=12)

    ax_coverage.set_xscale("log")
    ax_coverage.set_xlabel("Per-dimension samples $M$", fontsize=15)
    ax_coverage.set_ylabel(r"$S_z=0$ basis coverage (\%)", fontsize=15)
    ax_coverage.set_title("(b) Sampled-subspace coverage", fontsize=15)
    ax_coverage.grid(True, which="both", alpha=0.3)
    ax_coverage.legend(fontsize=12)

    fig.suptitle(
        f"v3 SKQD sample budget (4x4 J1-J2, J2={config['J2']}, "
        f"D={config['krylov_dim']}, {config['repeats']} repeats)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"skqd_m_sweep.{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
        print(f"Wrote figure: {path}")
    plt.close(fig)
    return paths


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = args.results_json or os.path.join(args.output_dir, "results.json")

    result = None
    if args.mode in ("compute", "both"):
        config, raw = run_sweep(args)
        aggregated = aggregate_results(raw, args.m_values)
        result = save_results(args.output_dir, config, raw, aggregated)

    if args.mode in ("plot", "both"):
        if result is None:
            with open(results_path, "r") as handle:
                result = json.load(handle)
        plot_results(result, args.output_dir)


if __name__ == "__main__":
    main()
