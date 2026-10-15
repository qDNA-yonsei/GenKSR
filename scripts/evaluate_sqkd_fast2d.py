#!/usr/bin/env python3
"""Fast 2D SKQD evaluator for computational-basis samples.

This is a focused evaluator for the 4x4 J1-J2 Heisenberg SQKD experiments.  It
constructs the projected Hamiltonian directly in the sampled bitstring subspace:

    H = sum_ij J_ij (X_i X_j + Y_i Y_j + Z_i Z_j)

In the computational basis, ZZ is diagonal and XX+YY flips anti-aligned pairs
with matrix element 2*J_ij.  This matches the qiskit_addon_sqd projection but is
much faster for the large dt=0.15 subspaces used in the legacy SKQD notebooks.
"""
import argparse
import os
import pickle

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

from genksr.evaluation.skqd import exact_ground_energy
from genksr.utils.io import filter_folder_for_files
from genksr.utils.quantum import build_2d_xxz_hamiltonian


def parse_args():
    p = argparse.ArgumentParser(description="Fast 2D SKQD evaluator")
    p.add_argument("--data-root", required=True,
                   help="Root containing {n_qubit}qubits/{split}/.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--transformer-results", type=str, default=None)
    p.add_argument("--mamba-results", type=str, default=None)
    p.add_argument("--models", nargs="+", default=["device"],
                   choices=["device", "transformer", "mamba"])
    p.add_argument("--snapshots", type=int, default=10000,
                   help="Sample count in model_samples_D*_snapshots_id*.npy filenames.")
    p.add_argument("--n-qubit", type=int, default=16)
    p.add_argument("--max-krylov-dim", type=int, default=30)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--h-z", type=float, default=0.0)
    return p.parse_args()


def hamiltonian_ids(data_dir):
    fps = filter_folder_for_files(data_dir, "coupling_matrix_id*.npy")
    return sorted(int(fp[(fp.find("id") + 2):fp.find(".npy")]) for fp in fps)


def samples_to_ints(samples, n_qubit):
    mask = samples.sum(axis=1) == (n_qubit // 2)
    samples = samples[mask]
    if len(samples) == 0:
        return np.array([], dtype=np.int64)
    powers = (1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64))
    return samples.astype(np.int64) @ powers


def coupling_edges(coupling):
    rows, cols = np.triu_indices(coupling.shape[0], k=1)
    return [(int(i), int(j), float(coupling[i, j]))
            for i, j in zip(rows, cols) if coupling[i, j] != 0.0]


def model_results_dir(model, args):
    if model == "transformer":
        return args.transformer_results
    if model == "mamba":
        return args.mamba_results
    return None


def load_samples_for_model(model, args, split, d, hid):
    if model == "device":
        fp = os.path.join(args.data_root, f"{args.n_qubit}qubits", split,
                          f"data_D{d}_id{hid}.npy")
    else:
        rdir = model_results_dir(model, args)
        if rdir is None:
            return None
        fp = os.path.join(rdir, "samples", split, "model",
                          f"model_samples_D{d}_{args.snapshots}_id{hid}.npy")
    if not os.path.exists(fp):
        return None
    samples = np.load(fp)
    mask = np.all((samples == 0) | (samples == 1), axis=1)
    samples = samples[mask]
    return samples if len(samples) else None


def projected_energy(bitstrings, edges, n_qubit):
    states = sorted(bitstrings)
    dim = len(states)
    if dim == 0:
        return np.nan
    state_to_row = {state: row for row, state in enumerate(states)}

    rows = []
    cols = []
    data = []
    for row, state in enumerate(states):
        diag = 0.0
        for i, j, weight in edges:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_j = 1.0 if bit_j == 0 else -1.0
            diag += weight * z_i * z_j

            if bit_i != bit_j:
                flipped = state ^ ((1 << i) | (1 << j))
                col = state_to_row.get(flipped)
                if col is not None:
                    rows.append(row)
                    cols.append(col)
                    data.append(2.0 * weight)

        rows.append(row)
        cols.append(row)
        data.append(diag)

    mat = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float)
    if dim <= 2:
        return float(np.linalg.eigvalsh(mat.toarray())[0])
    eigvals = spla.eigsh(mat, k=1, which="SA", return_eigenvectors=False, tol=1e-10)
    return float(eigvals[0])


def evaluate(args):
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    results = {
        split: {"exact_energy": {}, "device": {}, "transformer": {}, "mamba": {}}
        for split in args.splits
    }

    for split in args.splits:
        data_dir = os.path.join(args.data_root, f"{args.n_qubit}qubits", split)
        for hid in tqdm(hamiltonian_ids(data_dir), desc=f"fast SQKD {split}"):
            coupling = np.load(os.path.join(data_dir, f"coupling_matrix_id{hid}.npy"))
            H_op = build_2d_xxz_hamiltonian(
                N=args.n_qubit, J=coupling, delta=args.delta, h_z=args.h_z)
            results[split]["exact_energy"][hid] = exact_ground_energy(H_op)

            edges = coupling_edges(coupling)
            for model in args.models:
                if model in ("transformer", "mamba") and model_results_dir(model, args) is None:
                    print(f"Skipping {model}: results dir not provided")
                    continue

                cumulative = set()
                energies = []
                for d in range(args.max_krylov_dim):
                    samples = load_samples_for_model(model, args, split, d, hid)
                    if samples is not None and args.num_samples is not None \
                            and args.num_samples < len(samples):
                        idx = rng.choice(len(samples), size=args.num_samples, replace=False)
                        samples = samples[idx]
                    if samples is not None:
                        cumulative.update(samples_to_ints(samples, args.n_qubit).tolist())
                    energies.append(projected_energy(cumulative, edges, args.n_qubit))

                results[split][model][hid] = {"energies": energies}

    with open(os.path.join(out, "all_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"Fast SQKD results saved to {out}/")


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
