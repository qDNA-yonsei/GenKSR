"""Unified evaluation for the canonical GenKSR v3 protocol.

KQD (Section 4.1, Figure 2 / Table 1):
    Estimate the KQD generalized-eigenvalue energy at every Krylov dimension for the
    ideal (exact-simulation) reference, the raw classical-shadow device data, and the
    Transformer / Mamba generated samples. Writes all_results.pkl + per-method
    *_errors.txt into a data_comparison_{N}qubit_{M}samples/ directory.

    python scripts/evaluate.py --task kqd --n-qubit 15 --max-krylov-dim 15 \
        --snapshots 10000 --num-samples 10000 \
        --data-root ./data/heisenberg/ \
        --transformer-results results/<tf_run>/ --mamba-results results/<mamba_run>/ \
        --models ideal shadow transformer mamba

SQKD (Section 4.2, Figure 3 / Table 2 / Figure 8):
    Compute the cumulative SKQD ground-state energy convergence curve (per Krylov
    dimension) for the exact-simulation device data and for the generated samples,
    against the exact diagonalisation energy. Writes all_results.pkl.

    python scripts/evaluate.py --task sqkd --n-qubit 16 --max-krylov-dim 30 \
        --snapshots 1000 --num-samples 1000 \
        --data-root ./data/heisenberg_SQKD_2d/ \
        --transformer-results results/<tf_run>/ --mamba-results results/<mamba_run>/ \
        --models device transformer mamba
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
from tqdm import tqdm

from genksr.constants import measurement_outcomes_for_task
from genksr.data.hamiltonian import validate_protocol_metadata
from genksr.protocol import PROTOCOL_VERSION
from genksr.utils.io import filter_folder_for_files
from genksr.utils.quantum import (
    build_hamiltonian_from_couplings,
    build_2d_xxz_hamiltonian,
    convert_to_counts_dict,
)
from genksr.evaluation.kqd import (
    exact_ground_state,
    ideal_krylov_energies,
    shadow_krylov_energies,
)
from genksr.evaluation.skqd import (
    exact_ground_energy,
    projected_xxz_energies_per_dim,
    samples_to_basis_ints,
    skqd_energies_per_dim,
)


def parse_args():
    p = argparse.ArgumentParser(description='GenKSR v3 evaluation')
    p.add_argument('--task', type=str, default='sqkd', choices=['kqd', 'sqkd'])
    p.add_argument('--lattice', type=str, default='1d', choices=['1d', '2d'],
                   help='KQD lattice. SQKD is always 2d.')
    p.add_argument('--n-qubit', type=int, default=16)
    p.add_argument('--max-krylov-dim', type=int, default=15,
                   help='Evaluate Krylov dimensions 1 .. max-krylov-dim.')
    p.add_argument('--num-trotter-steps', type=int, default=6)
    p.add_argument('--snapshots', type=int, default=10000,
                   help='Sample count in the model_samples filename.')
    p.add_argument('--num-samples', type=int, default=None,
                   help='Subsample this many shots per Krylov dim (default: use all).')
    p.add_argument('--k', type=int, default=10, help='Median-of-means splits (KQD shadow estimator).')
    p.add_argument('--splits', nargs='+', default=['test'], choices=['train', 'test'])
    p.add_argument('--models', nargs='+', default=['ideal', 'shadow', 'transformer', 'mamba'],
                   help="KQD: ideal/shadow/transformer/mamba.  SQKD: device/transformer/mamba.")
    p.add_argument('--data-root', type=str, default='./data/heisenberg/',
                   help='KQD: ./data/heisenberg/  SQKD: ./data/heisenberg_SQKD_2d/')
    p.add_argument('--transformer-results', type=str, default=None)
    p.add_argument('--mamba-results', type=str, default=None)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--delta', type=float, default=1.0)
    p.add_argument('--h-z', type=float, default=0.0)
    p.add_argument('--sqkd-dt', type=float, default=None,
                   help='SQKD physical timestep override used during data generation.')
    p.add_argument('--skqd-solver', choices=['fast', 'addon'], default='fast',
                   help='SKQD projected solver. fast directly constructs the sampled XXZ '
                        'subspace; addon uses qiskit-addon-sqd for cross-checks.')
    p.add_argument('--energy-dims', type=int, nargs='+', default=None,
                   help='SQKD: solve only these 1-indexed cumulative Krylov dimensions. '
                        'Other entries are stored as NaN (default: solve every dimension).')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _hamiltonian_ids(data_dir):
    fps = filter_folder_for_files(data_dir, 'coupling_matrix_id*.npy')
    return sorted(int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in fps)


def _model_results_dir(model, args):
    if model == 'transformer':
        return args.transformer_results
    if model == 'mamba':
        return args.mamba_results
    return None


def _validate_model_results(results_dir, task, n_qubit, lattice):
    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        raise ValueError(f'Missing model config: {config_path}')
    with open(config_path, 'r') as f:
        data_config = json.load(f).get('data', {})

    expected_outcomes = measurement_outcomes_for_task(task)
    expected = {
        'protocol_version': PROTOCOL_VERSION,
        'task': task,
        'n_qubit': n_qubit,
        'lattice': lattice,
        'n_outcomes': expected_outcomes,
    }
    mismatches = {
        key: (data_config.get(key), value)
        for key, value in expected.items()
        if data_config.get(key) != value
    }
    if mismatches:
        details = ', '.join(
            f'{key}={actual!r} (expected {wanted!r})'
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f'Incompatible model results at {results_dir}: {details}')


def _load_model_samples(results_dir, split, kd, snapshots, hid, num_samples, rng, binary_only=False):
    """Load one Krylov-dim sample file for a generative model; returns array or None."""
    fp = os.path.join(results_dir, 'samples', split, 'model',
                      f'model_samples_D{kd}_{snapshots}_id{hid}.npy')
    if not os.path.exists(fp):
        return None
    samples = np.load(fp)
    if binary_only:
        mask = np.all((samples == 0) | (samples == 1), axis=1)
        if not np.all(mask):
            raise ValueError(
                f'{fp} contains {np.count_nonzero(~mask)} non-binary sample(s); '
                'canonical v3 SQKD sampling must use a two-outcome model')
    if len(samples) == 0:
        return None
    if num_samples is not None and num_samples < len(samples):
        samples = samples[rng.choice(len(samples), size=num_samples, replace=False)]
    return samples


def _sqkd_sample_statistics(samples_by_dim, n_qubit, krylov_dims):
    """Return per-dimension post-selection and cumulative basis statistics."""
    input_counts = []
    postselected_counts = []
    unique_counts = []
    cumulative_unique_counts = []
    cumulative_basis = set()

    for dimension in krylov_dims:
        samples = samples_by_dim.get(dimension)
        if samples is None:
            input_counts.append(0)
            postselected_counts.append(0)
            unique_counts.append(0)
        else:
            input_counts.append(int(len(samples)))
            selected_mask = np.sum(samples, axis=1) == n_qubit // 2
            postselected_counts.append(int(np.count_nonzero(selected_mask)))
            basis = samples_to_basis_ints(samples, n_qubit=n_qubit)
            unique_counts.append(int(len(basis)))
            cumulative_basis.update(basis.tolist())
        cumulative_unique_counts.append(len(cumulative_basis))

    return {
        'num_input_samples_per_dim': input_counts,
        'num_postselected_samples_per_dim': postselected_counts,
        'num_unique_postselected_per_dim': unique_counts,
        'num_cumulative_unique_postselected_per_dim': cumulative_unique_counts,
    }


# ---------------------------------------------------------------------------
# KQD evaluation (Section 4.1)
# ---------------------------------------------------------------------------

def evaluate_kqd(args):
    rng = np.random.default_rng(args.seed)
    n = args.n_qubit
    maxd = args.max_krylov_dim
    suffix = '_2d' if args.lattice == '2d' else ''
    out = args.output_dir or f'data_comparison_{n}qubit_{args.num_samples}samples{suffix}'
    os.makedirs(out, exist_ok=True)

    results = {s: {m: {} for m in ['ideal', 'shadow', 'transformer', 'mamba']} for s in args.splits}

    for model in ('transformer', 'mamba'):
        if model in args.models and _model_results_dir(model, args) is not None:
            _validate_model_results(
                _model_results_dir(model, args), 'kqd', n, args.lattice)

    for split in args.splits:
        data_dir = os.path.join(args.data_root, f'{n}qubits_pennylane', split)
        validate_protocol_metadata(
            data_dir, task='kqd', lattice=args.lattice, n_qubit=n,
            num_trotter_steps=args.num_trotter_steps)
        ham_ids = _hamiltonian_ids(data_dir)

        for hid in tqdm(ham_ids, desc=f'KQD {split}'):
            coupling = np.load(os.path.join(data_dir, f'coupling_matrix_id{hid}.npy'))
            if args.lattice == '2d':
                H_op = build_2d_xxz_hamiltonian(N=n, J=coupling,
                                                delta=args.delta, h_z=args.h_z)
                gs_energy, _ = exact_ground_state(H_op, n, canonicalize=True)
            else:
                H_op = build_hamiltonian_from_couplings(n_qubits=n, J=coupling)
                gs_energy, _ = exact_ground_state(H_op, n)

            if 'ideal' in args.models:
                energies = ideal_krylov_energies(
                    coupling, n, args.num_trotter_steps, maxd,
                    lattice=args.lattice, delta=args.delta, h_z=args.h_z)
                results[split]['ideal'][hid] = {
                    'energies': energies, 'exact_energy': gs_energy, 'coupling_matrix': coupling,
                }

            if 'shadow' in args.models:
                samples_by_dim = {}
                for kd in range(maxd - 1):
                    fp = os.path.join(data_dir, f'data_D{kd}_id{hid}.npy')
                    if os.path.exists(fp):
                        samples_by_dim[kd] = np.load(fp)
                energies = shadow_krylov_energies(samples_by_dim, coupling, n,
                                                  args.num_trotter_steps, maxd,
                                                  num_samples=args.num_samples, k=args.k,
                                                  rng=rng, lattice=args.lattice,
                                                  delta=args.delta, h_z=args.h_z)
                results[split]['shadow'][hid] = {'energies': energies}

            for model in ('transformer', 'mamba'):
                if model not in args.models:
                    continue
                rdir = _model_results_dir(model, args)
                if rdir is None:
                    warnings.warn(f'--{model}-results not provided; skipping {model}')
                    continue
                samples_by_dim = {}
                for kd in range(maxd - 1):
                    s = _load_model_samples(rdir, split, kd, args.snapshots, hid,
                                            args.num_samples, rng)
                    if s is not None:
                        samples_by_dim[kd] = s
                energies = shadow_krylov_energies(samples_by_dim, coupling, n,
                                                  args.num_trotter_steps, maxd,
                                                  num_samples=args.num_samples, k=args.k,
                                                  rng=rng, lattice=args.lattice,
                                                  delta=args.delta, h_z=args.h_z)
                results[split][model][hid] = {'energies': energies}

    _save_kqd(results, out, args)
    print(f'KQD results saved to {out}/')


def _save_kqd(results, out, args):
    with open(os.path.join(out, 'all_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    for split in results:
        ideal = results[split].get('ideal', {})
        for method in ['ideal', 'shadow', 'transformer', 'mamba']:
            rows = []
            for hid, entry in sorted(results[split][method].items()):
                if hid not in ideal:
                    continue
                exact = ideal[hid]['exact_energy']
                rows.append([abs(exact - e) for e in entry['energies']])
            if rows:
                np.savetxt(os.path.join(out, f'{split}_{method}_errors.txt'), np.array(rows),
                           header=f'{split} {method} |E_exact - E_KQD| per (Hamiltonian, Krylov dim)')


# ---------------------------------------------------------------------------
# SQKD evaluation (Section 4.2)
# ---------------------------------------------------------------------------

def evaluate_sqkd(args):
    rng = np.random.default_rng(args.seed)
    n = args.n_qubit
    maxd = args.max_krylov_dim
    krylov_dims = list(range(maxd))
    if args.energy_dims is None:
        evaluate_dims = None
    else:
        invalid_dims = sorted({dimension for dimension in args.energy_dims
                               if dimension < 1 or dimension > maxd})
        if invalid_dims:
            raise ValueError(
                f'--energy-dims must be between 1 and {maxd}, got {invalid_dims}'
            )
        evaluate_dims = {dimension - 1 for dimension in args.energy_dims}
    out = args.output_dir or f'evaluation_sqkd_{n}qubit_kd{maxd}'
    os.makedirs(out, exist_ok=True)

    results = {s: {'exact_energy': {}, 'device': {}, 'transformer': {}, 'mamba': {}}
               for s in args.splits}

    for model in ('transformer', 'mamba'):
        if model in args.models and _model_results_dir(model, args) is not None:
            _validate_model_results(
                _model_results_dir(model, args), 'sqkd', n, '2d')

    for split in args.splits:
        data_dir = os.path.join(args.data_root, f'{n}qubits', split)
        validate_protocol_metadata(
            data_dir, task='sqkd', lattice='2d', n_qubit=n,
            delta=args.delta, h_z=args.h_z, sqkd_dt=args.sqkd_dt,
            num_trotter_steps=args.num_trotter_steps)
        ham_ids = _hamiltonian_ids(data_dir)

        for hid in tqdm(ham_ids, desc=f'SQKD {split}'):
            coupling = np.load(os.path.join(data_dir, f'coupling_matrix_id{hid}.npy'))
            H_op = build_2d_xxz_hamiltonian(N=n, J=coupling, delta=args.delta, h_z=args.h_z)
            results[split]['exact_energy'][hid] = exact_ground_energy(H_op)

            for model in args.models:
                samples_by_dim = {}
                for d in krylov_dims:
                    if model == 'device':
                        fp = os.path.join(data_dir, f'data_D{d}_id{hid}.npy')
                        samples = np.load(fp) if os.path.exists(fp) else None
                        if samples is not None and args.num_samples is not None \
                                and args.num_samples < len(samples):
                            samples = samples[rng.choice(len(samples), args.num_samples, replace=False)]
                    else:
                        rdir = _model_results_dir(model, args)
                        if rdir is None:
                            continue
                        samples = _load_model_samples(rdir, split, d, args.snapshots, hid,
                                                      args.num_samples, rng, binary_only=True)
                    if samples is not None and len(samples) > 0:
                        samples_by_dim[d] = samples

                if not samples_by_dim:
                    continue
                if args.skqd_solver == 'fast':
                    energies = projected_xxz_energies_per_dim(
                        samples_by_dim, coupling, n, krylov_dims,
                        delta=args.delta, h_z=args.h_z,
                        evaluate_dims=evaluate_dims)
                else:
                    counts_by_dim = {
                        d: convert_to_counts_dict(samples)
                        for d, samples in samples_by_dim.items()
                    }
                    energies = skqd_energies_per_dim(
                        counts_by_dim, H_op, n, krylov_dims,
                        evaluate_dims=evaluate_dims)
                results[split][model][hid] = {
                    'energies': energies,
                    **_sqkd_sample_statistics(samples_by_dim, n, krylov_dims),
                }

    with open(os.path.join(out, 'all_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    evaluation_args = vars(args).copy()
    evaluation_args['lattice'] = '2d'
    with open(os.path.join(out, 'evaluation_args.json'), 'w') as f:
        json.dump(evaluation_args, f, indent=2)
    print(f'SQKD results saved to {out}/')


def main():
    args = parse_args()
    if args.task == 'kqd':
        evaluate_kqd(args)
    else:
        evaluate_sqkd(args)


if __name__ == '__main__':
    main()
