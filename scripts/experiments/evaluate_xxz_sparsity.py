#!/usr/bin/env python3
"""Equal-raw-budget physical evaluation for the XXZ sparsity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genksr.evaluation.sparsity import (  # noqa: E402
    basis_state_probability,
    fixed_hamming_weight_basis,
    lowest_eigenpairs,
    sparsity_metrics,
    xxz_open_chain_sector_components,
)
from genksr.protocol import (  # noqa: E402
    neel_reference_bitstring,
    xxz_chain_pauli_l1_bound,
    xxz_chain_timestep,
)


METHODS = (
    'exact_state',
    'transformer',
    'mamba',
    'knn5_train',
    'uniform_full_raw',
    'uniform_sector',
)
FIXED_SAMPLE_METHODS = ('exact_state', 'transformer', 'mamba')
REGIME_ORDER = {'sparse': 0, 'middle': 1, 'dense': 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sampling-root', type=Path,
        default=Path('artifacts/v3_xxz_sparsity_scaling/'
                     'formal_sampling_best_trainseed0_sampleseed20260716'))
    parser.add_argument(
        '--data-root', type=Path, default=Path('data/xxz_sparsity_1d'))
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('artifacts/v3_xxz_sparsity_scaling/'
                     'formal_evaluation_single_sampling_20260716'))
    parser.add_argument('--sizes', type=int, nargs='+', default=[12, 16, 20])
    parser.add_argument(
        '--hamiltonian-ids', type=int, nargs='+', default=None,
        help='Optional per-size test-ID subset for a smoke run.')
    parser.add_argument('--depth', type=int, default=15)
    parser.add_argument('--budgets', type=int, nargs='+', default=[3000, 15000])
    parser.add_argument('--knn-k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260716)
    parser.add_argument('--jzz', type=float, default=1.0)
    parser.add_argument('--eig-tol', type=float, default=1e-10)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args):
    require(args.sizes and len(set(args.sizes)) == len(args.sizes),
            'sizes must be non-empty and unique')
    require(all(n > 0 and n % 2 == 0 for n in args.sizes),
            'sizes must be positive and even')
    require(args.depth > 0, 'depth must be positive')
    require(args.budgets and len(set(args.budgets)) == len(args.budgets),
            'budgets must be non-empty and unique')
    require(all(b > 0 and b % args.depth == 0 for b in args.budgets),
            'every budget must be positive and divisible by depth')
    require(args.knn_k > 0, 'knn-k must be positive')
    require(args.seed >= 0, 'seed must be non-negative')
    require(args.jzz > 0 and np.isfinite(args.jzz),
            'jzz must be positive and finite')
    require(args.eig_tol > 0, 'eig-tol must be positive')
    require(args.sampling_root.is_dir(),
            f'sampling root not found: {args.sampling_root}')
    require(args.data_root.is_dir(), f'data root not found: {args.data_root}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f'Output directory is non-empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)


def load_instances(split_dir):
    rows = read_csv(Path(split_dir) / 'instances.csv')
    indexed = {int(row['split_id']): row for row in rows}
    require(len(rows) == len(indexed), f'duplicate split IDs in {split_dir}')
    return indexed


def samples_to_fixed_rows(samples, n_qubit):
    """Convert displayed bitstrings to integers, using -1 for rejected rows."""
    samples = np.asarray(samples)
    require(samples.ndim == 2 and samples.shape[1] == n_qubit,
            f'unexpected sample shape {samples.shape}, N={n_qubit}')
    require(np.all((samples == 0) | (samples == 1)),
            'samples contain non-binary outcomes')
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    states = samples.astype(np.int64) @ powers
    selected = samples.sum(axis=1) == n_qubit // 2
    states[~selected] = -1
    return states


def fixed_stream(rows_by_dimension, depth, samples_per_dimension):
    pieces = []
    raw_draws = 0
    for dimension in range(depth):
        raw = np.asarray(rows_by_dimension[dimension])[:samples_per_dimension]
        require(len(raw) == samples_per_dimension,
                f'not enough samples at D={dimension}')
        raw_draws += len(raw)
        pieces.append(raw[raw >= 0])
    return np.concatenate(pieces), raw_draws


def unique_basis(stream):
    stream = np.asarray(stream, dtype=np.int64)
    return np.unique(stream[stream >= 0])


def uniform_control_streams(seed, n_qubit, hid, max_budget, sector_basis):
    """Return nested random-control streams for one test Hamiltonian."""
    full_rng = np.random.default_rng(
        np.random.SeedSequence([seed, n_qubit, hid, 0]))
    sector_rng = np.random.default_rng(
        np.random.SeedSequence([seed, n_qubit, hid, 1]))
    return {
        'uniform_full_raw': full_rng.integers(
            0, 1 << n_qubit, size=max_budget, dtype=np.int64),
        'uniform_sector': sector_rng.choice(
            sector_basis, size=max_budget, replace=True),
    }


class TrainingSampleCache:
    def __init__(self, train_dir, n_qubit, max_samples):
        self.train_dir = Path(train_dir)
        self.n_qubit = int(n_qubit)
        self.max_samples = int(max_samples)
        self.cache = {}

    def get(self, hid, dimension):
        key = (int(hid), int(dimension))
        if key not in self.cache:
            path = self.train_dir / f'data_D{dimension}_id{hid}.npy'
            samples = np.load(path, mmap_mode='r')[:self.max_samples]
            require(len(samples) == self.max_samples,
                    f'not enough training samples in {path}')
            states = samples_to_fixed_rows(samples, self.n_qubit)
            require(np.all(states >= 0),
                    f'training data left the half-filled sector: {path}')
            self.cache[key] = states
        return self.cache[key]


def build_condition_library(train_instances, depth):
    records = []
    jxy_values = np.asarray(
        [float(row['J_xy']) for row in train_instances.values()])
    jxy_min = float(jxy_values.min())
    jxy_max = float(jxy_values.max())
    require(jxy_max > jxy_min, 'training J_xy range is degenerate')
    time_scale = max(
        (depth - 1) * float(row['dt']) for row in train_instances.values())
    require(time_scale > 0, 'training physical-time scale is not positive')

    for hid, row in sorted(train_instances.items()):
        jxy = float(row['J_xy'])
        dt = float(row['dt'])
        for dimension in range(depth):
            physical_time = dimension * dt
            records.append({
                'hamiltonian_id': int(hid),
                'dimension': int(dimension),
                'J_xy': jxy,
                'time': physical_time,
                'features': np.asarray([
                    (jxy - jxy_min) / (jxy_max - jxy_min),
                    physical_time / time_scale,
                ]),
            })
    scaling = {
        'jxy_min': jxy_min,
        'jxy_max': jxy_max,
        'time_scale': float(time_scale),
    }
    return records, scaling


def select_knn_conditions(
        target_jxy, target_time, conditions, scaling, knn_k):
    target_features = np.asarray([
        (target_jxy - scaling['jxy_min'])
        / (scaling['jxy_max'] - scaling['jxy_min']),
        target_time / scaling['time_scale'],
    ])
    ranked = sorted(
        conditions,
        key=lambda condition: (
            float(np.linalg.norm(condition['features'] - target_features)),
            condition['hamiltonian_id'],
            condition['dimension'],
        ),
    )
    require(len(ranked) >= knn_k, 'condition library is smaller than k')
    return target_features, ranked[:knn_k]


def generate_knn_streams(
        n_qubit, hid, target_jxy, target_dt, depth, max_samples,
        seed, conditions, scaling, knn_k, sample_cache):
    streams = {}
    mapping = []
    for dimension in range(depth):
        target_time = dimension * target_dt
        target_features, neighbors = select_knn_conditions(
            target_jxy, target_time, conditions, scaling, knn_k)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, n_qubit, hid, dimension, 2]))
        choices = rng.integers(0, len(neighbors), size=max_samples)
        stream = np.empty(max_samples, dtype=np.int64)
        for neighbor_index, neighbor in enumerate(neighbors):
            positions = np.flatnonzero(choices == neighbor_index)
            if len(positions) == 0:
                continue
            source = sample_cache.get(
                neighbor['hamiltonian_id'], neighbor['dimension'])
            source_indices = rng.integers(0, len(source), size=len(positions))
            stream[positions] = source[source_indices]
        streams[dimension] = stream

        for rank, neighbor in enumerate(neighbors):
            mapping.append({
                'N': int(n_qubit),
                'test_hamiltonian_id': int(hid),
                'target_dimension': int(dimension),
                'target_J_xy': float(target_jxy),
                'target_time': float(target_time),
                'target_feature_J_xy': float(target_features[0]),
                'target_feature_time': float(target_features[1]),
                'neighbor_rank': int(rank),
                'train_hamiltonian_id': int(neighbor['hamiltonian_id']),
                'train_dimension': int(neighbor['dimension']),
                'train_J_xy': float(neighbor['J_xy']),
                'train_time': float(neighbor['time']),
                'feature_distance': float(np.linalg.norm(
                    neighbor['features'] - target_features)),
                'mixture_weight': float(1.0 / len(neighbors)),
            })
    return streams, mapping


def lowest_energy(matrix, tol):
    dimension = int(matrix.shape[0])
    if dimension == 0:
        return np.nan
    if dimension <= 4:
        return float(np.linalg.eigvalsh(matrix.toarray())[0])
    v0 = np.linspace(1.0, 2.0, dimension, dtype=float)
    v0 /= np.linalg.norm(v0)
    return float(sparse_linalg.eigsh(
        matrix, k=1, which='SA', return_eigenvectors=False,
        v0=v0, tol=tol,
    )[0])


def basis_metrics(
        basis, sector_hamiltonian, sector_index, ground_probabilities,
        doublet_probabilities,
        exact_energy, coefficient_l1, eig_tol):
    basis = np.asarray(basis, dtype=np.int64)
    require(len(basis) > 0, 'sampled basis is empty')
    positions = sector_index[basis]
    require(np.all(positions >= 0), 'basis contains a state outside the sector')
    projected = sector_hamiltonian[positions][:, positions].tocsr()
    start = time.perf_counter()
    energy = lowest_energy(projected, tol=eig_tol)
    solve_seconds = time.perf_counter() - start
    delta_e = float(energy - exact_energy)
    require(delta_e >= -1e-7 * coefficient_l1,
            f'variational bound violation: delta={delta_e}')
    delta_e = max(delta_e, 0.0)
    captured_mass = float(np.sum(ground_probabilities[positions]))
    captured_doublet_mass = float(np.sum(doublet_probabilities[positions]))
    return {
        'energy': float(energy),
        'exact_energy': float(exact_energy),
        'coefficient_l1': float(coefficient_l1),
        'delta_e': delta_e,
        'normalized_delta_e': float(delta_e / coefficient_l1),
        'captured_ground_mass': captured_mass,
        'missing_ground_mass': float(max(1.0 - captured_mass, 0.0)),
        'captured_doublet_mass': captured_doublet_mass,
        'missing_doublet_mass': float(max(1.0 - captured_doublet_mass, 0.0)),
        'projected_solve_seconds': float(solve_seconds),
    }


def fixed_method_rows(
        method, n_qubit, hid, depth, max_samples, test_dir,
        sampling_root):
    rows = {}
    for dimension in range(depth):
        if method == 'exact_state':
            path = test_dir / f'data_D{dimension}_id{hid}.npy'
        else:
            path = (sampling_root / f'n{n_qubit}_{method}' / 'samples' /
                    'test' / 'model' /
                    f'model_samples_D{dimension}_1000_id{hid}.npy')
        samples = np.load(path, mmap_mode='r')[:max_samples]
        require(len(samples) == max_samples, f'not enough samples in {path}')
        rows[dimension] = samples_to_fixed_rows(samples, n_qubit)
    return rows


def record_result(
        n_qubit, row, method, budget, depth, samples_per_dimension,
        raw_draws, accepted_draws, basis, sector_size, metrics):
    return {
        'N': int(n_qubit),
        'hamiltonian_id': int(row['split_id']),
        'master_id': int(row['master_id']),
        'paired_key': row['paired_key'],
        'regime': row['regime'],
        'regime_id': int(row['regime_id']),
        'J_xy': float(row['J_xy']),
        'J_zz': float(row['J_zz']),
        'method': method,
        'budget': int(budget),
        'D': int(depth),
        'M': int(samples_per_dimension),
        'raw_draws': int(raw_draws),
        'accepted_draws': int(accepted_draws),
        'rejected_draws': int(raw_draws - accepted_draws),
        'acceptance_rate': float(accepted_draws / raw_draws),
        'basis_size': int(len(basis)),
        'sector_size': int(sector_size),
        'coverage': float(len(basis) / sector_size),
        **metrics,
    }


def ground_metrics(
        n_qubit, row, sector_basis, sector_hamiltonian,
        coefficient_l1, eig_tol):
    start = time.perf_counter()
    eigenvalues, eigenvectors = lowest_eigenpairs(
        sector_hamiltonian, k=2, tol=eig_tol)
    eigensolve_seconds = time.perf_counter() - start
    probabilities = np.abs(eigenvectors[:, 0]) ** 2
    probabilities /= probabilities.sum()
    doublet = np.sum(np.abs(eigenvectors[:, :2]) ** 2, axis=1)
    doublet /= doublet.sum()
    single_metrics = sparsity_metrics(probabilities, sector_size=len(sector_basis))
    doublet_metrics = sparsity_metrics(doublet, sector_size=len(sector_basis))
    neel_integer = int(neel_reference_bitstring(n_qubit), 2)
    gap = float(eigenvalues[1] - eigenvalues[0])
    output = {
        'N': int(n_qubit),
        'hamiltonian_id': int(row['split_id']),
        'master_id': int(row['master_id']),
        'paired_key': row['paired_key'],
        'regime': row['regime'],
        'regime_id': int(row['regime_id']),
        'J_xy': float(row['J_xy']),
        'J_zz': float(row['J_zz']),
        'sector_size': int(len(sector_basis)),
        'coefficient_l1': float(coefficient_l1),
        'dt': float(row['dt']),
        'E0': float(eigenvalues[0]),
        'E1': float(eigenvalues[1]),
        'gap': gap,
        'normalized_gap': float(gap / coefficient_l1),
        'L95': int(single_metrics['l95']),
        'f95': float(single_metrics['f95']),
        'L99': int(single_metrics['l99']),
        'f99': float(single_metrics['f99']),
        'effective_support': float(single_metrics['effective_support']),
        'effective_support_fraction': float(
            single_metrics['effective_support_fraction']),
        'doublet_L95': int(doublet_metrics['l95']),
        'doublet_f95': float(doublet_metrics['f95']),
        'doublet_L99': int(doublet_metrics['l99']),
        'doublet_f99': float(doublet_metrics['f99']),
        'neel_probability': basis_state_probability(
            probabilities, sector_basis, neel_integer),
        'neel_doublet_average_probability': basis_state_probability(
            doublet, sector_basis, neel_integer),
        'eigensolve_seconds': float(eigensolve_seconds),
    }
    return output, probabilities, doublet


def summarize_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['N'], row['regime'], row['method'], row['budget'])].append(row)
    metrics = (
        'acceptance_rate', 'basis_size', 'coverage',
        'captured_ground_mass', 'captured_doublet_mass',
        'delta_e', 'normalized_delta_e',
        'projected_solve_seconds',
    )
    summary = []
    for key in sorted(
            grouped,
            key=lambda value: (
                value[0], REGIME_ORDER[value[1]], METHODS.index(value[2]), value[3])):
        group = grouped[key]
        output = {
            'N': int(key[0]),
            'regime': key[1],
            'method': key[2],
            'budget': int(key[3]),
            'D': int(group[0]['D']),
            'M': int(group[0]['M']),
            'num_hamiltonians': int(len(group)),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            output[f'{metric}_mean'] = float(np.mean(values))
            output[f'{metric}_std'] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0)
        summary.append(output)
    return summary


def paired_model_rows(rows):
    lookup = {
        (row['N'], row['hamiltonian_id'], row['budget'], row['method']): row
        for row in rows
    }
    paired = []
    keys = sorted({
        (row['N'], row['hamiltonian_id'], row['budget']) for row in rows
    })
    for n_qubit, hid, budget in keys:
        exact = lookup[(n_qubit, hid, budget, 'exact_state')]
        for model in ('transformer', 'mamba'):
            candidate = lookup[(n_qubit, hid, budget, model)]
            paired.append({
                'N': int(n_qubit),
                'hamiltonian_id': int(hid),
                'paired_key': candidate['paired_key'],
                'regime': candidate['regime'],
                'J_xy': float(candidate['J_xy']),
                'budget': int(budget),
                'M': int(candidate['M']),
                'model': model,
                'model_normalized_delta_e': float(
                    candidate['normalized_delta_e']),
                'exact_state_normalized_delta_e': float(
                    exact['normalized_delta_e']),
                'surrogate_gap': float(
                    candidate['normalized_delta_e']
                    - exact['normalized_delta_e']),
                'model_basis_size': int(candidate['basis_size']),
                'exact_state_basis_size': int(exact['basis_size']),
                'model_captured_ground_mass': float(
                    candidate['captured_ground_mass']),
                'exact_state_captured_ground_mass': float(
                    exact['captured_ground_mass']),
                'model_captured_doublet_mass': float(
                    candidate['captured_doublet_mass']),
                'exact_state_captured_doublet_mass': float(
                    exact['captured_doublet_mass']),
            })
    return paired


def validate_sampling_inputs(args):
    metadata = {}
    expected_m = max(args.budgets) // args.depth
    for n_qubit in args.sizes:
        for decoder in ('transformer', 'mamba'):
            model_dir = args.sampling_root / f'n{n_qubit}_{decoder}'
            require(model_dir.is_dir(), f'missing model samples: {model_dir}')
            sampling = json.loads((model_dir / 'sampling_config.json').read_text())
            require(sampling['checkpoint'] == 'model_best.pth.tar',
                    f'{model_dir.name}: not a best checkpoint')
            require(sampling['splits'] == ['test'],
                    f'{model_dir.name}: sampling is not test-only')
            require(sampling['krylov_dims'] == list(range(args.depth)),
                    f'{model_dir.name}: Krylov dimensions mismatch')
            require(int(sampling['snapshots']) >= expected_m,
                    f'{model_dir.name}: insufficient snapshots')
            require(sampling['coupling_normalization'] == 'none',
                    f'{model_dir.name}: coupling normalization mismatch')
            checkpoint_path = Path(sampling['checkpoint_path'])
            require(sha256_file(checkpoint_path) == sampling['checkpoint_sha256'],
                    f'{model_dir.name}: checkpoint hash mismatch')
            metadata[model_dir.name] = sampling
    return metadata


def evaluate_size(args, n_qubit, all_ground_rows, all_result_rows, all_mapping_rows):
    print(f'=== N={n_qubit}: building sector components ===', flush=True)
    sector_basis = fixed_hamming_weight_basis(n_qubit, n_qubit // 2)
    sector_size = len(sector_basis)
    sector_index = np.full(1 << n_qubit, -1, dtype=np.int32)
    sector_index[sector_basis] = np.arange(sector_size, dtype=np.int32)
    construction_start = time.perf_counter()
    h_xy, h_zz = xxz_open_chain_sector_components(n_qubit, sector_basis)
    construction_seconds = time.perf_counter() - construction_start
    print(
        f'N={n_qubit}: sector={sector_size:,}, components={construction_seconds:.1f}s',
        flush=True,
    )

    train_dir = args.data_root / f'{n_qubit}qubits' / 'train'
    test_dir = args.data_root / f'{n_qubit}qubits' / 'test'
    train_instances = load_instances(train_dir)
    test_instances = load_instances(test_dir)
    if args.hamiltonian_ids is None:
        hamiltonian_ids = sorted(test_instances)
    else:
        hamiltonian_ids = sorted(set(args.hamiltonian_ids))
        require(set(hamiltonian_ids) <= set(test_instances),
                f'N={n_qubit}: requested test ID is unavailable')

    max_samples = max(args.budgets) // args.depth
    conditions, scaling = build_condition_library(train_instances, args.depth)
    sample_cache = TrainingSampleCache(train_dir, n_qubit, max_samples)
    max_budget = max(args.budgets)

    for position, hid in enumerate(hamiltonian_ids, start=1):
        row = test_instances[hid]
        jxy = float(row['J_xy'])
        jzz = float(row['J_zz'])
        require(np.isclose(jzz, args.jzz), f'N={n_qubit}, id={hid}: J_zz mismatch')
        coefficient_l1 = xxz_chain_pauli_l1_bound(n_qubit, jxy, jzz)
        require(np.isclose(coefficient_l1, float(row['coefficient_l1'])),
                f'N={n_qubit}, id={hid}: coefficient L1 mismatch')
        require(np.isclose(xxz_chain_timestep(n_qubit, jxy, jzz), float(row['dt'])),
                f'N={n_qubit}, id={hid}: dt mismatch')

        sector_hamiltonian = (jxy * h_xy + jzz * h_zz).tocsr()
        ground_row, ground_probabilities, doublet_probabilities = ground_metrics(
            n_qubit, row, sector_basis, sector_hamiltonian,
            coefficient_l1, args.eig_tol)
        ground_row['component_construction_seconds'] = float(construction_seconds)
        all_ground_rows.append(ground_row)

        fixed_rows = {
            method: fixed_method_rows(
                method, n_qubit, hid, args.depth, max_samples,
                test_dir, args.sampling_root)
            for method in FIXED_SAMPLE_METHODS
        }
        knn_streams, mapping = generate_knn_streams(
            n_qubit=n_qubit,
            hid=hid,
            target_jxy=jxy,
            target_dt=float(row['dt']),
            depth=args.depth,
            max_samples=max_samples,
            seed=args.seed,
            conditions=conditions,
            scaling=scaling,
            knn_k=args.knn_k,
            sample_cache=sample_cache,
        )
        all_mapping_rows.extend(mapping)
        uniform_streams = uniform_control_streams(
            args.seed, n_qubit, hid, max_budget, sector_basis)

        for budget in sorted(args.budgets):
            samples_per_dimension = budget // args.depth
            for method in FIXED_SAMPLE_METHODS:
                stream, raw_draws = fixed_stream(
                    fixed_rows[method], args.depth, samples_per_dimension)
                basis = unique_basis(stream)
                metrics = basis_metrics(
                    basis, sector_hamiltonian, sector_index,
                    ground_probabilities, doublet_probabilities,
                    float(ground_row['E0']),
                    coefficient_l1, args.eig_tol)
                all_result_rows.append(record_result(
                    n_qubit, row, method, budget, args.depth,
                    samples_per_dimension, raw_draws, len(stream),
                    basis, sector_size, metrics))

            knn_stream, raw_draws = fixed_stream(
                knn_streams, args.depth, samples_per_dimension)
            knn_basis = unique_basis(knn_stream)
            knn_metrics = basis_metrics(
                knn_basis, sector_hamiltonian, sector_index,
                ground_probabilities, doublet_probabilities,
                float(ground_row['E0']),
                coefficient_l1, args.eig_tol)
            all_result_rows.append(record_result(
                n_qubit, row, 'knn5_train', budget, args.depth,
                samples_per_dimension, raw_draws, len(knn_stream),
                knn_basis, sector_size, knn_metrics))

            uniform_raw = uniform_streams['uniform_full_raw'][:budget]
            uniform_selected = uniform_raw[sector_index[uniform_raw] >= 0]
            uniform_basis = unique_basis(uniform_selected)
            uniform_metrics = basis_metrics(
                uniform_basis, sector_hamiltonian, sector_index,
                ground_probabilities, doublet_probabilities,
                float(ground_row['E0']),
                coefficient_l1, args.eig_tol)
            all_result_rows.append(record_result(
                n_qubit, row, 'uniform_full_raw', budget, args.depth,
                samples_per_dimension, budget, len(uniform_selected),
                uniform_basis, sector_size, uniform_metrics))

            sector_stream = uniform_streams['uniform_sector'][:budget]
            sector_random_basis = unique_basis(sector_stream)
            sector_random_metrics = basis_metrics(
                sector_random_basis, sector_hamiltonian, sector_index,
                ground_probabilities, doublet_probabilities,
                float(ground_row['E0']),
                coefficient_l1, args.eig_tol)
            all_result_rows.append(record_result(
                n_qubit, row, 'uniform_sector', budget, args.depth,
                samples_per_dimension, budget, budget,
                sector_random_basis, sector_size, sector_random_metrics))

        print(
            f'N={n_qubit} [{position}/{len(hamiltonian_ids)}] id={hid} '
            f'{row["regime"]} Jxy={jxy:.4f} complete',
            flush=True,
        )
        write_csv(args.output_dir / 'ground_metrics.csv', all_ground_rows)
        write_csv(args.output_dir / 'evaluation_metrics.csv', all_result_rows)
        write_csv(args.output_dir / 'knn_source_mapping.csv', all_mapping_rows)

    return {
        'N': int(n_qubit),
        'sector_size': int(sector_size),
        'component_construction_seconds': float(construction_seconds),
        'condition_feature_scaling': scaling,
        'test_hamiltonian_ids': hamiltonian_ids,
    }


def main():
    args = parse_args()
    validate_args(args)
    sampling_metadata = validate_sampling_inputs(args)
    start = time.perf_counter()

    protocol = {
        'task': 'xxz_sparsity_equal_raw_budget',
        'sizes': args.sizes,
        'depth': args.depth,
        'budgets': sorted(args.budgets),
        'samples_per_dimension': {
            str(budget): budget // args.depth for budget in sorted(args.budgets)
        },
        'methods': list(METHODS),
        'model_training_seed': 0,
        'model_sampling_seed': int(next(iter(sampling_metadata.values()))['seed']),
        'baseline_seed': args.seed,
        'sampling_repeats': 1,
        'knn_k': args.knn_k,
        'knn_features': (
            'normalized (J_xy, physical time) over the train condition library'),
        'knn_sampling': (
            'uniform empirical mixture of five nearest train-only conditions; '
            'bootstrap with replacement'),
        'uniform_full_raw': (
            'B draws from all 2**N bitstrings followed by half-filled-sector '
            'postselection'),
        'uniform_sector': (
            'B direct uniform draws from the half-filled sector'),
        'budget_definition': 'raw draws B=15*M before postselection',
        'nested_budget': (
            'M=200 uses the first 200 rows of each saved/generated M=1000 '
            'dimension stream; random controls use the B=3000 prefix of B=15000'),
        'data_root': str(args.data_root.resolve()),
        'sampling_root': str(args.sampling_root.resolve()),
        'ratio_manifest_sha256': sha256_file(
            args.data_root / 'ratio_manifest.csv'),
        'evaluator_sha256': sha256_file(Path(__file__)),
        'hamiltonian_id_subset': args.hamiltonian_ids,
        'status': 'running',
    }
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    ground_rows = []
    result_rows = []
    mapping_rows = []
    size_reports = []
    for n_qubit in args.sizes:
        size_reports.append(evaluate_size(
            args, n_qubit, ground_rows, result_rows, mapping_rows))

    summary_rows = summarize_results(result_rows)
    paired_rows = paired_model_rows(result_rows)
    write_csv(args.output_dir / 'ground_metrics.csv', ground_rows)
    write_csv(args.output_dir / 'evaluation_metrics.csv', result_rows)
    write_csv(args.output_dir / 'summary.csv', summary_rows)
    write_csv(args.output_dir / 'paired_model_vs_exact.csv', paired_rows)
    write_csv(args.output_dir / 'knn_source_mapping.csv', mapping_rows)

    protocol['size_reports'] = size_reports
    protocol['wall_seconds'] = float(time.perf_counter() - start)
    protocol['row_counts'] = {
        'ground_metrics': len(ground_rows),
        'evaluation_metrics': len(result_rows),
        'summary': len(summary_rows),
        'paired_model_vs_exact': len(paired_rows),
        'knn_source_mapping': len(mapping_rows),
    }
    protocol['status'] = 'complete'
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(
        f"Complete: {len(ground_rows)} Hamiltonians, {len(result_rows)} "
        f"method-budget rows, {protocol['wall_seconds']:.1f}s",
        flush=True,
    )


if __name__ == '__main__':
    main()
