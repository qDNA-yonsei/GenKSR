#!/usr/bin/env python3
"""Validate formal XXZ model samples and summarize simple health metrics."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/xxz_sparsity_1d'))
    return parser.parse_args()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_instances(path):
    with path.open(newline='') as f:
        return {int(row['split_id']): row for row in csv.DictReader(f)}


def validate_model(model_dir, data_root):
    sampling = json.loads((model_dir / 'sampling_config.json').read_text())
    config = json.loads((model_dir / 'config.json').read_text())
    n_qubit = int(config['data']['n_qubit'])
    decoder = config['decoder']['type']
    snapshots = int(sampling['snapshots'])
    dims = [int(d) for d in sampling['krylov_dims']]

    require(sampling['splits'] == ['test'], f'{model_dir.name}: expected test-only sampling')
    require(sampling['checkpoint'] == 'model_best.pth.tar',
            f'{model_dir.name}: expected best checkpoint')
    require(sampling['coupling_normalization'] == 'none',
            f'{model_dir.name}: raw coupling mode is not active')
    checkpoint_path = Path(sampling['checkpoint_path'])
    require(checkpoint_path.is_file(), f'{model_dir.name}: checkpoint path is missing')
    require(sha256_file(checkpoint_path) == sampling['checkpoint_sha256'],
            f'{model_dir.name}: checkpoint SHA-256 mismatch')

    source_dir = data_root / f'{n_qubit}qubits' / 'test'
    instances = load_instances(source_dir / 'instances.csv')
    expected_ids = set(instances)
    require(len(expected_ids) == 30, f'{model_dir.name}: expected 30 test Hamiltonians')

    records = json.loads((model_dir / 'sampling_conditions.json').read_text())
    keyed_records = {
        (int(record['krylov_dim']), int(record['hamiltonian_id'])): record
        for record in records
    }
    expected_keys = {(d, hid) for d in dims for hid in expected_ids}
    require(len(records) == len(keyed_records) == len(expected_keys),
            f'{model_dir.name}: duplicate or missing condition records')
    require(set(keyed_records) == expected_keys,
            f'{model_dir.name}: condition-record key mismatch')

    sample_dir = model_dir / 'samples' / 'test' / 'model'
    actual_sample_files = set(sample_dir.glob('model_samples_*.npy'))
    expected_sample_files = {
        sample_dir / f'model_samples_D{d}_{snapshots}_id{hid}.npy'
        for d, hid in expected_keys
    }
    require(actual_sample_files == expected_sample_files,
            f'{model_dir.name}: sample file set mismatch')

    actual_coupling_files = set(sample_dir.glob('coupling_matrix_id*.npy'))
    expected_coupling_files = {
        sample_dir / f'coupling_matrix_id{hid}.npy' for hid in expected_ids
    }
    require(actual_coupling_files == expected_coupling_files,
            f'{model_dir.name}: coupling file set mismatch')

    reference_state = np.tile(np.array([1, 0], dtype=np.uint8), n_qubit // 2)
    aggregate = defaultdict(lambda: {
        'raw_samples': 0,
        'sector_samples': 0,
        'reference_samples': 0,
        'unique_per_hamiltonian': [],
    })

    for hid in sorted(expected_ids):
        row = instances[hid]
        dt = float(row['dt'])
        regime = row['regime']
        source_coupling = np.load(source_dir / f'coupling_matrix_id{hid}.npy')
        copied_coupling = np.load(sample_dir / f'coupling_matrix_id{hid}.npy')
        require(np.array_equal(source_coupling, copied_coupling),
                f'{model_dir.name}: coupling mismatch for id={hid}')

        for d in dims:
            record = keyed_records[(d, hid)]
            require(abs(float(record['dt']) - dt) < 1e-15,
                    f'{model_dir.name}: dt mismatch for D={d}, id={hid}')
            require(abs(float(record['condition_time']) - d * dt) < 1e-13,
                    f'{model_dir.name}: condition-time mismatch for D={d}, id={hid}')

            samples = np.load(sample_dir / f'model_samples_D{d}_{snapshots}_id{hid}.npy')
            require(samples.shape == (snapshots, n_qubit),
                    f'{model_dir.name}: shape mismatch for D={d}, id={hid}')
            require(samples.dtype == np.uint8,
                    f'{model_dir.name}: dtype is {samples.dtype}, expected uint8')
            require(np.all((samples == 0) | (samples == 1)),
                    f'{model_dir.name}: non-binary sample for D={d}, id={hid}')

            stats = aggregate[(d, regime)]
            stats['raw_samples'] += snapshots
            stats['sector_samples'] += int(np.sum(samples.sum(axis=1) == n_qubit // 2))
            stats['reference_samples'] += int(
                np.sum(np.all(samples == reference_state, axis=1)))
            stats['unique_per_hamiltonian'].append(len(np.unique(samples, axis=0)))

    rows = []
    for (d, regime), stats in sorted(aggregate.items()):
        raw = stats['raw_samples']
        rows.append({
            'model': model_dir.name,
            'n_qubit': n_qubit,
            'decoder': decoder,
            'checkpoint_epoch': int(sampling['checkpoint_epoch']),
            'sampling_seed': int(sampling['seed']),
            'krylov_dim': d,
            'regime': regime,
            'raw_samples': raw,
            'sector_samples': stats['sector_samples'],
            'sector_rate': stats['sector_samples'] / raw,
            'reference_rate': stats['reference_samples'] / raw,
            'mean_unique_per_hamiltonian': float(
                np.mean(stats['unique_per_hamiltonian'])),
        })

    model_report = {
        'model': model_dir.name,
        'n_qubit': n_qubit,
        'decoder': decoder,
        'checkpoint_epoch': int(sampling['checkpoint_epoch']),
        'checkpoint_sha256': sampling['checkpoint_sha256'],
        'sampling_seed': int(sampling['seed']),
        'snapshots': snapshots,
        'krylov_dims': dims,
        'hamiltonians': len(expected_ids),
        'sample_files': len(actual_sample_files),
        'coupling_files': len(actual_coupling_files),
        'condition_records': len(records),
    }
    return model_report, rows


def main():
    args = parse_args()
    sampling_root = args.sampling_root.resolve()
    data_root = args.data_root.resolve()
    model_dirs = sorted(path for path in sampling_root.glob('n*_*') if path.is_dir())
    require(len(model_dirs) == 6, f'Expected 6 model directories, found {len(model_dirs)}')

    model_reports = []
    summary_rows = []
    for model_dir in model_dirs:
        report, rows = validate_model(model_dir, data_root)
        model_reports.append(report)
        summary_rows.extend(rows)
        print(f"PASS {model_dir.name}: {report['sample_files']} sample files")

    summary_path = sampling_root / 'validation_summary.csv'
    with summary_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        'status': 'pass',
        'sampling_root': str(sampling_root),
        'data_root': str(data_root),
        'models': model_reports,
        'total_sample_files': sum(row['sample_files'] for row in model_reports),
        'total_condition_records': sum(row['condition_records'] for row in model_reports),
    }
    (sampling_root / 'validation_report.json').write_text(json.dumps(report, indent=2))
    print(f"PASS all: {report['total_sample_files']} sample files")


if __name__ == '__main__':
    main()
