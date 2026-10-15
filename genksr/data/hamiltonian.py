"""Data loading utilities for Hamiltonian-based datasets."""

import json
import os
from operator import itemgetter

import numpy as np

from genksr.protocol import (
    PROTOCOL_VERSION,
    SQKD_TASK,
    evolution_timestep,
    physical_krylov_time,
)
from genksr.constants import measurement_outcomes_for_task
from genksr.utils.io import filter_folder_for_files
from genksr.utils.quantum import build_2d_xxz_hamiltonian, build_hamiltonian_from_couplings


def validate_protocol_metadata(data_dir, task, lattice, n_qubit,
                               delta=None, h_z=None, sqkd_dt=None,
                               num_trotter_steps=None):
    metadata_path = os.path.join(data_dir, 'args.json')
    if not os.path.exists(metadata_path):
        raise ValueError(
            f"Missing {metadata_path}; refusing to treat an unversioned dataset as v3")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    version = metadata.get('protocol_version')
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Dataset protocol_version={version!r}, expected {PROTOCOL_VERSION}; "
            "regenerate it instead of mixing v2 and v3 artifacts")
    if metadata.get('task') != task:
        raise ValueError(
            f"Dataset task={metadata.get('task')!r}, but training requested {task!r}")
    if metadata.get('n_qubit') != n_qubit:
        raise ValueError(
            f"Dataset n_qubit={metadata.get('n_qubit')!r}, expected {n_qubit}")
    expected_outcomes = measurement_outcomes_for_task(task)
    if metadata.get('n_outcomes') != expected_outcomes:
        raise ValueError(
            f"Dataset n_outcomes={metadata.get('n_outcomes')!r}, expected "
            f"{expected_outcomes} for task {task!r}")
    if (num_trotter_steps is not None
            and metadata.get('num_trotter_steps') != num_trotter_steps):
        raise ValueError(
            f"Dataset num_trotter_steps={metadata.get('num_trotter_steps')!r}, "
            f"expected {num_trotter_steps}")
    dataset_lattice = metadata.get('lattice', '2d' if task == SQKD_TASK else None)
    if dataset_lattice != lattice:
        raise ValueError(
            f"Dataset lattice={dataset_lattice!r}, expected {lattice!r}")
    if task == SQKD_TASK:
        for name, expected in (('delta', delta), ('h_z', h_z)):
            actual = metadata.get(name)
            if expected is not None and (
                    actual is None or not np.isclose(actual, expected)):
                raise ValueError(
                    f"Dataset {name}={actual!r}, expected {expected!r}")
        dataset_dt = metadata.get('sqkd_dt')
        if (dataset_dt is None) != (sqkd_dt is None) or (
                dataset_dt is not None and not np.isclose(dataset_dt, sqkd_dt)):
            raise ValueError(
                f"Dataset sqkd_dt={dataset_dt!r}, expected {sqkd_dt!r}")


def load_data_for_all_dims(data_dir, num_samples_per, num_hamiltonians, rng,
                           krylov_dims, n_qubit, task, lattice='1d',
                           delta=1.0, h_z=0.0, sqkd_dt=None):
    """Load data for multiple Krylov dimensions and concatenate into a single dataset.

    Args:
        data_dir: Directory containing data files.
        num_samples_per: Number of samples per Hamiltonian per Krylov dimension.
        num_hamiltonians: Number of Hamiltonians to use (None for all).
        rng: NumPy random number generator.
        krylov_dims: List of Krylov dimensions to load.
        n_qubit: Number of qubits.
        task: ``kqd`` or ``sqkd``; determines the stored-index/time mapping.
        lattice: Hamiltonian lattice used to calculate the canonical timestep.

    Returns:
        Tuple of (measurements, coupling_matrices, hamiltonian_ids, krylov_dt).
    """
    validate_protocol_metadata(
        data_dir, task, lattice, n_qubit, delta=delta, h_z=h_z, sqkd_dt=sqkd_dt)
    coupling_mats_fps = filter_folder_for_files(data_dir, file_pattern='coupling_matrix_id*.npy')
    coupling_mats_ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in coupling_mats_fps]

    if num_hamiltonians is not None:
        indices = rng.choice(range(len(coupling_mats_ids)), size=num_hamiltonians, replace=False)
        coupling_mats_fps = list(itemgetter(*indices)(coupling_mats_fps)) if len(indices) > 1 else [coupling_mats_fps[indices[0]]]
        coupling_mats_ids = list(itemgetter(*indices)(coupling_mats_ids)) if len(indices) > 1 else [coupling_mats_ids[indices[0]]]

    all_measurements = []
    all_hamiltonian_ids = []
    all_krylov_dt = []
    coupling_matrices_by_id = {}

    for d in krylov_dims:
        print(f"Loading data for Krylov dimension D = {d}...")
        for cid in coupling_mats_ids:
            data_filename = f'data_D{d}_id{cid}.npy'
            data_path = os.path.join(data_dir, data_filename)

            if not os.path.exists(data_path):
                print(f"Warning: Data file not found, skipping: {data_path}")
                continue

            ham_mmts = np.load(data_path)
            coup_mat = np.load(os.path.join(data_dir, f'coupling_matrix_id{cid}.npy'))
            if lattice == '2d' or task == SQKD_TASK:
                H_op = build_2d_xxz_hamiltonian(
                    N=n_qubit, J=coup_mat, delta=delta, h_z=h_z)
            else:
                H_op = build_hamiltonian_from_couplings(n_qubits=n_qubit, J=coup_mat)
            dt_override = sqkd_dt if task == SQKD_TASK else None
            dt = evolution_timestep(H_op, override=dt_override)

            num_available = ham_mmts.shape[0]
            indices = rng.choice(num_available, num_samples_per, replace=False)
            ham_mmts_sampled = ham_mmts[indices]

            all_measurements.append(ham_mmts_sampled)
            coupling_matrices_by_id[int(cid)] = coup_mat
            all_hamiltonian_ids.append(cid * np.ones(shape=num_samples_per, dtype=np.int64))
            condition_time = physical_krylov_time(task, d, dt)
            all_krylov_dt.append(condition_time * np.ones(shape=num_samples_per))

    if not all_measurements:
        raise ValueError("No data was loaded. Check data paths and file names.")

    unique_ids = sorted(coupling_matrices_by_id)

    return (
        np.concatenate(all_measurements, axis=0),
        np.stack([coupling_matrices_by_id[cid] for cid in unique_ids], axis=0),
        np.concatenate(all_hamiltonian_ids, axis=0),
        np.concatenate(all_krylov_dt, axis=0),
    )
