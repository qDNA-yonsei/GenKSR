"""Unified sampling script for the canonical GenKSR v3 protocol.

Usage:
    python scripts/sample.py --results-dir /path/to/results/ --snapshots 10000 --krylov-dims 0 1 2 3 4
    python scripts/sample.py --results-dir /path/to/results/ --snapshots 10000 --krylov-dim-range 30
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from genksr.constants import WDEGREE_FEATURE, measurement_outcomes_for_task
from genksr.protocol import (
    PROTOCOL_VERSION,
    SQKD_TASK,
    evolution_timestep,
    physical_krylov_time,
)
from genksr.data.hamiltonian import validate_protocol_metadata
from genksr.data.factory import XXZ_SPARSITY_FAMILY, xxz_instance_dt_by_id
from genksr.data.xxz_sparsity import validate_xxz_sparsity_metadata
from genksr.data.sample_structure import get_sample_structure
from genksr.nn.encoder import get_graph_encoder
from genksr.nn.gc_model import build_gc_model
from genksr.sampling import generate_samples
from genksr.utils.io import filter_folder_for_files, process_weight_names
from genksr.utils.quantum import build_2d_xxz_hamiltonian, build_hamiltonian_from_couplings


def parse_args():
    parser = argparse.ArgumentParser(description='GenKSR v3 Sampling')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory where the trained model and its config are stored.')
    parser.add_argument('--output-results-dir', type=str, default=None,
                        help='Directory where generated samples are saved (defaults to --results-dir).')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override the data root from the training config, useful for fixed-J2 evaluation sets.')
    parser.add_argument('--snapshots', type=int, default=10000,
                        help='Number of generated samples per Hamiltonian.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-qubit', type=int, default=None,
                        help='Number of qubits (defaults to the value in the training config).')
    parser.add_argument('--sqkd-dt', type=float, default=None,
                        help='(SQKD) Override the physical time per Krylov step. '
                             'Legacy runs may use 0.15.')
    parser.add_argument('--krylov-dim-range', type=int, default=None,
                        help='Generate for range(krylov_dim_range)')
    parser.add_argument('--krylov-dims', type=int, nargs='+', default=None,
                        help='Explicit list of Krylov dimensions')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth.tar',
                        help='Checkpoint filename')
    parser.add_argument('--splits', type=str, nargs='+', default=['test', 'train'],
                        choices=['test', 'train', 'validation'],
                        help='Which splits to sample (default both). Use "test" only for figures.')
    return parser.parse_args()


def load_couplings(data_dir):
    """Load coupling matrices from a directory."""
    coupling_mats_fps = filter_folder_for_files(data_dir, file_pattern='coupling_matrix_id*.npy')
    coupling_mats_ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in coupling_mats_fps]

    coupling_matrices_array = []
    coupling_matrices_ids_list = []

    for cid, cfp in zip(coupling_mats_ids, coupling_mats_fps):
        coup_mat = np.load(cfp)
        coupling_matrices_array.append(coup_mat)
        coupling_matrices_ids_list.append(cid)

    coupling_matrices_array = np.stack(coupling_matrices_array, axis=0)
    return coupling_matrices_array, coupling_matrices_ids_list


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of a sampling input artifact."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def measurement_samples_uint8(samples, n_qubit, task):
    """Validate the task's measurement width and alphabet before compact storage."""
    n_outcomes = measurement_outcomes_for_task(task)
    # KQD measures an ancilla in addition to the Hamiltonian's physical qubits.
    expected_qubits = n_qubit + (task == 'kqd')
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != expected_qubits:
        raise ValueError(
            f'Generated samples have shape {samples.shape}, expected (*, {expected_qubits})')
    if not np.all(np.isin(samples, np.arange(n_outcomes))):
        if task == SQKD_TASK:
            raise ValueError('Generated SQKD samples contain non-binary outcomes')
        raise ValueError(
            f'Generated KQD samples must contain integer Pauli-6 outcomes in '
            f'[0, {n_outcomes - 1}]')
    return samples.astype(np.uint8, copy=False)


def binary_samples_uint8(samples, expected_qubits):
    """Validate SKQD bitstrings and store them in the compact canonical dtype."""
    return measurement_samples_uint8(samples, n_qubit=expected_qubits, task=SQKD_TASK)


def run_sampling_for_dim(model, results_dir, data_root, data_subdir, snapshots, rng,
                         gcn_features, device, krylov_dim, n_qubit, task,
                         lattice='1d', delta=1.0, h_z=0.0,
                         splits=('test', 'train'), sqkd_dt=None,
                         hamiltonian_family='canonical',
                         coupling_normalization='per_instance_minmax'):
    """Run sampling for a specific Krylov dimension on the requested splits."""
    print(f"\n===== Sampling for Krylov Dimension D = {krylov_dim} =====")
    condition_records = []

    for split in splits:
        split_data_dir = os.path.join(data_root, data_subdir, split)
        if not os.path.exists(split_data_dir):
            print(f"  {split} data dir not found: {split_data_dir}, skipping")
            continue

        if hamiltonian_family == XXZ_SPARSITY_FAMILY:
            metadata = validate_xxz_sparsity_metadata(split_data_dir)
            if int(metadata['n_qubit']) != n_qubit:
                raise ValueError(
                    f"Dataset n_qubit={metadata['n_qubit']}, expected {n_qubit}")
            dt_by_id = xxz_instance_dt_by_id(split_data_dir)
        else:
            validate_protocol_metadata(
                split_data_dir, task=task, lattice=lattice, n_qubit=n_qubit)
            dt_by_id = None
        couplings, couplings_ids = load_couplings(split_data_dir)
        all_samples = {}

        print(f"  Generating {snapshots} samples for {len(couplings_ids)} {split} Hamiltonians...")

        for i, hid in enumerate(tqdm(couplings_ids, desc=f"{split} D={krylov_dim}")):
            coupling = couplings[i:i+1]

            # Use the same physical-time condition as generation and training.
            if hamiltonian_family == XXZ_SPARSITY_FAMILY:
                if hid not in dt_by_id:
                    raise ValueError(f"Missing per-instance dt for id={hid}")
                condition_dt = dt_by_id[hid]
                current_time = krylov_dim * condition_dt
            else:
                if lattice == '2d' or task == SQKD_TASK:
                    H_op = build_2d_xxz_hamiltonian(
                        N=n_qubit, J=coupling[0], delta=delta, h_z=h_z)
                else:
                    H_op = build_hamiltonian_from_couplings(
                        n_qubits=n_qubit, J=coupling[0])
                dt_override = sqkd_dt if task == SQKD_TASK else None
                dt = evolution_timestep(H_op, override=dt_override)
                condition_dt = dt
                current_time = physical_krylov_time(task, krylov_dim, dt)

            condition_records.append({
                'split': split,
                'hamiltonian_id': int(hid),
                'krylov_dim': int(krylov_dim),
                'dt': float(condition_dt),
                'condition_time': float(current_time),
            })

            samples_single = generate_samples(
                model, coupling, [hid], snapshots,
                device, rng, gcn_features, noise_factor=current_time, task=task,
                coupling_normalization=coupling_normalization,
            )
            all_samples.update(samples_single)

        # Save samples
        save_dir = os.path.join(results_dir, 'samples', split, 'model')
        os.makedirs(save_dir, exist_ok=True)
        for k, s in all_samples.items():
            compact_samples = measurement_samples_uint8(s, n_qubit=n_qubit, task=task)
            np.save(
                os.path.join(save_dir, f'model_samples_D{krylov_dim}_{snapshots}_id{k}.npy'),
                compact_samples,
            )
            original_index = couplings_ids.index(k)
            np.save(os.path.join(save_dir, f'coupling_matrix_id{k}.npy'), couplings[original_index])

        print(f"  {split} samples for D={krylov_dim} saved to {save_dir}")
    return condition_records


def main():
    args = parse_args()

    # Load config from training results
    config_path = os.path.join(args.results_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    sample_seed = cfg.get('seed', 0) if args.seed is None else args.seed
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed=sample_seed)

    dec = cfg['decoder']
    enc = cfg['encoder']
    data_cfg = cfg['data']
    hamiltonian_family = data_cfg.get('hamiltonian_family', 'canonical')
    coupling_normalization = data_cfg.get(
        'coupling_normalization', 'per_instance_minmax')
    if hamiltonian_family == XXZ_SPARSITY_FAMILY and coupling_normalization != 'none':
        raise ValueError(
            "XXZ sparsity checkpoints must use coupling_normalization='none'")
    config_protocol_version = data_cfg.get('protocol_version')
    if config_protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f'Checkpoint protocol_version={config_protocol_version!r}, expected '
            f'{PROTOCOL_VERSION}; v2 checkpoints cannot sample canonical v3 data')
    task = data_cfg['task']
    n_outcomes = measurement_outcomes_for_task(task)
    if data_cfg.get('n_outcomes') != n_outcomes:
        raise ValueError(
            f'Checkpoint n_outcomes={data_cfg.get("n_outcomes")!r}, but task '
            f'{task!r} requires {n_outcomes}; retrain with the canonical v3 vocabulary')

    n_qubit = args.n_qubit if args.n_qubit is not None else data_cfg['n_qubit']
    sample_struct = get_sample_structure(version=data_cfg['sample_structure'])
    d_model = dec['d_model']
    in_node_dim = 1 if enc['features'] == WDEGREE_FEATURE else n_qubit

    # Build encoder
    encoder = get_graph_encoder(
        enc['arch'], in_node_dim=in_node_dim, gcn_dim=enc['gcn_dim'],
        gcn_layers=enc['gcn_layers'], d_model=d_model, qubits=n_qubit
    )

    # Build model
    model = build_gc_model(
        decoder_type=dec['type'],
        n_outcomes=n_outcomes,
        encoder=encoder,
        n_layers=dec['n_layers'],
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=dec.get('n_heads', 4),
        dropout=dec['dropout'],
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift,
        d_state=dec.get('d_state', 16),
        d_conv=dec.get('d_conv', 4),
        expand=dec.get('expand', 2),
    )

    # Load checkpoint
    ckpt_fp = os.path.join(args.results_dir, 'checkpoints', args.checkpoint)
    print(f"Loading model weights from {ckpt_fp}...")
    ckpt = torch.load(ckpt_fp, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    print("Weights loaded successfully.")

    # Determine Krylov dimensions
    if args.krylov_dims is not None:
        krylov_dims = args.krylov_dims
    elif args.krylov_dim_range is not None:
        krylov_dims = list(range(args.krylov_dim_range))
    else:
        krylov_dims = list(range(data_cfg.get('krylov_dim', 15)))

    # Determine data subdir
    lattice = data_cfg.get('lattice', '2d' if task == SQKD_TASK else '1d')
    if task == 'sqkd':
        data_subdir = f'{n_qubit}qubits'
    else:
        data_subdir = f'{n_qubit}qubits_pennylane'

    # Read XXZ parameters from config
    delta = data_cfg.get('delta', 1.0)
    h_z = data_cfg.get('h_z', 0.0)

    data_root = args.data_root if args.data_root is not None else data_cfg['data_root']
    output_results_dir = args.output_results_dir if args.output_results_dir is not None else args.results_dir
    sqkd_dt = args.sqkd_dt if args.sqkd_dt is not None else data_cfg.get('sqkd_dt')

    # A separate sampling directory must remain self-describing so that the
    # evaluator can validate checkpoint compatibility without reaching back to
    # the training run.  Also record the actual data override and dimensions;
    # these are experiment provenance rather than model hyperparameters.
    os.makedirs(output_results_dir, exist_ok=True)
    output_config_path = os.path.join(output_results_dir, 'config.json')
    if os.path.abspath(output_config_path) != os.path.abspath(config_path):
        with open(output_config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
    sampling_metadata = {
        'source_results_dir': os.path.abspath(args.results_dir),
        'checkpoint': args.checkpoint,
        'checkpoint_path': os.path.abspath(ckpt_fp),
        'checkpoint_sha256': sha256_file(ckpt_fp),
        'checkpoint_epoch': ckpt.get('epoch'),
        'data_root': os.path.abspath(data_root),
        'output_results_dir': os.path.abspath(output_results_dir),
        'snapshots': args.snapshots,
        'seed': sample_seed,
        'krylov_dims': krylov_dims,
        'splits': args.splits,
        'sqkd_dt': sqkd_dt,
        'hamiltonian_family': hamiltonian_family,
        'coupling_normalization': coupling_normalization,
    }
    with open(os.path.join(output_results_dir, 'sampling_config.json'), 'w') as f:
        json.dump(sampling_metadata, f, indent=2)

    # Run sampling for each Krylov dimension
    condition_records = []
    for dim in krylov_dims:
        condition_records.extend(run_sampling_for_dim(
            model=model,
            results_dir=output_results_dir,
            data_root=data_root,
            data_subdir=data_subdir,
            snapshots=args.snapshots,
            rng=rng,
            gcn_features=enc['features'],
            device=device,
            krylov_dim=dim,
            n_qubit=n_qubit,
            task=task,
            lattice=lattice,
            delta=delta,
            h_z=h_z,
            splits=args.splits,
            sqkd_dt=sqkd_dt,
            hamiltonian_family=hamiltonian_family,
            coupling_normalization=coupling_normalization,
        ))
    with open(os.path.join(output_results_dir, 'sampling_conditions.json'), 'w') as f:
        json.dump(condition_records, f, indent=2)


if __name__ == '__main__':
    main()
