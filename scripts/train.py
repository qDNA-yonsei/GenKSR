"""Unified training script for the canonical GenKSR v3 protocol.

Usage:
    python scripts/train.py --config configs/experiment/sqkd_transformer.yaml
    python scripts/train.py --config configs/experiment/sqkd_mamba.yaml --training.lr 5e-4
    python scripts/train.py --config configs/experiment/kqd_1d_transformer.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from genksr.config import load_config, config_to_dict
from genksr.constants import WDEGREE_FEATURE, measurement_outcomes_for_task
from genksr.data.dataset import DatasetGCModel
from genksr.data.factory import (
    CANONICAL_FAMILY,
    XXZ_SPARSITY_FAMILY,
    load_configured_split,
    validate_data_config,
)
from genksr.data.sample_structure import get_sample_structure
from genksr.nn.encoder import get_graph_encoder
from genksr.nn.gc_model import build_gc_model
from genksr.training import Trainer, dir_setup, Logger
from genksr.utils import timestamp


def parse_args():
    parser = argparse.ArgumentParser(description='GenKSR v3 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                val = unknown[i + 1]
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        if val.lower() in ('true', 'false'):
                            val = val.lower() == 'true'
                overrides[key] = val
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1

    return args.config, overrides


def print_dict(d, tag=None):
    print(f'--------{tag or ""}----------')
    for k, v in d.items():
        print('{0:27}: {1}'.format(k, v))
    print(f'--------{tag or ""}----------\n')


def main():
    config_path, overrides = parse_args()

    # Layer shared defaults (configs/defaults.yaml) under the experiment config so the
    # resolved config comes entirely from the YAML files (+ optional CLI), never from
    # hidden dataclass-only values. defaults.yaml is the sibling of the experiment/ dir.
    defaults_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(config_path))),
                                 'defaults.yaml')
    base_yaml = defaults_yaml if os.path.exists(defaults_yaml) else None
    cfg = load_config(config_path, overrides, base_yaml=base_yaml)

    if cfg.debug:
        cfg.data.train_samples = 20
        cfg.training.iterations = 2
        cfg.training.batch_size = 20

    # Seed model initialization, dropout, and autoregressive CUDA/CPU operations.
    # The NumPy generator below controls dataset subsampling and shuffling; both
    # RNG families must be seeded for a repeatable experiment.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    n_outcomes = measurement_outcomes_for_task(cfg.data.task)
    cfg.data.n_outcomes = n_outcomes
    validate_data_config(cfg.data)

    print(f"\n{'='*50}")
    print(f"[{timestamp()}] GenKSR v3 Training")
    print(f"  Decoder: {cfg.decoder.type} | Task: {cfg.data.task} | Lattice: {cfg.data.lattice}")
    print(f"  Krylov dims: 0 to {cfg.data.krylov_dim - 1}")
    print(f"{'='*50}\n")

    # --- Build model_id and dataset_id based on task ---
    if cfg.decoder.type == 'mamba':
        model_id = (f'{cfg.encoder.arch}-mamba_l{cfg.decoder.n_layers}_d{cfg.decoder.d_model}'
                    f'_feat{cfg.encoder.features}_D_conditional')
    else:
        model_id = (f'{cfg.encoder.arch}-{cfg.decoder.type}_l{cfg.decoder.n_layers}'
                    f'_d{cfg.decoder.d_model}_h{cfg.decoder.n_heads}'
                    f'_feat{cfg.encoder.features}_D_conditional')

    train_id = (f'iter{cfg.training.iterations}_lr{cfg.training.lr}_wd{cfg.training.weight_decay}'
                f'_bs{cfg.training.batch_size}_dropout{cfg.decoder.dropout}'
                f'_samplestruct{cfg.data.sample_structure}_lrschedule{cfg.training.lr_scheduler}')

    if cfg.decoder.type == 'mamba':
        train_id += f'_dstate{cfg.decoder.d_state}_dconv{cfg.decoder.d_conv}_expand{cfg.decoder.expand}'
    if cfg.data.hamiltonian_family != CANONICAL_FAMILY:
        train_id += f'_cnorm{cfg.data.coupling_normalization}'
    # Keep independent random initializations in separate result directories.
    # The current formal study uses seed 0 only, but including it here prevents
    # accidental overwrites if additional seeds are run later.
    train_id += f'_seed{cfg.seed}'

    # Dataset ID
    if cfg.data.hamiltonian_family == XXZ_SPARSITY_FAMILY:
        state_type = 'conditional_xxz_sparsity_SQKD_1d'
        dataset_id = f'{state_type}_{cfg.data.n_qubit}qubit'
    elif cfg.data.task == 'sqkd':
        state_type = 'conditional_heisenberg_SQKD_2d'
        dataset_id = f'{state_type}_{cfg.data.n_qubit}qubit'
    elif cfg.data.lattice == '2d':
        state_type = 'conditional_heisenberg_KQD_2d'
        dataset_id = f'{state_type}_{cfg.data.n_qubit}qubit_pennylane'
    else:
        state_type = 'conditional_heisenberg'
        dataset_id = f'{state_type}_{cfg.data.n_qubit}qubit_pennylane'

    results_dir, _, train_id = dir_setup(
        results_root=cfg.results_dir, dataset_id=dataset_id, model_id=model_id,
        num_train=f'ns{cfg.data.train_samples}', train_id=train_id, verbose=cfg.verbose
    )

    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'train_out.txt'))

    # Save config
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print_dict(config_to_dict(cfg), tag='config')

    # Setup
    rng = np.random.default_rng(seed=cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_struct = get_sample_structure(version=cfg.data.sample_structure)

    d_model = cfg.decoder.d_model
    in_node_dim = 1 if cfg.encoder.features == WDEGREE_FEATURE else cfg.data.n_qubit

    # Build encoder
    encoder = get_graph_encoder(
        cfg.encoder.arch, in_node_dim=in_node_dim, gcn_dim=cfg.encoder.gcn_dim,
        gcn_layers=cfg.encoder.gcn_layers, d_model=d_model, qubits=cfg.data.n_qubit
    )

    # Build model
    model = build_gc_model(
        decoder_type=cfg.decoder.type,
        n_outcomes=n_outcomes,
        encoder=encoder,
        n_layers=cfg.decoder.n_layers,
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=cfg.decoder.n_heads,
        dropout=cfg.decoder.dropout,
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift,
        d_state=cfg.decoder.d_state,
        d_conv=cfg.decoder.d_conv,
        expand=cfg.decoder.expand,
    )
    model.to(device)

    # --- Load data ---
    krylov_dims = range(cfg.data.krylov_dim)

    def build_dataset(split, num_hamiltonians=None):
        measurements, coupling_matrices, hamiltonian_ids, condition_times = (
            load_configured_split(
                cfg.data,
                split=split,
                rng=rng,
                krylov_dims=krylov_dims,
                num_hamiltonians=num_hamiltonians,
            )
        )
        dataset = DatasetGCModel(
            measurements=measurements,
            coupling_matrices=coupling_matrices,
            hamiltonian_ids=hamiltonian_ids,
            noise_factors=condition_times,
            rng=rng,
            device=device,
            nodes_feature_type=cfg.encoder.features,
            start_token=sample_struct.start_token,
            end_token=sample_struct.end_token,
            token_shift=sample_struct.token_shift,
            n_outcomes=n_outcomes,
            coupling_normalization=cfg.data.coupling_normalization,
        )
        return dataset, measurements, hamiltonian_ids

    train_dataset, train_measurements, train_hamiltonian_ids = build_dataset(
        'train', num_hamiltonians=cfg.data.hamiltonians)
    val_dataset = None
    if cfg.data.eval_validation:
        val_dataset, _, _ = build_dataset('validation')
    test_dataset = None
    if cfg.data.eval_test:
        test_dataset, _, _ = build_dataset('test')

    # Save training Hamiltonian IDs
    train_ids_unique = list(map(int, np.unique(train_hamiltonian_ids)))
    with open(os.path.join(results_dir, 'train_hamiltonian_ids.json'), 'w') as f:
        json.dump(train_ids_unique, f)

    print(f'* Total training samples: {len(train_measurements)}')
    print(f'* Training Hamiltonians: {train_ids_unique}')
    print(f'* Validation samples: {0 if val_dataset is None else len(val_dataset)}')
    print(f'* Test samples: {0 if test_dataset is None else len(test_dataset)}')

    # Train
    print(f'[{timestamp()}] Starting training, saving to {results_dir}')
    trainer = Trainer(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset,
        test_dataset=test_dataset, save_dir=results_dir, iterations=cfg.training.iterations,
        lr=cfg.training.lr, final_lr=cfg.training.final_lr, lr_scheduler=cfg.training.lr_scheduler,
        warmup_frac=cfg.training.warmup_frac, weight_decay=cfg.training.weight_decay,
        device=device, batch_size=cfg.training.batch_size, rng=rng,
        eval_every=cfg.training.eval_every,
        checkpoint_every=cfg.training.checkpoint_every,
        epoch_mode=cfg.training.epoch_mode
    )

    train_loss, val_loss, test_loss = trainer.train()
    print(f'[{timestamp()}] Training complete: train={train_loss:.4f}, val={val_loss:.4f}, test={test_loss:.4f}')

    trainer.save_model('final', is_best=False)


if __name__ == '__main__':
    main()
