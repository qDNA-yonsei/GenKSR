"""Dataset-family dispatch shared by training and sampling entrypoints."""

from __future__ import annotations

import os

import numpy as np

from genksr.data.hamiltonian import load_data_for_all_dims
from genksr.data.xxz_sparsity import (
    HAMILTONIAN_FAMILY as XXZ_SPARSITY_FAMILY,
    load_xxz_sparsity_data_for_all_dims,
    read_instance_rows,
)


CANONICAL_FAMILY = "canonical"
SUPPORTED_FAMILIES = (CANONICAL_FAMILY, XXZ_SPARSITY_FAMILY)
SUPPORTED_COUPLING_NORMALIZATIONS = ("per_instance_minmax", "none")
SUPPORTED_HAMILTONIAN_SUBSET_MODES = ("random", "stratified_nested")


def validate_data_config(data_config):
    family = data_config.hamiltonian_family
    normalization = data_config.coupling_normalization
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unknown hamiltonian_family {family!r}; expected {SUPPORTED_FAMILIES}")
    if normalization not in SUPPORTED_COUPLING_NORMALIZATIONS:
        raise ValueError(
            f"Unknown coupling_normalization {normalization!r}; expected "
            f"{SUPPORTED_COUPLING_NORMALIZATIONS}")
    if data_config.hamiltonian_subset_mode not in SUPPORTED_HAMILTONIAN_SUBSET_MODES:
        raise ValueError(
            f"Unknown hamiltonian_subset_mode "
            f"{data_config.hamiltonian_subset_mode!r}; expected "
            f"{SUPPORTED_HAMILTONIAN_SUBSET_MODES}")
    if family == XXZ_SPARSITY_FAMILY:
        if data_config.task != "sqkd" or data_config.lattice != "1d":
            raise ValueError(
                "The XXZ sparsity family requires task='sqkd' and lattice='1d'")
        if normalization != "none":
            raise ValueError(
                "The XXZ sparsity family requires coupling_normalization='none' "
                "so the J_xy scale is not erased")
        if data_config.sqkd_dt is not None:
            raise ValueError(
                "The XXZ sparsity family uses per-instance dt; sqkd_dt must be null")
        if not np.isclose(data_config.delta, 1.0) or not np.isclose(
                data_config.h_z, 0.0):
            raise ValueError(
                "The pure-XXZ sparsity family requires delta=1 and h_z=0")
    return True


def data_subdirectory(data_config):
    if data_config.hamiltonian_family == XXZ_SPARSITY_FAMILY:
        return f"{data_config.n_qubit}qubits"
    if data_config.task == "sqkd":
        return f"{data_config.n_qubit}qubits"
    return f"{data_config.n_qubit}qubits_pennylane"


def load_configured_split(
        data_config, split, rng, krylov_dims, num_hamiltonians=None):
    """Load one configured split using its family's time convention."""
    validate_data_config(data_config)
    data_dir = os.path.join(
        data_config.data_root, data_subdirectory(data_config), split)
    if data_config.hamiltonian_family == XXZ_SPARSITY_FAMILY:
        return load_xxz_sparsity_data_for_all_dims(
            data_dir=data_dir,
            num_samples_per=data_config.train_samples,
            num_hamiltonians=num_hamiltonians,
            rng=rng,
            krylov_dims=krylov_dims,
            hamiltonian_subset_mode=data_config.hamiltonian_subset_mode,
            hamiltonian_subset_seed=data_config.hamiltonian_subset_seed,
            measurement_subset_seed=data_config.measurement_subset_seed,
            split=split,
        )
    return load_data_for_all_dims(
        data_dir=data_dir,
        num_samples_per=data_config.train_samples,
        num_hamiltonians=num_hamiltonians,
        rng=rng,
        krylov_dims=krylov_dims,
        n_qubit=data_config.n_qubit,
        task=data_config.task,
        lattice=data_config.lattice,
        delta=data_config.delta,
        h_z=data_config.h_z,
        sqkd_dt=data_config.sqkd_dt,
    )


def xxz_instance_dt_by_id(split_data_dir):
    """Return the frozen per-Hamiltonian timestep map for model sampling."""
    rows = read_instance_rows(split_data_dir)
    mapping = {int(row["split_id"]): float(row["dt"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Duplicate split_id in XXZ instances.csv")
    return mapping
