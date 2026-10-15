from pathlib import Path

import numpy as np
import pytest

from genksr.config import load_config
from genksr.data.factory import (
    XXZ_SPARSITY_FAMILY,
    data_subdirectory,
    load_configured_split,
    validate_data_config,
    xxz_instance_dt_by_id,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("n_qubit", [12, 16, 20])
def test_xxz_transformer_configs_are_explicit_and_valid(n_qubit):
    config_path = ROOT / "configs" / "experiment" / (
        f"skqd_1d_xxz_sparsity_n{n_qubit}_transformer.yaml")
    config = load_config(
        str(config_path),
        base_yaml=str(ROOT / "configs" / "defaults.yaml"),
    )
    assert config.data.n_qubit == n_qubit
    assert config.data.hamiltonian_family == XXZ_SPARSITY_FAMILY
    assert config.data.coupling_normalization == "none"
    assert config.data.eval_validation is True
    assert config.data.eval_test is False
    assert data_subdirectory(config.data) == f"{n_qubit}qubits"
    assert validate_data_config(config.data)


def test_xxz_family_rejects_scale_erasing_normalization():
    config_path = ROOT / "configs" / "experiment" / (
        "skqd_1d_xxz_sparsity_n12_transformer.yaml")
    config = load_config(
        str(config_path),
        base_yaml=str(ROOT / "configs" / "defaults.yaml"),
    )
    config.data.coupling_normalization = "per_instance_minmax"
    with pytest.raises(ValueError, match="J_xy scale"):
        validate_data_config(config.data)


def test_family_loader_preserves_raw_jxy_and_per_instance_times():
    config_path = ROOT / "configs" / "experiment" / (
        "skqd_1d_xxz_sparsity_n12_transformer.yaml")
    config = load_config(
        str(config_path),
        base_yaml=str(ROOT / "configs" / "defaults.yaml"),
    )
    config.data.data_root = str(ROOT / "data" / "xxz_sparsity_1d")
    config.data.train_samples = 2
    measurements, couplings, ids, times = load_configured_split(
        config.data,
        split="train",
        rng=np.random.default_rng(9),
        krylov_dims=[0, 14],
        num_hamiltonians=None,
    )
    assert measurements.shape == (45 * 2 * 2, 12)
    assert couplings.shape == (45, 12, 12)
    assert len(np.unique(ids)) == 45
    assert np.all(times[:90] == 0.0)
    assert len(np.unique(times[90:])) == 45
    edge_values = couplings[:, 0, 1]
    assert len(np.unique(edge_values)) == 45

    dt_by_id = xxz_instance_dt_by_id(
        ROOT / "data" / "xxz_sparsity_1d" / "12qubits" / "train")
    assert len(dt_by_id) == 45
    for identifier in range(45):
        selected_times = times[90:][ids[90:] == identifier]
        np.testing.assert_allclose(selected_times, 14 * dt_by_id[identifier])


def test_stratified_hamiltonian_and_measurement_subsets_are_nested():
    config_path = ROOT / "configs" / "experiment" / (
        "skqd_1d_xxz_sparsity_n12_transformer.yaml")
    config = load_config(
        str(config_path),
        base_yaml=str(ROOT / "configs" / "defaults.yaml"),
    )
    config.data.data_root = str(ROOT / "data" / "xxz_sparsity_1d")
    config.data.hamiltonian_subset_mode = "stratified_nested"
    config.data.hamiltonian_subset_seed = 71
    config.data.measurement_subset_seed = 93

    def load(num_hamiltonians, samples):
        config.data.train_samples = samples
        return load_configured_split(
            config.data,
            split="train",
            rng=np.random.default_rng(999),
            krylov_dims=[1],
            num_hamiltonians=num_hamiltonians,
        )

    measurements_9_125, _, ids_9_125, _ = load(9, 125)
    measurements_18_125, _, ids_18_125, _ = load(18, 125)
    measurements_9_250, _, ids_9_250, _ = load(9, 250)

    selected_9 = set(map(int, np.unique(ids_9_125)))
    selected_18 = set(map(int, np.unique(ids_18_125)))
    assert selected_9 < selected_18
    assert sum(identifier < 15 for identifier in selected_9) == 3
    assert sum(15 <= identifier < 30 for identifier in selected_9) == 3
    assert sum(identifier >= 30 for identifier in selected_9) == 3

    for block_index, identifier in enumerate(sorted(selected_9)):
        assert np.all(ids_9_125[block_index * 125:(block_index + 1) * 125] == identifier)
        assert np.all(ids_9_250[block_index * 250:(block_index + 1) * 250] == identifier)
        np.testing.assert_array_equal(
            measurements_9_125[block_index * 125:(block_index + 1) * 125],
            measurements_9_250[block_index * 250:block_index * 250 + 125],
        )
