from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from genksr.experiments.pauli_cross_family.dataset import (
    PauliSampleDataset,
    generate_dataset,
    validate_dataset,
)
from genksr.experiments.pauli_cross_family.model import PauliConditionalARModel
from genksr.experiments.pauli_cross_family.protocol import (
    SOURCE_FAMILIES,
    TARGET_FAMILY,
    build_family_terms,
    build_protocol,
    encode_terms,
    family_support,
    load_experiment_config,
    validate_protocol,
)
from genksr.experiments.pauli_cross_family.quantum import (
    exact_krylov_probabilities,
    ground_diagnostics,
    neel_bitstring,
    pauli_sum_matrix,
    projected_ground_energy,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_cross_family_n8.yaml"
)
STAGGERED_XXZ_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_cross_family_staggered_xxz_n8.yaml"
)
TARGET_TRAINED_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_target_trained_staggered_xxz_n8.yaml"
)
N12_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_cross_family_staggered_xxz_n12.yaml"
)
N12_TARGET_TRAINED_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_target_trained_staggered_xxz_n12.yaml"
)
N12_TRANSVERSE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_cross_family_transverse_xxz_n12.yaml"
)
N12_TRANSVERSE_TARGET_TRAINED_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiment/pauli_target_trained_transverse_xxz_n12.yaml"
)


def _condition(n_qubit=4):
    terms = build_family_terms("xxz", n_qubit, {"xy": 0.8, "zz": 1.1})
    codes, coefficients, _ = encode_terms(terms)
    return (
        torch.as_tensor(codes, dtype=torch.long).unsqueeze(0),
        torch.as_tensor(coefficients, dtype=torch.float32).unsqueeze(0),
        torch.ones((1, len(terms)), dtype=torch.bool),
    )


def test_protocol_has_strict_target_template_holdout():
    config = load_experiment_config(CONFIG, smoke=True)
    protocol = build_protocol(config)
    assert validate_protocol(protocol)
    n_qubit = config["data"]["n_qubit"]
    source_union = frozenset().union(
        *(family_support(family, n_qubit) for family in SOURCE_FAMILIES)
    )
    target = family_support(TARGET_FAMILY, n_qubit)
    assert target <= source_union
    assert all(target != family_support(family, n_qubit) for family in SOURCE_FAMILIES)
    assert all(
        row["split"] == "target_test"
        for row in protocol["instances"]
        if row["family"] == TARGET_FAMILY
    )


def test_protocol_supports_a_configured_heldout_target_family():
    config = load_experiment_config(STAGGERED_XXZ_CONFIG, smoke=True)
    protocol = build_protocol(config)
    assert validate_protocol(protocol)
    assert protocol["target_family"] == "staggered_field_xxz"
    assert protocol["source_families"] == list(SOURCE_FAMILIES)
    target_rows = [
        row for row in protocol["instances"] if row["split"] == "target_test"
    ]
    assert target_rows
    assert {row["family"] for row in target_rows} == {"staggered_field_xxz"}
    n_qubit = config["data"]["n_qubit"]
    source_union = frozenset().union(
        *(family_support(family, n_qubit) for family in protocol["source_families"])
    )
    target = family_support(protocol["target_family"], n_qubit)
    assert target <= source_union
    assert all(
        target != family_support(family, n_qubit)
        for family in protocol["source_families"]
    )


def test_target_trained_control_is_budget_matched_and_reuses_only_the_test_manifest():
    cross = build_protocol(load_experiment_config(STAGGERED_XXZ_CONFIG))
    control = build_protocol(load_experiment_config(TARGET_TRAINED_CONFIG))
    assert validate_protocol(control)
    assert control["evaluation_regime"] == "in_family_target_trained_control"
    assert control["target_template_seen_in_source"] is True
    assert control["source_families"] == ["staggered_field_xxz"]

    cross_test = [row for row in cross["instances"] if row["split"] == "target_test"]
    control_test = [
        row for row in control["instances"] if row["split"] == "target_test"
    ]
    assert control_test == cross_test

    train = [row for row in control["instances"] if row["split"] == "source_train"]
    assert len(train) == 192
    assert len(control_test) == 64
    train_parameters = {tuple(sorted(row["parameters"].items())) for row in train}
    test_parameters = {
        tuple(sorted(row["parameters"].items())) for row in control_test
    }
    assert train_parameters.isdisjoint(test_parameters)
    data = control["config"]["data"]
    assert len(train) * data["depth"] * data["train_shots_per_condition"] == 768_000


def test_n12_protocols_are_budget_matched_and_share_target_manifest():
    cross = build_protocol(load_experiment_config(N12_CONFIG))
    control = build_protocol(load_experiment_config(N12_TARGET_TRAINED_CONFIG))
    assert validate_protocol(cross)
    assert validate_protocol(control)
    cross_test = [row for row in cross["instances"] if row["split"] == "target_test"]
    control_test = [
        row for row in control["instances"] if row["split"] == "target_test"
    ]
    assert cross_test == control_test
    assert len(cross_test) == 32
    assert {row["num_terms"] for row in cross_test} == {45}
    cross_train = [row for row in cross["instances"] if row["split"] == "source_train"]
    control_train = [
        row for row in control["instances"] if row["split"] == "source_train"
    ]
    cross_data = cross["config"]["data"]
    control_data = control["config"]["data"]
    cross_budget = (
        len(cross_train)
        * cross_data["depth"]
        * cross_data["train_shots_per_condition"]
    )
    control_budget = (
        len(control_train)
        * control_data["depth"]
        * control_data["train_shots_per_condition"]
    )
    assert cross_budget == control_budget == 768_000


def test_n12_transverse_target_is_a_budget_matched_unseen_composition():
    cross = build_protocol(load_experiment_config(N12_TRANSVERSE_CONFIG))
    control = build_protocol(
        load_experiment_config(N12_TRANSVERSE_TARGET_TRAINED_CONFIG)
    )
    assert validate_protocol(cross)
    assert validate_protocol(control)
    assert cross["target_family"] == "transverse_field_xxz"
    assert cross["target_template_seen_in_source"] is False
    assert cross["config"]["evaluation"]["random_baselines"] == ["uniform_full"]

    n_qubit = cross["config"]["data"]["n_qubit"]
    target = family_support("transverse_field_xxz", n_qubit)
    source_union = frozenset().union(
        *(family_support(family, n_qubit) for family in cross["source_families"])
    )
    assert target <= source_union
    assert all(
        target != family_support(family, n_qubit)
        for family in cross["source_families"]
    )

    cross_test = [row for row in cross["instances"] if row["split"] == "target_test"]
    control_test = [
        row for row in control["instances"] if row["split"] == "target_test"
    ]
    assert cross_test == control_test
    assert len(cross_test) == 32
    assert {row["num_terms"] for row in cross_test} == {45}

    cross_train = [
        row for row in cross["instances"] if row["split"] == "source_train"
    ]
    control_train = [
        row for row in control["instances"] if row["split"] == "source_train"
    ]
    cross_data = cross["config"]["data"]
    control_data = control["config"]["data"]
    cross_budget = (
        len(cross_train)
        * cross_data["depth"]
        * cross_data["train_shots_per_condition"]
    )
    control_budget = (
        len(control_train)
        * control_data["depth"]
        * control_data["train_shots_per_condition"]
    )
    assert cross_budget == control_budget == 768_000


def test_exact_evolution_uses_displayed_neel_bit_order():
    n_qubit = 4
    terms = build_family_terms("tfim", n_qubit, {"zz": 1.0, "x": 0.7})
    hamiltonian = pauli_sum_matrix(terms)
    np.testing.assert_allclose(hamiltonian.toarray(), hamiltonian.toarray().conj().T)
    probabilities, error = exact_krylov_probabilities(
        hamiltonian, n_qubit=n_qubit, depth=3, dt=0.1
    )
    expected = np.zeros(1 << n_qubit)
    expected[int("".join(map(str, neel_bitstring(n_qubit))), 2)] = 1.0
    np.testing.assert_allclose(probabilities[0], expected, atol=1e-12)
    assert error < 1e-9


def test_pauli_set_encoder_is_term_permutation_invariant():
    torch.manual_seed(0)
    model = PauliConditionalARModel(
        n_qubit=4,
        d_model=32,
        decoder_layers=2,
        decoder_heads=4,
        encoder_layers=2,
        encoder_heads=4,
        symbol_dim=8,
        dropout=0.0,
    ).eval()
    codes, coefficients, mask = _condition(4)
    permutation = torch.randperm(codes.shape[1])
    with torch.no_grad():
        original = model.hamiltonian_encoder(codes, coefficients, mask)
        shuffled = model.hamiltonian_encoder(
            codes[:, permutation], coefficients[:, permutation], mask[:, permutation]
        )
    torch.testing.assert_close(original, shuffled, rtol=1e-5, atol=1e-6)


def test_coefficient_sum_encoder_is_permutation_invariant_and_coefficient_sensitive():
    torch.manual_seed(3)
    model = PauliConditionalARModel(
        n_qubit=4,
        d_model=32,
        decoder_layers=1,
        decoder_heads=4,
        encoder_layers=1,
        encoder_heads=4,
        symbol_dim=8,
        dropout=0.0,
        encoder_mode="coefficient_sum",
    ).eval()
    codes, coefficients, mask = _condition(4)
    permutation = torch.randperm(codes.shape[1])
    with torch.no_grad():
        original = model.hamiltonian_encoder(codes, coefficients, mask)
        shuffled = model.hamiltonian_encoder(
            codes[:, permutation], coefficients[:, permutation], mask[:, permutation]
        )
        reassigned = model.hamiltonian_encoder(
            codes, coefficients.flip(dims=(1,)), mask
        )
    torch.testing.assert_close(original, shuffled, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(original, reassigned, rtol=1e-4, atol=1e-5)


def test_autoregressive_distribution_normalizes_exactly():
    torch.manual_seed(1)
    model = PauliConditionalARModel(
        n_qubit=4,
        d_model=32,
        decoder_layers=1,
        decoder_heads=4,
        encoder_layers=1,
        encoder_heads=4,
        symbol_dim=4,
        dropout=0.0,
    ).eval()
    codes, coefficients, mask = _condition(4)
    probability = model.exact_distribution(
        pauli_codes=codes,
        coefficients=coefficients,
        term_mask=mask,
        tau=torch.as_tensor([np.pi], dtype=torch.float32),
        batch_size=16,
    )
    assert probability.shape == (16,)
    assert np.all(probability > 0.0)
    assert np.isclose(probability.sum(), 1.0, atol=1e-7)


def test_full_basis_projected_energy_matches_exact():
    terms = build_family_terms(
        "mixed_field_xxz",
        4,
        {"xy": 0.8, "zz": 1.0, "x": 0.4, "zs": 0.3},
    )
    hamiltonian = pauli_sum_matrix(terms)
    exact = float(np.linalg.eigvalsh(hamiltonian.toarray())[0])
    projected = projected_ground_energy(hamiltonian, range(16))
    assert np.isclose(projected, exact, atol=1e-10)


def test_sparse_ground_diagnostics_for_large_diagonal_hamiltonian():
    n_qubit = 10
    terms = []
    for site in range(n_qubit):
        symbols = ["I"] * n_qubit
        symbols[site] = "Z"
        terms.append(("".join(symbols), -1.0))
    hamiltonian = pauli_sum_matrix(terms)
    diagnostics = ground_diagnostics(hamiltonian, n_qubit=n_qubit)
    assert np.isclose(diagnostics["ground_energy"], -n_qubit, atol=1e-9)
    assert diagnostics["ground_space_degeneracy"] == 1
    assert np.isclose(diagnostics["spectral_gap_above_ground_space"], 2.0, atol=1e-8)


def test_tiny_dataset_round_trip(tmp_path):
    config = load_experiment_config(CONFIG, smoke=True)
    config = deepcopy(config)
    config["data"].update(
        {
            "n_qubit": 4,
            "depth": 2,
            "train_shots_per_condition": 4,
            "source_train_per_family": 1,
            "source_holdout_per_family": 1,
            "target_test_hamiltonians": 1,
        }
    )
    protocol = build_protocol(config)
    report = generate_dataset(protocol, tmp_path, workers=1)
    assert report["num_instances"] == 7
    assert validate_dataset(tmp_path, verify_hashes=True)["hashes_verified"]
    dataset = PauliSampleDataset(tmp_path)
    assert len(dataset) == 3 * 2 * 4
    batch = dataset.collate([0, 1, 2])
    assert batch["bits"].shape == (3, 4)
    assert batch["pauli_codes"].shape[0] == 3
