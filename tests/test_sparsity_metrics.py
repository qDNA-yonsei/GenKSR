import numpy as np

from genksr.evaluation.sparsity import (
    basis_state_probability,
    effective_support,
    exact_krylov_states,
    fixed_hamming_weight_basis,
    l_alpha,
    lowest_eigenpairs,
    sample_sector_bitstrings,
    sparsity_metrics,
)


def test_fixed_hamming_weight_basis_is_sorted_and_complete():
    basis = fixed_hamming_weight_basis(4, num_ones=2)
    np.testing.assert_array_equal(basis, [3, 5, 6, 9, 10, 12])


def test_product_state_has_unit_support():
    probabilities = np.asarray([1.0, 0.0, 0.0, 0.0])
    metrics = sparsity_metrics(probabilities)
    assert metrics["l95"] == 1
    assert metrics["l99"] == 1
    assert metrics["effective_support"] == 1.0


def test_uniform_distribution_has_expected_support():
    probabilities = np.full(20, 1.0 / 20)
    assert l_alpha(probabilities, 0.95) == 19
    assert np.isclose(effective_support(probabilities), 20.0)
    metrics = sparsity_metrics(probabilities, sector_size=20)
    assert metrics["l95"] == 19
    assert np.isclose(metrics["f95"], 0.95)


def test_metrics_normalize_unnormalized_probabilities():
    probabilities = np.asarray([6.0, 3.0, 1.0])
    assert l_alpha(probabilities, 0.9) == 2
    assert np.isclose(basis_state_probability(probabilities, [1, 2, 4], 2), 0.3)
    assert basis_state_probability(probabilities, [1, 2, 4], 3) == 0.0


def test_lowest_eigenpairs_are_sorted():
    matrix = np.diag([2.0, -1.0, 3.0])
    values, vectors = lowest_eigenpairs(matrix, k=2)
    np.testing.assert_allclose(values, [-1.0, 2.0])
    assert vectors.shape == (3, 2)


def test_exact_krylov_states_start_at_reference_and_preserve_norm():
    from scipy import sparse
    hamiltonian = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    states = exact_krylov_states(
        hamiltonian, initial_state=[1.0, 0.0], depth=4, dt=0.2)
    np.testing.assert_allclose(states[0], [1.0, 0.0])
    np.testing.assert_allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0)


def test_sector_sampling_returns_displayed_order_half_filled_bits():
    basis = fixed_hamming_weight_basis(4, 2)
    state = np.zeros(len(basis), dtype=complex)
    # Integer 10 is displayed-order bitstring 1010.
    state[np.searchsorted(basis, 10)] = 1.0
    samples = sample_sector_bitstrings(
        state, basis, n_qubit=4, shots=7, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(samples, np.tile([1, 0, 1, 0], (7, 1)))
    assert samples.dtype == np.uint8
