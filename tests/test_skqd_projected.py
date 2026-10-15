import numpy as np
import pytest

from genksr.evaluation.skqd import (
    projected_xxz_energy,
    projected_xxz_energies_per_dim,
    projected_xxz_hamiltonian,
    samples_to_basis_ints,
    skqd_energy_from_counts,
    skqd_energies_per_dim,
)


def _two_qubit_coupling():
    coupling = np.zeros((2, 2), dtype=float)
    coupling[0, 1] = coupling[1, 0] = 1.0
    return coupling


def test_samples_to_basis_ints_uses_qiskit_display_order_and_postselects():
    samples = np.asarray([
        [0, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ])
    states = samples_to_basis_ints(samples, n_qubit=2, num_ones=1)
    np.testing.assert_array_equal(states, [1, 2])


def test_projected_xxz_energy_handles_one_dimensional_subspace():
    energy = projected_xxz_energy([1], _two_qubit_coupling(), n_qubit=2)
    assert np.isclose(energy, -1.0)


def test_projected_xxz_energy_recovers_two_qubit_singlet():
    energy = projected_xxz_energy([1, 2], _two_qubit_coupling(), n_qubit=2)
    assert np.isclose(energy, -3.0)


def test_projected_xxz_hamiltonian_uses_sorted_unique_basis_order():
    projected = projected_xxz_hamiltonian(
        [2, 1, 2], _two_qubit_coupling(), n_qubit=2)
    np.testing.assert_allclose(
        projected.toarray(),
        [[-1.0, 2.0], [2.0, -1.0]],
    )


def test_projected_xxz_energy_is_variational_under_nested_basis_growth():
    small = projected_xxz_energy([1], _two_qubit_coupling(), n_qubit=2)
    large = projected_xxz_energy([1, 2], _two_qubit_coupling(), n_qubit=2)
    assert large <= small


def test_fast_cumulative_curve_can_solve_selected_dimensions_only():
    samples_by_dim = {
        0: np.asarray([[0, 1]]),
        1: np.asarray([[1, 0]]),
    }
    energies = projected_xxz_energies_per_dim(
        samples_by_dim,
        _two_qubit_coupling(),
        n_qubit=2,
        krylov_dims=range(2),
        evaluate_dims={1},
    )
    assert np.isnan(energies[0])
    assert np.isclose(energies[1], -3.0)


def test_fast_cumulative_curve_matches_qiskit_addon_curve():
    pytest.importorskip("qiskit_addon_sqd")
    pytest.importorskip("qiskit")
    from genksr.utils.quantum import build_2d_xxz_hamiltonian, convert_to_counts_dict

    coupling = np.zeros((4, 4), dtype=float)
    coupling[0, 1] = coupling[1, 0] = 1.0
    coupling[1, 2] = coupling[2, 1] = 0.7
    coupling[2, 3] = coupling[3, 2] = 1.2
    coupling[0, 3] = coupling[3, 0] = 0.4
    samples_by_dim = {
        0: np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]]),
        1: np.asarray([[0, 1, 1, 0], [1, 0, 0, 1]]),
        2: np.asarray([[1, 0, 1, 0], [1, 1, 0, 0]]),
    }
    hamiltonian = build_2d_xxz_hamiltonian(N=4, J=coupling)
    counts_by_dim = {
        d: convert_to_counts_dict(samples)
        for d, samples in samples_by_dim.items()
    }

    fast = projected_xxz_energies_per_dim(
        samples_by_dim, coupling, n_qubit=4, krylov_dims=range(3))
    addon = skqd_energies_per_dim(
        counts_by_dim, hamiltonian, n_qubit=4, krylov_dims=range(3))

    np.testing.assert_allclose(fast, addon, atol=1e-10)
    assert np.all(np.diff(fast) <= 1e-10)


def test_fast_projected_solver_matches_qiskit_addon_sqd():
    pytest.importorskip("qiskit_addon_sqd")
    pytest.importorskip("qiskit")
    from genksr.utils.quantum import build_2d_xxz_hamiltonian

    coupling = np.zeros((4, 4), dtype=float)
    coupling[0, 1] = coupling[1, 0] = 1.0
    coupling[1, 2] = coupling[2, 1] = 0.7
    coupling[2, 3] = coupling[3, 2] = 1.2
    coupling[0, 3] = coupling[3, 0] = 0.4
    basis = [3, 5, 6, 9, 10, 12]  # the complete four-qubit, two-one sector
    counts = {format(state, "04b"): 1 for state in basis}
    hamiltonian = build_2d_xxz_hamiltonian(N=4, J=coupling)

    fast = projected_xxz_energy(basis, coupling, n_qubit=4)
    addon = skqd_energy_from_counts(counts, hamiltonian, n_qubit=4)
    assert np.isclose(fast, addon, atol=1e-10)


def test_qiskit_addon_path_handles_single_basis_state():
    pytest.importorskip("qiskit_addon_sqd")
    pytest.importorskip("qiskit")
    from genksr.utils.quantum import build_2d_xxz_hamiltonian

    coupling = _two_qubit_coupling()
    hamiltonian = build_2d_xxz_hamiltonian(N=2, J=coupling)
    energy = skqd_energy_from_counts({"01": 1}, hamiltonian, n_qubit=2)
    assert np.isclose(energy, -1.0)
