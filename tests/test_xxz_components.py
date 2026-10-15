import numpy as np
import pytest

from genksr.evaluation.skqd import (
    projected_xxz_hamiltonian,
    projected_xxz_hamiltonian_from_couplings,
)
from genksr.protocol import xxz_chain_pauli_l1_bound, xxz_chain_timestep
from genksr.evaluation.sparsity import (
    fixed_hamming_weight_basis,
    xxz_open_chain_sector_components,
)


def _edge(weight):
    matrix = np.zeros((2, 2), dtype=float)
    matrix[0, 1] = matrix[1, 0] = weight
    return matrix


def test_separate_projected_xxz_has_expected_two_qubit_matrix():
    projected = projected_xxz_hamiltonian_from_couplings(
        basis_states=[0, 1, 2, 3],
        jxy_coupling_matrix=_edge(2.0),
        jzz_coupling_matrix=_edge(3.0),
        n_qubit=2,
    )
    np.testing.assert_allclose(
        projected.toarray(),
        [[3.0, 0.0, 0.0, 0.0],
         [0.0, -3.0, 4.0, 0.0],
         [0.0, 4.0, -3.0, 0.0],
         [0.0, 0.0, 0.0, 3.0]],
    )


def test_separate_projection_matches_legacy_delta_parameterization():
    basis = [0, 1, 2, 3]
    coupling = _edge(0.7)
    legacy = projected_xxz_hamiltonian(
        basis, coupling, n_qubit=2, delta=1.4, h_z=0.2)
    separate = projected_xxz_hamiltonian_from_couplings(
        basis,
        jxy_coupling_matrix=coupling,
        jzz_coupling_matrix=1.4 * coupling,
        n_qubit=2,
        z_fields=np.full(2, 0.2),
    )
    np.testing.assert_allclose(separate.toarray(), legacy.toarray())


def test_qiskit_separate_builder_matches_projected_matrix():
    pytest.importorskip("qiskit")
    from genksr.utils.quantum import build_xxz_hamiltonian

    operator = build_xxz_hamiltonian(
        N=2,
        jxy_couplings=_edge(2.0),
        jzz_couplings=_edge(3.0),
    )
    projected = projected_xxz_hamiltonian_from_couplings(
        [0, 1, 2, 3], _edge(2.0), _edge(3.0), n_qubit=2)
    np.testing.assert_allclose(operator.to_matrix(), projected.toarray())


def test_uniform_open_chain_l1_and_timestep():
    bound = xxz_chain_pauli_l1_bound(n_qubits=4, j_xy=0.5, j_zz=1.0)
    assert np.isclose(bound, 6.0)
    assert np.isclose(xxz_chain_timestep(4, 0.5, 1.0), np.pi / 6.0)


def test_open_chain_l1_and_timestep_include_site_fields():
    fields = np.asarray([-0.1, 0.0, 0.0, 0.0])
    bound = xxz_chain_pauli_l1_bound(
        n_qubits=4, j_xy=0.5, j_zz=1.0, z_fields=fields)
    assert np.isclose(bound, 6.1)
    assert np.isclose(
        xxz_chain_timestep(4, 0.5, 1.0, z_fields=fields), np.pi / 6.1)


def test_negative_z0_pin_favors_neel_orientation_with_q0_zero():
    projected = projected_xxz_hamiltonian_from_couplings(
        basis_states=[1, 2],
        jxy_coupling_matrix=_edge(0.0),
        jzz_coupling_matrix=_edge(0.0),
        n_qubit=2,
        z_fields=np.asarray([-0.1, 0.0]),
    )
    # Integer 2 is displayed bitstring "10", the canonical N=2 Neel state;
    # its physical q0 bit is zero and therefore receives energy -0.1.
    np.testing.assert_allclose(projected.toarray(), [[0.1, 0.0], [0.0, -0.1]])


def test_separate_couplings_reject_non_symmetric_matrix():
    asymmetric = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        projected_xxz_hamiltonian_from_couplings(
            [1, 2], asymmetric, _edge(1.0), n_qubit=2)


def test_open_chain_components_match_generic_projected_builder():
    n_qubit = 4
    basis = fixed_hamming_weight_basis(n_qubit, n_qubit // 2)
    h_xy, h_zz = xxz_open_chain_sector_components(n_qubit, basis)
    jxy = np.zeros((n_qubit, n_qubit))
    jzz = np.zeros((n_qubit, n_qubit))
    for site in range(n_qubit - 1):
        jxy[site, site + 1] = jxy[site + 1, site] = 0.37
        jzz[site, site + 1] = jzz[site + 1, site] = 1.2
    generic = projected_xxz_hamiltonian_from_couplings(
        basis, jxy, jzz, n_qubit=n_qubit)
    component_form = 0.37 * h_xy + 1.2 * h_zz
    np.testing.assert_allclose(component_form.toarray(), generic.toarray())
