import math

import numpy as np
import pytest
from scipy.linalg import expm

from genksr.protocol import (
    build_sqkd_evolution_gate,
    evolution_timestep,
    kqd_shadow_sampling_seeds,
    kqd_trotter_rotation_angle,
    neel_reference_bitstring,
    pauli_l1_bound,
    physical_krylov_time,
)


class _HamiltonianStub:
    def __init__(self, coeffs):
        self.coeffs = np.asarray(coeffs)


def test_timestep_uses_pauli_coefficient_l1_bound():
    hamiltonian = _HamiltonianStub([1.0, -2.0j, -3.0])

    assert pauli_l1_bound(hamiltonian) == pytest.approx(6.0)
    assert evolution_timestep(hamiltonian) == pytest.approx(math.pi / 6.0)
    assert evolution_timestep(hamiltonian, override=0.15) == pytest.approx(0.15)


def test_stored_dimension_maps_to_physical_time_by_task():
    dt = 0.125

    assert physical_krylov_time('kqd', 0, dt) == pytest.approx(dt)
    assert physical_krylov_time('kqd', 4, dt) == pytest.approx(5 * dt)
    assert physical_krylov_time('sqkd', 0, dt) == pytest.approx(0.0)
    assert physical_krylov_time('sqkd', 4, dt) == pytest.approx(4 * dt)


def test_kqd_angle_accounts_for_pennylane_half_angle_convention():
    assert kqd_trotter_rotation_angle(2, dt=0.3, reps=6) == pytest.approx(0.3)


def test_kqd_shadow_sampling_seeds_are_coordinate_deterministic():
    coordinates = (12345, 4, 2)
    first = kqd_shadow_sampling_seeds(*coordinates)

    assert first == kqd_shadow_sampling_seeds(*coordinates)
    assert first != kqd_shadow_sampling_seeds(12345, 5, 2)
    assert first != kqd_shadow_sampling_seeds(12345, 4, 3)
    assert first[0] != first[1]


@pytest.mark.parametrize("coordinates", [(-1, 0, 0), (1, -1, 0), (1, 0, -1)])
def test_kqd_shadow_sampling_seeds_reject_negative_coordinates(coordinates):
    with pytest.raises(ValueError):
        kqd_shadow_sampling_seeds(*coordinates)


def test_neel_reference_states_use_displayed_bitstring_order():
    assert neel_reference_bitstring(5) == '10101'
    assert neel_reference_bitstring(4, lattice_side=2) == '1001'


def test_protocol_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        physical_krylov_time('unknown', 0, 0.1)
    with pytest.raises(ValueError):
        physical_krylov_time('kqd', -1, 0.1)
    with pytest.raises(ValueError):
        neel_reference_bitstring(8, lattice_side=3)


def test_sqkd_gate_represents_full_dt_not_dt_divided_by_reps():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator, SparsePauliOp

    dt = 0.6
    reps = 6
    hamiltonian = SparsePauliOp.from_list([('X', 1.0)])
    actual = Operator(build_sqkd_evolution_gate(hamiltonian, dt, reps, order=2)).data

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    expected = expm(-1j * dt * pauli_x)
    formerly_wrong = expm(-1j * (dt / reps) * pauli_x)

    assert np.linalg.norm(actual - expected) < 1e-10
    assert np.linalg.norm(actual - formerly_wrong) > 0.1


def test_sqkd_synthesized_second_order_formula_converges():
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, SparsePauliOp

    hamiltonian = SparsePauliOp.from_list([
        ('XXI', 0.7), ('YYI', 0.7), ('ZZI', 0.7),
        ('IXX', 1.3), ('IYY', 1.3), ('IZZ', 1.3),
    ])
    dt = 0.35
    exact = expm(-1j * dt * hamiltonian.to_matrix())

    def synthesized_matrix(reps):
        circuit = QuantumCircuit(3)
        circuit.append(
            build_sqkd_evolution_gate(hamiltonian, dt, reps=reps, order=2),
            range(3),
        )
        return Operator(circuit.decompose()).data

    error_one = np.linalg.norm(synthesized_matrix(1) - exact)
    error_four = np.linalg.norm(synthesized_matrix(4) - exact)

    assert error_four < error_one / 5
