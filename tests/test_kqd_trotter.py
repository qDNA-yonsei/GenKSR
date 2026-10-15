import numpy as np
import pytest
from scipy.linalg import expm

qml = pytest.importorskip('pennylane')

from genksr.protocol import kqd_trotter_rotation_angle
from genksr.utils.quantum import create_trotter_evolution_circuit_pennylane


def _heisenberg_term(n_qubits, wire_a, wire_b):
    identity = np.eye(2, dtype=complex)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    term = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    for pauli in paulis:
        factors = [identity] * n_qubits
        factors[wire_a] = pauli
        factors[wire_b] = pauli
        product = factors[0]
        for factor in factors[1:]:
            product = np.kron(product, factor)
        term += product
    return term


def _trotter_matrix(couplings, total_time, reps):
    n_qubits = len(couplings)
    angle = kqd_trotter_rotation_angle(0, total_time, reps)

    def circuit():
        create_trotter_evolution_circuit_pennylane(
            n_qubits, couplings, reps, angle, wire_offset=0)

    return np.asarray(qml.matrix(circuit, wire_order=range(n_qubits))())


def test_each_kqd_edge_receives_its_hamiltonian_weight():
    couplings = np.zeros((3, 3))
    couplings[0, 1] = couplings[1, 0] = 0.25
    couplings[1, 2] = couplings[2, 1] = 1.5

    with qml.queuing.AnnotatedQueue() as queue:
        create_trotter_evolution_circuit_pennylane(
            3, couplings, num_trotter_steps=1, angle=0.4, wire_offset=0)

    total_xx_angle = {}
    for op in queue.queue:
        if op.name != 'IsingXX':
            continue
        edge = tuple(op.wires.tolist())
        total_xx_angle[edge] = total_xx_angle.get(edge, 0.0) + float(op.data[0])

    assert total_xx_angle[(0, 1)] == pytest.approx(0.25 * 0.4)
    assert total_xx_angle[(1, 2)] == pytest.approx(1.5 * 0.4)


def test_weighted_second_order_trotter_converges_to_exact_unitary():
    couplings = np.zeros((3, 3))
    couplings[0, 1] = couplings[1, 0] = 0.7
    couplings[1, 2] = couplings[2, 1] = 1.3
    total_time = 0.35

    hamiltonian = (
        0.7 * _heisenberg_term(3, 0, 1)
        + 1.3 * _heisenberg_term(3, 1, 2)
    )
    exact = expm(-1j * total_time * hamiltonian)
    error_one = np.linalg.norm(_trotter_matrix(couplings, total_time, reps=1) - exact)
    error_four = np.linalg.norm(_trotter_matrix(couplings, total_time, reps=4) - exact)

    assert error_four < error_one / 5
