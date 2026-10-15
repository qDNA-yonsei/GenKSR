"""Canonical v3 physics conventions shared across generation, training, and sampling."""

from __future__ import annotations

import math
import numbers

import numpy as np


PROTOCOL_VERSION = 3
KQD_TASK = "kqd"
SQKD_TASK = "sqkd"
SUPPORTED_TASKS = (KQD_TASK, SQKD_TASK)


def pauli_l1_bound(hamiltonian) -> float:
    """Return the Pauli coefficient 1-norm used as the computable ``||H||`` bound."""
    bound = float(np.sum(np.abs(hamiltonian.coeffs)))
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"Hamiltonian Pauli 1-norm must be positive and finite, got {bound}")
    return bound


def evolution_timestep(hamiltonian, override: float | None = None) -> float:
    """Return the canonical Krylov step ``dt``.

    v3 uses ``pi / sum_p |c_p|`` for ``H = sum_p c_p P_p``.  This is a
    computable Pauli 1-norm upper bound, not an exact spectral norm.
    """
    dt = float(override) if override is not None else math.pi / pauli_l1_bound(hamiltonian)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Evolution timestep must be positive and finite, got {dt}")
    return dt


def xxz_chain_pauli_l1_bound(
        n_qubits: int, j_xy: float, j_zz: float = 1.0,
        z_fields=None) -> float:
    """Pauli coefficient 1-norm of a uniform open XXZ chain.

    ``H = sum_i [j_xy (XX + YY) + j_zz ZZ]`` has ``n_qubits - 1``
    bonds and hence the exact coefficient sum used by protocol v3 is
    ``(n_qubits - 1) * (2*abs(j_xy) + abs(j_zz))``.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, numbers.Integral):
        raise TypeError("n_qubits must be an integer")
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be at least 2, got {n_qubits}")
    if not np.isfinite(j_xy) or not np.isfinite(j_zz):
        raise ValueError("j_xy and j_zz must be finite")
    bound = (int(n_qubits) - 1) * (2.0 * abs(float(j_xy)) + abs(float(j_zz)))
    if z_fields is not None:
        fields = np.asarray(z_fields, dtype=float)
        if fields.shape != (int(n_qubits),):
            raise ValueError(
                f"z_fields must have shape {(int(n_qubits),)}, got {fields.shape}")
        if not np.all(np.isfinite(fields)):
            raise ValueError("z_fields must contain only finite values")
        bound += float(np.sum(np.abs(fields)))
    if bound <= 0.0:
        raise ValueError("XXZ chain Pauli 1-norm must be positive")
    return float(bound)


def xxz_chain_timestep(
        n_qubits: int, j_xy: float, j_zz: float = 1.0,
        z_fields=None) -> float:
    """Canonical v3 timestep for a uniform open XXZ chain."""
    return math.pi / xxz_chain_pauli_l1_bound(
        n_qubits, j_xy, j_zz, z_fields=z_fields)


def physical_krylov_time(task: str, data_index: int, dt: float) -> float:
    """Map a stored ``data_D{index}`` file to its physical evolution time.

    KQD stores Hadamard-test data for non-zero time differences, so ``data_D0``
    corresponds to ``dt``.  SQKD stores the initial state, so ``data_D0``
    corresponds to time zero.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {SUPPORTED_TASKS}")
    if data_index < 0:
        raise ValueError(f"data_index must be non-negative, got {data_index}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")

    krylov_power = data_index + 1 if task == KQD_TASK else data_index
    return float(krylov_power * dt)


def kqd_trotter_rotation_angle(data_index: int, dt: float, reps: int) -> float:
    """Return the PennyLane Ising rotation angle used per KQD Trotter repeat.

    PennyLane's Ising gates implement ``exp(-i * angle * P / 2)``.  Therefore
    evolving for physical time ``t`` requires angle ``2*t/reps`` per repeat.
    """
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    return 2.0 * physical_krylov_time(KQD_TASK, data_index, dt) / reps


def kqd_shadow_sampling_seeds(
    hamiltonian_seed: int,
    data_index: int,
    chunk_index: int,
) -> tuple[int, int]:
    """Return deterministic, independent device and shadow-recipe seeds.

    PennyLane's ``classical_shadow`` measurement owns a random generator for
    selecting Pauli recipes that is separate from the device generator used to
    sample measurement outcomes.  Both therefore need explicit seeds.  The
    coordinate-based derivation also makes the result independent of process
    scheduling when Hamiltonians are generated in parallel.
    """
    coordinates = {
        "hamiltonian_seed": hamiltonian_seed,
        "data_index": data_index,
        "chunk_index": chunk_index,
    }
    for name, value in coordinates.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    state = np.random.SeedSequence(
        [int(hamiltonian_seed), int(data_index), int(chunk_index)]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]), int(state[1])


def neel_reference_bitstring(n_qubits: int, lattice_side: int | None = None) -> str:
    """Return the canonical Neel reference state in displayed bitstring order.

    For a 1D chain this is exactly ``1010...`` as specified in the paper.  For
    a square lattice, physical qubit ``q=r*L+c`` is occupied on the even
    checkerboard sublattice and the result is reversed into Qiskit's displayed
    ``q[N-1]...q[0]`` order.
    """
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    if lattice_side is None:
        return "".join("1" if i % 2 == 0 else "0" for i in range(n_qubits))

    if lattice_side * lattice_side != n_qubits:
        raise ValueError(
            f"lattice_side={lattice_side} gives {lattice_side ** 2} sites, "
            f"not n_qubits={n_qubits}"
        )
    qubit_order_bits = [
        "1" if (r + c) % 2 == 0 else "0"
        for r in range(lattice_side)
        for c in range(lattice_side)
    ]
    return "".join(reversed(qubit_order_bits))


def build_sqkd_evolution_gate(hamiltonian, dt: float, reps: int, order: int = 2):
    """Build one SQKD Krylov step representing total evolution ``exp(-i H dt)``."""
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}")

    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter, SuzukiTrotter

    synthesis = LieTrotter(reps=reps) if order == 1 else SuzukiTrotter(order=2, reps=reps)
    return PauliEvolutionGate(hamiltonian, time=float(dt), synthesis=synthesis)
