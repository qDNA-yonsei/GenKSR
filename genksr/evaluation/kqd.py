"""KQD generalized-eigenvalue energy estimation (paper Section 4.1, Figure 2 / Table 1).

The Krylov Quantum Diagonalization (KQD) energy at Krylov dimension ``d`` is obtained
by estimating the first row of the overlap matrix ``S`` and projected Hamiltonian ``H``
(via Hadamard-test / QSD circuits), building their Toeplitz forms, and solving the
regularized generalized eigenvalue problem ``H v = E S v``.

Three estimators are supported, all sharing the same post-processing:

* ``ideal_krylov_energies``    -- noiseless exact-simulation reference (statevector overlaps).
* ``shadow_krylov_energies``   -- from Pauli-6 classical-shadow measurements, used both for
  the raw-device classical-shadow baseline (``data_D*.npy``) and for the generative-model
  samples (``model_samples_D*.npy``); both share the identical Pauli-6 encoding.

The v3 evaluator deliberately shares its reference state, edge weights, timestep, and
second-order Trotter circuit with the data generator. Median-of-means ``k=10`` and the
generalized-eigenvalue regularization threshold ``0.9`` remain unchanged.
"""
import numpy as np
import scipy as sp

from genksr.protocol import (
    evolution_timestep,
    kqd_trotter_rotation_angle,
    neel_reference_bitstring,
)
from genksr.utils.quantum import (
    build_hamiltonian_from_couplings,
    build_2d_xxz_hamiltonian,
    build_hamiltonian_from_couplings_pennylane,
    build_hamiltonian_from_couplings_2d_pennylane,
    build_interaction_list_from_couplings,
    create_trotter_evolution_circuit_pennylane,
    create_trotter_evolution_circuit_pennylane_2d,
    build_toeplitz_matrix,
    canonicalize_dominant_state,
    decode_pauli6_shadow,
    estimate_shadow_observable,
    solve_regularized_gen_eig,
)

GEN_EIG_THRESHOLD = 0.9
MEDIAN_OF_MEANS_K = 10


def _statevector_device(qml, wires):
    """Prefer Lightning when installed, otherwise use PennyLane's reference simulator."""
    try:
        return qml.device("lightning.qubit", wires=wires)
    except Exception:
        return qml.device("default.qubit", wires=wires)


def exact_ground_state(H_op, n_qubit, canonicalize=False):
    """Return (exact ground-state energy, dominant computational-basis bitstring)."""
    H_mat = H_op.to_matrix(sparse=True)
    eigvals, eigvecs = sp.sparse.linalg.eigsh(H_mat, which='SA', k=1)
    gs_energy = float(eigvals[0].real)
    ground_state = eigvecs[:, int(np.argmin(eigvals))]
    ground_state = ground_state / np.linalg.norm(ground_state)
    probs = np.abs(ground_state) ** 2
    dominant_idx = int(np.argmax(probs))
    dominant_state = format(dominant_idx, f'0{n_qubit}b')
    if canonicalize:
        dominant_state, _, _ = canonicalize_dominant_state(dominant_state, n_qubit)
    return gs_energy, dominant_state


def _qiskit_hamiltonian(coupling_matrix, n_qubit, lattice='1d', delta=1.0, h_z=0.0):
    if lattice == '2d':
        return build_2d_xxz_hamiltonian(N=n_qubit, J=coupling_matrix, delta=delta, h_z=h_z)
    return build_hamiltonian_from_couplings(n_qubits=n_qubit, J=coupling_matrix)


def _pennylane_hamiltonian(coupling_matrix, n_qubit, lattice='1d'):
    if lattice == '2d':
        return build_hamiltonian_from_couplings_2d_pennylane(n_qubit, coupling_matrix)
    return build_hamiltonian_from_couplings_pennylane(n_qubit, coupling_matrix)


def _prefactors(H_op, dt, max_krylov_dim):
    z_sum = sum(c for p, c in H_op.to_list() if "Z" in p)
    return [np.exp(-1j * z_sum * i * dt) for i in range(1, max_krylov_dim)]


def _energies_from_first_rows(H_row, S_row, max_krylov_dim):
    energies = []
    for d in range(1, max_krylov_dim + 1):
        S_circ = build_toeplitz_matrix(S_row, d)
        H_circ = build_toeplitz_matrix(H_row, d)
        energies.append(float(solve_regularized_gen_eig(H_circ, S_circ, threshold=GEN_EIG_THRESHOLD)))
    return energies


def _time_evolved_sv(n_qubit, reference_str, num_trotter_steps, angle, coupling_matrix):
    """Apply the canonical weighted KQD circuit to a 1D reference state.

    ``angle`` is the base PennyLane Ising rotation angle per repeat. The shared
    circuit helper multiplies it by each edge's ``J_ij``, matching generation.
    """
    import pennylane as qml

    dev = _statevector_device(qml, wires=n_qubit)

    @qml.qnode(dev)
    def circuit():
        for i, bit in enumerate(reversed(reference_str)):
            if bit == '1':
                qml.X(wires=i)
        if angle > 0:
            create_trotter_evolution_circuit_pennylane(
                n_qubit, coupling_matrix, num_trotter_steps, angle, wire_offset=0)
        return qml.state()

    return np.array(circuit())


def _bit_reverse_permutation(n_qubit):
    perm = np.zeros(2 ** n_qubit, dtype=int)
    for k in range(2 ** n_qubit):
        perm[k] = int(format(k, f'0{n_qubit}b')[::-1], 2)
    return perm


def _qsd_statevector_2d(n_qubit, reference_str, interaction_list, num_trotter_steps, angle):
    """Control+system QSD statevector for the 2D KQD Hadamard-test circuit."""
    import pennylane as qml
    from genksr.utils.quantum import controlled_state_prep

    dev = _statevector_device(qml, wires=n_qubit + 1)

    @qml.qnode(dev)
    def circuit():
        control = 0
        system = list(range(1, n_qubit + 1))
        qml.Hadamard(wires=control)
        controlled_state_prep(reference_str, system, control)
        create_trotter_evolution_circuit_pennylane_2d(
            interaction_list, num_trotter_steps, angle)
        qml.X(wires=control)
        controlled_state_prep(reference_str, system, control)
        qml.X(wires=control)
        return qml.state()

    return np.array(circuit())


def _h0_element(coupling_matrix, n_qubit, reference_str, num_trotter_steps, lattice='1d',
                delta=1.0, h_z=0.0):
    H_op = _qiskit_hamiltonian(coupling_matrix, n_qubit, lattice=lattice,
                               delta=delta, h_z=h_z)
    if lattice == '2d':
        interaction_list = build_interaction_list_from_couplings(
            n_qubit, coupling_matrix, wire_offset=1)
        sv = _qsd_statevector_2d(
            n_qubit, reference_str, interaction_list, num_trotter_steps, 0)
        branch0 = sv[:2 ** n_qubit]
        branch1 = sv[2 ** n_qubit:]
        perm = _bit_reverse_permutation(n_qubit)
        return 2.0 * np.vdot(branch0[perm], H_op.to_matrix(sparse=True) @ branch1[perm])

    sv0 = _time_evolved_sv(n_qubit, reference_str, 1, 0, coupling_matrix)
    return np.vdot(sv0, H_op.to_matrix(sparse=True) @ sv0)


def _ideal_krylov_energies_2d(coupling_matrix, n_qubit, num_trotter_steps,
                              max_krylov_dim, delta=1.0, h_z=0.0):
    H_op = _qiskit_hamiltonian(coupling_matrix, n_qubit, lattice='2d',
                               delta=delta, h_z=h_z)
    lattice_side = int(round(np.sqrt(n_qubit)))
    reference_str = neel_reference_bitstring(n_qubit, lattice_side=lattice_side)

    dt = evolution_timestep(H_op)
    H_sparse = H_op.to_matrix(sparse=True)
    prefactors = _prefactors(H_op, dt, max_krylov_dim)
    interaction_list = build_interaction_list_from_couplings(
        n_qubit, coupling_matrix, wire_offset=1)
    perm = _bit_reverse_permutation(n_qubit)

    S_row = np.zeros(max_krylov_dim, dtype=complex)
    H_row = np.zeros(max_krylov_dim, dtype=complex)
    S_row[0] = 1 + 0j

    sv0 = _qsd_statevector_2d(
        n_qubit, reference_str, interaction_list, num_trotter_steps, 0)
    branch0 = sv0[:2 ** n_qubit]
    branch1 = sv0[2 ** n_qubit:]
    H_row[0] = 2.0 * np.vdot(branch0[perm], H_sparse @ branch1[perm])

    for kd in range(max_krylov_dim - 1):
        angle = kqd_trotter_rotation_angle(kd, dt, num_trotter_steps)
        sv = _qsd_statevector_2d(
            n_qubit, reference_str, interaction_list, num_trotter_steps, angle)
        branch0 = sv[:2 ** n_qubit]
        branch1 = sv[2 ** n_qubit:]
        S_row[kd + 1] = prefactors[kd] * (2.0 * np.vdot(branch0, branch1))
        H_row[kd + 1] = prefactors[kd] * (
            2.0 * np.vdot(branch0[perm], H_sparse @ branch1[perm]))

    return _energies_from_first_rows(H_row, S_row, max_krylov_dim)


def ideal_krylov_energies(coupling_matrix, n_qubit, num_trotter_steps, max_krylov_dim,
                          lattice='1d', delta=1.0, h_z=0.0):
    """Noiseless exact-simulation KQD energies for d = 1 .. max_krylov_dim."""
    if lattice == '2d':
        return _ideal_krylov_energies_2d(coupling_matrix, n_qubit, num_trotter_steps,
                                         max_krylov_dim, delta=delta, h_z=h_z)

    H_op = _qiskit_hamiltonian(coupling_matrix, n_qubit, lattice=lattice,
                               delta=delta, h_z=h_z)
    reference_str = neel_reference_bitstring(n_qubit)

    dt = evolution_timestep(H_op)
    H_sparse = H_op.to_matrix(sparse=True)
    prefactors = _prefactors(H_op, dt, max_krylov_dim)

    angles = [0.0] + [
        kqd_trotter_rotation_angle(k - 1, dt, num_trotter_steps)
        for k in range(1, max_krylov_dim)
    ]
    svs = [
        _time_evolved_sv(
            n_qubit, reference_str, num_trotter_steps, angle, coupling_matrix)
        for angle in angles
    ]

    S_row = np.zeros(max_krylov_dim, dtype=complex)
    H_row = np.zeros(max_krylov_dim, dtype=complex)
    S_row[0] = 1 + 0j
    H_row[0] = np.vdot(svs[0], H_sparse @ svs[0])
    for k in range(1, max_krylov_dim):
        S_row[k] = prefactors[k - 1] * np.vdot(svs[0], svs[k])
        H_row[k] = prefactors[k - 1] * np.vdot(svs[0], H_sparse @ svs[k])

    return _energies_from_first_rows(H_row, S_row, max_krylov_dim)


def shadow_krylov_energies(samples_by_dim, coupling_matrix, n_qubit, num_trotter_steps,
                           max_krylov_dim, num_samples=None, k=MEDIAN_OF_MEANS_K,
                           rng=None, lattice='1d', delta=1.0, h_z=0.0):
    """KQD energies from Pauli-6 classical-shadow samples (device baseline or model samples).

    Args:
        samples_by_dim: mapping ``{kd: array}`` of Pauli-6 encoded shadow samples for
            kd = 0 .. max_krylov_dim-2. Missing dims are skipped (their contribution stays 0).
        coupling_matrix: the 1D chain coupling matrix for this Hamiltonian.
        num_samples: optionally subsample this many shots per Krylov dim.
        k: median-of-means splits for the shadow estimator.
        rng: numpy Generator for subsampling (deterministic if provided).
    """
    import pennylane as qml

    rng = np.random.default_rng() if rng is None else rng
    H_op = _qiskit_hamiltonian(coupling_matrix, n_qubit, lattice=lattice,
                               delta=delta, h_z=h_z)
    if lattice == '2d':
        lattice_side = int(round(np.sqrt(n_qubit)))
        reference_str = neel_reference_bitstring(n_qubit, lattice_side=lattice_side)
    else:
        reference_str = neel_reference_bitstring(n_qubit)

    dt = evolution_timestep(H_op)
    prefactors = _prefactors(H_op, dt, max_krylov_dim)

    # Hamiltonian Pauli terms shifted onto system wires 1..n (control qubit is wire 0).
    hamiltonian = _pennylane_hamiltonian(coupling_matrix, n_qubit, lattice=lattice)
    hamiltonian_on_system = qml.Hamiltonian(
        hamiltonian.coeffs,
        [qml.map_wires(op, {w: w + 1 for w in op.wires}) for op in hamiltonian.ops],
    )
    coeffs, pauli_ops = hamiltonian_on_system.terms()

    S_row = np.zeros(max_krylov_dim, dtype=complex)
    H_row = np.zeros(max_krylov_dim, dtype=complex)
    S_row[0] = 1 + 0j
    H_row[0] = _h0_element(coupling_matrix, n_qubit, reference_str, num_trotter_steps,
                           lattice=lattice, delta=delta, h_z=h_z)

    for kd in range(max_krylov_dim - 1):
        samples = samples_by_dim.get(kd)
        if samples is None or len(samples) == 0:
            continue
        samples = np.asarray(samples)
        if num_samples is not None and num_samples < len(samples):
            idx = rng.choice(len(samples), size=num_samples, replace=False)
            samples = samples[idx]

        outcomes, bases = decode_pauli6_shadow(samples)
        shadow = (outcomes, bases)

        ev_real = estimate_shadow_observable(shadow, qml.PauliX(0), k=k)
        ev_imag = estimate_shadow_observable(shadow, qml.PauliY(0), k=k)
        S_row[kd + 1] += prefactors[kd] * (ev_real + 1j * ev_imag)

        for coeff, op in zip(coeffs, pauli_ops):
            ev_real = estimate_shadow_observable(shadow, op @ qml.PauliX(0), k=k)
            ev_imag = estimate_shadow_observable(shadow, op @ qml.PauliY(0), k=k)
            H_row[kd + 1] += prefactors[kd] * coeff * (ev_real + 1j * ev_imag)

    return _energies_from_first_rows(H_row, S_row, max_krylov_dim)
