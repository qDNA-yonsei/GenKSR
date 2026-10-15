"""Quantum physics utilities: Hamiltonian construction, coupling matrices, eigenvalue solvers."""

import itertools as it
from typing import Union, List

import numpy as np
import scipy as sp


def canonicalize_dominant_state(dominant_state_str, n_qubits):
    """Choose a stable representative when the dominant state and its complement tie.

    The 2D antiferromagnetic Heisenberg instances often have two Neel-like dominant
    bitstrings related by a global bit flip. The legacy 2D KQD scripts fixed this
    ambiguity by requiring the Qiskit qubit-0 bit (last character) to be 0.
    """
    if dominant_state_str[-1] == '1':
        canonical = ''.join('1' if bit == '0' else '0' for bit in dominant_state_str)
        return canonical, int(canonical, 2), True
    return dominant_state_str, int(dominant_state_str, 2), False


def solve_regularized_gen_eig(h, s, threshold, k=1, return_dimn=False):
    """Solve the generalized eigenvalue problem with regularization."""
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = np.array([vec for val, vec in zip(s_vals, s_vecs) if val > threshold])
    h_reg = good_vecs.conj() @ h @ good_vecs.T
    s_reg = good_vecs.conj() @ s @ good_vecs.T
    if k == 1:
        result = sp.linalg.eigh(h_reg, s_reg)[0][0]
    else:
        result = sp.linalg.eigh(h_reg, s_reg)[0][:k]
    if return_dimn:
        return result, len(good_vecs)
    return result


def sample_coupling_matrix_1d(n_qubits, low=0.0, high=2.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits - 1):
        w = rng.uniform(low, high)
        J[i, i + 1] = J[i + 1, i] = w
    return J


def sample_coupling_matrix_2d_nn(L, J_range=(0.5, 1.5), rng=None):
    """Sample coupling matrix for 2D square lattice with NN only (no NNN).

    Each bond gets an independent random coupling strength.

    Args:
        L: Lattice size (L x L), with PBC.
        J_range: (low, high) range for each bond coupling.
        rng: Random number generator.

    Returns:
        NxN symmetric coupling matrix (N = L*L).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = L * L
    J = np.zeros((N, N))

    for i in range(L):
        for j in range(L):
            idx = i * L + j
            # Right neighbor (horizontal, PBC)
            right = i * L + ((j + 1) % L)
            if J[idx, right] == 0.0:  # avoid overwriting symmetric pair
                w = rng.uniform(*J_range)
                J[idx, right] = J[right, idx] = w
            # Down neighbor (vertical, PBC)
            down = ((i + 1) % L) * L + j
            if J[idx, down] == 0.0:
                w = rng.uniform(*J_range)
                J[idx, down] = J[down, idx] = w

    return J


def sample_coupling_matrix_2d(L, n_dim=2, J1_range=(0.0, 1.0), J2_range=(0.0, 1.0), rng=None):
    """Sample coupling matrix for 2D lattice with nearest and next-nearest neighbors.

    Args:
        L: Lattice size (L x L)
        n_dim: Dimension (default 2)
        J1_range: Range for nearest neighbor couplings
        J2_range: Range for next-nearest neighbor couplings
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    N = L ** n_dim
    J = np.zeros((N, N))
    J_val = rng.uniform(*J1_range)

    for i in range(L):
        for j in range(L):
            idx = i * L + j
            right = i * L + ((j + 1) % L)
            J[idx, right] = J[right, idx] = J_val
            down = ((i + 1) % L) * L + j
            J[idx, down] = J[down, idx] = J_val

    J_val = rng.uniform(*J2_range)
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            diag1 = ((i + 1) % L) * L + ((j + 1) % L)
            J[idx, diag1] = J[diag1, idx] = J_val
            diag2 = ((i + 1) % L) * L + ((j - 1) % L)
            J[idx, diag2] = J[diag2, idx] = J_val

    return J


# --- Qiskit-based Hamiltonian builders ---

def build_hamiltonian_from_couplings(n_qubits, J):
    """Build 1D Heisenberg Hamiltonian (Qiskit SparsePauliOp)."""
    from qiskit.quantum_info import SparsePauliOp

    pauli_strings = []
    coeffs = []
    for i in range(n_qubits - 1):
        w = J[i, i + 1]
        if w == 0.0:
            continue
        for pauli_char in ["Z", "X", "Y"]:
            p = ["I"] * n_qubits
            p[i] = p[i + 1] = pauli_char
            pauli_strings.append("".join(p))
            coeffs.append(w)

    return SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))


def build_2d_xxz_hamiltonian(N, J, delta=1.0, h_z=0.0):
    """Build 2D XXZ Heisenberg Hamiltonian with external field.

    H = Σ_{<i,j>} J_ij [ S^x_i S^x_j + S^y_i S^y_j + Δ · S^z_i S^z_j ] + h_z Σ_i S^z_i

    Args:
        N: Number of qubits.
        J: NxN coupling matrix.
        delta: Anisotropy parameter. 1.0 = isotropic Heisenberg, >1 = Ising-like.
        h_z: External magnetic field strength along Z.

    Uses Qiskit reversed bit ordering.
    """
    coupling = np.asarray(J, dtype=float)
    return build_xxz_hamiltonian(
        N=N,
        jxy_couplings=coupling,
        jzz_couplings=delta * coupling,
        z_fields=np.full(N, h_z, dtype=float),
    )


def _validated_xxz_coupling_matrix(values, N, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (N, N):
        raise ValueError(f"{name} must have shape {(N, N)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError(f"{name} must have a zero diagonal")
    return matrix


def build_xxz_hamiltonian(
        N, jxy_couplings, jzz_couplings, z_fields=None):
    """Build an XXZ Hamiltonian with independent XY and ZZ couplings.

    The operator convention is

    ``H = sum_{i<j} Jxy_ij (X_i X_j + Y_i Y_j)``
    ``    + Jzz_ij Z_i Z_j + sum_i h_i Z_i``.

    Unlike :func:`build_2d_xxz_hamiltonian`, this function does not tie the
    ZZ coefficients to the XY coefficients through one global anisotropy.
    This is needed by the 1D sparsity sweep, where ``Jzz=1`` is fixed while
    ``Jxy`` changes between Hamiltonian instances.  Qiskit's displayed Pauli
    ordering is used, so physical qubit ``i`` is written at string position
    ``N - 1 - i``.
    """
    from qiskit.quantum_info import SparsePauliOp

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    jxy = _validated_xxz_coupling_matrix(jxy_couplings, N, "jxy_couplings")
    jzz = _validated_xxz_coupling_matrix(jzz_couplings, N, "jzz_couplings")

    if z_fields is None:
        fields = np.zeros(N, dtype=float)
    else:
        fields = np.asarray(z_fields, dtype=float)
        if fields.shape != (N,):
            raise ValueError(f"z_fields must have shape {(N,)}, got {fields.shape}")
        if not np.all(np.isfinite(fields)):
            raise ValueError("z_fields must contain only finite values")

    pauli_terms = []

    def pauli_term(i, j, pauli_type):
        pauli_str = ['I'] * N
        pauli_str[i] = pauli_type
        pauli_str[j] = pauli_type
        return ''.join(reversed(pauli_str))

    for i in range(N):
        for j in range(i + 1, N):
            xy_weight = float(jxy[i, j])
            zz_weight = float(jzz[i, j])
            if not np.isclose(xy_weight, 0.0):
                pauli_terms.append((pauli_term(i, j, 'X'), xy_weight))
                pauli_terms.append((pauli_term(i, j, 'Y'), xy_weight))
            if not np.isclose(zz_weight, 0.0):
                pauli_terms.append((pauli_term(i, j, 'Z'), zz_weight))

    for i, field in enumerate(fields):
        if np.isclose(field, 0.0):
            continue
        pauli_str = ['I'] * N
        pauli_str[i] = 'Z'
        pauli_terms.append((''.join(reversed(pauli_str)), float(field)))

    if not pauli_terms:
        pauli_terms = [('I' * N, 0.0)]
    return SparsePauliOp.from_list(pauli_terms)


# --- PennyLane-based Hamiltonian builders ---

def build_hamiltonian_from_couplings_pennylane(n_qubits, J):
    """Build 1D Heisenberg Hamiltonian as PennyLane Hamiltonian."""
    import pennylane as qml

    coeffs = []
    ops = []
    for i in range(n_qubits - 1):
        w = J[i, i + 1]
        if w == 0.0:
            continue
        coeffs.extend([w, w, w])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX(i + 1),
            qml.PauliY(i) @ qml.PauliY(i + 1),
            qml.PauliZ(i) @ qml.PauliZ(i + 1)
        ])
    return qml.Hamiltonian(coeffs, ops)


def build_hamiltonian_from_couplings_2d_pennylane(n_qubits, J):
    """Build the 2D isotropic Heisenberg Hamiltonian as a PennyLane Hamiltonian."""
    import pennylane as qml

    coeffs = []
    ops = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            w = J[i, j]
            if np.isclose(w, 0.0):
                continue
            coeffs.extend([w, w, w])
            ops.extend([
                qml.PauliX(i) @ qml.PauliX(j),
                qml.PauliY(i) @ qml.PauliY(j),
                qml.PauliZ(i) @ qml.PauliZ(j),
            ])
    return qml.Hamiltonian(coeffs, ops)


# --- Utility functions ---

def convert_to_counts_dict(data_gen):
    """Convert measurement data array to bitstring count dictionary."""
    from collections import Counter
    bitstring_generator = ("".join(map(str, measurement)) for measurement in data_gen)
    counts_dict = Counter(bitstring_generator)
    return dict(counts_dict)


def postselect_counts(counts, num_ones):
    """Filter counts to only include bitstrings with specific number of ones."""
    filtered_counts = {}
    for bitstring, freq in counts.items():
        if bitstring.count("1") == num_ones:
            filtered_counts[bitstring] = freq
    return filtered_counts


# --- J1-J2 coupling sampler (paper Section 4.2) ---

def sample_coupling_matrix_2d_j1j2(L, J1=1.0, J2_range=(0.0, 1.0), rng=None):
    """Sample the paper's 2D J1-J2 Heisenberg couplings on an LxL torus (Section 4.2).

    Nearest-neighbour bonds all get the fixed coupling ``J1`` (default 1.0); the
    next-nearest-neighbour (diagonal) bonds all get a single value sampled from
    ``J2_range`` (default U[0, 1]). This matches H = J1 Σ_<r,r'> S·S + J2 Σ_<<r,r'>> S·S.

    Returns an NxN symmetric coupling matrix (N = L*L).
    """
    return sample_coupling_matrix_2d(L, J1_range=(J1, J1), J2_range=J2_range, rng=rng)


# --- Classical-shadow Krylov (KQD) building blocks, Section 4.1 ---
# Ported from the original GenKSR/utils.py. PennyLane is imported lazily so that
# `import genksr.utils.quantum` stays light for the SKQD-only pipeline.

def controlled_state_prep(dominant_state_str, system_wires, control_wire):
    """Controlled preparation of |dominant_state> on system wires, conditioned on control."""
    import pennylane as qml
    for i, bit in enumerate(dominant_state_str[::-1]):
        if bit == '1':
            qml.CNOT(wires=[control_wire, system_wires[i]])


def _rxyz(wires, angle):
    import pennylane as qml
    qml.IsingXX(angle, wires=wires)
    qml.IsingYY(angle, wires=wires)
    qml.IsingZZ(angle, wires=wires)


def _apply_second_order_trotter(interaction_list, num_trotter_steps, angle):
    """Apply a symmetric second-order product formula to edge-colored groups."""
    import pennylane as qml

    if num_trotter_steps <= 0:
        raise ValueError(f"num_trotter_steps must be positive, got {num_trotter_steps}")
    if not interaction_list:
        return

    def apply_group(group, scale):
        for wire_a, wire_b, weight in group:
            _rxyz(wires=[wire_a, wire_b], angle=weight * angle * scale)

    # exp(A_0/2)...exp(A_m)...exp(A_0/2) for every Trotter repeat.
    # Edges within one color group are disjoint and can be applied in parallel.
    for _ in range(num_trotter_steps):
        for group in interaction_list[:-1]:
            apply_group(group, 0.5)
            qml.Barrier()
        apply_group(interaction_list[-1], 1.0)
        for group in reversed(interaction_list[:-1]):
            qml.Barrier()
            apply_group(group, 0.5)


def create_trotter_evolution_circuit_pennylane(
        n_qubits, coupling_matrix, num_trotter_steps, angle, wire_offset=1):
    """Weighted 2nd-order Trotter evolution of a 1D Heisenberg chain.

    ``angle`` is the base PennyLane Ising rotation angle per repeat.  Each edge
    receives ``J_ij * angle`` so the implemented evolution uses the same weighted
    Hamiltonian as the graph and evaluator.  ``wire_offset=1`` reserves wire 0 for
    the Hadamard-test control; statevector-only evaluation uses ``wire_offset=0``.
    """
    interaction_list = [
        [
            (wire_offset + i, wire_offset + i + 1, float(coupling_matrix[i, i + 1]))
            for i in range(0, n_qubits - 1, 2)
            if not np.isclose(coupling_matrix[i, i + 1], 0.0)
        ],
        [
            (wire_offset + i, wire_offset + i + 1, float(coupling_matrix[i, i + 1]))
            for i in range(1, n_qubits - 1, 2)
            if not np.isclose(coupling_matrix[i, i + 1], 0.0)
        ],
    ]
    _apply_second_order_trotter(interaction_list, num_trotter_steps, angle)


def build_interaction_list_from_couplings(n_qubits, J, wire_offset=1):
    """Greedily edge-color a weighted graph into parallel Trotter groups.

    Each interaction is returned as ``(wire_a, wire_b, J_ij)``.  Keeping the
    weight beside the edge prevents the circuit topology from silently discarding
    Hamiltonian coefficients.
    """
    edges = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if not np.isclose(J[i, j], 0.0):
                edges.append((i, j))

    edge_colors = {}
    for i, j in edges:
        used_colors = set()
        for a, b in edge_colors:
            if a == i or a == j or b == i or b == j:
                used_colors.add(edge_colors[(a, b)])
        color = 0
        while color in used_colors:
            color += 1
        edge_colors[(i, j)] = color

    groups = [[] for _ in range(max(edge_colors.values()) + 1)] if edge_colors else []
    for (i, j), color in edge_colors.items():
        groups[color].append((wire_offset + i, wire_offset + j, float(J[i, j])))
    return groups


def create_trotter_evolution_circuit_pennylane_2d(
        interaction_list, num_trotter_steps, angle):
    """Weighted second-order graph KQD circuit from edge-colored interactions."""
    _apply_second_order_trotter(interaction_list, num_trotter_steps, angle)


def calculate_classical_shadow(circuit_template, dominant_state_str, shadow_size,
                               num_qubits, num_trotter_steps, t, rng=None):
    """Collect a classical shadow of ``shadow_size`` snapshots from a QSD circuit.

    Returns (outcomes, bases) where outcomes are in {-1, +1} and bases in {0,1,2}
    (0=X, 1=Y, 2=Z). ``num_qubits`` includes the control qubit.
    """
    import pennylane as qml
    rng = np.random.default_rng() if rng is None else rng
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = rng.integers(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    for ns in range(shadow_size):
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(
            n_qubits=num_qubits - 1, dominant_state_str=dominant_state_str,
            num_trotter_steps=num_trotter_steps, t=t, observable=obs,
        )
    return outcomes, unitary_ids


def estimate_shadow_observable(shadow, observable, k=10):
    """Median-of-means estimator E[O] from a classical shadow (k splits)."""
    import pennylane as qml
    shadow_size, num_qubits = shadow[0].shape
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs = np.array([map_name_to_int[observable.name]])
        target_locs = np.array([observable.wires[0]])
    else:
        target_obs = np.array([map_name_to_int[o.name] for o in observable.operands])
        target_locs = np.array([o.wires[0] for o in observable.operands])

    b_lists, obs_lists = shadow
    means = []
    for i in range(0, shadow_size, shadow_size // k):
        b_k = b_lists[i: i + shadow_size // k]
        o_k = obs_lists[i: i + shadow_size // k]
        indices = np.all(o_k[:, target_locs] == target_obs, axis=1)
        if sum(indices) > 0:
            product = np.prod(b_k[indices][:, target_locs], axis=1)
            means.append(np.sum(product) / sum(indices))
        else:
            means.append(0)
    return np.median(means)


# Backwards-compatible spelling used in the original code.
estimate_shadow_obervable = estimate_shadow_observable


def build_toeplitz_matrix(first_row, dim):
    """Build a Hermitian Toeplitz matrix of size ``dim`` from the first row."""
    matrix = np.zeros((dim, dim), dtype=complex)
    for i, j in it.product(range(dim), repeat=2):
        if i >= j:
            matrix[j, i] = first_row[i - j]
        else:
            matrix[j, i] = np.conj(first_row[j - i])
    return matrix


def decode_pauli6_shadow(samples):
    """Decode Pauli-6 encoded shadow samples into (outcomes in {-1,+1}, bases in {0,1,2}).

    The generator stores each measurement as ``outcome_bit * 3 + basis`` (values 0..5).
    """
    samples = np.asarray(samples)
    outcome_bits = samples // 3
    bases = samples % 3
    # Compact uint8 files need signed arithmetic: otherwise 0 * 2 - 1 wraps to 255.
    outcomes = outcome_bits.astype(int) * 2 - 1
    return outcomes, bases
