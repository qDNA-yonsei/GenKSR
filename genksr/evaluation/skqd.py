"""Sample-based Krylov Quantum Diagonalization (SKQD) energy estimation.

Used for the paper's 2D J1-J2 results (Section 4.2, Figure 3 / Table 2 / Figure 8).
Given computational-basis measurement bitstrings collected across Krylov dimensions,
SKQD builds the subspace spanned by the unique sampled bitstrings, projects the
Hamiltonian onto it, and classically diagonalises it.  The canonical 2D XXZ path
constructs that projection directly; ``qiskit_addon_sqd`` remains as a reference
implementation for cross-checks.

The Sz = 0 sector post-selection (n/2 ones) matches the original GenKSR SKQD
evaluator.
"""
from collections import Counter
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp_sparse

from genksr.utils.quantum import postselect_counts


def exact_ground_energy(H_op):
    """Exact ground-state energy of a Qiskit SparsePauliOp via sparse Lanczos."""
    H_mat = H_op.to_matrix(sparse=True)
    v0 = np.linspace(1.0, 2.0, H_mat.shape[0], dtype=float)
    v0 /= np.linalg.norm(v0)
    eigvals, _ = sp_sparse.linalg.eigsh(
        H_mat, which='SA', k=1, v0=v0, tol=1e-12
    )
    return float(eigvals.real[0])


def skqd_energy_from_counts(counts, H_op, n_qubit, num_ones=None):
    """SKQD ground-state energy from a single counts dict (one Krylov subspace).

    Returns ``np.nan`` if no bitstring survives the Sz post-selection.
    """
    from qiskit_addon_sqd.counts import counts_to_arrays
    from qiskit_addon_sqd.qubit import solve_qubit

    num_ones = n_qubit // 2 if num_ones is None else num_ones
    counts = postselect_counts(counts, num_ones=num_ones)
    if len(counts) == 0:
        return np.nan

    # ``solve_qubit(..., k=2)`` cannot handle a one-dimensional projected
    # subspace and may also fall through SciPy's sparse-to-dense fallback for
    # two dimensions.  Project these tiny cases explicitly.  Apart from fixing
    # D=1 for a Neel-state SKQD run, this preserves the variational energy.
    if len(counts) <= 2:
        indices = np.asarray(sorted(int(bitstring, 2) for bitstring in counts), dtype=int)
        h_sparse = H_op.to_matrix(sparse=True).tocsr()
        projected = h_sparse[indices][:, indices].toarray()
        return float(np.linalg.eigvalsh(projected).real[0])

    bitstring_matrix, _ = counts_to_arrays(counts=counts)
    try:
        eigvals, _ = solve_qubit(bitstring_matrix, H_op, verbose=False, k=2, which='SA')
    except Exception:
        # The sampled subspace can be too small for the sparse solver (e.g. very few
        # bitstrings survive post-selection at low Krylov dimension). Skip gracefully.
        return np.nan
    return float(np.min(eigvals))


def samples_to_basis_ints(samples, n_qubit, num_ones=None):
    """Convert displayed-order bitstring samples to unique basis-state integers.

    Array columns are interpreted as Qiskit's displayed ``q[n-1]...q[0]``
    ordering.  The returned integer therefore has qubit ``i`` at bit position
    ``i``, which is the convention used by :func:`projected_xxz_energy`.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != n_qubit:
        raise ValueError(
            f"samples must have shape (n_samples, {n_qubit}), got {samples.shape}"
        )
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must contain only binary outcomes 0 and 1")

    num_ones = n_qubit // 2 if num_ones is None else num_ones
    selected = samples[np.sum(samples, axis=1) == num_ones]
    if len(selected) == 0:
        return np.empty(0, dtype=np.int64)

    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    return np.unique(selected.astype(np.int64) @ powers)


def projected_xxz_hamiltonian(
        basis_states: Iterable[int], coupling_matrix, n_qubit, delta=1.0, h_z=0.0):
    """Construct the XXZ Hamiltonian in a computational-basis subspace.

    This constructs the same projected Hamiltonian as ``qiskit_addon_sqd`` for
    ``H = sum_ij J_ij (XX + YY + delta*ZZ) + h_z*sum_i Z_i``.  In the
    computational basis, ``ZZ`` and the field are diagonal while ``XX+YY``
    flips an anti-aligned pair with matrix element ``2*J_ij``.  Direct
    construction is substantially faster for the 16-qubit J1-J2 experiments.

    Matrix rows and columns follow the sorted unique integer labels in
    ``basis_states``.  Exposing the matrix separately lets support-controlled
    experiments build the complete symmetry-sector Hamiltonian once and slice
    many sampled subspaces from it without changing the canonical energy
    convention.

    Args:
        basis_states: Integer computational-basis labels. Qubit ``i`` is bit ``i``.
        coupling_matrix: Symmetric ``(n_qubit, n_qubit)`` coupling matrix.
        n_qubit: Number of qubits.
        delta: XXZ anisotropy multiplying the ZZ term.
        h_z: Uniform longitudinal field.

    Returns:
        A CSR matrix.  An empty basis produces a ``(0, 0)`` matrix.
    """
    coupling = np.asarray(coupling_matrix, dtype=float)
    return projected_xxz_hamiltonian_from_couplings(
        basis_states=basis_states,
        jxy_coupling_matrix=coupling,
        jzz_coupling_matrix=delta * coupling,
        n_qubit=n_qubit,
        z_fields=np.full(n_qubit, h_z, dtype=float),
    )


def _validated_projected_couplings(values, n_qubit, name):
    coupling = np.asarray(values, dtype=float)
    if coupling.shape != (n_qubit, n_qubit):
        raise ValueError(
            f"{name} must have shape {(n_qubit, n_qubit)}, got {coupling.shape}"
        )
    if not np.all(np.isfinite(coupling)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(coupling, coupling.T):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(coupling), 0.0):
        raise ValueError(f"{name} must have a zero diagonal")
    return coupling


def projected_xxz_hamiltonian_from_couplings(
        basis_states: Iterable[int], jxy_coupling_matrix,
        jzz_coupling_matrix, n_qubit, z_fields=None):
    """Construct a projected XXZ matrix with independent XY and ZZ edges.

    This is the separate-coupling counterpart of
    :func:`projected_xxz_hamiltonian`.  The input integer convention is the
    same: physical qubit ``i`` is bit ``i``.  In the computational basis the
    XY term flips an anti-aligned pair with matrix element ``2*Jxy_ij``;
    the ZZ terms and optional site-resolved Z fields are diagonal.
    """
    if n_qubit <= 0:
        raise ValueError(f"n_qubit must be positive, got {n_qubit}")
    jxy = _validated_projected_couplings(
        jxy_coupling_matrix, n_qubit, "jxy_coupling_matrix")
    jzz = _validated_projected_couplings(
        jzz_coupling_matrix, n_qubit, "jzz_coupling_matrix")
    if z_fields is None:
        fields = np.zeros(n_qubit, dtype=float)
    else:
        fields = np.asarray(z_fields, dtype=float)
        if fields.shape != (n_qubit,):
            raise ValueError(
                f"z_fields must have shape {(n_qubit,)}, got {fields.shape}")
        if not np.all(np.isfinite(fields)):
            raise ValueError("z_fields must contain only finite values")

    states = sorted({int(state) for state in basis_states})
    dim = len(states)
    if dim == 0:
        return sp_sparse.csr_matrix((0, 0), dtype=float)
    if states[0] < 0 or states[-1] >= 1 << n_qubit:
        raise ValueError(f"basis state outside the {n_qubit}-qubit Hilbert space")

    upper_i, upper_j = np.triu_indices(n_qubit, k=1)
    xy_edges = [
        (int(i), int(j), float(jxy[i, j]))
        for i, j in zip(upper_i, upper_j)
        if not np.isclose(jxy[i, j], 0.0)
    ]
    zz_edges = [
        (int(i), int(j), float(jzz[i, j]))
        for i, j in zip(upper_i, upper_j)
        if not np.isclose(jzz[i, j], 0.0)
    ]
    state_to_row = {state: row for row, state in enumerate(states)}

    rows = []
    cols = []
    values = []
    for row, state in enumerate(states):
        diag = 0.0
        for qubit, field in enumerate(fields):
            if not np.isclose(field, 0.0):
                z_value = 1.0 if ((state >> qubit) & 1) == 0 else -1.0
                diag += float(field) * z_value

        for i, j, weight in zz_edges:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_j = 1.0 if bit_j == 0 else -1.0
            diag += weight * z_i * z_j

        for i, j, weight in xy_edges:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i != bit_j:
                flipped = state ^ ((1 << i) | (1 << j))
                col = state_to_row.get(flipped)
                if col is not None:
                    rows.append(row)
                    cols.append(col)
                    values.append(2.0 * weight)

        rows.append(row)
        cols.append(row)
        values.append(diag)

    return sp_sparse.csr_matrix(
        (values, (rows, cols)), shape=(dim, dim), dtype=float)


def projected_xxz_energy_from_couplings(
        basis_states: Iterable[int], jxy_coupling_matrix,
        jzz_coupling_matrix, n_qubit, z_fields=None):
    """Lowest projected energy for independent XY and ZZ couplings."""
    projected = projected_xxz_hamiltonian_from_couplings(
        basis_states=basis_states,
        jxy_coupling_matrix=jxy_coupling_matrix,
        jzz_coupling_matrix=jzz_coupling_matrix,
        n_qubit=n_qubit,
        z_fields=z_fields,
    )
    dim = projected.shape[0]
    if dim == 0:
        return np.nan
    if dim <= 2:
        return float(np.linalg.eigvalsh(projected.toarray())[0])
    v0 = np.linspace(1.0, 2.0, dim, dtype=float)
    v0 /= np.linalg.norm(v0)
    eigval = sp_sparse.linalg.eigsh(
        projected, k=1, which="SA", return_eigenvectors=False,
        tol=1e-10, v0=v0,
    )[0]
    return float(eigval)


def projected_xxz_energy(
        basis_states: Iterable[int], coupling_matrix, n_qubit, delta=1.0, h_z=0.0):
    """Lowest XXZ energy in a computational-basis subspace.

    Returns ``np.nan`` for an empty subspace.  See
    :func:`projected_xxz_hamiltonian` for the matrix and bit-ordering
    conventions.
    """
    projected = projected_xxz_hamiltonian(
        basis_states,
        coupling_matrix=coupling_matrix,
        n_qubit=n_qubit,
        delta=delta,
        h_z=h_z,
    )
    dim = projected.shape[0]
    if dim == 0:
        return np.nan
    if dim <= 2:
        return float(np.linalg.eigvalsh(projected.toarray())[0])

    # A deterministic starting vector makes repeated seeded experiments stable
    # without relying on ARPACK's internal random initialization.
    v0 = np.linspace(1.0, 2.0, dim, dtype=float)
    v0 /= np.linalg.norm(v0)
    eigval = sp_sparse.linalg.eigsh(
        projected, k=1, which="SA", return_eigenvectors=False,
        tol=1e-10, v0=v0,
    )[0]
    return float(eigval)


def projected_xxz_energies_per_dim(
        samples_by_dim, coupling_matrix, n_qubit, krylov_dims,
        num_ones=None, delta=1.0, h_z=0.0, evaluate_dims=None):
    """Cumulative SKQD curve using the direct projected XXZ solver.

    The sampled basis is accumulated across Krylov dimensions exactly as in
    :func:`skqd_energies_per_dim`, but this path avoids converting samples to
    counts and avoids the general-purpose ``qiskit_addon_sqd`` solver.  It is
    the canonical fast path for the 16-qubit 2D J1-J2 experiments.
    """
    evaluate_dims = None if evaluate_dims is None else set(evaluate_dims)
    energies = []
    cumulative = set()
    for d in krylov_dims:
        samples = samples_by_dim.get(d)
        if samples is not None:
            cumulative.update(
                samples_to_basis_ints(
                    samples, n_qubit=n_qubit, num_ones=num_ones
                ).tolist()
            )
        if evaluate_dims is None or d in evaluate_dims:
            energies.append(projected_xxz_energy(
                cumulative,
                coupling_matrix=coupling_matrix,
                n_qubit=n_qubit,
                delta=delta,
                h_z=h_z,
            ))
        else:
            energies.append(np.nan)
    return energies


def skqd_energies_per_dim(
        counts_by_dim, H_op, n_qubit, krylov_dims, num_ones=None, evaluate_dims=None):
    """Cumulative SKQD energy at each Krylov dimension.

    The subspace at dimension d is the union of all unique bitstrings sampled at
    Krylov dimensions 0 .. d (matching the SKQD protocol), so the returned list is
    the energy convergence curve used for Figure 3.

    Args:
        counts_by_dim: mapping ``{d: counts_dict}``.
        krylov_dims: ordered iterable of d values to accumulate over.
    """
    evaluate_dims = None if evaluate_dims is None else set(evaluate_dims)
    energies = []
    cumulative = Counter()
    for d in krylov_dims:
        cumulative.update(counts_by_dim.get(d, {}))
        if evaluate_dims is None or d in evaluate_dims:
            energies.append(
                skqd_energy_from_counts(dict(cumulative), H_op, n_qubit, num_ones)
            )
        else:
            energies.append(np.nan)
    return energies
