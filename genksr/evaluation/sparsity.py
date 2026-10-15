"""Exact-sector and ground-state sparsity utilities.

The 1D XXZ sparsity experiment works directly in a fixed-Hamming-weight
sector.  These helpers remain independent of Qiskit so the N=20 pilot stores
only the physically relevant ``C(N, N/2)`` amplitudes rather than ``2**N``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


def fixed_hamming_weight_basis(n_qubit: int, num_ones: int | None = None):
    """Return sorted integer basis labels with exactly ``num_ones`` set bits."""
    if isinstance(n_qubit, bool) or not isinstance(n_qubit, int):
        raise TypeError("n_qubit must be an integer")
    if n_qubit <= 0 or n_qubit >= 63:
        raise ValueError("n_qubit must be between 1 and 62 for int64 basis labels")
    num_ones = n_qubit // 2 if num_ones is None else num_ones
    if isinstance(num_ones, bool) or not isinstance(num_ones, int):
        raise TypeError("num_ones must be an integer")
    if not 0 <= num_ones <= n_qubit:
        raise ValueError(f"num_ones must be in [0, {n_qubit}], got {num_ones}")

    dimension = math.comb(n_qubit, num_ones)
    states = np.fromiter(
        (
            sum(1 << qubit for qubit in occupied)
            for occupied in itertools.combinations(range(n_qubit), num_ones)
        ),
        dtype=np.int64,
        count=dimension,
    )
    states.sort()
    return states


def l_alpha(probabilities, alpha: float) -> int:
    """Smallest number of largest probabilities accumulating to ``alpha``."""
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size == 0:
        raise ValueError("probabilities must not be empty")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    normalized = probs / total
    cumulative = np.cumsum(np.sort(normalized)[::-1])
    # Guard against floating-point normalization leaving cumsum[-1] just below 1.
    cumulative[-1] = 1.0
    index = int(np.searchsorted(cumulative, alpha, side="left"))
    # Decimal inputs such as [0.6, 0.3, 0.1] can represent the exact target
    # 0.9 as 0.8999999999999999.  Equality at the requested mass should not
    # spuriously require one additional basis state.
    if index > 0 and np.isclose(
            cumulative[index - 1], alpha, rtol=0.0, atol=1e-12):
        index -= 1
    return index + 1


def effective_support(probabilities) -> float:
    """Inverse-participation-ratio effective number of basis states."""
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size == 0 or not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be non-empty, finite, and non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    normalized = probs / total
    return float(1.0 / np.sum(normalized ** 2))


def sparsity_metrics(
        probabilities, sector_size: int | None = None,
        alphas=(0.95, 0.99)) -> dict:
    """Compute normalized support metrics for a ground-state distribution."""
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if sector_size is None:
        sector_size = probs.size
    if isinstance(sector_size, bool) or not isinstance(sector_size, int):
        raise TypeError("sector_size must be an integer")
    if sector_size < probs.size or sector_size <= 0:
        raise ValueError("sector_size must be positive and at least len(probabilities)")

    result = {
        "sector_size": int(sector_size),
        "effective_support": effective_support(probs),
    }
    result["effective_support_fraction"] = (
        result["effective_support"] / sector_size)
    for alpha in alphas:
        label = int(round(100 * float(alpha)))
        support = l_alpha(probs, float(alpha))
        result[f"l{label}"] = int(support)
        result[f"f{label}"] = float(support / sector_size)
    return result


def lowest_eigenpairs(matrix, k: int = 2, tol: float = 1e-10):
    """Return deterministic lowest eigenpairs of a real Hermitian matrix."""
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix must be non-empty and square")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    dimension = int(matrix.shape[0])
    if k >= dimension or dimension <= 4:
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        values, vectors = np.linalg.eigh(dense)
        order = np.argsort(values)[:min(k, dimension)]
        return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order])

    v0 = np.linspace(1.0, 2.0, dimension, dtype=float)
    v0 /= np.linalg.norm(v0)
    values, vectors = sparse_linalg.eigsh(
        matrix, k=k, which="SA", v0=v0, tol=tol)
    order = np.argsort(values)
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order])


def basis_state_probability(
        probabilities, sector_basis, state: int) -> float:
    """Return one integer basis state's probability from a sorted sector vector."""
    basis = np.asarray(sector_basis, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if basis.ndim != 1 or len(basis) != len(probs):
        raise ValueError("sector_basis and probabilities must be aligned 1D arrays")
    position = int(np.searchsorted(basis, int(state)))
    if position >= len(basis) or int(basis[position]) != int(state):
        return 0.0
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    return float(probs[position] / total)


def xxz_open_chain_sector_components(n_qubit: int, sector_basis=None):
    """Return unit-coupling ``(H_xy, H_zz)`` for an open XXZ chain sector.

    The returned CSR matrices satisfy

    ``H(Jxy, Jzz) = Jxy * H_xy + Jzz * H_zz``

    for ``sum_i Jxy(XX+YY) + Jzz*ZZ``.  Constructing these topology-only
    components once is much faster than rebuilding the same N=20 sparse
    pattern for every scalar ``Jxy`` in a sweep.
    """
    if isinstance(n_qubit, bool) or not isinstance(n_qubit, int):
        raise TypeError("n_qubit must be an integer")
    if n_qubit < 2:
        raise ValueError("n_qubit must be at least 2")
    if sector_basis is None:
        sector_basis = fixed_hamming_weight_basis(n_qubit, n_qubit // 2)
    states = np.asarray(sector_basis, dtype=np.int64)
    if states.ndim != 1 or len(states) == 0:
        raise ValueError("sector_basis must be a non-empty 1D array")
    if not np.all(states[1:] > states[:-1]):
        raise ValueError("sector_basis must be strictly increasing")
    if states[0] < 0 or states[-1] >= 1 << n_qubit:
        raise ValueError("sector_basis contains a state outside the Hilbert space")

    dimension = len(states)
    zz_diagonal = np.zeros(dimension, dtype=float)
    row_parts = []
    col_parts = []
    for site in range(n_qubit - 1):
        left = (states >> site) & 1
        right = (states >> (site + 1)) & 1
        zz_diagonal += (1.0 - 2.0 * left) * (1.0 - 2.0 * right)

        rows = np.flatnonzero(left != right).astype(np.int64, copy=False)
        flipped = states[rows] ^ ((1 << site) | (1 << (site + 1)))
        cols = np.searchsorted(states, flipped).astype(np.int64, copy=False)
        if (np.any(cols >= dimension)
                or not np.array_equal(states[cols], flipped)):
            raise ValueError(
                "sector_basis is not closed under nearest-neighbour XY flips")
        row_parts.append(rows)
        col_parts.append(cols)

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    h_xy = sparse.csr_matrix(
        (np.full(len(rows), 2.0), (rows, cols)),
        shape=(dimension, dimension),
        dtype=float,
    )
    h_zz = sparse.diags(zz_diagonal, format="csr")
    return h_xy, h_zz


def exact_krylov_states(hamiltonian, initial_state, depth: int, dt: float):
    """Return exact states at times ``d*dt`` for ``d=0,...,depth-1``."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    state = np.asarray(initial_state, dtype=complex).reshape(-1)
    if hamiltonian.shape != (len(state), len(state)):
        raise ValueError("hamiltonian and initial_state dimensions do not match")
    norm = float(np.linalg.norm(state))
    if norm <= 0:
        raise ValueError("initial_state must have nonzero norm")
    state = state / norm
    if depth == 1:
        return state[None, :]
    generator = (-1j * hamiltonian).tocsr()
    trace_generator = complex(np.sum(generator.diagonal()))
    states = sparse_linalg.expm_multiply(
        generator,
        state,
        start=0.0,
        stop=(depth - 1) * float(dt),
        num=depth,
        endpoint=True,
        traceA=trace_generator,
    )
    return np.asarray(states)


def sample_sector_bitstrings(
        state, sector_basis, n_qubit: int, shots: int, rng):
    """Sample displayed-order full bitstrings from a sector statevector."""
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    basis = np.asarray(sector_basis, dtype=np.int64)
    amplitudes = np.asarray(state, dtype=complex).reshape(-1)
    if basis.ndim != 1 or len(basis) != len(amplitudes):
        raise ValueError("sector_basis and state must be aligned 1D arrays")
    probabilities = np.abs(amplitudes) ** 2
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("state probabilities must have positive finite mass")
    cumulative = np.cumsum(probabilities / total)
    cumulative[-1] = 1.0
    rows = np.searchsorted(cumulative, rng.random(shots), side="right")
    integers = basis[rows]
    shifts = np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    return ((integers[:, None] >> shifts) & 1).astype(np.uint8)
