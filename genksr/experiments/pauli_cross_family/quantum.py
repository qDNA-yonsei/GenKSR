"""Exact small-system quantum backend for the Pauli cross-family experiment."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .protocol import canonicalize_terms, coefficient_l1


@lru_cache(maxsize=None)
def _single_pauli(symbol: str) -> sp.csr_matrix:
    matrices = {
        "I": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    if symbol not in matrices:
        raise ValueError(f"unknown Pauli symbol {symbol!r}")
    return sp.csr_matrix(matrices[symbol])


@lru_cache(maxsize=4096)
def pauli_string_matrix(pauli: str) -> sp.csr_matrix:
    if not pauli:
        raise ValueError("Pauli string cannot be empty")
    matrix = _single_pauli(pauli[0])
    for symbol in pauli[1:]:
        matrix = sp.kron(matrix, _single_pauli(symbol), format="csr")
    return matrix


def pauli_sum_matrix(terms: Sequence[tuple[str, float]]) -> sp.csr_matrix:
    canonical = canonicalize_terms(terms)
    n_qubit = len(canonical[0][0])
    dimension = 1 << n_qubit
    matrix = sp.csr_matrix((dimension, dimension), dtype=np.complex128)
    for pauli, coefficient in canonical:
        matrix = matrix + float(coefficient) * pauli_string_matrix(pauli)
    matrix = matrix.tocsr()
    hermiticity_error = float(sp.linalg.norm(matrix - matrix.getH()))
    if hermiticity_error > 1e-10:
        raise RuntimeError(f"Hamiltonian is not Hermitian: error={hermiticity_error}")
    return matrix


def neel_bitstring(n_qubit: int) -> np.ndarray:
    if n_qubit <= 0 or n_qubit % 2:
        raise ValueError("Neel reference requires a positive even number of qubits")
    return np.asarray([site % 2 for site in range(n_qubit)], dtype=np.uint8)


def bits_to_integers(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] <= 0:
        raise ValueError("bits must have shape (samples, n_qubit)")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be binary")
    powers = 1 << np.arange(bits.shape[1] - 1, -1, -1, dtype=np.int64)
    return bits.astype(np.int64) @ powers


def integers_to_bits(indices: np.ndarray, n_qubit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= 1 << n_qubit):
        raise ValueError("basis index outside Hilbert space")
    shifts = np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def neel_statevector(n_qubit: int) -> np.ndarray:
    bits = neel_bitstring(n_qubit)[None, :]
    index = int(bits_to_integers(bits)[0])
    state = np.zeros(1 << n_qubit, dtype=np.complex128)
    state[index] = 1.0
    return state


def exact_krylov_probabilities(
    hamiltonian: sp.spmatrix, *, n_qubit: int, depth: int, dt: float
) -> tuple[np.ndarray, float]:
    if depth <= 0 or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("depth and dt must be positive")
    initial = neel_statevector(n_qubit)
    if hamiltonian.shape != (1 << n_qubit, 1 << n_qubit):
        raise ValueError("Hamiltonian shape does not match n_qubit")
    if depth == 1:
        states = initial[None, :]
    else:
        states = expm_multiply(
            -1.0j * hamiltonian,
            initial,
            start=0.0,
            stop=(depth - 1) * float(dt),
            num=depth,
            endpoint=True,
        )
    probabilities = np.abs(np.asarray(states)) ** 2
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if normalization_error > 1e-9:
        raise RuntimeError(f"state normalization error {normalization_error}")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float64), normalization_error


def ground_diagnostics(
    hamiltonian: sp.spmatrix, *, n_qubit: int, degeneracy_tol: float = 1e-9
) -> dict[str, float | int]:
    dimension = 1 << n_qubit
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("Hamiltonian shape does not match n_qubit")

    # Dense diagonalization is faster and maximally robust for the tiny smoke
    # tests, but scales cubically and becomes the dominant cost already at
    # n=12.  Only the lowest eigenspace and the first gap are needed here.
    if dimension <= 512:
        dense = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        hermiticity_error = float(np.max(np.abs(dense - dense.conj().T)))
    else:
        hermiticity_error = float(sp.linalg.norm(hamiltonian - hamiltonian.getH()))
        requested = min(8, dimension - 1)
        while True:
            eigenvalues, eigenvectors = sp.linalg.eigsh(
                hamiltonian,
                k=requested,
                which="SA",
                tol=1e-11,
                maxiter=max(10_000, 20 * dimension),
            )
            order = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[order].real
            eigenvectors = eigenvectors[:, order]
            ground = float(eigenvalues[0])
            degeneracy = int(
                np.count_nonzero(eigenvalues <= ground + degeneracy_tol)
            )
            if degeneracy < requested or requested >= min(64, dimension - 1):
                break
            requested = min(2 * requested, 64, dimension - 1)
        if degeneracy == requested:
            raise RuntimeError(
                "could not resolve the first spectral gap with sparse eigensolver"
            )

    ground = float(eigenvalues[0].real)
    ground_mask = eigenvalues <= eigenvalues[0] + degeneracy_tol
    degeneracy = int(np.count_nonzero(ground_mask))
    if degeneracy < len(eigenvalues):
        gap = float((eigenvalues[degeneracy] - eigenvalues[0]).real)
    else:
        gap = 0.0
    initial = neel_statevector(n_qubit)
    overlaps = np.abs(eigenvectors[:, ground_mask].conj().T @ initial) ** 2
    return {
        "ground_energy": ground,
        "spectral_gap_above_ground_space": gap,
        "ground_space_degeneracy": degeneracy,
        "initial_ground_space_overlap": float(np.sum(overlaps).real),
        "hermiticity_error": hermiticity_error,
    }


def sample_probabilities(
    probabilities: np.ndarray, *, shots: int, n_qubit: int, rng: np.random.Generator
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (1 << n_qubit,):
        raise ValueError("probability vector shape mismatch")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if np.any(probabilities < -1e-14) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
        raise ValueError("invalid probability vector")
    clean = np.clip(probabilities, 0.0, None)
    clean /= clean.sum()
    indices = rng.choice(len(clean), size=int(shots), replace=True, p=clean)
    return integers_to_bits(indices, n_qubit)


def projected_ground_energy(
    hamiltonian: sp.spmatrix, basis_indices: Iterable[int]
) -> float:
    dimension = hamiltonian.shape[0]
    basis = np.asarray(sorted({int(index) for index in basis_indices}), dtype=np.int64)
    if basis.size == 0:
        return float("nan")
    if basis[0] < 0 or basis[-1] >= dimension:
        raise ValueError("sampled basis index outside Hamiltonian dimension")
    projected = hamiltonian.tocsr()[basis][:, basis].tocsr()
    if basis.size == 1:
        return float(projected[0, 0].real)
    if basis.size <= 512:
        lowest = la.eigh(
            projected.toarray(),
            eigvals_only=True,
            subset_by_index=(0, 0),
            check_finite=False,
            driver="evr",
        )[0]
    else:
        lowest = sp.linalg.eigsh(
            projected,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1e-10,
            maxiter=max(10_000, 20 * basis.size),
        )[0]
    return float(np.real(lowest))


def cumulative_sampled_energy(
    hamiltonian: sp.spmatrix, samples_by_dimension: Mapping[int, np.ndarray]
) -> float:
    cumulative: set[int] = set()
    for dimension in sorted(samples_by_dimension):
        cumulative.update(bits_to_integers(samples_by_dimension[dimension]).tolist())
    return projected_ground_energy(hamiltonian, cumulative)


def generate_instance_quantum_data(
    terms: Sequence[tuple[str, float]],
    *,
    n_qubit: int,
    depth: int,
    train_shots: int,
    sample_seed: int,
    include_training_samples: bool,
) -> dict[str, np.ndarray | float | int]:
    canonical = canonicalize_terms(terms)
    scale = coefficient_l1(canonical)
    dt = np.pi / scale
    hamiltonian = pauli_sum_matrix(canonical)
    probabilities, norm_error = exact_krylov_probabilities(
        hamiltonian, n_qubit=n_qubit, depth=depth, dt=dt
    )
    if include_training_samples:
        sample_blocks = []
        for dimension, probability in enumerate(probabilities):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(sample_seed), int(dimension), 0x5041554C])
            )
            sample_blocks.append(
                sample_probabilities(
                    probability, shots=train_shots, n_qubit=n_qubit, rng=rng
                )
            )
        samples = np.stack(sample_blocks, axis=0)
    else:
        samples = np.empty((0, 0, n_qubit), dtype=np.uint8)
    diagnostics = ground_diagnostics(hamiltonian, n_qubit=n_qubit)
    return {
        "probabilities": probabilities,
        "samples": samples,
        "normalization_error": norm_error,
        **diagnostics,
    }
