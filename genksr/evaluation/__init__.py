"""Evaluation utilities for the paper's energy results.

* ``kqd``  — KQD generalized-eigenvalue energy (Section 4.1, Figure 2 / Table 1).
* ``skqd`` — sample-based KQD energy via SQD diagonalization (Section 4.2, Figure 3 / Table 2 / Figure 8).

Submodules import their heavy dependency (pennylane) lazily, so ``import genksr.evaluation``
succeeds even in an SKQD-only environment.
"""
from .kqd import (
    exact_ground_state,
    ideal_krylov_energies,
    shadow_krylov_energies,
)
from .skqd import (
    exact_ground_energy,
    skqd_energy_from_counts,
    skqd_energies_per_dim,
)

__all__ = [
    "exact_ground_state",
    "ideal_krylov_energies",
    "shadow_krylov_energies",
    "exact_ground_energy",
    "skqd_energy_from_counts",
    "skqd_energies_per_dim",
]
