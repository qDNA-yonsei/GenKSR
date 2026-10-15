#!/usr/bin/env python
"""Reproduce paper Appendix A.3: Table 3 and Figure 5.

Ground-state *sparsity* of the 2D J1-J2 Heisenberg model at J2=0.5 on an
LxL torus (periodic boundary conditions), J1=1, isotropic Heisenberg
(delta=1). This script is fully self-contained: it performs exact
diagonalisation only -- no trained model and no measurement data are
required.

Sparsity definition (Yu et al., Appendix A.3)
---------------------------------------------
Given the exact ground-state vector |psi>, sort the squared amplitudes
|<x|psi>|^2 in descending order and define

    L_alpha = smallest number of top-ranked basis states whose
              cumulative probability is >= alpha.

The relevant denominator is the size of the Sz=0 symmetry sector, which
is C(N, N/2) (N = L*L). We report L_alpha together with the fraction
L_alpha / |Sz=0 sector|; a fraction far below alpha means the ground
state concentrates its weight on a tiny corner of the Hilbert space.

Outputs
-------
- table3_sparsity.csv : Table 3 (alpha, L_alpha, |Sz=0|, fraction) for --L.
- fig5_sparsity.pdf / .png : Figure 5, two panels --
    (a) L_0.95 vs |Sz=0 sector| (log-log) across system sizes,
    (b) L_0.95 / |Sz=0 sector| vs system size N.

Feasibility caveat
------------------
Dense-vector exact diagonalisation via scipy.sparse.linalg.eigsh stores a
2^N-dimensional ground-state vector. 3x3 (N=9, 512), 4x4 (N=16, 65536)
are cheap. 5x5 (N=25) is a 2^25 = 33,554,432-dimensional problem whose
sparse Hamiltonian and Lanczos workspace are far too large for this
approach (tens of GB), so it is skipped by default via --max-n 16. Pass a
larger --max-n only if you have the resources; larger sizes are logged and
skipped rather than crashing with OOM.
"""

import argparse
import math
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

import matplotlib
matplotlib.use("Agg")  # headless / deterministic backend
import matplotlib.pyplot as plt  # noqa: E402

from genksr.utils.quantum import (  # noqa: E402
    sample_coupling_matrix_2d_j1j2,
    build_2d_xxz_hamiltonian,
)

J1 = 1.0
J2 = 0.5  # frustrated point studied in Appendix A.3


def compute_ground_state(L, seed=0):
    """Exact ground-state vector of the LxL J1-J2 Heisenberg model at J2=0.5.

    Returns (gs_vec, E_gs, N). The coupling matrix is built with a fixed
    J2_range=(0.5, 0.5), so the result is deterministic regardless of the
    RNG, but we still pass a seeded generator for full reproducibility.
    """
    N = L * L
    rng = np.random.default_rng(seed)
    J = sample_coupling_matrix_2d_j1j2(L, J1=J1, J2_range=(J2, J2), rng=rng)
    H_op = build_2d_xxz_hamiltonian(N=N, J=J, delta=1.0)
    H_sparse = H_op.to_matrix(sparse=True)
    # Deterministic Lanczos start vector. This matters when the ground state
    # is (nearly) degenerate -- e.g. the 3x3 torus -- where the particular
    # eigenvector returned would otherwise depend on the Lanczos RNG and the
    # sorted-probability profile (and thus L_alpha) would not be reproducible.
    v0 = np.random.default_rng(seed).standard_normal(H_sparse.shape[0])
    # 'SA' = smallest algebraic eigenvalue (the ground state).
    eigenvalues, eigenvectors = sla.eigsh(H_sparse, k=1, which="SA", v0=v0)
    order = np.argsort(eigenvalues)
    gs_vec = np.asarray(eigenvectors[:, order[0]]).ravel()
    E_gs = float(eigenvalues[order[0]])
    return gs_vec, E_gs, N


def sorted_cumulative_probs(gs_vec):
    """Probabilities |amplitude|^2 sorted descending, plus their cumsum.

    The full 2^N vector is used. Basis states outside the Sz=0 sector carry
    essentially zero weight for this Sz-conserving Hamiltonian, so sorting
    the full vector is equivalent to restricting to the sector for the
    top-ranked states; the denominator used for the *fraction* is, per the
    paper, the Sz=0 sector size C(N, N/2).
    """
    probs = np.abs(gs_vec) ** 2
    probs = probs / probs.sum()  # guard against tiny normalisation drift
    probs_sorted = np.sort(probs)[::-1]
    cumsum = np.cumsum(probs_sorted)
    return probs_sorted, cumsum


def l_alpha(cumsum, alpha):
    """Smallest count of top states with cumulative probability >= alpha."""
    # searchsorted on a non-decreasing cumsum: first index where cumsum>=alpha.
    idx = int(np.searchsorted(cumsum, alpha))
    return idx + 1  # convert 0-based index to a count


def sz0_sector_size(N):
    return math.comb(N, N // 2)


def build_table3(L, alphas, seed=0):
    """Compute Table 3 rows for a single lattice side length L."""
    print(f"[Table 3] L={L} (N={L * L}) -- exact diagonalisation ...", flush=True)
    t0 = time.time()
    gs_vec, E_gs, N = compute_ground_state(L, seed=seed)
    _, cumsum = sorted_cumulative_probs(gs_vec)
    sector = sz0_sector_size(N)
    print(f"           E_gs = {E_gs:.6f}, |Sz=0| = {sector:,}, "
          f"ED took {time.time() - t0:.1f}s", flush=True)
    rows = []
    for alpha in alphas:
        La = l_alpha(cumsum, alpha)
        frac = La / sector
        rows.append({
            "L": L,
            "N": N,
            "alpha": alpha,
            "L_alpha": La,
            "sz0_sector_size": sector,
            "fraction": frac,
        })
        print(f"           alpha={alpha:.2f}: L_alpha={La:,} "
              f"({100 * frac:.2f}% of Sz=0 sector, "
              f"{frac / alpha:.3f}x uniform)", flush=True)
    return rows


def write_table3_csv(rows, path):
    header = "L,N,alpha,L_alpha,sz0_sector_size,fraction"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['L']},{r['N']},{r['alpha']:.2f},{r['L_alpha']},"
            f"{r['sz0_sector_size']},{r['fraction']:.8e}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[output] wrote {path}", flush=True)


def collect_scaling(sizes, max_n, alpha=0.95, seed=0):
    """L_alpha scaling data for Figure 5 across lattice side lengths.

    Skips (with a log message) any size whose Hilbert-space dimension 2^N
    exceeds 2^max_n, rather than attempting an infeasible dense ED.
    """
    data = []  # list of dicts: L, N, sector, L_alpha, fraction
    for L in sorted(sizes):
        N = L * L
        if N > max_n:
            dim = 2 ** N
            print(f"[Fig 5] SKIP L={L} (N={N}): 2^{N} = {dim:,} > 2^{max_n}; "
                  f"dense eigsh is infeasible (would OOM). "
                  f"Raise --max-n to attempt it.", flush=True)
            continue
        print(f"[Fig 5] L={L} (N={N}) -- exact diagonalisation ...", flush=True)
        t0 = time.time()
        gs_vec, E_gs, _ = compute_ground_state(L, seed=seed)
        _, cumsum = sorted_cumulative_probs(gs_vec)
        sector = sz0_sector_size(N)
        La = l_alpha(cumsum, alpha)
        frac = La / sector
        data.append({
            "L": L, "N": N, "sector": sector,
            "L_alpha": La, "fraction": frac, "E_gs": E_gs,
        })
        print(f"        E_gs={E_gs:.6f}, |Sz=0|={sector:,}, "
              f"L_{alpha}={La:,} ({100 * frac:.2f}%), "
              f"{time.time() - t0:.1f}s", flush=True)
    return data


def make_figure5(scaling, alpha, pdf_path, png_path):
    if len(scaling) == 0:
        print("[Fig 5] no feasible sizes computed; skipping figure.", flush=True)
        return

    Ns = np.array([d["N"] for d in scaling])
    sectors = np.array([d["sector"] for d in scaling], dtype=float)
    Las = np.array([d["L_alpha"] for d in scaling], dtype=float)
    fracs = np.array([d["fraction"] for d in scaling], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) L_alpha vs |Sz=0 sector|, log-log
    ax = axes[0]
    ax.plot(sectors, Las, "o-", color="tab:red", markersize=8, linewidth=2,
            label=fr"$L_{{{alpha:g}}}$ (ground state)")
    # Uniform reference: L_alpha = alpha * |sector| if weight were flat.
    ax.plot(sectors, alpha * sectors, "k--", alpha=0.4,
            label=r"uniform: $\alpha\cdot|S_z{=}0|$")
    for d in scaling:
        ax.annotate(f"{d['L']}x{d['L']}",
                    (d["sector"], d["L_alpha"]),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=10, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|S_z=0|$ sector size $= \binom{N}{N/2}$", fontsize=13)
    ax.set_ylabel(fr"$L_{{{alpha:g}}}$  (states for {100 * alpha:g}% prob.)",
                  fontsize=13)
    ax.set_title("(a) $L_{0.95}$ vs Hilbert-space sector size", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)

    # (b) fraction vs system size N
    ax = axes[1]
    ax.plot(Ns, 100 * fracs, "s-", color="tab:blue", markersize=8, linewidth=2,
            label=fr"$L_{{{alpha:g}}} / |S_z{{=}}0|$")
    ax.axhline(100 * alpha, color="gray", linestyle=":", alpha=0.6,
               label=fr"uniform ({100 * alpha:g}%)")
    for d in scaling:
        ax.annotate(f"{d['L']}x{d['L']}",
                    (d["N"], 100 * d["fraction"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=10, color="gray")
    ax.set_xlabel(r"system size $N = L^2$ (number of qubits)", fontsize=13)
    ax.set_ylabel(fr"$L_{{{alpha:g}}} / |S_z{{=}}0|$  (%)", fontsize=13)
    ax.set_title("(b) ground-state concentration vs $N$", fontsize=13)
    ax.set_xticks(Ns)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.suptitle(
        fr"Figure 5: ground-state sparsity of the 2D $J_1$-$J_2$ Heisenberg "
        fr"model ($J_2={J2}$, isotropic, torus)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[output] wrote {pdf_path}", flush=True)
    print(f"[output] wrote {png_path}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce Appendix A.3 Table 3 and Figure 5 "
                    "(ground-state sparsity of the 2D J1-J2 Heisenberg "
                    "model at J2=0.5).")
    p.add_argument("--L", type=int, default=4,
                   help="lattice side length for Table 3 (default: 4)")
    p.add_argument("--sizes", type=int, nargs="+", default=[3, 4],
                   help="lattice side lengths for Figure 5 (default: 3 4)")
    p.add_argument("--max-n", type=int, default=16,
                   help="skip any size whose N=L^2 exceeds this; guards "
                        "against OOM on infeasible dense ED (default: 16; "
                        "5x5=25 is skipped)")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.90, 0.95, 0.99],
                   help="cumulative-probability thresholds for Table 3 "
                        "(default: 0.90 0.95 0.99)")
    p.add_argument("--output-dir", type=str, default="figures/",
                   help="directory for outputs (default: figures/)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (result is deterministic regardless; "
                        "default: 0)")
    return p.parse_args(argv)


def main(argv=None):
    import os
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Appendix A.3 -- ground-state sparsity, 2D J1-J2 Heisenberg, "
          f"J1={J1}, J2={J2}")
    print("=" * 70)

    # --- Table 3 (single lattice size --L) ---
    table_rows = build_table3(args.L, args.alphas, seed=args.seed)
    csv_path = os.path.join(args.output_dir, "table3_sparsity.csv")
    write_table3_csv(table_rows, csv_path)

    # --- Figure 5 (scaling across --sizes, L_0.95) ---
    if 0.95 not in args.alphas:
        print("[Fig 5] note: Figure 5 always uses alpha=0.95 (L_0.95).",
              flush=True)
    scaling = collect_scaling(args.sizes, args.max_n, alpha=0.95,
                              seed=args.seed)
    pdf_path = os.path.join(args.output_dir, "fig5_sparsity.pdf")
    png_path = os.path.join(args.output_dir, "fig5_sparsity.png")
    make_figure5(scaling, 0.95, pdf_path, png_path)

    print("=" * 70)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
