#!/usr/bin/env python3
"""Reproduce Figure 3, Table 2, and Figure 8 of the GenKSR paper (Section 4.2).

These are the 2D 4x4 J1-J2 Heisenberg SKQD (Sample-based Krylov Quantum
Diagonalization) results.  The script is a *consumer* of the evaluation output
``all_results.pkl`` produced by ``scripts/evaluate.py --task sqkd`` -- it does no
quantum simulation itself, so it is fully deterministic and standalone (only
numpy + matplotlib + pickle, with pandas optional).

all_results.pkl structure (one Hamiltonian id = "hid"):

    {
      '<split>': {
        'exact_energy': { hid(int): float },                       # exact diag GS energy
        'device':       { hid(int): {'energies': [E_1 .. E_maxD]} },# exact-simulation SKQD baseline
        'transformer':  { hid(int): {'energies': [E_1 .. E_maxD]} },
        'mamba':        { hid(int): {'energies': [E_1 .. E_maxD]} },
      }
    }

``energies[i]`` is the CUMULATIVE SKQD energy using the union of all bitstrings
sampled up to Krylov dimension ``i`` -- so the x-axis Krylov dimension is
``D = i + 1`` (D = 1 .. len(energies)).  Some entries may be ``np.nan`` (too few
post-selected bitstrings); those are skipped in every plot/aggregation.

  - 'device'      -> exact-simulation SKQD curve  ("Exact Simulation" / "Exact SKQD")
  - 'transformer' -> Transformer generated samples
  - 'mamba'       -> Mamba generated samples
  - exact_energy[hid] -> flat "Exact Ground State" reference line.

Figures (Section 4.2):
  Figure 3  -- energy convergence vs Krylov dimension D for a chosen Hamiltonian
               at J2 = 0.5: (a) energy vs D, (b) semilog |E(D) - E_exact| vs D.
  Table 2   -- final (largest-D) ground-state energy PER SITE at J2 = 0.5 for the
               chosen Hamiltonian (Exact SKQD / Transformer / Mamba [/ ViT-NQS]).
  Figure 8  -- box plots of the dE = |E(D) - E_exact| distribution over the
               random-J2 test Hamiltonians, for three sample budgets
               {1000, 5000, 10000}, at two Krylov dimensions D_train
               (unhatched) and D_extrap (hatched).

Style follows GenKSR/'compar_skqd copy.ipynb' (energy-vs-D cell ~38, semilog
|dE| cell ~44, per-site energy cells 45-49).

Usage examples
--------------
    python scripts/figures/fig3_table2_fig8_skqd.py --mode fig3 \
        --results evaluation_sqkd_16qubit_kd30/all_results.pkl \
        --output-dir figures/

    python scripts/figures/fig3_table2_fig8_skqd.py --mode table2 \
        --results .../all_results.pkl --vit-energy-per-site -0.51359 \
        --output-dir figures/

    python scripts/figures/fig3_table2_fig8_skqd.py --mode fig8 \
        --results-by-samples 1000:run1k/all_results.pkl \
                             5000:run5k/all_results.pkl \
                             10000:run10k/all_results.pkl \
        --d-train 15 --d-extrap 30 --output-dir figures/

    python scripts/figures/fig3_table2_fig8_skqd.py --mode all \
        --results .../all_results.pkl --vit-energy-per-site -0.51359 \
        --results-by-samples 1000:r1k.pkl 5000:r5k.pkl 10000:r10k.pkl \
        --output-dir figures/
"""
import argparse
import csv
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless / deterministic rendering
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Normalisation convention (Table 2)
# ---------------------------------------------------------------------------
# The 4x4 lattice has L*L = 16 sites (= 16 qubits).  The SKQD Hamiltonian is built
# in the Pauli basis with terms J*(X_iX_j + Y_iY_j + Z_iZ_j) per bond.  The
# textbook spin-1/2 Heisenberg convention writes the bond term as J * S_i . S_j
# with S = sigma/2, i.e.  J * S_i.S_j = (J/4)*(X_iX_j + Y_iY_j + Z_iZ_j).
# Hence to convert the *total* Pauli-basis energy returned by the evaluator into
# the spin-1/2 energy *per site*, we divide by 4 (Pauli->spin) and by 16 (sites):
#
#       energy_per_site = total_energy / (L * L * 4) = total_energy / 64
#
# This exactly replicates the original notebook 'compar_skqd copy.ipynb'
# cells 45-49, which all use  `... / (4 * 4 * 4)` (= /64).
L = 4                       # 4x4 lattice
N_SITES = L * L            # 16 sites = 16 qubits
SPIN_FACTOR = 4           # Pauli (XX+YY+ZZ) -> spin-1/2 (S.S) conversion factor
PER_SITE_DIVISOR = N_SITES * SPIN_FACTOR   # = 64

# Consistent per-method plot styling across figures.
STYLE = {
    "device":      dict(color="red",   linestyle=":",  marker="^", label="Exact Simulation"),
    "transformer": dict(color="blue",  linestyle="-.", marker="o", label="Transformer"),
    "mamba":       dict(color="green", linestyle="--", marker="s", label="Mamba"),
}
METHODS = ["device", "transformer", "mamba"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_results(path):
    """Load an all_results.pkl, accepting either the file or its parent dir."""
    if os.path.isdir(path):
        path = os.path.join(path, "all_results.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"all_results.pkl not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def pick_split(results, preferred="test"):
    """Return (split_name, split_dict). Prefer 'test', else first available."""
    if not results:
        raise ValueError("Empty results object.")
    if preferred in results:
        return preferred, results[preferred]
    split = sorted(results.keys())[0]
    return split, results[split]


def pick_hamiltonian(split_dict, requested=None):
    """Choose the Hamiltonian id. Default = smallest hid with an exact energy."""
    hids = sorted(split_dict.get("exact_energy", {}).keys())
    if not hids:
        raise ValueError("No exact_energy entries found in results split.")
    if requested is None:
        return hids[0]
    if requested not in hids:
        raise ValueError(f"--hamiltonian {requested} not in available hids {hids}.")
    return requested


def energies_for(split_dict, method, hid):
    """Return the energies array for (method, hid) as float ndarray, or None."""
    entry = split_dict.get(method, {}).get(hid)
    if entry is None:
        return None
    energies = np.asarray(entry["energies"], dtype=float)
    return energies if energies.size else None


def last_valid(energies):
    """Final (largest-D, last non-nan) energy value, or None if all nan."""
    if energies is None:
        return None
    valid = np.where(np.isfinite(energies))[0]
    return float(energies[valid[-1]]) if valid.size else None


def _ensure_outdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Figure 3 : energy convergence at J2 = 0.5
# ---------------------------------------------------------------------------
def make_figure3(results, hamiltonian, output_dir, d_min=1):
    """Two-panel convergence figure for one Hamiltonian (style ~ notebook cells 38/44)."""
    split_name, split = pick_split(results)
    hid = pick_hamiltonian(split, hamiltonian)
    exact = float(split["exact_energy"][hid])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel (a): energy vs Krylov dimension D ---------------------------
    # Reference flat line at the exact ground-state energy, spanning the full
    # x-range so it reads as a target (notebook cell 38).
    d_max = 1
    for method in METHODS:
        e = energies_for(split, method, hid)
        if e is not None:
            d_max = max(d_max, e.size)

    ax_a.plot([d_min, d_max], [exact, exact], color="black", linestyle="-",
              linewidth=2, label="Exact Ground State", zorder=10)

    for method in METHODS:
        e = energies_for(split, method, hid)
        if e is None:
            continue
        d = np.arange(1, e.size + 1)
        mask = np.isfinite(e) & (d >= d_min)
        if not mask.any():
            continue
        st = STYLE[method]
        ax_a.plot(d[mask], e[mask], color=st["color"], linestyle=st["linestyle"],
                  marker=st["marker"], markersize=4, alpha=0.8, label=st["label"])

    ax_a.set_xlabel("Krylov Dimension $D$", fontsize=18)
    ax_a.set_ylabel("Energy", fontsize=18)
    ax_a.tick_params(axis="both", which="major", labelsize=14)
    ax_a.legend(fontsize=13, loc="best")
    ax_a.grid(True, alpha=0.3)
    ax_a.set_title("(a) Energy convergence", fontsize=16)

    # --- Panel (b): semilog |E(D) - E_exact| vs D --------------------------
    # Drop nan and non-positive |dE| so the log scale is well defined
    # (notebook cell 44).
    plotted_any = False
    for method in METHODS:
        e = energies_for(split, method, hid)
        if e is None:
            continue
        d = np.arange(1, e.size + 1)
        dE = np.abs(e - exact)
        mask = np.isfinite(dE) & (dE > 0) & (d >= d_min)
        if not mask.any():
            continue
        st = STYLE[method]
        ax_b.semilogy(d[mask], dE[mask], color=st["color"], linestyle=st["linestyle"],
                      marker=st["marker"], markersize=4, alpha=0.8, label=st["label"])
        plotted_any = True

    ax_b.set_xlabel("Krylov Dimension $D$", fontsize=18)
    ax_b.set_ylabel(r"$|E(D) - E_{\mathrm{exact}}|$ (log scale)", fontsize=18)
    ax_b.tick_params(axis="both", which="major", labelsize=14)
    if plotted_any:
        ax_b.legend(fontsize=13, loc="best")
    ax_b.grid(True, alpha=0.3, which="both")
    ax_b.set_title("(b) Absolute energy error", fontsize=16)

    fig.suptitle(
        f"SKQD convergence (4x4 J1-J2, J2=0.5)  -  split='{split_name}', Hamiltonian id={hid}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf = os.path.join(output_dir, "fig3_skqd.pdf")
    png = os.path.join(output_dir, "fig3_skqd.png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"[fig3] split='{split_name}' hid={hid} exact={exact:.6f}")
    print(f"[fig3] saved {pdf}")
    print(f"[fig3] saved {png}")
    return pdf, png


# ---------------------------------------------------------------------------
# Table 2 : ground-state energy per site at J2 = 0.5
# ---------------------------------------------------------------------------
def make_table2(results, hamiltonian, vit_energy_per_site, output_dir):
    """Per-site final SKQD energy table for one Hamiltonian (notebook cells 45-49)."""
    split_name, split = pick_split(results)
    hid = pick_hamiltonian(split, hamiltonian)
    exact_total = float(split["exact_energy"][hid])

    # Column order matches the paper Table 2.
    col_labels = {
        "device": "Exact SKQD",
        "transformer": "Transformer",
        "mamba": "Mamba",
    }

    rows = []  # (method_label, total_energy_or_None, per_site_or_None)

    # Exact diagonalisation reference (per-site), useful sanity row.
    rows.append(("Exact GS", exact_total, exact_total / PER_SITE_DIVISOR))

    for method in METHODS:
        e = energies_for(split, method, hid)
        total = last_valid(e)
        per_site = (total / PER_SITE_DIVISOR) if total is not None else None
        rows.append((col_labels[method], total, per_site))

    # ViT-NQS column supplied externally (already per-site); print N/A if absent.
    rows.append(("ViT-NQS", None, vit_energy_per_site))

    # ----- pretty print -----
    print("=" * 72)
    print(f"Table 2: Ground-state energy per site (J2=0.5)  "
          f"split='{split_name}' hid={hid}")
    print(f"  per-site divisor = L*L*4 = {N_SITES}*{SPIN_FACTOR} = {PER_SITE_DIVISOR}")
    print("=" * 72)
    print(f"{'Method':<14}{'Total energy':>18}{'Energy / site':>18}")
    print("-" * 72)
    for label, total, per_site in rows:
        total_str = f"{total:.6f}" if total is not None else "N/A"
        ps_str = f"{per_site:.6f}" if per_site is not None else "N/A"
        print(f"{label:<14}{total_str:>18}{ps_str:>18}")
    print("=" * 72)

    # ----- CSV -----
    csv_path = os.path.join(output_dir, "table2_energy_per_site.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "split", "hamiltonian_id",
                         "total_energy", "energy_per_site", "per_site_divisor"])
        for label, total, per_site in rows:
            writer.writerow([
                label, split_name, hid,
                "" if total is None else f"{total:.10f}",
                "" if per_site is None else f"{per_site:.10f}",
                PER_SITE_DIVISOR,
            ])
    print(f"[table2] saved {csv_path}")
    return csv_path, rows


# ---------------------------------------------------------------------------
# Figure 8 : box plots of dE distributions across sample budgets
# ---------------------------------------------------------------------------
def _de_distribution(split, method, d):
    """Collect dE = |E(D=d) - E_exact| over all Hamiltonians for one method/D.

    ``d`` is the Krylov dimension (1-indexed); the energies array index is d-1.
    nan / non-finite entries are dropped.
    """
    vals = []
    method_dict = split.get(method, {})
    exact_map = split.get("exact_energy", {})
    for hid, entry in method_dict.items():
        if hid not in exact_map:
            continue
        e = np.asarray(entry["energies"], dtype=float)
        idx = d - 1
        if idx < 0 or idx >= e.size:
            continue
        val = e[idx]
        if not np.isfinite(val):
            continue
        dE = abs(val - float(exact_map[hid]))
        if np.isfinite(dE):
            vals.append(dE)
    return np.asarray(vals, dtype=float)


def make_figure8(results_by_samples, d_train, d_extrap, output_dir):
    """One box-plot panel per sample budget; box dE over all test Hamiltonians.

    Within each panel, for each method we draw two adjacent boxes:
      - D = d_train  (unhatched)
      - D = d_extrap (hatched)
    Each box shows the median line, IQR box, and range whiskers over the set of
    test Hamiltonians (matplotlib default boxplot, whis=(0, 100) for full range).
    """
    budgets = sorted(results_by_samples.keys())
    n_panels = len(budgets)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    method_labels = [STYLE[m]["label"] for m in METHODS]
    method_colors = [STYLE[m]["color"] for m in METHODS]
    # Two boxes per method group: left = d_train, right = d_extrap.
    group_centres = np.arange(len(METHODS)) * 1.0
    offset = 0.18
    box_width = 0.30

    for ax, budget in zip(axes, budgets):
        results = load_results(results_by_samples[budget])
        split_name, split = pick_split(results)

        positions, data, facecolors, hatches = [], [], [], []
        for gi, method in enumerate(METHODS):
            color = STYLE[method]["color"]
            for di, (d, hatch) in enumerate([(d_train, ""), (d_extrap, "//")]):
                dist = _de_distribution(split, method, d)
                if dist.size == 0:
                    # Keep an empty placeholder so spacing stays aligned.
                    dist = np.array([np.nan])
                pos = group_centres[gi] + (offset if di else -offset)
                positions.append(pos)
                data.append(dist)
                facecolors.append(color)
                hatches.append(hatch)

        # Replace all-nan placeholders with empty arrays for boxplot.
        clean = [d[np.isfinite(d)] for d in data]
        plot_data = [(d if d.size else [0.0]) for d in clean]

        bp = ax.boxplot(
            plot_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            whis=(0, 100),                      # whiskers span the full range
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, fc, hatch, src in zip(bp["boxes"], facecolors, hatches, clean):
            patch.set_facecolor(fc)
            patch.set_alpha(0.45 if hatch == "" else 0.75)
            if hatch:
                patch.set_hatch(hatch)
            if src.size == 0:
                patch.set_facecolor("white")  # no data for this cell

        ax.set_yscale("log")
        ax.set_xticks(group_centres)
        ax.set_xticklabels(method_labels, fontsize=13)
        ax.tick_params(axis="y", which="major", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_title(f"{budget} samples", fontsize=15)

    axes[0].set_ylabel(r"$|E(D) - E_{\mathrm{exact}}|$ (log scale)", fontsize=16)

    # Shared legend: solid (D_train) vs hatched (D_extrap).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="gray", alpha=0.45, label=f"$D = {d_train}$ (training)"),
        Patch(facecolor="gray", alpha=0.75, hatch="//", label=f"$D = {d_extrap}$ (extrapolation)"),
    ]
    fig.legend(handles=legend_handles, fontsize=13, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("SKQD energy-error distributions over random-J2 test Hamiltonians "
                 "(4x4 J1-J2)", fontsize=15, y=1.07)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "fig8_skqd_boxplots.pdf")
    png = os.path.join(output_dir, "fig8_skqd_boxplots.png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig8] budgets={budgets} D_train={d_train} D_extrap={d_extrap}")
    print(f"[fig8] saved {pdf}")
    print(f"[fig8] saved {png}")
    return pdf, png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_results_by_samples(items):
    """Parse ['1000:pathA', '5000:pathB'] -> {1000: 'pathA', 5000: 'pathB'}."""
    out = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"--results-by-samples entry must be 'M:path', got: {item}")
        m_str, path = item.split(":", 1)
        out[int(m_str)] = path
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce GenKSR Figure 3 / Table 2 / Figure 8 (2D 4x4 J1-J2 SKQD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["fig3", "table2", "fig8", "all"], default="all")
    p.add_argument("--results", type=str, default=None,
                   help="all_results.pkl (or its dir) for fig3/table2.")
    p.add_argument("--hamiltonian", type=int, default=None,
                   help="Hamiltonian id for fig3/table2 (default: first hid).")
    p.add_argument("--vit-energy-per-site", type=float, default=None,
                   help="External ViT-NQS energy-per-site for Table 2 (per-site units).")
    p.add_argument("--results-by-samples", nargs="+", default=None,
                   help="For fig8: list of 'M:path' (e.g. 1000:a.pkl 5000:b.pkl 10000:c.pkl).")
    p.add_argument("--d-train", type=int, default=15,
                   help="Training Krylov dimension D for Figure 8 (unhatched boxes).")
    p.add_argument("--d-extrap", type=int, default=30,
                   help="Extrapolation Krylov dimension D for Figure 8 (hatched boxes).")
    p.add_argument("--output-dir", type=str, default="figures/")
    p.add_argument("--d-min", type=int, default=1,
                   help="Minimum Krylov dimension shown in Figure 3.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = _ensure_outdir(args.output_dir)

    want_fig3 = args.mode in ("fig3", "all")
    want_table2 = args.mode in ("table2", "all")
    want_fig8 = args.mode in ("fig8", "all")

    if want_fig3 or want_table2:
        if args.results is None:
            raise SystemExit("--results is required for fig3/table2/all.")
        results = load_results(args.results)
        if want_fig3:
            make_figure3(results, args.hamiltonian, output_dir, d_min=args.d_min)
        if want_table2:
            make_table2(results, args.hamiltonian, args.vit_energy_per_site, output_dir)

    if want_fig8:
        rbs = parse_results_by_samples(args.results_by_samples)
        if not rbs:
            if args.mode == "fig8":
                raise SystemExit("--results-by-samples is required for fig8.")
            print("[fig8] skipped: no --results-by-samples provided.")
        else:
            make_figure8(rbs, args.d_train, args.d_extrap, output_dir)


if __name__ == "__main__":
    main()
