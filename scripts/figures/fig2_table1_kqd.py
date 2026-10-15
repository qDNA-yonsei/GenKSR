#!/usr/bin/env python3
"""Reproduce paper Figure 2 and Table 1 (Section 4.1, 1D Heisenberg KQD).

This standalone, deterministic script consumes the output of
``scripts/evaluate.py --task kqd`` and reproduces:

Figure 2 (energy-error vs Krylov dimension, averaged over the *test*
Hamiltonians):
    Panel (a)  Exact-simulation ('ideal') vs Classical Shadow ('shadow').
    Panel (b)  Exact-simulation ('ideal') vs Transformer vs Mamba.
    For each method the per-Hamiltonian error is
        dE[hid, d] = | exact_energy(hid) - E_method(hid)[d] |
    where ``exact_energy`` is always taken from the 'ideal' entry of that
    Hamiltonian (it is the only place the true ground-state energy is stored).
    The mean over Hamiltonians is plotted with a +/- std shaded band
    (matplotlib ``fill_between``).  Saved as ``fig2_kqd.{pdf,png}``.

    Optional single-Hamiltonian mode (``--hamiltonian <hid>``) instead draws
    the literal Fig. 2 panel for one Hamiltonian: the KQD *energy* vs Krylov
    dimension for every method together with a horizontal line at the exact
    ground-state energy (``plt.axhline``).  This is the per-Hamiltonian
    energy-vs-D view used in ``GenKSR/plot_kqd.ipynb``.

Table 1 (RMSE, one row per (n_qubit, shots) data_comparison directory):
    For each method two scalars are reported, both in units of energy:
        rmse           = sqrt( mean over (test Ham, Krylov dim) of err^2 )
                         where err = | E_exact - E_method | is read from
                         ``test_{method}_errors.txt``.  This is the "error
                         against the true ground state" RMSE.
        rmse_vs_ideal  = sqrt( mean (E_method - E_ideal)^2 )
                         reconstructed from the *_errors.txt files as
                         (err_method - err_ideal) under the sign convention
                         E = E_exact - err (see ``_rmse_vs_ideal``).  This
                         matches the paper's RMSE formula, which references
                         the exact-simulation (ideal) energy rather than the
                         true ground state.
    Printed as a table and written to ``table1_rmse.csv``.

Dependencies: numpy, matplotlib, pickle, argparse, os, glob (pandas optional,
only used to pretty-print the table if available).  matplotlib runs on the Agg
backend so the module is import-safe and headless-safe.

Examples
--------
Both figure and table from a single run directory::

    python scripts/figures/fig2_table1_kqd.py \
        --results data_comparison_15qubit_10000samples/all_results.pkl \
        --data-comparison-dirs data_comparison_15qubit_10000samples \
        --output-dir figures/

Single-Hamiltonian energy-vs-D panel for test Hamiltonian 3::

    python scripts/figures/fig2_table1_kqd.py --mode figure \
        --results data_comparison_15qubit_10000samples \
        --hamiltonian 3
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")  # import-safe / headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# Plot style (mirrors GenKSR/plot_kqd.ipynb)
# --------------------------------------------------------------------------- #
METHODS = ["ideal", "shadow", "transformer", "mamba"]
COLORS = {"ideal": "black", "transformer": "red", "mamba": "blue", "shadow": "green"}
MARKERS = {"ideal": "o", "transformer": "s", "mamba": "^", "shadow": "d"}
LABELS = {
    "ideal": "Exact simulation",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "shadow": "Classical Shadow",
}

DEFAULT_OUTPUT_DIR = "figures"
_DIR_PATTERN = re.compile(r"data_comparison_(\d+)qubit_(\d+)samples")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _resolve_results_path(results):
    """Accept either an all_results.pkl path or a directory containing it."""
    if os.path.isdir(results):
        candidate = os.path.join(results, "all_results.pkl")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"No all_results.pkl in directory: {results}")
        return candidate
    if not os.path.exists(results):
        raise FileNotFoundError(f"Results file not found: {results}")
    return results


def load_results(results):
    """Load the KQD all_results.pkl. Returns (results_dict, source_dir)."""
    path = _resolve_results_path(results)
    with open(path, "rb") as f:
        all_results = pickle.load(f)
    return all_results, os.path.dirname(os.path.abspath(path))


def _infer_max_dim(all_results, split):
    """Length of the energies list (= max Krylov dimension)."""
    ideal = all_results[split].get("ideal", {})
    if not ideal:
        raise ValueError(
            f"Split '{split}' has no 'ideal' entries; cannot compute the "
            f"exact-energy reference. Re-run evaluate.py with --models ideal "
            f"on this split, or pick a split that is populated."
        )
    any_hid = next(iter(ideal))
    return len(ideal[any_hid]["energies"])


def parse_dir_tag(path):
    """Parse n_qubit/shots from a data_comparison_{N}qubit_{M}samples dir name.

    Returns (n_qubit, shots) as ints, or (None, None) if the pattern is absent.
    Works on the directory name regardless of trailing components (e.g. _2d).
    """
    name = os.path.basename(os.path.normpath(path))
    m = _DIR_PATTERN.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# --------------------------------------------------------------------------- #
# Error statistics for the figure (computed directly from the pkl)
# --------------------------------------------------------------------------- #
def compute_error_statistics(all_results, split, methods=None):
    """Per-method energy-error statistics vs Krylov dimension.

    dE[hid, d] = | exact_energy(hid) - E_method(hid)[d] |, with exact_energy
    taken from the 'ideal' entry of that Hamiltonian.  Hamiltonians are those
    present in BOTH the 'ideal' reference and the given method, ordered by hid.

    Returns: dict method -> {'mean','std','raw' (n_ham x maxD)}.
    """
    if methods is None:
        methods = METHODS
    split_data = all_results[split]
    ideal = split_data["ideal"]

    stats = {}
    for method in methods:
        if method not in split_data or len(split_data[method]) == 0:
            continue
        rows = []
        for hid in sorted(split_data[method].keys()):
            if hid not in ideal:
                continue
            exact = ideal[hid]["exact_energy"]
            energies = np.asarray(split_data[method][hid]["energies"], dtype=float)
            rows.append(np.abs(exact - energies))
        if not rows:
            continue
        arr = np.array(rows)
        stats[method] = {
            "mean": np.nanmean(arr, axis=0),
            "std": np.nanstd(arr, axis=0),
            "raw": arr,
        }
    return stats


# --------------------------------------------------------------------------- #
# Figure 2
# --------------------------------------------------------------------------- #
def _plot_mean_band(ax, krylov_dims, stats, methods):
    """Plot mean error with +/- std band for the given methods on one axis."""
    for method in methods:
        if method not in stats:
            continue
        mean = stats[method]["mean"]
        std = stats[method]["std"]
        ax.plot(
            krylov_dims, mean,
            color=COLORS[method], marker=MARKERS[method], linestyle="-",
            label=LABELS[method], linewidth=2, markersize=7,
        )
        ax.fill_between(
            krylov_dims, mean - std, mean + std,
            color=COLORS[method], alpha=0.2,
        )
    ax.set_xlabel("Krylov Dimension", fontsize=18)
    ax.set_ylabel(r"$\Delta E$", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=12)


def make_figure2(all_results, split, output_dir):
    """Two-panel mean-error figure. Returns list of written file paths."""
    stats = compute_error_statistics(all_results, split)
    maxd = _infer_max_dim(all_results, split)
    krylov_dims = np.arange(1, maxd + 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel (a): ideal vs Classical Shadow
    _plot_mean_band(ax_a, krylov_dims, stats, ["ideal", "shadow"])
    ax_a.set_title("(a) Exact simulation vs Classical Shadow", fontsize=15)

    # Panel (b): ideal vs Transformer vs Mamba
    _plot_mean_band(ax_b, krylov_dims, stats, ["ideal", "transformer", "mamba"])
    ax_b.set_title("(b) Exact simulation vs Transformer / Mamba", fontsize=15)

    fig.suptitle(
        f"Figure 2 — KQD energy error vs Krylov dimension ({split} set)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        fp = os.path.join(output_dir, f"fig2_kqd.{ext}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        written.append(fp)
    plt.close(fig)
    return written


def make_figure2_single_hamiltonian(all_results, split, hid, output_dir):
    """Literal Fig. 2 panel for a single Hamiltonian: energy vs Krylov dim for
    every method + an axhline at the exact ground-state energy.

    Returns list of written file paths.
    """
    split_data = all_results[split]
    if hid not in split_data["ideal"]:
        raise KeyError(
            f"Hamiltonian {hid} not in {split}['ideal']. "
            f"Available: {sorted(split_data['ideal'].keys())}"
        )
    exact = split_data["ideal"][hid]["exact_energy"]
    maxd = len(split_data["ideal"][hid]["energies"])
    krylov_dims = np.arange(1, maxd + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(
        y=exact, color="gray", linestyle="--", linewidth=2, alpha=0.8,
        label="Exact energy",
    )
    for method in METHODS:
        if method not in split_data or hid not in split_data[method]:
            continue
        energies = np.asarray(split_data[method][hid]["energies"], dtype=float)
        ax.plot(
            krylov_dims, energies,
            color=COLORS[method], marker=MARKERS[method], linestyle="-",
            label=LABELS[method], linewidth=2, markersize=6,
        )
    ax.set_xlabel("Krylov Dimension", fontsize=18)
    ax.set_ylabel("Energy", fontsize=18)
    ax.set_title(
        f"Figure 2 — {split} Hamiltonian {hid}: energy vs Krylov dimension",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        fp = os.path.join(output_dir, f"fig2_kqd_hamiltonian{hid}.{ext}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        written.append(fp)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Table 1 (RMSE)
# --------------------------------------------------------------------------- #
def _load_errors(data_dir, split, method):
    """Load test_{method}_errors.txt as a 2D array (n_ham x maxD) or None.

    np.loadtxt collapses a single-row file to 1D; restore the row axis so the
    downstream RMSE math is shape-stable.
    """
    fp = os.path.join(data_dir, f"{split}_{method}_errors.txt")
    if not os.path.exists(fp):
        return None
    arr = np.loadtxt(fp)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _rmse(errors):
    """sqrt(mean(errors^2)) over all entries; errors = |E_exact - E_method|."""
    return float(np.sqrt(np.mean(np.square(errors))))


def _rmse_vs_ideal(err_method, err_ideal):
    """RMSE of (E_method - E_ideal) reconstructed from the error files.

    The *_errors.txt store err = |E_exact - E| = E_exact - E because the KQD
    energy estimate sits above the true ground state (E >= E_exact). Hence
    E = E_exact - err and
        E_method - E_ideal = (E_exact - err_method) - (E_exact - err_ideal)
                           = err_ideal - err_method.
    The exact_energy cancels, so this is the paper's RMSE referenced to the
    exact-simulation (ideal) energy.  Falls back to None on shape mismatch.
    """
    if err_method is None or err_ideal is None:
        return None
    if err_method.shape != err_ideal.shape:
        return None
    return float(np.sqrt(np.mean(np.square(err_ideal - err_method))))


def compute_table1(data_dir, split):
    """RMSE row(s) for one data_comparison directory.

    Returns: dict with 'n_qubit','shots','dir' and per-method
    {'rmse','rmse_vs_ideal'} entries (None when a file is missing).
    """
    n_qubit, shots = parse_dir_tag(data_dir)
    err_ideal = _load_errors(data_dir, split, "ideal")

    row = {
        "dir": os.path.basename(os.path.normpath(data_dir)),
        "n_qubit": n_qubit,
        "shots": shots,
        "methods": {},
    }
    for method in METHODS:
        err = _load_errors(data_dir, split, method)
        if err is None:
            row["methods"][method] = {"rmse": None, "rmse_vs_ideal": None}
            continue
        rmse = _rmse(err)
        # ideal-vs-ideal is trivially 0; report it for completeness.
        rmse_vi = _rmse_vs_ideal(err, err_ideal)
        row["methods"][method] = {"rmse": rmse, "rmse_vs_ideal": rmse_vi}
    return row


def _fmt(x):
    return "n/a" if x is None else f"{x:.6e}"


def print_table1(rows, split):
    """Pretty-print the RMSE table. Uses pandas if available, else plain text."""
    header_cols = ["dir", "n_qubit", "shots", "method", "rmse", "rmse_vs_ideal"]
    records = []
    for row in rows:
        for method in METHODS:
            m = row["methods"].get(method)
            if m is None:
                continue
            records.append({
                "dir": row["dir"],
                "n_qubit": row["n_qubit"],
                "shots": row["shots"],
                "method": method,
                "rmse": m["rmse"],
                "rmse_vs_ideal": m["rmse_vs_ideal"],
            })

    print(f"\nTable 1 — KQD RMSE ({split} set)")
    print("  rmse          = sqrt(mean |E_exact - E_method|^2)")
    print("  rmse_vs_ideal = sqrt(mean (E_method - E_ideal)^2)  [paper formula]")
    print("-" * 92)

    try:
        import pandas as pd  # optional
        df = pd.DataFrame.from_records(records, columns=header_cols)
        with pd.option_context("display.float_format", lambda v: f"{v:.6e}"):
            print(df.to_string(index=False))
    except Exception:
        widths = {"dir": 40, "n_qubit": 8, "shots": 8, "method": 12,
                  "rmse": 16, "rmse_vs_ideal": 16}
        head = "".join(h.ljust(widths[h]) for h in header_cols)
        print(head)
        for r in records:
            line = (
                str(r["dir"]).ljust(widths["dir"])
                + str(r["n_qubit"]).ljust(widths["n_qubit"])
                + str(r["shots"]).ljust(widths["shots"])
                + str(r["method"]).ljust(widths["method"])
                + _fmt(r["rmse"]).ljust(widths["rmse"])
                + _fmt(r["rmse_vs_ideal"]).ljust(widths["rmse_vs_ideal"])
            )
            print(line)
    print("-" * 92)
    return records


def write_table1_csv(records, output_dir):
    """Write table1_rmse.csv (stdlib csv; no pandas dependency)."""
    import csv
    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, "table1_rmse.csv")
    cols = ["dir", "n_qubit", "shots", "method", "rmse", "rmse_vs_ideal"]
    with open(fp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in records:
            out = dict(r)
            out["rmse"] = "" if r["rmse"] is None else r["rmse"]
            out["rmse_vs_ideal"] = "" if r["rmse_vs_ideal"] is None else r["rmse_vs_ideal"]
            writer.writerow(out)
    return fp


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_figure(args):
    all_results, _ = load_results(args.results)
    if args.split not in all_results:
        raise KeyError(f"Split '{args.split}' not in results "
                       f"(available: {list(all_results.keys())})")
    if args.hamiltonian is not None:
        written = make_figure2_single_hamiltonian(
            all_results, args.split, args.hamiltonian, args.output_dir)
    else:
        written = make_figure2(all_results, args.split, args.output_dir)
    for fp in written:
        print(f"Wrote figure: {fp}")
    return written


def run_table(args):
    if not args.data_comparison_dirs:
        raise ValueError("--data-comparison-dirs is required for the table.")
    rows = [compute_table1(d, args.split) for d in args.data_comparison_dirs]
    records = print_table1(rows, args.split)
    csv_path = write_table1_csv(records, args.output_dir)
    print(f"Wrote table: {csv_path}")
    return csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce paper Figure 2 and Table 1 (Section 4.1, 1D KQD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["figure", "table", "both"], default="both",
                   help="Which artifact(s) to produce.")
    p.add_argument("--results", type=str, default=None,
                   help="Path to all_results.pkl or its directory (for the figure).")
    p.add_argument("--data-comparison-dirs", nargs="+", default=None,
                   help="One or more data_comparison_{N}qubit_{M}samples dirs "
                        "(for the table).")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"],
                   help="Which split to use.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for output figures and CSV.")
    p.add_argument("--hamiltonian", type=int, default=None,
                   help="If set, draw the single-Hamiltonian energy-vs-D panel "
                        "instead of the averaged two-panel figure.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    do_figure = args.mode in ("figure", "both")
    do_table = args.mode in ("table", "both")

    if do_figure:
        if args.results is None:
            if args.mode == "figure":
                raise ValueError("--results is required for --mode figure.")
            print("[skip figure] --results not provided.")
        else:
            run_figure(args)

    if do_table:
        if not args.data_comparison_dirs:
            if args.mode == "table":
                raise ValueError(
                    "--data-comparison-dirs is required for --mode table.")
            print("[skip table] --data-comparison-dirs not provided.")
        else:
            run_table(args)


if __name__ == "__main__":
    main()
