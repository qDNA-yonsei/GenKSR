#!/usr/bin/env python3
"""Plot the registered XXZ training-budget accuracy criteria."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path,
        default=Path(
            "artifacts/v3_xxz_training_budget_20260801/"
            "formal_evaluation_seed0"))
    return parser.parse_args()


def as_bool(value):
    return str(value).lower() == "true"


def main():
    args = parse_args()
    with (args.evaluation_dir / "cell_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    protocol = json.loads((args.evaluation_dir / "protocol.json").read_text())
    criterion = protocol["primary_accuracy_criterion"]
    nll_limit = float(
        criterion[
            "nll_gap_to_same_N_full_budget_anchor_nats_per_qubit_max"])
    energy_limit = float(
        criterion[
            "mean_absolute_normalized_energy_difference_vs_direct_max"])

    sizes = [12, 16, 20]
    hamiltonians = [9, 18, 36, 45]
    colors = dict(zip(hamiltonians, plt.cm.viridis([0.05, 0.35, 0.65, 0.9])))
    markers = {125: "o", 250: "s", 500: "^", 1000: "D"}
    fig, axes = plt.subplots(
        2, 3, figsize=(11.2, 6.6), sharex="col", constrained_layout=True)

    for column, n_qubit in enumerate(sizes):
        selected = [row for row in rows if int(row["N"]) == n_qubit]
        for num_hamiltonians in hamiltonians:
            group = sorted(
                [row for row in selected
                 if int(row["num_train_hamiltonians"]) == num_hamiltonians],
                key=lambda row: int(row["train_shots_per_condition"]))
            xs = [int(row["fit_samples"]) for row in group]
            axes[0, column].plot(
                xs, [float(row["nll_gap_to_full_budget_anchor"]) for row in group],
                color=colors[num_hamiltonians], alpha=0.55, linewidth=1.0)
            axes[1, column].plot(
                xs,
                [float(row["absolute_normalized_energy_difference_mean"])
                 for row in group],
                color=colors[num_hamiltonians], alpha=0.55, linewidth=1.0)
            for row in group:
                x = int(row["fit_samples"])
                train_shots = int(row["train_shots_per_condition"])
                passed = as_bool(row["passes_primary_joint_criterion"])
                style = {
                    "marker": markers[train_shots],
                    "color": colors[num_hamiltonians],
                    "markeredgecolor": "black" if passed else "white",
                    "markeredgewidth": 1.2 if passed else 0.6,
                    "markersize": 6.5,
                    "capsize": 2.0,
                    "linestyle": "none",
                }
                axes[0, column].errorbar(
                    x, float(row["nll_gap_to_full_budget_anchor"]),
                    yerr=float(row["nll_gap_hamiltonian_sem"]), **style)
                axes[1, column].errorbar(
                    x,
                    float(row["absolute_normalized_energy_difference_mean"]),
                    yerr=float(row[
                        "absolute_normalized_energy_difference_hamiltonian_sem"]),
                    **style)

        axes[0, column].axhline(
            nll_limit, color="tab:red", linestyle="--", linewidth=1.0)
        axes[1, column].axhline(
            energy_limit, color="tab:red", linestyle="--", linewidth=1.0)
        axes[0, column].set_title(f"N = {n_qubit}")
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel(
            "Training samples / QPU-equivalent shots\n"
            r"$N_{H,\mathrm{train}}D M_{\mathrm{train}}$")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)

    axes[0, 0].set_ylabel(
        "Held-out NLL gap to\nfull-budget anchor (nat/qubit)")
    axes[1, 0].set_ylabel(
        "Mean absolute normalized energy\ndifference vs direct SKQD")
    color_handles = [
        Line2D([0], [0], color=colors[value], marker="o", linestyle="-",
               label=rf"$N_H={value}$")
        for value in hamiltonians
    ]
    marker_handles = [
        Line2D([0], [0], color="0.35", marker=marker, linestyle="none",
               label=rf"$M_{{train}}={value}$")
        for value, marker in markers.items()
    ]
    pass_handle = Line2D(
        [0], [0], color="0.5", marker="o", markeredgecolor="black",
        markeredgewidth=1.3, linestyle="none", label="passes both criteria")
    fig.legend(
        handles=color_handles + marker_handles + [pass_handle],
        loc="outside lower center", ncol=9, frameon=False)
    fig.suptitle(
        "Single-seed finite-size training-budget evidence",
        fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.evaluation_dir / f"training_budget_accuracy.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
