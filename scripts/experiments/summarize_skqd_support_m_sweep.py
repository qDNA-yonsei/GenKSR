#!/usr/bin/env python3
"""Aggregate support-controlled SKQD runs across sample budgets.

Inputs are the ``m{M}/per_hamiltonian.csv`` artifacts emitted by
``skqd_support_controlled.py``.  Uniform-sampling repeats have already been
collapsed to their per-Hamiltonian medians in those files, so Hamiltonians—not
baseline seeds—remain the statistical unit in this summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402


METHODS = ("device", "transformer", "mamba", "uniform_full", "uniform_sector")
FIXED_METHODS = ("device", "transformer", "mamba")
BASELINES = ("uniform_full", "uniform_sector")
DISPLAY_NAMES = {
    "device": "Exact-state",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "uniform_full": "Uniform full",
    "uniform_sector": "Uniform Sz=0",
}
COLORS = {
    "device": "tab:blue",
    "transformer": "tab:orange",
    "mamba": "tab:red",
    "uniform_full": "tab:green",
    "uniform_sector": "tab:purple",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate the D=30 support-controlled SKQD M sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        default="artifacts/v3_skqd_support_m_sweep_d30",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument(
        "--high-j2-threshold",
        type=float,
        default=0.8,
        help="Exploratory split motivated by the pilot's observed crossover",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args(argv)


def _read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(args):
    root = Path(args.input_root)
    combined = []
    source_configs = {}
    expected_ids = None
    expected_d = None
    for m in sorted(args.m_values):
        run_dir = root / f"m{m}"
        rows_path = run_dir / "per_hamiltonian.csv"
        config_path = run_dir / "config.json"
        if not rows_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Missing completed M={m} run in {run_dir}")
        with config_path.open() as handle:
            config = json.load(handle)
        if int(config["samples_per_dim"]) != m:
            raise ValueError(f"M mismatch in {config_path}")
        if int(config["baseline_repeats"]) < 5:
            raise ValueError(f"M={m} has fewer than five uniform baseline repeats")
        ids = tuple(int(value) for value in config["hamiltonian_ids"])
        if expected_ids is None:
            expected_ids = ids
            expected_d = int(config["krylov_dim"])
        if ids != expected_ids or int(config["krylov_dim"]) != expected_d:
            raise ValueError("Hamiltonian ids or D differ across M runs")

        rows = _read_csv(rows_path)
        if len(rows) != 2 * len(METHODS) * len(ids):
            raise ValueError(f"Unexpected per-Hamiltonian row count at M={m}")
        for row in rows:
            row = dict(row)
            row["M"] = int(m)
            row["hamiltonian_id"] = int(row["hamiltonian_id"])
            row["J2"] = float(row["J2"])
            row["n_repeats"] = int(row["n_repeats"])
            for field in (
                "delta_e",
                "basis_size",
                "coverage",
                "acceptance_rate",
                "captured_ground_mass",
                "missing_ground_mass",
            ):
                row[field] = float(row[field])
            combined.append(row)
        source_configs[m] = config
    return combined, source_configs


def aggregate_sweep(combined, m_values):
    summary = []
    for m in sorted(m_values):
        for matching in ("raw", "unique"):
            for method in METHODS:
                subset = [
                    row for row in combined
                    if row["M"] == m and row["matching"] == matching
                    and row["method"] == method
                ]
                record = {
                    "M": int(m),
                    "matching": matching,
                    "method": method,
                    "n_hamiltonians": len(subset),
                }
                for field in (
                    "delta_e",
                    "basis_size",
                    "coverage",
                    "acceptance_rate",
                    "captured_ground_mass",
                    "missing_ground_mass",
                ):
                    values = np.asarray([row[field] for row in subset], dtype=float)
                    record[f"{field}_median"] = float(np.median(values))
                    record[f"{field}_q25"] = float(np.quantile(values, 0.25))
                    record[f"{field}_q75"] = float(np.quantile(values, 0.75))
                    record[f"{field}_mean"] = float(np.mean(values))
                summary.append(record)
    return summary


def _bootstrap_median_ci(values, repeats, seed_sequence):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_sequence)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    return (
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def paired_advantages(combined, args):
    lookup = {
        (row["M"], row["matching"], row["method"], row["hamiltonian_id"]): row
        for row in combined
    }
    ids = sorted({row["hamiltonian_id"] for row in combined})
    j2_by_id = {
        hid: lookup[(args.m_values[0], "unique", "device", hid)]["J2"]
        for hid in ids
    }
    regimes = {
        "all": ids,
        "J2_lt_threshold": [
            hid for hid in ids if j2_by_id[hid] < args.high_j2_threshold
        ],
        "J2_ge_threshold": [
            hid for hid in ids if j2_by_id[hid] >= args.high_j2_threshold
        ],
    }

    rows = []
    comparison_index = 0
    for m in sorted(args.m_values):
        for matching in ("raw", "unique"):
            for method in FIXED_METHODS:
                for baseline in BASELINES:
                    for regime, regime_ids in regimes.items():
                        energy_diffs = np.asarray([
                            lookup[(m, matching, method, hid)]["delta_e"]
                            - lookup[(m, matching, baseline, hid)]["delta_e"]
                            for hid in regime_ids
                        ])
                        mass_diffs = np.asarray([
                            lookup[(m, matching, method, hid)]["captured_ground_mass"]
                            - lookup[(m, matching, baseline, hid)]["captured_ground_mass"]
                            for hid in regime_ids
                        ])
                        ci_low, ci_high = _bootstrap_median_ci(
                            energy_diffs,
                            args.bootstrap_repeats,
                            np.random.SeedSequence(
                                [args.seed, comparison_index, 0]),
                        )
                        mass_low, mass_high = _bootstrap_median_ci(
                            mass_diffs,
                            args.bootstrap_repeats,
                            np.random.SeedSequence(
                                [args.seed, comparison_index, 1]),
                        )
                        nonzero = energy_diffs[
                            ~np.isclose(energy_diffs, 0.0, atol=1e-14)]
                        p_value = 1.0 if len(nonzero) == 0 else float(
                            wilcoxon(nonzero, alternative="two-sided").pvalue)
                        rows.append({
                            "M": int(m),
                            "matching": matching,
                            "method": method,
                            "baseline": baseline,
                            "regime": regime,
                            "j2_threshold": float(args.high_j2_threshold),
                            "n_hamiltonians": len(regime_ids),
                            "median_delta_e_difference": float(
                                np.median(energy_diffs)),
                            "delta_e_difference_ci95_low": ci_low,
                            "delta_e_difference_ci95_high": ci_high,
                            "energy_win_rate": float(np.mean(energy_diffs < 0.0)),
                            "wilcoxon_p_two_sided": p_value,
                            "median_captured_mass_difference": float(
                                np.median(mass_diffs)),
                            "captured_mass_difference_ci95_low": mass_low,
                            "captured_mass_difference_ci95_high": mass_high,
                            "captured_mass_win_rate": float(
                                np.mean(mass_diffs > 0.0)),
                        })
                        comparison_index += 1
    return rows


def j2_correlations(combined, m_values):
    lookup = {
        (row["M"], row["matching"], row["method"], row["hamiltonian_id"]): row
        for row in combined
    }
    ids = sorted({row["hamiltonian_id"] for row in combined})
    rows = []
    for m in sorted(m_values):
        j2_values = np.asarray([
            lookup[(m, "unique", "device", hid)]["J2"] for hid in ids
        ])
        for method in FIXED_METHODS:
            differences = np.asarray([
                lookup[(m, "unique", method, hid)]["delta_e"]
                - lookup[(m, "unique", "uniform_sector", hid)]["delta_e"]
                for hid in ids
            ])
            rho, p_value = spearmanr(j2_values, differences)
            rows.append({
                "M": int(m),
                "method": method,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            })
    return rows


def _summary_lookup(summary):
    return {
        (row["M"], row["matching"], row["method"]): row
        for row in summary
    }


def plot_sweep(summary, advantages, output_dir, config):
    output_dir = Path(output_dir)
    lookup = _summary_lookup(summary)
    m_values = config["m_values"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.reshape(-1)

    for method in METHODS:
        medians = np.asarray([
            max(lookup[(m, "raw", method)]["delta_e_median"], 1e-8)
            for m in m_values
        ])
        q25 = np.asarray([
            max(lookup[(m, "raw", method)]["delta_e_q25"], 1e-8)
            for m in m_values
        ])
        q75 = np.asarray([
            max(lookup[(m, "raw", method)]["delta_e_q75"], 1e-8)
            for m in m_values
        ])
        axes[0].plot(
            m_values, medians, marker="o", color=COLORS[method],
            label=DISPLAY_NAMES[method])
        axes[0].fill_between(m_values, q25, q75, color=COLORS[method], alpha=0.12)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Samples per Krylov dimension M")
    axes[0].set_ylabel(r"Median energy error $E_{\mathrm{proj}}-E_0$")
    axes[0].set_title("(a) Equal raw-draw budget")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=9)

    for method in METHODS:
        basis = np.asarray([
            lookup[(m, "unique", method)]["basis_size_median"]
            for m in m_values
        ])
        medians = np.asarray([
            max(lookup[(m, "unique", method)]["delta_e_median"], 1e-8)
            for m in m_values
        ])
        axes[1].plot(
            basis, medians, marker="o", color=COLORS[method],
            label=DISPLAY_NAMES[method])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Common unique basis size K")
    axes[1].set_ylabel(r"Median energy error $E_{\mathrm{proj}}-E_0$")
    axes[1].set_title("(b) Equal unique-basis budget")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=9)

    for method in METHODS:
        basis = np.asarray([
            lookup[(m, "unique", method)]["basis_size_median"]
            for m in m_values
        ])
        missing = np.asarray([
            max(lookup[(m, "unique", method)]["missing_ground_mass_median"], 1e-8)
            for m in m_values
        ])
        axes[2].plot(
            basis, missing, marker="o", color=COLORS[method],
            label=DISPLAY_NAMES[method])
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Common unique basis size K")
    axes[2].set_ylabel("Median missing ground-state mass")
    axes[2].set_title("(c) Support quality at equal K")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(fontsize=9)

    line_styles = {
        "J2_lt_threshold": "-",
        "J2_ge_threshold": "--",
    }
    for method in ("transformer", "mamba"):
        for regime in ("J2_lt_threshold", "J2_ge_threshold"):
            subset = [
                row for row in advantages
                if row["matching"] == "unique" and row["method"] == method
                and row["baseline"] == "uniform_sector"
                and row["regime"] == regime
            ]
            subset = sorted(subset, key=lambda row: row["M"])
            label_regime = (
                f"J2<{config['high_j2_threshold']}" if regime == "J2_lt_threshold"
                else f"J2>={config['high_j2_threshold']}"
            )
            axes[3].plot(
                [row["M"] for row in subset],
                [row["median_delta_e_difference"] for row in subset],
                marker="o",
                linestyle=line_styles[regime],
                color=COLORS[method],
                label=f"{DISPLAY_NAMES[method]}, {label_regime}",
            )
    axes[3].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("Samples per Krylov dimension M")
    axes[3].set_ylabel(
        r"Median $\Delta E_{\mathrm{model}}"
        r"-\Delta E_{\mathrm{uniform}\ S_z=0}$"
    )
    axes[3].set_title("(d) Exploratory J2-regime crossover")
    axes[3].grid(True, which="both", alpha=0.25)
    axes[3].legend(fontsize=9)

    figure.suptitle(
        f"Support-controlled SKQD M sweep: D={config['krylov_dim']}, "
        f"{config['n_hamiltonians']} held-out Hamiltonians"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"support_m_sweep.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_heatmap(combined, output_dir, config):
    output_dir = Path(output_dir)
    m_values = config["m_values"]
    lookup = {
        (row["M"], row["matching"], row["method"], row["hamiltonian_id"]): row
        for row in combined
    }
    ids = sorted(
        {row["hamiltonian_id"] for row in combined},
        key=lambda hid: lookup[(m_values[0], "unique", "device", hid)]["J2"],
    )
    matrices = {}
    for method in ("transformer", "mamba"):
        matrices[method] = np.asarray([
            [
                lookup[(m, "unique", method, hid)]["delta_e"]
                - lookup[(m, "unique", "uniform_sector", hid)]["delta_e"]
                for m in m_values
            ]
            for hid in ids
        ])
    minimum = min(float(np.min(matrix)) for matrix in matrices.values())
    maximum = max(float(np.max(matrix)) for matrix in matrices.values())
    norm = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)

    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    image = None
    for axis, method in zip(axes, ("transformer", "mamba")):
        image = axis.imshow(
            matrices[method], aspect="auto", cmap="coolwarm", norm=norm)
        axis.set_xticks(range(len(m_values)), [str(m) for m in m_values], rotation=40)
        axis.set_xlabel("M")
        axis.set_title(DISPLAY_NAMES[method])
    axes[0].set_yticks(
        range(len(ids)),
        [f"{lookup[(m_values[0], 'unique', 'device', hid)]['J2']:.3f}" for hid in ids],
    )
    axes[0].set_ylabel("Held-out J2")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label(
        r"Unique-matched $\Delta E_{\mathrm{model}}"
        r"-\Delta E_{\mathrm{uniform}\ S_z=0}$ (negative is better)"
    )
    figure.suptitle("Per-Hamiltonian support advantage across sample budgets")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.08)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"support_advantage_j2_m.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir or args.input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, source_configs = load_runs(args)
    summary = aggregate_sweep(combined, args.m_values)
    advantages = paired_advantages(combined, args)
    correlations = j2_correlations(combined, args.m_values)

    first_config = source_configs[min(source_configs)]
    config = {
        "task": "sqkd_support_controlled_m_sweep_summary",
        "input_root": str(Path(args.input_root).resolve()),
        "m_values": sorted(args.m_values),
        "krylov_dim": int(first_config["krylov_dim"]),
        "n_hamiltonians": len(first_config["hamiltonian_ids"]),
        "baseline_repeats_per_m": int(first_config["baseline_repeats"]),
        "high_j2_threshold": float(args.high_j2_threshold),
        "high_j2_split_note": "Exploratory/post-pilot, not preregistered",
        "model_uncertainty_note": first_config["model_uncertainty_note"],
    }
    _write_csv(output_dir / "combined_per_hamiltonian.csv", combined)
    _write_csv(output_dir / "sweep_summary.csv", summary)
    _write_csv(output_dir / "paired_advantages.csv", advantages)
    _write_csv(output_dir / "j2_correlations.csv", correlations)
    with (output_dir / "sweep_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    plot_sweep(summary, advantages, output_dir, config)
    plot_heatmap(combined, output_dir, config)
    print(f"Wrote aggregate M-sweep artifacts to {output_dir}")


if __name__ == "__main__":
    main()
