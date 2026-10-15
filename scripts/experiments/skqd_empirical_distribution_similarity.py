#!/usr/bin/env python3
"""Compare exact-simulation and generated SKQD bitstring histograms across D.

The main comparison uses the actual saved 10,000-shot arrays for exact simulation,
Transformer, and Mamba.  Raw distances penalize samples outside Sz=0; conditional
distances compare histogram shape after post-selection.  Independent statevector
draws provide equal-budget and effective-postselection-matched sampling-noise
baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from genksr.data.hamiltonian import validate_protocol_metadata
from genksr.protocol import PROTOCOL_VERSION, SQKD_TASK
from scripts.experiments.skqd_distribution_learning import (
    coupling_j2,
    exact_probabilities_for_dimensions,
)


MODEL_STYLE = {
    "transformer": {"color": "tab:blue", "marker": "o", "label": "Transformer"},
    "mamba": {"color": "tab:green", "marker": "s", "label": "Mamba"},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--data-root", default="data/heisenberg_SQKD_2d")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--transformer-results", required=True)
    parser.add_argument("--mamba-results", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/v3_skqd_empirical_distribution_similarity"
    )
    parser.add_argument("--hamiltonian-ids", type=int, nargs="+", default=None)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--n-qubit", type=int, default=16)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--snapshots", type=int, default=10_000)
    parser.add_argument("--num-trotter-steps", type=int, default=6)
    parser.add_argument("--trotter-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--h-z", type=float, default=0.0)
    parser.add_argument("--sqkd-dt", type=float, default=None)
    parser.add_argument("--baseline-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument(
        "--exhaustive-metrics",
        default="artifacts/v3_skqd_distribution_learning/distribution_metrics.csv",
        help="Optional exhaustive q(z) metrics used for sampler-consistency validation.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.L * args.L != args.n_qubit:
        raise ValueError("L*L must equal n-qubit")
    if args.snapshots <= 0 or args.baseline_repeats <= 0:
        raise ValueError("snapshots and baseline-repeats must be positive")
    if not args.dimensions or min(args.dimensions) < 0:
        raise ValueError("dimensions must be non-empty and non-negative")
    if len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("dimensions must not contain duplicates")
    args.dimensions = sorted(args.dimensions)


def samples_to_histogram(samples, n_qubit, postselect=False):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != n_qubit:
        raise ValueError(f"Expected samples with shape (M, {n_qubit}), got {samples.shape}")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("Samples must be binary")
    if postselect:
        samples = samples[samples.sum(axis=1) == n_qubit // 2]
    if len(samples) == 0:
        return np.zeros(1 << n_qubit, dtype=float), 0
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    states = samples.astype(np.int64) @ powers
    counts = np.bincount(states, minlength=1 << n_qubit).astype(float)
    return counts / counts.sum(), len(samples)


def states_to_histogram(states, hilbert_size):
    counts = np.bincount(np.asarray(states, dtype=np.int64), minlength=hilbert_size)
    return counts.astype(float) / len(states)


def distribution_distances(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("Distance inputs must be normalized")
    mixture = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    jsd = 0.5 * (
        np.sum(p[p_mask] * np.log2(p[p_mask] / mixture[p_mask]))
        + np.sum(q[q_mask] * np.log2(q[q_mask] / mixture[q_mask]))
    )
    coefficient = float(np.sum(np.sqrt(p * q)))
    return {
        "jsd": float(np.clip(jsd, 0.0, 1.0)),
        "hellinger": float(np.sqrt(max(0.0, 1.0 - coefficient))),
        "total_variation": float(0.5 * np.sum(np.abs(p - q))),
    }


def baseline_statistics(reference_hist, exact_probabilities, sample_count, repeats, rng):
    if sample_count <= 0:
        return {
            f"{field}_{stat}": None
            for field in ("jsd", "hellinger", "total_variation")
            for stat in ("median", "q25", "q75")
        }
    records = []
    for _ in range(repeats):
        states = rng.choice(
            len(exact_probabilities), size=sample_count, replace=True,
            p=exact_probabilities,
        )
        sampled_hist = states_to_histogram(states, len(exact_probabilities))
        records.append(distribution_distances(reference_hist, sampled_hist))
    summary = {}
    for field in ("jsd", "hellinger", "total_variation"):
        values = np.asarray([record[field] for record in records])
        summary[f"{field}_median"] = float(np.median(values))
        summary[f"{field}_q25"] = float(np.quantile(values, 0.25))
        summary[f"{field}_q75"] = float(np.quantile(values, 0.75))
    return summary


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute(args):
    data_dir = Path(args.data_root) / f"{args.n_qubit}qubits" / args.split
    validate_protocol_metadata(
        data_dir, task=SQKD_TASK, lattice="2d", n_qubit=args.n_qubit,
        delta=args.delta, h_z=args.h_z, sqkd_dt=args.sqkd_dt,
        num_trotter_steps=args.num_trotter_steps,
    )
    available_ids = sorted(
        int(path.stem.split("id")[-1]) for path in data_dir.glob("coupling_matrix_id*.npy")
    )
    hamiltonian_ids = available_ids if args.hamiltonian_ids is None else args.hamiltonian_ids
    missing = sorted(set(hamiltonian_ids) - set(available_ids))
    if missing:
        raise ValueError(f"Unknown Hamiltonian ids: {missing}")

    model_paths = {
        "transformer": Path(args.transformer_results),
        "mamba": Path(args.mamba_results),
    }
    rows = []
    for hamiltonian_id in tqdm(hamiltonian_ids, desc="Hamiltonians"):
        coupling = np.load(data_dir / f"coupling_matrix_id{hamiltonian_id}.npy")
        j2 = coupling_j2(coupling)
        exact_by_dimension, _ = exact_probabilities_for_dimensions(
            coupling, args.n_qubit, args.L, args.dimensions,
            args.num_trotter_steps, args.trotter_order,
            args.delta, args.h_z, args.sqkd_dt,
        )
        for dimension in args.dimensions:
            exact_samples = np.load(
                data_dir / f"data_D{dimension}_id{hamiltonian_id}.npy",
                allow_pickle=False,
            )
            exact_hist, exact_count = samples_to_histogram(
                exact_samples, args.n_qubit, postselect=False
            )
            exact_probabilities = exact_by_dimension[dimension]
            raw_baseline_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, hamiltonian_id, dimension, 0])
            )
            same_budget = baseline_statistics(
                exact_hist, exact_probabilities, exact_count,
                args.baseline_repeats, raw_baseline_rng,
            )

            for model_index, (model_name, results_dir) in enumerate(model_paths.items(), 1):
                sample_path = (
                    results_dir / "samples" / args.split / "model"
                    / f"model_samples_D{dimension}_{args.snapshots}_id{hamiltonian_id}.npy"
                )
                model_samples = np.load(sample_path, allow_pickle=False)
                model_raw_hist, input_count = samples_to_histogram(
                    model_samples, args.n_qubit, postselect=False
                )
                model_sector_hist, postselected_count = samples_to_histogram(
                    model_samples, args.n_qubit, postselect=True
                )
                if postselected_count == 0:
                    raise RuntimeError(
                        f"No Sz=0 samples for {model_name}, id={hamiltonian_id}, D={dimension}"
                    )

                raw = distribution_distances(exact_hist, model_raw_hist)
                sector = distribution_distances(exact_hist, model_sector_hist)
                population_raw = distribution_distances(exact_probabilities, model_raw_hist)
                population_sector = distribution_distances(exact_probabilities, model_sector_hist)
                matched_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.seed, hamiltonian_id, dimension, model_index]
                    )
                )
                matched = baseline_statistics(
                    exact_hist, exact_probabilities, postselected_count,
                    args.baseline_repeats, matched_rng,
                )

                row = {
                    "hamiltonian_id": int(hamiltonian_id),
                    "J2": j2,
                    "data_dimension": int(dimension),
                    "krylov_dimension": int(dimension + 1),
                    "model": model_name,
                    "exact_samples": int(exact_count),
                    "model_input_samples": int(input_count),
                    "model_postselected_samples": int(postselected_count),
                    "model_sector_rate": float(postselected_count / input_count),
                    "raw_jsd": raw["jsd"],
                    "raw_hellinger": raw["hellinger"],
                    "raw_total_variation": raw["total_variation"],
                    "sector_jsd": sector["jsd"],
                    "sector_hellinger": sector["hellinger"],
                    "sector_total_variation": sector["total_variation"],
                    "population_raw_jsd": population_raw["jsd"],
                    "population_sector_jsd": population_sector["jsd"],
                    **{
                        f"exact_same_budget_{key}": value
                        for key, value in same_budget.items()
                    },
                    **{
                        f"exact_matched_budget_{key}": value
                        for key, value in matched.items()
                    },
                }
                row["sector_jsd_excess_over_matched_noise"] = max(
                    0.0,
                    row["sector_jsd"]
                    - row["exact_matched_budget_jsd_median"],
                )
                rows.append(row)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "empirical_distribution_metrics.csv", rows)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "skqd_empirical_distribution_similarity",
        "comparison": "saved exact-simulation samples vs saved model-generated samples",
        "n_qubit": args.n_qubit,
        "split": args.split,
        "hamiltonian_ids": hamiltonian_ids,
        "data_dimensions": args.dimensions,
        "krylov_dimensions": [dimension + 1 for dimension in args.dimensions],
        "snapshots": args.snapshots,
        "baseline_repeats": args.baseline_repeats,
        "seed": args.seed,
        "transformer_results": str(Path(args.transformer_results).resolve()),
        "mamba_results": str(Path(args.mamba_results).resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir / "empirical_distribution_metrics.csv"


def summarize(rows):
    fields = [
        "model_sector_rate", "raw_jsd", "sector_jsd", "raw_hellinger",
        "sector_hellinger", "raw_total_variation", "sector_total_variation",
        "population_raw_jsd", "population_sector_jsd",
        "exact_same_budget_jsd_median", "exact_matched_budget_jsd_median",
        "raw_jsd_excess_over_finite_shot_baseline",
        "sector_jsd_excess_over_matched_noise",
    ]
    records = []
    for model_name in MODEL_STYLE:
        for dimension in sorted({int(row["krylov_dimension"]) for row in rows}):
            selected = [
                row for row in rows
                if row["model"] == model_name
                and int(row["krylov_dimension"]) == dimension
            ]
            record = {
                "model": model_name,
                "krylov_dimension": dimension,
                "n_hamiltonians": len(selected),
            }
            for field in fields:
                values = np.asarray([float(row[field]) for row in selected])
                record[f"{field}_mean"] = float(np.mean(values))
                record[f"{field}_median"] = float(np.median(values))
                record[f"{field}_q25"] = float(np.quantile(values, 0.25))
                record[f"{field}_q75"] = float(np.quantile(values, 0.75))
            records.append(record)
    return records


def grouped(rows, field):
    result = {}
    for row in rows:
        key = (row["model"], int(row["krylov_dimension"]))
        result.setdefault(key, []).append(float(row[field]))
    return result


def plot_line(ax, rows, field, model_name, linestyle="-", label=None, alpha=0.18):
    values_by_group = grouped(rows, field)
    dimensions = sorted({int(row["krylov_dimension"]) for row in rows})
    values = [np.asarray(values_by_group[(model_name, dimension)]) for dimension in dimensions]
    median = np.asarray([np.median(value) for value in values])
    q25 = np.asarray([np.quantile(value, 0.25) for value in values])
    q75 = np.asarray([np.quantile(value, 0.75) for value in values])
    style = MODEL_STYLE[model_name]
    ax.plot(dimensions, median, color=style["color"], linestyle=linestyle,
            marker=style["marker"] if linestyle == "-" else None,
            markersize=3, label=label or style["label"])
    if alpha:
        ax.fill_between(dimensions, q25, q75, color=style["color"], alpha=alpha)


def plot_exact_baseline(ax, rows, field="exact_same_budget_jsd_median"):
    dimensions = sorted({int(row["krylov_dimension"]) for row in rows})
    median, q25, q75 = [], [], []
    for dimension in dimensions:
        # The same-budget baseline is identical in the two model rows.
        by_condition = {}
        for row in rows:
            if int(row["krylov_dimension"]) == dimension:
                by_condition[int(row["hamiltonian_id"])] = float(row[field])
        values = np.asarray(list(by_condition.values()))
        median.append(np.median(values))
        q25.append(np.quantile(values, 0.25))
        q75.append(np.quantile(values, 0.75))
    ax.plot(dimensions, median, color="black", linestyle=":",
            label="Exact vs exact (10k)")
    ax.fill_between(dimensions, q25, q75, color="gray", alpha=0.16)


def add_boundary(ax):
    ax.axvline(15.5, color="black", linestyle=":", linewidth=1)
    ax.axvspan(15.5, 30.5, color="gray", alpha=0.08)


def plot_finite_shot_excess_raw_jsd(rows, output_dir):
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    for model_name in MODEL_STYLE:
        plot_line(
            ax, rows, "raw_jsd_excess_over_finite_shot_baseline", model_name
        )
    add_boundary(ax)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(-0.01, 0.75)
    ax.set_xlabel("Krylov dimension D")
    ax.set_ylabel("Finite-shot excess raw JSD")
    ax.set_title("Generated-distribution error beyond the 10,000-shot sampling floor")
    ax.text(
        8.25, 0.97, "Training range", ha="center", va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        23.0, 0.97, "Extrapolation", ha="center", va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "finite_shot_excess_raw_jsd.png", dpi=250)
    fig.savefig(output_dir / "finite_shot_excess_raw_jsd.pdf")
    plt.close(fig)


def plot_heatmap(args, rows, output_dir):
    dimensions = sorted({int(row["krylov_dimension"]) for row in rows})
    hamiltonian_ids = sorted(
        {int(row["hamiltonian_id"]) for row in rows},
        key=lambda hid: next(
            float(row["J2"]) for row in rows if int(row["hamiltonian_id"]) == hid
        ),
    )
    j2 = {
        int(row["hamiltonian_id"]): float(row["J2"])
        for row in rows
    }
    lookup = {
        (row["model"], int(row["hamiltonian_id"]), int(row["krylov_dimension"])): row
        for row in rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    vmax = max(float(row["sector_jsd"]) for row in rows)
    for ax, model_name in zip(axes, MODEL_STYLE):
        matrix = np.asarray([
            [float(lookup[(model_name, hid, dimension)]["sector_jsd"])
             for dimension in dimensions]
            for hid in hamiltonian_ids
        ])
        image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0,
                          vmax=vmax, cmap="magma")
        ax.set_title(f"{MODEL_STYLE[model_name]['label']}: postselected JSD")
        ax.set_xlabel("Krylov dimension D")
        n_ticks = min(7, len(dimensions))
        tick_columns = np.unique(
            np.linspace(0, len(dimensions) - 1, n_ticks, dtype=int)
        )
        ax.set_xticks(
            tick_columns,
            [dimensions[index] for index in tick_columns],
        )
        training_columns = [
            index for index, dimension in enumerate(dimensions) if dimension <= 15
        ]
        extrapolation_columns = [
            index for index, dimension in enumerate(dimensions) if dimension > 15
        ]
        if training_columns and extrapolation_columns:
            boundary = 0.5 * (max(training_columns) + min(extrapolation_columns))
            ax.axvline(boundary, color="white", linestyle=":", linewidth=1.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    tick_rows = np.linspace(0, len(hamiltonian_ids) - 1, 6, dtype=int)
    axes[0].set_yticks(tick_rows)
    axes[0].set_yticklabels([f"{j2[hamiltonian_ids[index]]:.3f}" for index in tick_rows])
    axes[0].set_ylabel(r"Held-out Hamiltonians sorted by $J_2$")
    fig.suptitle("Exact-simulation vs generated bitstring histogram distance")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "empirical_distribution_jsd_j2_heatmap.png", dpi=250)
    fig.savefig(output_dir / "empirical_distribution_jsd_j2_heatmap.pdf")
    plt.close(fig)


def write_validation(args, rows, output_dir):
    expected = (
        len({row["hamiltonian_id"] for row in rows})
        * len({row["krylov_dimension"] for row in rows}) * 2
    )
    numeric_fields = [
        "raw_jsd", "sector_jsd", "raw_hellinger", "sector_hellinger",
        "raw_total_variation", "sector_total_variation", "model_sector_rate",
    ]
    all_finite = all(
        np.isfinite(float(row[field])) for row in rows for field in numeric_fields
    )
    bounded = all(
        0 <= float(row[field]) <= 1 + 1e-12
        for row in rows for field in numeric_fields
    )
    report = {
        "row_count": len(rows),
        "expected_row_count": expected,
        "all_metrics_finite": bool(all_finite),
        "all_distances_and_rates_in_unit_interval": bool(bounded),
        "exact_sample_count_range": [
            min(int(row["exact_samples"]) for row in rows),
            max(int(row["exact_samples"]) for row in rows),
        ],
        "model_input_count_range": [
            min(int(row["model_input_samples"]) for row in rows),
            max(int(row["model_input_samples"]) for row in rows),
        ],
    }

    exhaustive_path = Path(args.exhaustive_metrics) if args.exhaustive_metrics else None
    if exhaustive_path is not None and exhaustive_path.exists():
        exhaustive_rows = read_csv(exhaustive_path)
        exhaustive_lookup = {
            (row["model"], int(row["hamiltonian_id"]), int(row["krylov_dimension"])):
                float(row["model_sector_mass"])
            for row in exhaustive_rows
        }
        differences = []
        for row in rows:
            key = (
                row["model"], int(row["hamiltonian_id"]),
                int(row["krylov_dimension"]),
            )
            if key in exhaustive_lookup:
                differences.append(
                    float(row["model_sector_rate"]) - exhaustive_lookup[key]
                )
        report["exhaustive_sector_consistency_conditions"] = len(differences)
        report["exhaustive_sector_consistency_max_abs_error"] = (
            float(np.max(np.abs(differences))) if differences else None
        )
    report["passed"] = bool(
        len(rows) == expected and all_finite and bounded
        and report["exact_sample_count_range"] == [args.snapshots, args.snapshots]
        and report["model_input_count_range"] == [args.snapshots, args.snapshots]
    )
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def plot(args, metrics_path):
    rows = read_csv(metrics_path)
    for row in rows:
        row["raw_jsd_excess_over_finite_shot_baseline"] = max(
            0.0,
            float(row["raw_jsd"])
            - float(row["exact_same_budget_jsd_median"]),
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "aggregate_empirical_distribution_metrics.csv", summarize(rows))
    write_validation(args, rows, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for model_name in MODEL_STYLE:
        plot_line(axes[0, 0], rows, "raw_jsd", model_name)
        plot_line(axes[0, 1], rows, "sector_jsd", model_name)
        plot_line(axes[1, 0], rows, "sector_hellinger", model_name)
        plot_line(axes[1, 1], rows, "model_sector_rate", model_name)
        plot_line(
            axes[0, 1], rows, "exact_matched_budget_jsd_median", model_name,
            linestyle="--", label=f"{MODEL_STYLE[model_name]['label']} exact-noise match",
            alpha=0,
        )
    plot_exact_baseline(axes[0, 0], rows)
    plot_exact_baseline(axes[0, 1], rows)
    axes[0, 0].set_ylabel("Raw histogram JSD")
    axes[0, 1].set_ylabel(r"Postselected $S_z=0$ histogram JSD")
    axes[1, 0].set_ylabel(r"Postselected $S_z=0$ Hellinger distance")
    axes[1, 1].set_ylabel(r"Generated-sample $S_z=0$ rate")
    for ax in axes.flat:
        ax.set_ylim(-0.015, 1.015)
        ax.grid(True, alpha=0.3)
        add_boundary(ax)
    axes[1, 0].set_xlabel("Krylov dimension D")
    axes[1, 1].set_xlabel("Krylov dimension D")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Similarity of exact-simulation and model-generated 10,000-shot distributions"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "empirical_distribution_similarity.png", dpi=250)
    fig.savefig(output_dir / "empirical_distribution_similarity.pdf")
    plt.close(fig)

    plot_finite_shot_excess_raw_jsd(rows, output_dir)
    plot_heatmap(args, rows, output_dir)
    print(f"Saved empirical similarity results to {output_dir}")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    metrics_path = Path(
        args.metrics_csv
        or Path(args.output_dir) / "empirical_distribution_metrics.csv"
    )
    if args.mode in ("compute", "both"):
        metrics_path = compute(args)
    if args.mode in ("plot", "both"):
        plot(args, metrics_path)


if __name__ == "__main__":
    main()
