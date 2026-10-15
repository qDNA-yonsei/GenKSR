"""Aggregate multiple model seeds into reviewer-ready tables and a diagnostic figure."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty summary {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample_sd(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else float("nan")


def _sem(values: Sequence[float]) -> float:
    return _sample_sd(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")


def _aggregate_distribution(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metrics = ("excess_nll_per_qubit", "tv", "jsd")
    primary = [row for row in rows if int(row["primary_nonzero_time"]) == 1]
    per_hamiltonian: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in primary:
        key = (
            int(row["model_seed"]),
            row["split"],
            row["method"],
            row["instance_id"],
        )
        for metric in metrics:
            per_hamiltonian[key][metric].append(float(row[metric]))

    h_rows = []
    for (seed, split, method, identifier), values in sorted(per_hamiltonian.items()):
        h_rows.append(
            {
                "model_seed": seed,
                "split": split,
                "method": method,
                "instance_id": identifier,
                **{metric: mean(values[metric]) for metric in metrics},
            }
        )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in h_rows:
        grouped[(row["split"], row["method"])].append(row)

    summary = []
    for (split, method), group in sorted(grouped.items()):
        seeds = sorted({int(row["model_seed"]) for row in group})
        output = {
            "split": split,
            "method": method,
            "num_model_seeds": len(seeds),
            "num_seed_hamiltonian_pairs": len(group),
            "num_hamiltonians_per_seed": len(
                {row["instance_id"] for row in group}
            ),
        }
        for metric in metrics:
            seed_means = [
                mean(
                    float(row[metric])
                    for row in group
                    if int(row["model_seed"]) == seed
                )
                for seed in seeds
            ]
            all_values = [float(row[metric]) for row in group]
            output[f"mean_{metric}"] = mean(seed_means)
            output[f"seed_sd_{metric}"] = _sample_sd(seed_means)
            output[f"pooled_h_sem_{metric}"] = _sem(all_values)
        summary.append(output)

    paired = []
    conditioned = {
        (row["model_seed"], row["instance_id"]): row
        for row in h_rows
        if row["split"] == "target_test" and row["method"] == "pauli_conditioned"
    }
    for baseline in (
        "mismatched_target_hamiltonian",
        "nearest_source_exact_distribution",
    ):
        baseline_rows = {
            (row["model_seed"], row["instance_id"]): row
            for row in h_rows
            if row["split"] == "target_test" and row["method"] == baseline
        }
        keys = sorted(set(conditioned) & set(baseline_rows))
        output = {
            "baseline": baseline,
            "num_pairs": len(keys),
        }
        for metric in metrics:
            improvements = [
                float(baseline_rows[key][metric]) - float(conditioned[key][metric])
                for key in keys
            ]
            output[f"mean_improvement_{metric}"] = mean(improvements)
            output[f"paired_sem_{metric}"] = _sem(improvements)
            output[f"fraction_conditioned_better_{metric}"] = float(
                np.mean(np.asarray(improvements) > 0.0)
            )
        paired.append(output)
    return summary, paired


def _aggregate_energy(rows: list[dict]) -> list[dict]:
    metrics = (
        "direct_1_total_normalized_error",
        "model_total_normalized_error",
        "model_direct_abs_normalized_gap",
        "direct_direct_abs_normalized_gap",
        "model_direct_signed_normalized_gap",
    )
    per_hamiltonian: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            int(row["model_seed"]),
            int(row["shots_per_condition"]),
            row["instance_id"],
        )
        for metric in metrics:
            per_hamiltonian[key][metric].append(float(row[metric]))

    h_rows = []
    for (seed, shots, identifier), values in sorted(per_hamiltonian.items()):
        h_rows.append(
            {
                "model_seed": seed,
                "shots_per_condition": shots,
                "instance_id": identifier,
                **{metric: mean(values[metric]) for metric in metrics},
            }
        )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in h_rows:
        grouped[int(row["shots_per_condition"])].append(row)
    summary = []
    for shots, group in sorted(grouped.items()):
        seeds = sorted({int(row["model_seed"]) for row in group})
        output = {
            "shots_per_condition": shots,
            "num_model_seeds": len(seeds),
            "num_seed_hamiltonian_pairs": len(group),
            "num_hamiltonians_per_seed": len({row["instance_id"] for row in group}),
        }
        for metric in metrics:
            seed_means = [
                mean(
                    float(row[metric])
                    for row in group
                    if int(row["model_seed"]) == seed
                )
                for seed in seeds
            ]
            all_values = [float(row[metric]) for row in group]
            output[f"mean_{metric}"] = mean(seed_means)
            output[f"seed_sd_{metric}"] = _sample_sd(seed_means)
            output[f"pooled_h_sem_{metric}"] = _sem(all_values)
        denominator = output["mean_direct_direct_abs_normalized_gap"]
        output["surrogate_to_sampling_gap_ratio"] = (
            output["mean_model_direct_abs_normalized_gap"] / denominator
            if denominator > 0.0
            else float("inf")
        )
        summary.append(output)
    return summary


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_evaluations(
    evaluation_dirs: Sequence[str | Path], output_dir: str | Path
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_rows = []
    energy_rows = []
    metadata = []
    for directory in map(Path, evaluation_dirs):
        distribution_rows.extend(_read_csv(directory / "distribution_context.csv"))
        energy_rows.extend(_read_csv(directory / "skqd_pairs.csv"))
        metadata.append(json.loads((directory / "evaluation.json").read_text()))
    protocol_hashes = {row["protocol_sha256"] for row in metadata}
    seeds = {int(row["model_seed"]) for row in metadata}
    regimes = {
        row.get("evaluation_regime", "zero_shot_operator_template_holdout")
        for row in metadata
    }
    if len(protocol_hashes) != 1:
        raise ValueError("cannot summarize evaluations from different protocols")
    if len(seeds) != len(metadata):
        raise ValueError("duplicate model seed evaluation directory")
    if len(regimes) != 1:
        raise ValueError("cannot summarize evaluations from different regimes")
    regime = next(iter(regimes))
    in_family_control = regime == "in_family_target_trained_control"
    target_panel_title = (
        "In-family target holdout" if in_family_control else "Zero-shot target family"
    )

    distribution_summary, paired_summary = _aggregate_distribution(distribution_rows)
    energy_summary = _aggregate_energy(energy_rows)
    _write_csv(output_dir / "distribution_summary.csv", distribution_summary)
    _write_csv(output_dir / "paired_target_improvement.csv", paired_summary)
    _write_csv(output_dir / "energy_summary.csv", energy_summary)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    target_methods = [
        "pauli_conditioned",
        "mismatched_target_hamiltonian",
        "nearest_source_exact_distribution",
    ]
    labels = ["Pauli-conditioned", "Mismatched H", "Nearest source"]
    for axis, metric, ylabel in (
        (axes[0], "excess_nll_per_qubit", "Excess NLL / qubit"),
        (axes[1], "tv", "Total variation"),
    ):
        lookup = {
            row["method"]: row
            for row in distribution_summary
            if row["split"] == "target_test"
        }
        values = [float(lookup[method][f"mean_{metric}"]) for method in target_methods]
        errors = [float(lookup[method][f"pooled_h_sem_{metric}"]) for method in target_methods]
        axis.bar(np.arange(3), values, yerr=errors, capsize=3)
        axis.set_xticks(np.arange(3), labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(target_panel_title)

    shots = [int(row["shots_per_condition"]) for row in energy_summary]
    axes[2].plot(
        shots,
        [float(row["mean_model_direct_abs_normalized_gap"]) for row in energy_summary],
        "o-",
        label="GenKSR–direct",
    )
    axes[2].plot(
        shots,
        [float(row["mean_direct_direct_abs_normalized_gap"]) for row in energy_summary],
        "s--",
        label="direct–direct",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Samples / (H,t)")
    axes[2].set_ylabel("Mean absolute normalized gap")
    axes[2].set_title("Matched SKQD variability")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cross_family_transfer.png", dpi=200)
    plt.close(fig)

    target_lookup = {
        row["method"]: row
        for row in distribution_summary
        if row["split"] == "target_test"
    }
    report = [
        (
            "# Target-trained in-family control results"
            if in_family_control
            else "# Pauli cross-family transfer results"
        ),
        "",
        f"Protocol SHA-256: `{next(iter(protocol_hashes))}`",
        f"Model seeds: `{sorted(seeds)}`",
        "",
        "Primary distribution metrics exclude the common trivial `t=0` context.",
        "",
        (
            "## In-family held-out target Hamiltonians"
            if in_family_control
            else "## Zero-shot held-out target family"
        ),
        "",
        _markdown_table(
            ["Method", "Excess NLL / qubit", "TV", "JSD"],
            [
                (
                    label,
                    f"{float(target_lookup[method]['mean_excess_nll_per_qubit']):.6g}",
                    f"{float(target_lookup[method]['mean_tv']):.6g}",
                    f"{float(target_lookup[method]['mean_jsd']):.6g}",
                )
                for method, label in zip(target_methods, labels)
            ],
        ),
        "",
        "## Matched full-space SKQD",
        "",
        _markdown_table(
            ["M", "GenKSR–direct gap", "direct–direct gap", "ratio"],
            [
                (
                    row["shots_per_condition"],
                    f"{float(row['mean_model_direct_abs_normalized_gap']):.6g}",
                    f"{float(row['mean_direct_direct_abs_normalized_gap']):.6g}",
                    f"{float(row['surrogate_to_sampling_gap_ratio']):.3f}",
                )
                for row in energy_summary
            ],
        ),
        "",
        (
            "This is a budget-matched in-family performance control trained on the "
            "target template; it is not a zero-shot transfer result."
            if in_family_control
            else "This is a fixed-size held-out operator-template experiment. It does "
            "not test unseen Pauli vocabularies, coupling topologies, system sizes, "
            "or asymptotic scaling."
        ),
        "",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(report))
    summary_metadata = {
        "protocol_sha256": next(iter(protocol_hashes)),
        "evaluation_regime": regime,
        "model_seeds": sorted(seeds),
        "evaluation_dirs": [str(Path(path).resolve()) for path in evaluation_dirs],
        "distribution_summary_rows": len(distribution_summary),
        "energy_summary_rows": len(energy_summary),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_metadata, indent=2, sort_keys=True)
    )
    return summary_metadata
