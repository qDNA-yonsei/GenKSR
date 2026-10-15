"""Joint, budget-matched downstream comparison for Pauli-conditioned models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Mapping, Sequence

import numpy as np
import torch

from .dataset import load_instance, load_protocol, pad_instance_terms, protocol_rows
from .evaluation import distribution_metrics
from .protocol import canonicalize_terms
from .quantum import (
    bits_to_integers,
    neel_bitstring,
    pauli_sum_matrix,
    projected_ground_energy,
)
from .training import load_trained_model, resolve_device


CORE_METHODS = (
    "direct_skqd",
    "zero_shot_genksr",
    "target_trained_genksr",
)

METHODS = (
    *CORE_METHODS,
    "uniform_sector_random",
    "uniform_full_random",
)

METHOD_LABELS = {
    "direct_skqd": "Exact-simulation SKQD",
    "zero_shot_genksr": "Source-family-trained GenKSR",
    "target_trained_genksr": "Target-family-trained GenKSR",
    "uniform_sector_random": "Uniform random (sector)",
    "uniform_full_random": "Random-sampling SKQD",
}

RANDOM_BASELINE_METHODS = {
    "uniform_sector": "uniform_sector_random",
    "uniform_full": "uniform_full_random",
}


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must have a stable field order")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _raw_terms(record: Mapping) -> list[tuple[str, float]]:
    return canonicalize_terms(
        [
            (str(pauli), float(coefficient))
            for pauli, coefficient in zip(
                record["pauli_strings"], record["raw_coefficients"]
            )
        ]
    )


def _condition_distribution(
    model,
    record: Mapping,
    *,
    tau: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    condition = pad_instance_terms(record, device=device)
    return model.exact_distribution(
        **condition,
        tau=torch.as_tensor([tau], dtype=torch.float32, device=device),
        batch_size=batch_size,
    )


def _basis_from_conditioned_distributions(
    distributions: Mapping[int, np.ndarray],
    *,
    initial_index: int,
    shots_per_nonzero_time: int,
    seed_parts: Sequence[object],
) -> set[int]:
    basis = {int(initial_index)}
    for dimension, probability in sorted(distributions.items()):
        clean = np.clip(np.asarray(probability, dtype=np.float64), 0.0, None)
        normalization = float(clean.sum())
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("conditioned distribution is not normalizable")
        clean /= normalization
        rng = np.random.default_rng(_stable_seed(*seed_parts, dimension))
        draws = rng.choice(
            len(clean),
            size=int(shots_per_nonzero_time),
            replace=True,
            p=clean,
        )
        basis.update(int(value) for value in draws)
    return basis


def _basis_from_uniform_candidates(
    candidates: np.ndarray,
    *,
    initial_index: int,
    nonzero_dimensions: Sequence[int],
    shots_per_nonzero_time: int,
    seed_parts: Sequence[object],
) -> set[int]:
    basis = {int(initial_index)}
    for dimension in nonzero_dimensions:
        rng = np.random.default_rng(_stable_seed(*seed_parts, dimension))
        draws = rng.choice(
            candidates,
            size=int(shots_per_nonzero_time),
            replace=True,
        )
        basis.update(int(value) for value in draws)
    return basis


def _sem(values: Sequence[float]) -> float:
    return float(stdev(values) / math.sqrt(len(values))) if len(values) > 1 else float("nan")


def _aggregate_distribution(rows: Sequence[Mapping]) -> list[dict]:
    per_hamiltonian: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (str(row["method"]), str(row["instance_id"]))
        for metric in (
            "excess_nll_per_qubit",
            "tv",
            "jsd",
            "target_sector_probability_mass",
            "exact_target_sector_probability_mass",
        ):
            per_hamiltonian[key][metric].append(float(row[metric]))

    h_rows = []
    for (method, identifier), metrics in sorted(per_hamiltonian.items()):
        h_rows.append(
            {
                "method": method,
                "instance_id": identifier,
                **{name: mean(values) for name, values in metrics.items()},
            }
        )

    output = []
    for method in ("zero_shot_genksr", "target_trained_genksr"):
        group = [row for row in h_rows if row["method"] == method]
        summary = {
            "method": method,
            "num_hamiltonians": len(group),
        }
        for metric in (
            "excess_nll_per_qubit",
            "tv",
            "jsd",
            "target_sector_probability_mass",
            "exact_target_sector_probability_mass",
        ):
            values = [float(row[metric]) for row in group]
            summary[f"mean_{metric}"] = mean(values)
            summary[f"hamiltonian_sem_{metric}"] = _sem(values)
        output.append(summary)
    return output


def _aggregate_paired_energy(
    rows: Sequence[Mapping], *, methods: Sequence[str]
) -> list[dict]:
    """Aggregate within-Hamiltonian differences from exact-simulation SKQD."""
    raw: dict[tuple[int, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["shots_per_nonzero_time"]),
            str(row["instance_id"]),
            int(row["repeat"]),
        )
        raw[key][str(row["method"])] = float(row["delta_e"])

    output = []
    budgets = sorted({key[0] for key in raw})
    identifiers = sorted({key[1] for key in raw})
    for shots in budgets:
        for method in methods:
            if method == "direct_skqd":
                continue
            per_hamiltonian = []
            for identifier in identifiers:
                differences = [
                    values[method] - values["direct_skqd"]
                    for (row_shots, row_identifier, _), values in raw.items()
                    if row_shots == shots and row_identifier == identifier
                ]
                if not differences:
                    raise ValueError("paired energy comparison is missing repetitions")
                per_hamiltonian.append(mean(differences))
            output.append(
                {
                    "shots_per_nonzero_time": shots,
                    "method": method,
                    "reference_method": "direct_skqd",
                    "num_hamiltonians": len(per_hamiltonian),
                    "mean_paired_delta_e_difference": mean(per_hamiltonian),
                    "hamiltonian_sem_paired_delta_e_difference": _sem(per_hamiltonian),
                    "fraction_hamiltonians_with_lower_delta_e": mean(
                        value < 0.0 for value in per_hamiltonian
                    ),
                }
            )
    return output


def _aggregate_energy(
    rows: Sequence[Mapping], *, methods: Sequence[str] = METHODS
) -> list[dict]:
    metrics = (
        "delta_e",
        "delta_e_per_qubit",
        "delta_e_normalized",
        "unique_basis",
        "unique_in_sector_basis",
        "unique_out_of_sector_basis",
    )
    per_hamiltonian: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            int(row["shots_per_nonzero_time"]),
            str(row["method"]),
            str(row["instance_id"]),
        )
        for metric in metrics:
            per_hamiltonian[key][metric].append(float(row[metric]))

    h_rows = []
    for (shots, method, identifier), values in sorted(per_hamiltonian.items()):
        h_rows.append(
            {
                "shots_per_nonzero_time": shots,
                "method": method,
                "instance_id": identifier,
                **{metric: mean(values[metric]) for metric in metrics},
            }
        )

    output = []
    budgets = sorted({int(row["shots_per_nonzero_time"]) for row in h_rows})
    for shots in budgets:
        for method in methods:
            group = [
                row
                for row in h_rows
                if int(row["shots_per_nonzero_time"]) == shots
                and row["method"] == method
            ]
            summary = {
                "shots_per_nonzero_time": shots,
                "method": method,
                "num_hamiltonians": len(group),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                summary[f"mean_{metric}"] = mean(values)
                summary[f"hamiltonian_sem_{metric}"] = _sem(values)
                if metric in ("delta_e", "delta_e_normalized"):
                    summary[f"median_{metric}"] = median(values)
                    summary[f"q25_{metric}"] = float(np.quantile(values, 0.25))
                    summary[f"q75_{metric}"] = float(np.quantile(values, 0.75))
            output.append(summary)
    return output


def _format_mean_sem(mean_value: float, sem_value: float) -> str:
    return f"{mean_value:.4e} +/- {sem_value:.1e}"


def _markdown_report(
    *,
    distribution_summary: Sequence[Mapping],
    energy_summary: Sequence[Mapping],
    n_qubit: int,
    depth: int,
    sector_dimension: int,
    repeats: int,
    methods: Sequence[str],
    uses_sector_random: bool,
) -> str:
    distribution_lookup = {row["method"]: row for row in distribution_summary}
    energy_lookup = {
        (int(row["shots_per_nonzero_time"]), row["method"]): row
        for row in energy_summary
    }
    budgets = sorted({key[0] for key in energy_lookup})
    lines = [
        f"# Joint n={n_qubit} Pauli cross-family SKQD comparison",
        "",
        (
            f"Known initial state is included once; M samples are drawn at each of "
            f"the {depth - 1} nonzero Krylov times. Total candidate draws per method "
            f"and Hamiltonian are therefore 1 + {depth - 1}M."
        ),
        "",
        f"Sampling repeats per target Hamiltonian: {repeats}.",
        (
            "Source-family-trained GenKSR is trained on the three source families only and "
            "uses no target-family training Hamiltonians or bitstrings."
        ),
        "",
        "## Exact target-distribution metrics (nonzero times)",
        "",
        (
            "| Method | Excess NLL / qubit | TV | JSD | Target-sector mass |"
            if uses_sector_random
            else "| Method | Excess NLL / qubit | TV | JSD |"
        ),
        ("|---|---:|---:|---:|---:|" if uses_sector_random else "|---|---:|---:|---:|"),
    ]
    if uses_sector_random:
        lines.insert(9, f"Reachable fixed-magnetization sector dimension: {sector_dimension}.")
    for method in ("zero_shot_genksr", "target_trained_genksr"):
        row = distribution_lookup[method]
        values = [
            METHOD_LABELS[method],
            _format_mean_sem(
                float(row["mean_excess_nll_per_qubit"]),
                float(row["hamiltonian_sem_excess_nll_per_qubit"]),
            ),
            _format_mean_sem(
                float(row["mean_tv"]), float(row["hamiltonian_sem_tv"])
            ),
            _format_mean_sem(
                float(row["mean_jsd"]), float(row["hamiltonian_sem_jsd"])
            ),
        ]
        if uses_sector_random:
            values.append(
                _format_mean_sem(
                    float(row["mean_target_sector_probability_mass"]),
                    float(row["hamiltonian_sem_target_sector_probability_mass"]),
                )
            )
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Absolute ground-energy error Delta E = |E_hat_0 - E_0|",
            "",
            "Values are mean +/- Hamiltonian-level SEM after averaging sampling repeats.",
            "",
            "| Samples per Krylov dimension M | "
            + " | ".join(METHOD_LABELS[method] for method in methods)
            + " |",
            "|---:|" + "---:|" * len(methods),
        ]
    )
    for shots in budgets:
        values = []
        for method in methods:
            row = energy_lookup[(shots, method)]
            values.append(
                _format_mean_sem(
                    float(row["mean_delta_e"]),
                    float(row["hamiltonian_sem_delta_e"]),
                )
            )
        lines.append(f"| {shots} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Scale-normalized ground-energy error Delta E / Lambda_H",
            "",
            "| Samples per Krylov dimension M | "
            + " | ".join(METHOD_LABELS[method] for method in methods)
            + " |",
            "|---:|" + "---:|" * len(methods),
        ]
    )
    for shots in budgets:
        values = []
        for method in methods:
            row = energy_lookup[(shots, method)]
            values.append(
                _format_mean_sem(
                    float(row["mean_delta_e_normalized"]),
                    float(row["hamiltonian_sem_delta_e_normalized"]),
                )
            )
        lines.append(f"| {shots} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            (
                "Uniform-sector random receives the same conserved Hamming-weight "
                "sector as exact-simulation SKQD; uniform-full random is secondary."
                if uses_sector_random
                else "Random-sampling SKQD draws uniformly from the full computational "
                "basis because the target dynamics do not conserve Hamming weight."
            ),
            "",
            (
                f"This is a single-system-size n={n_qubit} comparison and is not a "
                "system-size-generalization or asymptotic-scaling result."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_joint_comparison(
    cross_dataset_root: str | Path,
    cross_checkpoint_path: str | Path,
    target_dataset_root: str | Path,
    target_checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    cross_checkpoint_dataset_root: str | Path | None = None,
) -> dict:
    """Compare both learned models and matched direct/random SKQD baselines."""
    cross_dataset_root = Path(cross_dataset_root)
    target_dataset_root = Path(target_dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_protocol = load_protocol(cross_dataset_root)
    target_protocol = load_protocol(target_dataset_root)
    checkpoint_dataset_root = (
        cross_dataset_root
        if cross_checkpoint_dataset_root is None
        else Path(cross_checkpoint_dataset_root)
    )
    checkpoint_protocol = load_protocol(checkpoint_dataset_root)

    cross_config = cross_protocol["config"]
    target_config = target_protocol["config"]
    cross_data = cross_config["data"]
    target_data = target_config["data"]
    for key in ("n_qubit", "depth", "target_test_hamiltonians"):
        if int(cross_data[key]) != int(target_data[key]):
            raise ValueError(f"cross/control data mismatch for {key}")
    if cross_protocol.get("target_template_seen_in_source", False):
        raise ValueError("cross protocol is not a held-out target-template experiment")
    if not target_protocol.get("target_template_seen_in_source", False):
        raise ValueError("target protocol is not an in-family control")

    cross_rows = protocol_rows(cross_protocol, split="target_test")
    target_rows = protocol_rows(target_protocol, split="target_test")
    if cross_rows != target_rows:
        raise ValueError("cross and target-trained protocols do not share one target test manifest")

    # A source-only checkpoint may be reused for a new held-out target protocol,
    # but only when everything that affected its training is exactly unchanged.
    cross_source_rows = protocol_rows(cross_protocol, split="source_train")
    checkpoint_source_rows = protocol_rows(checkpoint_protocol, split="source_train")
    if cross_source_rows != checkpoint_source_rows:
        raise ValueError("source checkpoint and evaluation protocols have different training rows")
    if cross_protocol["source_families"] != checkpoint_protocol["source_families"]:
        raise ValueError("source checkpoint and evaluation protocols use different families")
    checkpoint_config = checkpoint_protocol["config"]
    if cross_config["model"] != checkpoint_config["model"]:
        raise ValueError("source checkpoint and evaluation model configurations differ")
    if cross_config["training"] != checkpoint_config["training"]:
        raise ValueError("source checkpoint and evaluation training configurations differ")
    for key in ("n_qubit", "depth", "train_shots_per_condition"):
        if int(cross_data[key]) != int(checkpoint_config["data"][key]):
            raise ValueError(f"source checkpoint data mismatch for {key}")

    n_qubit = int(cross_data["n_qubit"])
    depth = int(cross_data["depth"])
    if n_qubit % 2:
        raise ValueError("fixed-magnetization baseline requires even n_qubit")
    cross_train_count = len(cross_source_rows)
    target_train_count = len(protocol_rows(target_protocol, split="source_train"))
    cross_training_budget = (
        cross_train_count * depth * int(cross_data["train_shots_per_condition"])
    )
    target_training_budget = (
        target_train_count * depth * int(target_data["train_shots_per_condition"])
    )
    if cross_training_budget != target_training_budget:
        raise ValueError("cross and target-trained sampling budgets are not matched")

    cross_evaluation = cross_config["evaluation"]
    target_evaluation = target_config["evaluation"]
    budgets = [int(value) for value in cross_evaluation["sample_budgets"]]
    repeats = int(cross_evaluation["sampling_repeats"])
    comparison_seed = int(cross_evaluation.get("comparison_seed", 0))
    random_baselines = tuple(cross_evaluation.get("random_baselines", ()))
    target_random_baselines = tuple(target_evaluation.get("random_baselines", ()))
    if random_baselines != target_random_baselines:
        raise ValueError("cross/control random baselines differ")
    unknown_random = set(random_baselines) - set(RANDOM_BASELINE_METHODS)
    if unknown_random:
        raise ValueError(f"unknown random baselines: {sorted(unknown_random)}")
    methods = CORE_METHODS + tuple(
        RANDOM_BASELINE_METHODS[name] for name in random_baselines
    )
    uses_sector_random = "uniform_sector_random" in methods
    if budgets != [int(value) for value in target_evaluation["sample_budgets"]]:
        raise ValueError("cross/control evaluation budgets differ")
    if repeats != int(target_evaluation["sampling_repeats"]):
        raise ValueError("cross/control sampling repeats differ")
    if not cross_evaluation.get("known_initial_state_once", False):
        raise ValueError("joint comparison requires known_initial_state_once=true")

    resolved_device = resolve_device(device)
    cross_model, cross_checkpoint = load_trained_model(
        cross_checkpoint_path,
        protocol=checkpoint_protocol,
        device=resolved_device,
    )
    target_model, target_checkpoint = load_trained_model(
        target_checkpoint_path,
        protocol=target_protocol,
        device=resolved_device,
    )

    records = {
        row["instance_id"]: load_instance(cross_dataset_root, row)
        for row in cross_rows
    }
    dimension = 1 << n_qubit
    indices = np.arange(dimension, dtype=np.int64)
    shifts = np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    all_bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    hamming_weights = all_bits.sum(axis=1)
    target_weight = n_qubit // 2
    sector_indices = indices[hamming_weights == target_weight]
    sector_mask = hamming_weights == target_weight
    expected_sector_dimension = math.comb(n_qubit, target_weight)
    if len(sector_indices) != expected_sector_dimension:
        raise AssertionError("fixed-magnetization sector dimension mismatch")
    initial_index = int(bits_to_integers(neel_bitstring(n_qubit)[None, :])[0])
    nonzero_dimensions = list(range(1, depth))

    enumeration_batch_size = int(cross_evaluation.get("enumeration_batch_size", 512))
    model_distributions: dict[str, dict[tuple[str, int], np.ndarray]] = {
        "zero_shot_genksr": {},
        "target_trained_genksr": {},
    }
    distribution_rows = []
    for position, row in enumerate(cross_rows, start=1):
        identifier = row["instance_id"]
        record = records[identifier]
        for time_index in nonzero_dimensions:
            tau = float(record["tau"][time_index])
            exact = np.asarray(record["probabilities"][time_index], dtype=np.float64)
            if uses_sector_random and not np.isclose(
                exact[sector_mask].sum(), 1.0, atol=1e-9
            ):
                raise RuntimeError(f"exact target dynamics leaked outside sector: {identifier}")
            for method, model in (
                ("zero_shot_genksr", cross_model),
                ("target_trained_genksr", target_model),
            ):
                learned = _condition_distribution(
                    model,
                    record,
                    tau=tau,
                    device=resolved_device,
                    batch_size=enumeration_batch_size,
                )
                model_distributions[method][(identifier, time_index)] = learned
                metrics = distribution_metrics(exact, learned, n_qubit=n_qubit)
                distribution_rows.append(
                    {
                        "method": method,
                        "instance_id": identifier,
                        "time_index": time_index,
                        "tau": tau,
                        **metrics,
                        "target_sector_probability_mass": float(learned[sector_mask].sum()),
                        "exact_target_sector_probability_mass": float(
                            exact[sector_mask].sum()
                        ),
                    }
                )
        print(
            f"[joint distribution {position}/{len(cross_rows)}] {identifier}",
            flush=True,
        )

    energy_rows = []
    reachable_ground_rows = []
    full_candidates = indices
    for position, row in enumerate(cross_rows, start=1):
        identifier = row["instance_id"]
        record = records[identifier]
        hamiltonian = pauli_sum_matrix(_raw_terms(record))
        exact_ground = float(record["ground_energy"])
        if uses_sector_random:
            sector_ground = projected_ground_energy(hamiltonian, sector_indices)
            sector_ground_gap = abs(sector_ground - exact_ground)
            if sector_ground_gap > 1e-7:
                raise RuntimeError(
                    f"global ground state is outside the reachable sector for {identifier}: "
                    f"gap={sector_ground_gap}"
                )
            reachable_ground_rows.append(
                {
                    "instance_id": identifier,
                    "exact_ground_energy": exact_ground,
                    "reachable_sector_ground_energy": sector_ground,
                    "absolute_difference": sector_ground_gap,
                }
            )
        exact_by_time = {
            time_index: np.asarray(record["probabilities"][time_index], dtype=np.float64)
            for time_index in nonzero_dimensions
        }
        learned_by_method = {
            method: {
                time_index: model_distributions[method][(identifier, time_index)]
                for time_index in nonzero_dimensions
            }
            for method in ("zero_shot_genksr", "target_trained_genksr")
        }
        scale = float(record["coefficient_l1"])
        for shots in budgets:
            for repeat in range(repeats):
                bases = {
                    "direct_skqd": _basis_from_conditioned_distributions(
                        exact_by_time,
                        initial_index=initial_index,
                        shots_per_nonzero_time=shots,
                        seed_parts=(
                            comparison_seed,
                            identifier,
                            shots,
                            repeat,
                            "direct_skqd",
                        ),
                    ),
                    "zero_shot_genksr": _basis_from_conditioned_distributions(
                        learned_by_method["zero_shot_genksr"],
                        initial_index=initial_index,
                        shots_per_nonzero_time=shots,
                        seed_parts=(
                            comparison_seed,
                            identifier,
                            shots,
                            repeat,
                            "zero_shot_genksr",
                        ),
                    ),
                    "target_trained_genksr": _basis_from_conditioned_distributions(
                        learned_by_method["target_trained_genksr"],
                        initial_index=initial_index,
                        shots_per_nonzero_time=shots,
                        seed_parts=(
                            comparison_seed,
                            identifier,
                            shots,
                            repeat,
                            "target_trained_genksr",
                        ),
                    ),
                }
                if "uniform_sector_random" in methods:
                    bases["uniform_sector_random"] = _basis_from_uniform_candidates(
                        sector_indices,
                        initial_index=initial_index,
                        nonzero_dimensions=nonzero_dimensions,
                        shots_per_nonzero_time=shots,
                        seed_parts=(
                            comparison_seed,
                            identifier,
                            shots,
                            repeat,
                            "uniform_sector_random",
                        ),
                    )
                if "uniform_full_random" in methods:
                    bases["uniform_full_random"] = _basis_from_uniform_candidates(
                        full_candidates,
                        initial_index=initial_index,
                        nonzero_dimensions=nonzero_dimensions,
                        shots_per_nonzero_time=shots,
                        seed_parts=(
                            comparison_seed,
                            identifier,
                            shots,
                            repeat,
                            "uniform_full_random",
                        ),
                    )
                for method in methods:
                    basis = bases[method]
                    estimated = projected_ground_energy(hamiltonian, basis)
                    delta_e = abs(estimated - exact_ground)
                    in_sector = sum(bool(sector_mask[index]) for index in basis)
                    out_of_sector = len(basis) - in_sector
                    energy_rows.append(
                        {
                            "instance_id": identifier,
                            "shots_per_nonzero_time": shots,
                            "nonzero_time_conditions": len(nonzero_dimensions),
                            "total_candidate_draws": 1 + len(nonzero_dimensions) * shots,
                            "repeat": repeat,
                            "method": method,
                            "coefficient_l1": scale,
                            "exact_ground_energy": exact_ground,
                            "estimated_ground_energy": estimated,
                            "delta_e": delta_e,
                            "delta_e_per_qubit": delta_e / n_qubit,
                            "delta_e_normalized": delta_e / scale,
                            "unique_basis": len(basis),
                            "unique_in_sector_basis": in_sector,
                            "unique_out_of_sector_basis": out_of_sector,
                            "sector_dimension": expected_sector_dimension,
                        }
                    )
        print(f"[joint energy {position}/{len(cross_rows)}] {identifier}", flush=True)

    distribution_summary = _aggregate_distribution(distribution_rows)
    energy_summary = _aggregate_energy(energy_rows, methods=methods)
    paired_energy_summary = _aggregate_paired_energy(energy_rows, methods=methods)
    _write_csv(output_dir / "distribution_context.csv", distribution_rows)
    _write_csv(output_dir / "distribution_summary.csv", distribution_summary)
    _write_csv(output_dir / "skqd_absolute_error.csv", energy_rows)
    _write_csv(output_dir / "energy_summary.csv", energy_summary)
    _write_csv(output_dir / "paired_energy_difference.csv", paired_energy_summary)
    if reachable_ground_rows:
        _write_csv(output_dir / "reachable_sector_ground_check.csv", reachable_ground_rows)
    (output_dir / "RESULTS.md").write_text(
        _markdown_report(
            distribution_summary=distribution_summary,
            energy_summary=energy_summary,
            n_qubit=n_qubit,
            depth=depth,
            sector_dimension=expected_sector_dimension,
            repeats=repeats,
            methods=methods,
            uses_sector_random=uses_sector_random,
        )
    )
    metadata = {
        "cross_protocol_sha256": cross_protocol["protocol_sha256"],
        "cross_checkpoint_protocol_sha256": checkpoint_protocol["protocol_sha256"],
        "cross_checkpoint_dataset_root": str(checkpoint_dataset_root.resolve()),
        "target_trained_protocol_sha256": target_protocol["protocol_sha256"],
        "cross_checkpoint": str(Path(cross_checkpoint_path).resolve()),
        "target_trained_checkpoint": str(Path(target_checkpoint_path).resolve()),
        "cross_model_seed": int(cross_checkpoint["seed"]),
        "target_trained_model_seed": int(target_checkpoint["seed"]),
        "n_qubit": n_qubit,
        "depth": depth,
        "nonzero_time_conditions": len(nonzero_dimensions),
        "known_initial_state_included_once": True,
        "target_hamiltonians": len(cross_rows),
        "training_bitstrings_per_model": cross_training_budget,
        "sample_budgets_per_nonzero_time": budgets,
        "sampling_repeats": repeats,
        "comparison_seed": comparison_seed,
        "sector_hamming_weight": target_weight,
        "sector_dimension": expected_sector_dimension,
        "random_baselines": list(random_baselines),
        "methods": list(methods),
        "primary_energy_metric": "delta_e = abs(estimated_ground_energy - exact_ground_energy)",
        "secondary_energy_metric": "delta_e_normalized = delta_e / coefficient_l1",
        "device": str(resolved_device),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata
