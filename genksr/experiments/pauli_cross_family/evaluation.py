"""Exact distribution and matched full-space SKQD evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .dataset import load_instance, load_protocol, pad_instance_terms, protocol_rows
from .protocol import canonicalize_terms
from .quantum import (
    cumulative_sampled_energy,
    pauli_sum_matrix,
    sample_probabilities,
)
from .training import load_trained_model, resolve_device


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(values, dtype=np.float64), 1e-300, None))


def distribution_metrics(p: np.ndarray, q: np.ndarray, *, n_qubit: int) -> dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be one-dimensional distributions with equal shape")
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p /= p.sum()
    q /= q.sum()
    entropy = float(-np.sum(p * _safe_log(p)))
    cross_entropy = float(-np.sum(p * _safe_log(q)))
    kl = max(0.0, cross_entropy - entropy)
    tv = float(0.5 * np.sum(np.abs(p - q)))
    mixture = 0.5 * (p + q)
    jsd = float(
        0.5 * np.sum(p * (_safe_log(p) - _safe_log(mixture)))
        + 0.5 * np.sum(q * (_safe_log(q) - _safe_log(mixture)))
    )
    return {
        "entropy": entropy,
        "cross_entropy": cross_entropy,
        "nll_per_qubit": cross_entropy / n_qubit,
        "excess_nll_per_qubit": kl / n_qubit,
        "kl": kl,
        "tv": tv,
        "jsd": max(0.0, jsd),
    }


def _normalized_term_dict(record: Mapping) -> dict[str, float]:
    return {
        str(pauli): float(coefficient)
        for pauli, coefficient in zip(
            record["pauli_strings"], record["normalized_coefficients"]
        )
    }


def _term_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return float(math.sqrt(sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys)))


def _nearest_source_indices(
    target_rows: Sequence[Mapping], target_records: Mapping[str, Mapping],
    source_rows: Sequence[Mapping], source_records: Mapping[str, Mapping]
) -> dict[str, tuple[str, float]]:
    source_vectors = {
        row["instance_id"]: _normalized_term_dict(source_records[row["instance_id"]])
        for row in source_rows
    }
    output = {}
    for row in target_rows:
        identifier = row["instance_id"]
        target = _normalized_term_dict(target_records[identifier])
        ranked = sorted(
            (
                (_term_distance(target, source_vector), source_id)
                for source_id, source_vector in source_vectors.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        output[identifier] = (ranked[0][1], float(ranked[0][0]))
    return output


def _condition_distribution(
    model,
    record: Mapping,
    *,
    tau: float,
    device: torch.device,
    enumeration_batch_size: int,
) -> np.ndarray:
    condition = pad_instance_terms(record, device=device)
    return model.exact_distribution(
        **condition,
        tau=torch.as_tensor([tau], dtype=torch.float32, device=device),
        batch_size=enumeration_batch_size,
    )


def _raw_terms(record: Mapping) -> list[tuple[str, float]]:
    return canonicalize_terms(
        [
            (str(pauli), float(coefficient))
            for pauli, coefficient in zip(
                record["pauli_strings"], record["raw_coefficients"]
            )
        ]
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must share a stable field order")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> dict:
    dataset_root = Path(dataset_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(dataset_root)
    config = protocol["config"]
    n_qubit = int(config["data"]["n_qubit"])
    depth = int(config["data"]["depth"])
    evaluation_config = config["evaluation"]
    resolved_device = resolve_device(device)
    model, checkpoint = load_trained_model(
        checkpoint_path, protocol=protocol, device=resolved_device
    )
    model_seed = int(checkpoint["seed"])

    source_train_rows = protocol_rows(protocol, split="source_train")
    source_holdout_rows = protocol_rows(protocol, split="source_holdout")
    target_rows = protocol_rows(protocol, split="target_test")
    all_needed_rows = source_train_rows + source_holdout_rows + target_rows
    records = {
        row["instance_id"]: load_instance(dataset_root, row) for row in all_needed_rows
    }
    nearest = _nearest_source_indices(
        target_rows,
        records,
        source_train_rows,
        records,
    )
    target_next = {
        row["instance_id"]: target_rows[(position + 1) % len(target_rows)]["instance_id"]
        for position, row in enumerate(target_rows)
    }

    enumeration_batch_size = int(evaluation_config.get("enumeration_batch_size", 512))
    distribution_rows: list[dict] = []
    target_q_cache: dict[tuple[str, int], np.ndarray] = {}
    for split_rows in (source_holdout_rows, target_rows):
        for row in split_rows:
            identifier = row["instance_id"]
            record = records[identifier]
            for dimension in range(depth):
                tau = float(record["tau"][dimension])
                p = record["probabilities"][dimension]
                q = _condition_distribution(
                    model,
                    record,
                    tau=tau,
                    device=resolved_device,
                    enumeration_batch_size=enumeration_batch_size,
                )
                if row["split"] == "target_test":
                    target_q_cache[(identifier, dimension)] = q
                metrics = distribution_metrics(p, q, n_qubit=n_qubit)
                base = {
                    "protocol_sha256": protocol["protocol_sha256"],
                    "model_seed": model_seed,
                    "split": row["split"],
                    "family": row["family"],
                    "instance_id": identifier,
                    "dimension_index": dimension,
                    "tau": tau,
                    "primary_nonzero_time": int(dimension > 0),
                }
                distribution_rows.append({**base, "method": "pauli_conditioned", **metrics})

                if row["split"] == "target_test":
                    mismatch_id = target_next[identifier]
                    mismatch_record = records[mismatch_id]
                    q_mismatch = _condition_distribution(
                        model,
                        mismatch_record,
                        tau=tau,
                        device=resolved_device,
                        enumeration_batch_size=enumeration_batch_size,
                    )
                    mismatch_metrics = distribution_metrics(
                        p, q_mismatch, n_qubit=n_qubit
                    )
                    distribution_rows.append(
                        {
                            **base,
                            "method": "mismatched_target_hamiltonian",
                            **mismatch_metrics,
                        }
                    )
                    nearest_id, _ = nearest[identifier]
                    nearest_p = records[nearest_id]["probabilities"][dimension]
                    nearest_metrics = distribution_metrics(
                        p, nearest_p, n_qubit=n_qubit
                    )
                    distribution_rows.append(
                        {
                            **base,
                            "method": "nearest_source_exact_distribution",
                            **nearest_metrics,
                        }
                    )
            print(
                f"[distribution seed={model_seed}] {row['split']} {identifier}",
                flush=True,
            )

    energy_rows: list[dict] = []
    budgets = [int(value) for value in evaluation_config["sample_budgets"]]
    repeats = int(evaluation_config["sampling_repeats"])
    for row in target_rows:
        identifier = row["instance_id"]
        record = records[identifier]
        hamiltonian = pauli_sum_matrix(_raw_terms(record))
        exact_ground = float(record["ground_energy"])
        scale = float(record["coefficient_l1"])
        for shots in budgets:
            for repeat in range(repeats):
                sample_sets: dict[str, dict[int, np.ndarray]] = {
                    "direct_1": {},
                    "direct_2": {},
                    "pauli_conditioned": {},
                }
                for dimension in range(depth):
                    p = record["probabilities"][dimension]
                    q = target_q_cache[(identifier, dimension)]
                    for method, probabilities in (
                        ("direct_1", p),
                        ("direct_2", p),
                        ("pauli_conditioned", q),
                    ):
                        rng = np.random.default_rng(
                            _stable_seed(
                                protocol["protocol_sha256"],
                                model_seed,
                                identifier,
                                shots,
                                repeat,
                                dimension,
                                method,
                            )
                        )
                        sample_sets[method][dimension] = sample_probabilities(
                            probabilities,
                            shots=shots,
                            n_qubit=n_qubit,
                            rng=rng,
                        )
                energies = {
                    method: cumulative_sampled_energy(hamiltonian, samples)
                    for method, samples in sample_sets.items()
                }
                supports = {
                    method: len(
                        {
                            tuple(bits.tolist())
                            for sample_block in samples.values()
                            for bits in sample_block
                        }
                    )
                    for method, samples in sample_sets.items()
                }
                energy_rows.append(
                    {
                        "protocol_sha256": protocol["protocol_sha256"],
                        "model_seed": model_seed,
                        "family": row["family"],
                        "instance_id": identifier,
                        "shots_per_condition": shots,
                        "depth": depth,
                        "repeat": repeat,
                        "coefficient_l1": scale,
                        "exact_ground_energy": exact_ground,
                        "direct_1_energy": energies["direct_1"],
                        "direct_2_energy": energies["direct_2"],
                        "model_energy": energies["pauli_conditioned"],
                        "direct_1_total_normalized_error": abs(
                            energies["direct_1"] - exact_ground
                        )
                        / scale,
                        "direct_2_total_normalized_error": abs(
                            energies["direct_2"] - exact_ground
                        )
                        / scale,
                        "model_total_normalized_error": abs(
                            energies["pauli_conditioned"] - exact_ground
                        )
                        / scale,
                        "model_direct_abs_normalized_gap": abs(
                            energies["pauli_conditioned"] - energies["direct_1"]
                        )
                        / scale,
                        "direct_direct_abs_normalized_gap": abs(
                            energies["direct_2"] - energies["direct_1"]
                        )
                        / scale,
                        "model_direct_signed_normalized_gap": (
                            abs(energies["pauli_conditioned"] - exact_ground)
                            - abs(energies["direct_1"] - exact_ground)
                        )
                        / scale,
                        "direct_1_unique_basis": supports["direct_1"],
                        "direct_2_unique_basis": supports["direct_2"],
                        "model_unique_basis": supports["pauli_conditioned"],
                    }
                )
        print(f"[energy seed={model_seed}] {identifier}", flush=True)

    _write_csv(output_dir / "distribution_context.csv", distribution_rows)
    _write_csv(output_dir / "skqd_pairs.csv", energy_rows)
    nearest_rows = [
        {
            "target_instance_id": target_id,
            "nearest_source_instance_id": source_id,
            "normalized_pauli_l2_distance": distance,
        }
        for target_id, (source_id, distance) in sorted(nearest.items())
    ]
    _write_csv(output_dir / "nearest_source.csv", nearest_rows)
    metadata = {
        "protocol_sha256": protocol["protocol_sha256"],
        "experiment_name": protocol["experiment_name"],
        "evaluation_regime": protocol.get(
            "evaluation_regime", "zero_shot_operator_template_holdout"
        ),
        "source_families": protocol["source_families"],
        "target_family": protocol["target_family"],
        "target_template_seen_in_source": bool(
            protocol.get("target_template_seen_in_source", False)
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "model_seed": model_seed,
        "device": str(resolved_device),
        "n_qubit": n_qubit,
        "depth": depth,
        "distribution_rows": len(distribution_rows),
        "energy_rows": len(energy_rows),
        "target_hamiltonians": len(target_rows),
        "source_holdout_hamiltonians": len(source_holdout_rows),
        "sample_budgets": budgets,
        "sampling_repeats": repeats,
        "primary_distribution_scope": "all nonzero-time contexts only",
        "model_sampling_scope": (
            "exact enumeration of the n-qubit autoregressive distribution followed "
            "by categorical finite sampling"
        ),
        "skqd_scope": "full computational-basis sampled subspace; no symmetry postselection",
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata
