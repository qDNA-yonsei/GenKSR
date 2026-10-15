"""Dataset generation, integrity, and compact training loader."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import (
    build_family_terms,
    encode_terms,
    protocol_digest,
    validate_protocol,
)
from .quantum import generate_instance_quantum_data


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _instance_path(dataset_root: str | Path, row: Mapping) -> Path:
    return Path(dataset_root) / row["split"] / f"{row['instance_id']}.npz"


def _generate_instance_task(task: tuple[dict, dict]) -> tuple[dict, dict]:
    row, data_config = task
    terms = build_family_terms(
        row["family"], int(data_config["n_qubit"]), row["parameters"]
    )
    codes, normalized_coefficients, scale = encode_terms(terms, normalize=True)
    quantum = generate_instance_quantum_data(
        terms,
        n_qubit=int(data_config["n_qubit"]),
        depth=int(data_config["depth"]),
        train_shots=int(data_config["train_shots_per_condition"]),
        sample_seed=int(row["instance_seed"]),
        include_training_samples=row["split"] == "source_train",
    )
    payload = {
        "instance_json": np.asarray(json.dumps(row, sort_keys=True)),
        "pauli_strings": np.asarray([pauli for pauli, _ in terms]),
        "pauli_codes": codes,
        "raw_coefficients": np.asarray([value for _, value in terms], dtype=np.float64),
        "normalized_coefficients": normalized_coefficients,
        "coefficient_l1": np.asarray(scale, dtype=np.float64),
        "tau": np.arange(int(data_config["depth"]), dtype=np.float64) * np.pi,
        "probabilities": quantum["probabilities"],
        "samples": quantum["samples"],
        "normalization_error": np.asarray(quantum["normalization_error"], dtype=np.float64),
        "ground_energy": np.asarray(quantum["ground_energy"], dtype=np.float64),
        "spectral_gap_above_ground_space": np.asarray(
            quantum["spectral_gap_above_ground_space"], dtype=np.float64
        ),
        "ground_space_degeneracy": np.asarray(
            quantum["ground_space_degeneracy"], dtype=np.int64
        ),
        "initial_ground_space_overlap": np.asarray(
            quantum["initial_ground_space_overlap"], dtype=np.float64
        ),
        "hermiticity_error": np.asarray(quantum["hermiticity_error"], dtype=np.float64),
    }
    return row, payload


def _validate_instance_file(path: Path, row: Mapping, data_config: Mapping) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["instance_json"].item()))
        if metadata != dict(row):
            raise ValueError(f"instance metadata mismatch in {path}")
        n_qubit = int(data_config["n_qubit"])
        depth = int(data_config["depth"])
        probabilities = data["probabilities"]
        if probabilities.shape != (depth, 1 << n_qubit):
            raise ValueError(f"probability shape mismatch in {path}")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        expected_d0 = np.zeros(1 << n_qubit, dtype=float)
        neel_index = int("".join(str(site % 2) for site in range(n_qubit)), 2)
        expected_d0[neel_index] = 1.0
        np.testing.assert_allclose(probabilities[0], expected_d0, atol=1e-12)
        codes = data["pauli_codes"]
        coefficients = data["normalized_coefficients"]
        if codes.ndim != 2 or codes.shape[1] != n_qubit:
            raise ValueError(f"Pauli-code shape mismatch in {path}")
        if coefficients.shape != (codes.shape[0],):
            raise ValueError(f"coefficient shape mismatch in {path}")
        if not np.isclose(np.sum(np.abs(coefficients)), 1.0, atol=1e-12):
            raise ValueError(f"normalized coefficient L1 mismatch in {path}")
        samples = data["samples"]
        if row["split"] == "source_train":
            expected = (
                depth,
                int(data_config["train_shots_per_condition"]),
                n_qubit,
            )
            if samples.shape != expected or samples.dtype != np.uint8:
                raise ValueError(f"training sample mismatch in {path}")
        elif samples.shape != (0, 0, n_qubit):
            raise ValueError(f"non-training instance unexpectedly stores samples in {path}")
        return {
            "instance_id": row["instance_id"],
            "split": row["split"],
            "family": row["family"],
            "num_terms": int(codes.shape[0]),
            "ground_energy": float(data["ground_energy"]),
            "initial_ground_space_overlap": float(data["initial_ground_space_overlap"]),
            "spectral_gap_above_ground_space": float(data["spectral_gap_above_ground_space"]),
        }


def write_protocol(dataset_root: str | Path, protocol: Mapping) -> Path:
    validate_protocol(protocol)
    root = Path(dataset_root)
    path = root / "protocol.json"
    content = json.dumps(protocol, indent=2, sort_keys=True).encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"existing protocol differs: {path}")
    if not path.exists():
        _atomic_write_bytes(path, content)
    return path


def load_protocol(dataset_root: str | Path) -> dict:
    path = Path(dataset_root) / "protocol.json"
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    return protocol


def _write_instances_csv(dataset_root: Path, rows: list[dict], summaries: Mapping[str, dict]) -> None:
    fieldnames = [
        "instance_id",
        "split",
        "family",
        "index",
        "parameters_json",
        "num_terms",
        "coefficient_l1",
        "dt",
        "instance_seed",
        "ground_energy",
        "initial_ground_space_overlap",
        "spectral_gap_above_ground_space",
    ]
    output = []
    for row in rows:
        summary = summaries[row["instance_id"]]
        output.append(
            {
                **{name: row[name] for name in fieldnames if name in row},
                "parameters_json": json.dumps(row["parameters"], sort_keys=True),
                "ground_energy": summary["ground_energy"],
                "initial_ground_space_overlap": summary["initial_ground_space_overlap"],
                "spectral_gap_above_ground_space": summary[
                    "spectral_gap_above_ground_space"
                ],
            }
        )
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output)
    _atomic_write_bytes(dataset_root / "instances.csv", buffer.getvalue().encode())


def generate_dataset(
    protocol: Mapping, dataset_root: str | Path, *, workers: int = 1
) -> dict:
    """Generate or resume the exact small-system dataset."""
    validate_protocol(protocol)
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    protocol_path = write_protocol(dataset_root, protocol)
    data_config = protocol["config"]["data"]
    rows = list(protocol["instances"])

    pending = []
    summaries: dict[str, dict] = {}
    for row in rows:
        path = _instance_path(dataset_root, row)
        if path.exists():
            summaries[row["instance_id"]] = _validate_instance_file(
                path, row, data_config
            )
        else:
            pending.append((row, data_config))

    if workers <= 1:
        iterator: Iterable[tuple[dict, dict]] = map(_generate_instance_task, pending)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=int(workers))
        iterator = executor.map(_generate_instance_task, pending)

    try:
        for position, (row, payload) in enumerate(iterator, start=1):
            path = _instance_path(dataset_root, row)
            _atomic_save_npz(path, **payload)
            summaries[row["instance_id"]] = _validate_instance_file(
                path, row, data_config
            )
            print(
                f"[{position}/{len(pending)}] generated {row['instance_id']} "
                f"terms={row['num_terms']}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    _write_instances_csv(dataset_root, rows, summaries)
    files = [protocol_path, dataset_root / "instances.csv"] + [
        _instance_path(dataset_root, row) for row in rows
    ]
    dataset_manifest = {
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_content_sha256": protocol_digest(protocol),
        "num_instances": len(rows),
        "files": {
            str(path.relative_to(dataset_root)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    _atomic_write_bytes(
        dataset_root / "manifest.json",
        json.dumps(dataset_manifest, indent=2, sort_keys=True).encode(),
    )
    return validate_dataset(dataset_root, verify_hashes=True)


def validate_dataset(dataset_root: str | Path, *, verify_hashes: bool = False) -> dict:
    dataset_root = Path(dataset_root)
    protocol = load_protocol(dataset_root)
    data_config = protocol["config"]["data"]
    summaries = [
        _validate_instance_file(_instance_path(dataset_root, row), row, data_config)
        for row in protocol["instances"]
    ]
    if verify_hashes:
        manifest = json.loads((dataset_root / "manifest.json").read_text())
        if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("dataset manifest protocol mismatch")
        for relative, metadata in manifest["files"].items():
            path = dataset_root / relative
            if sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"dataset hash mismatch: {path}")
    split_counts: dict[str, int] = defaultdict(int)
    for summary in summaries:
        split_counts[summary["split"]] += 1
    return {
        "dataset_root": str(dataset_root.resolve()),
        "protocol_sha256": protocol["protocol_sha256"],
        "num_instances": len(summaries),
        "split_counts": dict(sorted(split_counts.items())),
        "min_initial_ground_space_overlap": min(
            row["initial_ground_space_overlap"] for row in summaries
        ),
        "min_spectral_gap": min(
            row["spectral_gap_above_ground_space"] for row in summaries
        ),
        "hashes_verified": bool(verify_hashes),
    }


def load_instance(dataset_root: str | Path, row: Mapping) -> dict:
    path = _instance_path(dataset_root, row)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def protocol_rows(protocol: Mapping, *, split: str | None = None) -> list[dict]:
    rows = list(protocol["instances"])
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


class PauliSampleDataset(Dataset):
    """Compact source-training dataset with Hamiltonian tensors stored once."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.protocol = load_protocol(self.dataset_root)
        rows = protocol_rows(self.protocol, split="source_train")
        if not rows:
            raise ValueError("source_train split is empty")

        records = [load_instance(self.dataset_root, row) for row in rows]
        n_qubit = int(self.protocol["config"]["data"]["n_qubit"])
        max_terms = max(record["pauli_codes"].shape[0] for record in records)
        num_hamiltonians = len(records)
        pauli_codes = np.zeros(
            (num_hamiltonians, max_terms, n_qubit), dtype=np.uint8
        )
        coefficients = np.zeros((num_hamiltonians, max_terms), dtype=np.float32)
        term_mask = np.zeros((num_hamiltonians, max_terms), dtype=bool)

        bit_blocks = []
        hamiltonian_blocks = []
        tau_blocks = []
        for hamiltonian_index, record in enumerate(records):
            count = record["pauli_codes"].shape[0]
            pauli_codes[hamiltonian_index, :count] = record["pauli_codes"]
            coefficients[hamiltonian_index, :count] = record[
                "normalized_coefficients"
            ]
            term_mask[hamiltonian_index, :count] = True
            samples = record["samples"]
            depth, shots, _ = samples.shape
            bit_blocks.append(samples.reshape(depth * shots, n_qubit))
            hamiltonian_blocks.append(
                np.full(depth * shots, hamiltonian_index, dtype=np.int32)
            )
            tau_blocks.append(np.repeat(record["tau"], shots).astype(np.float32))

        self.bits = torch.from_numpy(np.concatenate(bit_blocks, axis=0)).long()
        self.hamiltonian_indices = torch.from_numpy(
            np.concatenate(hamiltonian_blocks, axis=0)
        ).long()
        self.tau = torch.from_numpy(np.concatenate(tau_blocks, axis=0)).float()
        self.pauli_codes = torch.from_numpy(pauli_codes).long()
        self.coefficients = torch.from_numpy(coefficients).float()
        self.term_mask = torch.from_numpy(term_mask)
        self.rows = rows
        self.n_qubit = n_qubit
        self.max_terms = max_terms

    def __len__(self) -> int:
        return int(self.bits.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: list[int]) -> dict[str, torch.Tensor]:
        indices_tensor = torch.as_tensor(indices, dtype=torch.long)
        hamiltonian_indices = self.hamiltonian_indices[indices_tensor]
        return {
            "bits": self.bits[indices_tensor],
            "tau": self.tau[indices_tensor],
            "pauli_codes": self.pauli_codes[hamiltonian_indices],
            "coefficients": self.coefficients[hamiltonian_indices],
            "term_mask": self.term_mask[hamiltonian_indices],
        }


def pad_instance_terms(record: Mapping, *, device: torch.device | str) -> dict[str, torch.Tensor]:
    codes = torch.as_tensor(record["pauli_codes"], dtype=torch.long, device=device)
    coefficients = torch.as_tensor(
        record["normalized_coefficients"], dtype=torch.float32, device=device
    )
    return {
        "pauli_codes": codes.unsqueeze(0),
        "coefficients": coefficients.unsqueeze(0),
        "term_mask": torch.ones(
            (1, codes.shape[0]), dtype=torch.bool, device=device
        ),
    }
