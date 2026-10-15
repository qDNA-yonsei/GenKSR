"""Manifest-aware loading and integrity checks for the 1D XXZ sparsity data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from genksr.data.integrity import sha256_file
from genksr.protocol import PROTOCOL_VERSION, neel_reference_bitstring, xxz_chain_timestep


HAMILTONIAN_FAMILY = "xxz_1d_jxy_sparsity"
HAMILTONIAN_SUBSET_MODES = ("random", "stratified_nested")


def read_instance_rows(split_dir):
    path = Path(split_dir) / "instances.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty instance metadata: {path}")
    return rows


def expected_jxy_chain(n_qubit, j_xy):
    matrix = np.zeros((n_qubit, n_qubit), dtype=float)
    sites = np.arange(n_qubit - 1)
    matrix[sites, sites + 1] = float(j_xy)
    matrix[sites + 1, sites] = float(j_xy)
    return matrix


def validate_xxz_sparsity_metadata(split_dir):
    split_dir = Path(split_dir)
    args_path = split_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(args_path)
    args = json.loads(args_path.read_text())
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "sqkd",
        "lattice": "1d",
        "hamiltonian_family": HAMILTONIAN_FAMILY,
        "boundary_pin": 0.0,
        "n_outcomes": 2,
        "time_index": "data_Dd -> d*dt",
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise ValueError(
                f"Dataset {name}={args.get(name)!r}, expected {value!r}")
    n_qubit = int(args["n_qubit"])
    if n_qubit < 2 or n_qubit % 2:
        raise ValueError("XXZ sparsity data requires an even n_qubit >= 2")
    if int(args["krylov_dim"]) <= 0 or int(args["shots"]) <= 0:
        raise ValueError("krylov_dim and shots must be positive")
    if not np.isclose(float(args["J_zz"]), 1.0):
        raise ValueError("The frozen XXZ sparsity protocol requires J_zz=1")
    return args


def validate_xxz_sparsity_split(split_dir, verify_hashes=False):
    """Validate shapes, physics metadata, bit order, and optional file hashes."""
    split_dir = Path(split_dir)
    args = validate_xxz_sparsity_metadata(split_dir)
    rows = read_instance_rows(split_dir)
    n_qubit = int(args["n_qubit"])
    depth = int(args["krylov_dim"])
    shots = int(args["shots"])
    neel = np.asarray(
        [int(bit) for bit in neel_reference_bitstring(n_qubit)],
        dtype=np.uint8,
    )
    ids = [int(row["split_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("instances.csv contains duplicate split_id values")
    if len(rows) != int(args["num_hamiltonians"]):
        raise ValueError("instances.csv row count does not match args.json")

    expected_files = {"args.json", "instances.csv"}
    for row in rows:
        identifier = int(row["split_id"])
        j_xy = float(row["J_xy"])
        j_zz = float(row["J_zz"])
        expected_dt = xxz_chain_timestep(n_qubit, j_xy, j_zz)
        if not np.isclose(float(row["dt"]), expected_dt, rtol=1e-13):
            raise ValueError(f"Incorrect dt for split_id={identifier}")

        coupling_name = f"coupling_matrix_id{identifier}.npy"
        expected_files.add(coupling_name)
        coupling_path = split_dir / coupling_name
        if not coupling_path.exists():
            raise FileNotFoundError(coupling_path)
        coupling = np.load(coupling_path, allow_pickle=False)
        np.testing.assert_allclose(
            coupling, expected_jxy_chain(n_qubit, j_xy), rtol=0.0, atol=1e-14)

        for dimension in range(depth):
            name = f"data_D{dimension}_id{identifier}.npy"
            expected_files.add(name)
            path = split_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            samples = np.load(path, allow_pickle=False, mmap_mode="r")
            if samples.shape != (shots, n_qubit):
                raise ValueError(
                    f"{path} has shape {samples.shape}, expected {(shots, n_qubit)}")
            if samples.dtype != np.uint8:
                raise ValueError(f"{path} must use uint8, got {samples.dtype}")
            if not np.all((samples == 0) | (samples == 1)):
                raise ValueError(f"{path} contains a non-binary outcome")
            if not np.all(np.sum(samples, axis=1) == n_qubit // 2):
                raise ValueError(f"{path} contains an outcome outside S_z=0")
            if dimension == 0 and not np.all(samples == neel):
                raise ValueError(f"{path} does not contain the canonical Neel D0")

    actual_regular_files = {
        path.name for path in split_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    unexpected = actual_regular_files - expected_files
    missing = expected_files - actual_regular_files
    if unexpected or missing:
        raise ValueError(
            f"Dataset file mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}")

    manifest_path = split_dir / "manifest.json"
    if verify_hashes:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Invalid dataset manifest protocol version")
        if set(manifest.get("files", {})) != expected_files:
            raise ValueError("manifest.json file list does not match dataset")
        for filename, metadata in manifest["files"].items():
            if sha256_file(split_dir / filename) != metadata["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {filename}")

    return {
        "split": args["split"],
        "n_qubit": n_qubit,
        "num_hamiltonians": len(rows),
        "depth": depth,
        "shots": shots,
        "sample_files": len(rows) * depth,
        "hashes_verified": bool(verify_hashes),
    }


def _stratified_nested_rows(rows, num_hamiltonians, seed):
    """Return a regime-balanced prefix of deterministic per-regime orders."""
    if seed is None:
        raise ValueError(
            "stratified_nested selection requires hamiltonian_subset_seed")

    regime_order = []
    rows_by_regime = {}
    for row in rows:
        regime = row.get("regime")
        if not regime:
            raise ValueError("stratified_nested selection requires regime metadata")
        if regime not in rows_by_regime:
            regime_order.append(regime)
            rows_by_regime[regime] = []
        rows_by_regime[regime].append(row)

    n_regimes = len(regime_order)
    base, remainder = divmod(num_hamiltonians, n_regimes)
    selected = []
    for regime_index, regime in enumerate(regime_order):
        count = base + int(regime_index < remainder)
        candidates = sorted(
            rows_by_regime[regime], key=lambda row: int(row["split_id"]))
        if count > len(candidates):
            raise ValueError(
                f"Requested {count} {regime} Hamiltonians, only "
                f"{len(candidates)} are available")
        regime_seed = int.from_bytes(
            hashlib.sha256(
                f"xxz-hamiltonian-subset|{seed}|{regime}".encode()
            ).digest()[:8],
            byteorder="little",
            signed=False,
        )
        order = np.random.default_rng(regime_seed).permutation(len(candidates))
        selected.extend(candidates[index] for index in order[:count])
    return sorted(selected, key=lambda row: int(row["split_id"]))


def _nested_measurement_indices(
        num_available, num_samples, seed, split, n_qubit, identifier, dimension):
    payload = (
        f"xxz-measurement-subset|{seed}|{split}|{n_qubit}|"
        f"{identifier}|{dimension}"
    ).encode()
    file_seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)
    return np.random.default_rng(file_seed).permutation(num_available)[:num_samples]


def load_xxz_sparsity_data_for_all_dims(
        data_dir, num_samples_per, num_hamiltonians, rng, krylov_dims,
        hamiltonian_subset_mode="random", hamiltonian_subset_seed=None,
        measurement_subset_seed=None, split=None):
    """Load measurements with per-instance physical-time conditioning."""
    args = validate_xxz_sparsity_metadata(data_dir)
    rows = read_instance_rows(data_dir)
    n_qubit = int(args["n_qubit"])
    depth = int(args["krylov_dim"])
    dimensions = [int(value) for value in krylov_dims]
    if not dimensions or min(dimensions) < 0 or max(dimensions) >= depth:
        raise ValueError(f"krylov_dims must be within [0, {depth - 1}]")
    if num_samples_per <= 0 or num_samples_per > int(args["shots"]):
        raise ValueError("num_samples_per is outside the saved shot range")

    rows = sorted(rows, key=lambda row: int(row["split_id"]))
    if num_hamiltonians is not None:
        if num_hamiltonians <= 0 or num_hamiltonians > len(rows):
            raise ValueError("num_hamiltonians is outside the available range")
        if hamiltonian_subset_mode == "stratified_nested":
            rows = _stratified_nested_rows(
                rows, num_hamiltonians, hamiltonian_subset_seed)
        elif hamiltonian_subset_mode == "random":
            selected = rng.choice(
                len(rows), size=num_hamiltonians, replace=False)
            rows = [rows[index] for index in sorted(selected)]
        else:
            raise ValueError(
                f"Unknown hamiltonian_subset_mode "
                f"{hamiltonian_subset_mode!r}; expected "
                f"{HAMILTONIAN_SUBSET_MODES}")

    coupling_matrices = []
    for row in rows:
        identifier = int(row["split_id"])
        coupling_matrices.append(np.load(
            Path(data_dir) / f"coupling_matrix_id{identifier}.npy",
            allow_pickle=False,
        ))

    measurements = []
    hamiltonian_ids = []
    condition_times = []
    for dimension in dimensions:
        for row in rows:
            identifier = int(row["split_id"])
            data = np.load(
                Path(data_dir) / f"data_D{dimension}_id{identifier}.npy",
                allow_pickle=False,
            )
            if measurement_subset_seed is None:
                indices = rng.choice(
                    len(data), size=num_samples_per, replace=False)
            else:
                indices = _nested_measurement_indices(
                    len(data), num_samples_per, measurement_subset_seed,
                    split or args["split"], n_qubit, identifier, dimension)
            measurements.append(data[indices])
            hamiltonian_ids.append(np.full(
                num_samples_per, identifier, dtype=np.int64))
            condition_times.append(np.full(
                num_samples_per,
                dimension * float(row["dt"]),
                dtype=float,
            ))
    return (
        np.concatenate(measurements, axis=0),
        np.stack(coupling_matrices, axis=0),
        np.concatenate(hamiltonian_ids, axis=0),
        np.concatenate(condition_times, axis=0),
    )
