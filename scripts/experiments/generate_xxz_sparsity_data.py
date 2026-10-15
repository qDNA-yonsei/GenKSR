#!/usr/bin/env python3
"""Generate exact pure-XXZ Krylov measurements from the frozen ratio manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genksr.data.integrity import write_dataset_manifest  # noqa: E402
from genksr.data.xxz_sparsity import (  # noqa: E402
    HAMILTONIAN_FAMILY,
    expected_jxy_chain,
    validate_xxz_sparsity_split,
)
from genksr.evaluation.sparsity import (  # noqa: E402
    exact_krylov_states,
    fixed_hamming_weight_basis,
    sample_sector_bitstrings,
    xxz_open_chain_sector_components,
)
from genksr.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    neel_reference_bitstring,
    xxz_chain_pauli_l1_bound,
    xxz_chain_timestep,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact pure-XXZ sparsity dataset generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-qubit", type=int, default=12)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--manifest", default="data/xxz_sparsity_1d/ratio_manifest.csv")
    parser.add_argument("--output-root", default="data/xxz_sparsity_1d")
    parser.add_argument(
        "--limit-per-regime", type=int, default=None,
        help="Smoke-test subset only; select the first K rows in each regime/split",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Validate existing selected splits without generating files",
    )
    return parser.parse_args(argv)


def _validate_args(args):
    if args.n_qubit < 2 or args.n_qubit % 2:
        raise ValueError("n-qubit must be an even integer >= 2")
    if args.depth <= 0 or args.shots <= 0:
        raise ValueError("depth and shots must be positive")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("splits must be unique")
    if args.limit_per_regime is not None and args.limit_per_regime <= 0:
        raise ValueError("limit-per-regime must be positive")


def _read_frozen_manifest(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("csv_sha256"):
        raise ValueError("Frozen ratio manifest SHA-256 does not match metadata")
    if metadata.get("boundary_pin") != 0.0 or metadata.get("jzz") != 1.0:
        raise ValueError("Manifest is not the frozen pure-XXZ protocol")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90 or len({row["J_xy"] for row in rows}) != 90:
        raise ValueError("Frozen ratio manifest must contain 90 distinct J_xy rows")
    return rows, metadata, digest


def _select_rows(rows, split, limit_per_regime=None):
    selected = [row for row in rows if row["split"] == split]
    if limit_per_regime is not None:
        limited = []
        for regime in ("sparse", "middle", "dense"):
            limited.extend(
                [row for row in selected if row["regime"] == regime]
                [:limit_per_regime])
        selected = limited
    return sorted(selected, key=lambda row: int(row["split_id"]))


def _atomic_write_bytes(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_save_npy(path, array):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _csv_bytes(rows):
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_or_verify(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing metadata differs: {path}")
    else:
        _atomic_write_bytes(path, content)


def _sampling_rng(instance_seed, n_qubit, dimension):
    return np.random.default_rng(np.random.SeedSequence([
        int(instance_seed), int(n_qubit), int(dimension), 0x58585A,
    ]))


def _instance_metadata(master_rows, n_qubit, depth, shots):
    output = []
    for row in master_rows:
        j_xy = float(row["J_xy"])
        j_zz = float(row["J_zz"])
        output.append({
            "split_id": int(row["split_id"]),
            "master_id": int(row["master_id"]),
            "paired_key": row["paired_key"],
            "split": row["split"],
            "regime": row["regime"],
            "regime_id": int(row["regime_id"]),
            "J_xy": j_xy,
            "J_zz": j_zz,
            "n_qubit": int(n_qubit),
            "krylov_dim": int(depth),
            "shots": int(shots),
            "coefficient_l1": xxz_chain_pauli_l1_bound(
                n_qubit, j_xy, j_zz),
            "dt": xxz_chain_timestep(n_qubit, j_xy, j_zz),
            "instance_seed": int(row["instance_seed"]),
        })
    return output


def _split_args(
        split, selected, n_qubit, depth, shots, manifest_path,
        manifest_digest, limit_per_regime):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": "sqkd",
        "lattice": "1d",
        "hamiltonian_family": HAMILTONIAN_FAMILY,
        "n_qubit": int(n_qubit),
        "krylov_dim": int(depth),
        "shots": int(shots),
        "num_hamiltonians": len(selected),
        "split": split,
        "J_zz": 1.0,
        "boundary_pin": 0.0,
        "n_outcomes": 2,
        "reference_state": "displayed-order 1010... Neel",
        "time_index": "data_Dd -> d*dt",
        "dt_mode": "per_instance_pi_over_pauli_l1_bound",
        "evolution": "exact scipy.sparse.linalg.expm_multiply in S_z=0 sector",
        "trotterized": False,
        "measurement": "computational-basis full N-bit bitstrings",
        "bit_order": "displayed q[N-1]...q[0]",
        "coupling_matrix_semantics": "J_xy-weighted open-chain adjacency",
        "ratio_manifest": str(Path(manifest_path).resolve()),
        "ratio_manifest_sha256": manifest_digest,
        "subset_limit_per_regime": limit_per_regime,
    }


def _instance_complete(split_dir, identifier, depth, shots, n_qubit, j_xy):
    coupling_path = split_dir / f"coupling_matrix_id{identifier}.npy"
    data_paths = [
        split_dir / f"data_D{dimension}_id{identifier}.npy"
        for dimension in range(depth)
    ]
    paths = [coupling_path, *data_paths]
    if not any(path.exists() for path in paths):
        return False
    if not all(path.exists() for path in paths):
        return False
    coupling = np.load(coupling_path, allow_pickle=False)
    if not np.allclose(
            coupling, expected_jxy_chain(n_qubit, j_xy), rtol=0.0, atol=1e-14):
        raise ValueError(f"Existing coupling differs for id={identifier}")
    for path in data_paths:
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        if data.shape != (shots, n_qubit) or data.dtype != np.uint8:
            raise ValueError(f"Existing data is invalid: {path}")
    return True


def generate_split(
        split_dir, split, master_rows, args, manifest_digest,
        sector_basis, h_xy, h_zz, initial_state):
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = _instance_metadata(
        master_rows, args.n_qubit, args.depth, args.shots)
    split_args = _split_args(
        split, master_rows, args.n_qubit, args.depth, args.shots,
        args.manifest, manifest_digest, args.limit_per_regime)
    _write_or_verify(
        split_dir / "args.json",
        json.dumps(split_args, indent=2, sort_keys=True).encode("utf-8"),
    )
    _write_or_verify(split_dir / "instances.csv", _csv_bytes(metadata_rows))

    for position, row in enumerate(metadata_rows, start=1):
        identifier = int(row["split_id"])
        j_xy = float(row["J_xy"])
        if _instance_complete(
                split_dir, identifier, args.depth, args.shots,
                args.n_qubit, j_xy):
            print(
                f"[{split} {position}/{len(metadata_rows)}] id={identifier} "
                f"Jxy={j_xy:.6f} already complete", flush=True)
            continue

        start = time.perf_counter()
        hamiltonian = (j_xy * h_xy + float(row["J_zz"]) * h_zz).tocsr()
        states = exact_krylov_states(
            hamiltonian,
            initial_state=initial_state,
            depth=args.depth,
            dt=float(row["dt"]),
        )
        norm_error = float(np.max(
            np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)))
        if norm_error > 1e-8:
            raise RuntimeError(
                f"Evolution norm error {norm_error} for id={identifier}")

        _atomic_save_npy(
            split_dir / f"coupling_matrix_id{identifier}.npy",
            expected_jxy_chain(args.n_qubit, j_xy),
        )
        for dimension, state in enumerate(states):
            samples = sample_sector_bitstrings(
                state,
                sector_basis=sector_basis,
                n_qubit=args.n_qubit,
                shots=args.shots,
                rng=_sampling_rng(
                    int(row["instance_seed"]), args.n_qubit, dimension),
            )
            _atomic_save_npy(
                split_dir / f"data_D{dimension}_id{identifier}.npy",
                samples,
            )
        print(
            f"[{split} {position}/{len(metadata_rows)}] id={identifier} "
            f"regime={row['regime']} Jxy={j_xy:.6f} "
            f"dt={float(row['dt']):.6f} {time.perf_counter() - start:.2f}s",
            flush=True,
        )

    report = validate_xxz_sparsity_split(split_dir, verify_hashes=False)
    write_dataset_manifest(split_dir)
    validate_xxz_sparsity_split(split_dir, verify_hashes=True)
    print(f"Validated {split}: {report}", flush=True)


def run_generation(args):
    _validate_args(args)
    rows, _, manifest_digest = _read_frozen_manifest(args.manifest)
    size_root = Path(args.output_root) / f"{args.n_qubit}qubits"
    if args.verify_only:
        reports = []
        for split in args.splits:
            reports.append(validate_xxz_sparsity_split(
                size_root / split, verify_hashes=True))
        return reports

    print(f"Building N={args.n_qubit} sector components...", flush=True)
    sector_basis = fixed_hamming_weight_basis(
        args.n_qubit, args.n_qubit // 2)
    h_xy, h_zz = xxz_open_chain_sector_components(
        args.n_qubit, sector_basis)
    if not sparse.isspmatrix_csr(h_xy) or not sparse.isspmatrix_csr(h_zz):
        raise TypeError("Expected CSR XXZ components")
    neel_integer = int(neel_reference_bitstring(args.n_qubit), 2)
    neel_position = int(np.searchsorted(sector_basis, neel_integer))
    if (neel_position >= len(sector_basis)
            or int(sector_basis[neel_position]) != neel_integer):
        raise RuntimeError("Canonical Neel state is outside the sector basis")
    initial_state = np.zeros(len(sector_basis), dtype=complex)
    initial_state[neel_position] = 1.0
    print(
        f"Sector dimension={len(sector_basis):,}, Hxy nnz={h_xy.nnz:,}",
        flush=True,
    )

    reports = []
    for split in args.splits:
        selected = _select_rows(rows, split, args.limit_per_regime)
        if not selected:
            raise ValueError(f"No manifest rows selected for {split}")
        generate_split(
            size_root / split,
            split=split,
            master_rows=selected,
            args=args,
            manifest_digest=manifest_digest,
            sector_basis=sector_basis,
            h_xy=h_xy,
            h_zz=h_zz,
            initial_state=initial_state,
        )
        reports.append(validate_xxz_sparsity_split(
            size_root / split, verify_hashes=True))
    return reports


def main(argv=None):
    reports = run_generation(parse_args(argv))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
