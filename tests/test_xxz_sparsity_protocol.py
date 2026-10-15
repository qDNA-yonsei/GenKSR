import argparse
import csv

import numpy as np
import pytest

from genksr.data.xxz_sparsity import (
    load_xxz_sparsity_data_for_all_dims,
    validate_xxz_sparsity_split,
)
from scripts.experiments.generate_xxz_sparsity_data import run_generation
from scripts.experiments.generate_xxz_sparsity_manifest import (
    generate_manifest_rows,
    write_frozen_manifest,
)

from scripts.experiments.xxz_sparsity_pilot import (
    chain_coupling_matrix,
    sample_nested_exact_streams,
    validate_args,
)


def _args(tmp_path, **updates):
    values = {
        "sizes": [4, 6],
        "jxy_values": [0.3, 1.0],
        "regime_labels": ["sparse", "dense"],
        "jzz": 1.0,
        "boundary_pin": 0.0,
        "depth": 3,
        "budgets": [30, 60],
        "repeats": 2,
        "seed": 123,
        "eig_tol": 1e-10,
        "minimum_separation": 2.0,
        "gap_warning_threshold": 1e-8,
        "output_dir": str(tmp_path / "pilot"),
        "overwrite": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_chain_coupling_matrix_is_open_and_nearest_neighbour_only():
    matrix = chain_coupling_matrix(4, 0.3)
    np.testing.assert_allclose(
        matrix,
        [[0.0, 0.3, 0.0, 0.0],
         [0.3, 0.0, 0.3, 0.0],
         [0.0, 0.3, 0.0, 0.3],
         [0.0, 0.0, 0.3, 0.0]],
    )


def test_nested_exact_streams_are_deterministic_and_use_prefixes():
    probabilities = np.asarray([
        [1.0, 0.0, 0.0],
        [0.2, 0.3, 0.5],
    ])
    first = sample_nested_exact_streams(probabilities, 20, [7, 1, 0])
    second = sample_nested_exact_streams(probabilities, 20, [7, 1, 0])
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    np.testing.assert_array_equal(first[0], np.zeros(20, dtype=np.int64))
    # A smaller budget is represented by a prefix, not a second random draw.
    np.testing.assert_array_equal(first[1][:5], second[1][:5])


def test_pilot_budget_must_be_divisible_by_depth(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        validate_args(_args(tmp_path, budgets=[31]))


def test_pilot_rejects_negative_boundary_pin(tmp_path):
    with pytest.raises(ValueError, match="boundary-pin"):
        validate_args(_args(tmp_path, boundary_pin=-0.1))


def test_pilot_rejects_nonempty_output_without_overwrite(tmp_path):
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="non-empty"):
        validate_args(_args(tmp_path))


def test_exact_dataset_generator_smoke_and_instance_time_loader(tmp_path):
    manifest = tmp_path / "ratio_manifest.csv"
    write_frozen_manifest(
        generate_manifest_rows(seed=19), manifest, seed=19)
    output_root = tmp_path / "data"
    args = argparse.Namespace(
        n_qubit=4,
        depth=3,
        shots=20,
        splits=["train", "validation", "test"],
        manifest=str(manifest),
        output_root=str(output_root),
        limit_per_regime=1,
        verify_only=False,
    )
    reports = run_generation(args)
    assert len(reports) == 3
    for split in args.splits:
        report = validate_xxz_sparsity_split(
            output_root / "4qubits" / split, verify_hashes=True)
        assert report["num_hamiltonians"] == 3
        assert report["sample_files"] == 9

    train_dir = output_root / "4qubits" / "train"
    measurements, couplings, ids, times = load_xxz_sparsity_data_for_all_dims(
        train_dir,
        num_samples_per=5,
        num_hamiltonians=None,
        rng=np.random.default_rng(4),
        krylov_dims=[0, 1],
    )
    assert measurements.shape == (30, 4)
    assert couplings.shape == (3, 4, 4)
    assert len(np.unique(ids)) == 3
    np.testing.assert_allclose(times[:15], 0.0)
    assert np.all(times[15:] > 0.0)
    np.testing.assert_array_equal(
        measurements[:15], np.tile([1, 0, 1, 0], (15, 1)))

    with (train_dir / "instances.csv").open() as handle:
        instance_rows = list(csv.DictReader(handle))
    assert len({row["dt"] for row in instance_rows}) == 3

    args.verify_only = True
    verified = run_generation(args)
    assert all(report["hashes_verified"] for report in verified)
