import csv

import pytest

from scripts.experiments.generate_xxz_sparsity_manifest import (
    REGIME_RANGES,
    SPLIT_COUNTS_PER_REGIME,
    generate_manifest_rows,
    validate_manifest_rows,
    write_frozen_manifest,
)


def test_manifest_is_deterministic_balanced_and_disjoint():
    first = generate_manifest_rows(seed=17)
    second = generate_manifest_rows(seed=17)
    assert first == second
    assert validate_manifest_rows(first)
    assert len(first) == 90
    assert len({row["J_xy"] for row in first}) == 90
    for split, count in SPLIT_COUNTS_PER_REGIME.items():
        for regime, bounds in REGIME_RANGES.items():
            rows = [
                row for row in first
                if row["split"] == split and row["regime"] == regime]
            assert len(rows) == count
            assert all(bounds[0] <= row["J_xy"] < bounds[1] for row in rows)


def test_manifest_changes_with_seed():
    assert generate_manifest_rows(seed=17) != generate_manifest_rows(seed=18)


def test_frozen_manifest_is_idempotent_but_refuses_replacement(tmp_path):
    output = tmp_path / "ratio_manifest.csv"
    rows = generate_manifest_rows(seed=17)
    write_frozen_manifest(rows, output, seed=17)
    original = output.read_bytes()
    write_frozen_manifest(rows, output, seed=17)
    assert output.read_bytes() == original
    with pytest.raises(FileExistsError, match="Frozen manifest differs"):
        write_frozen_manifest(
            generate_manifest_rows(seed=18), output, seed=18)
    loaded = list(csv.DictReader(output.open()))
    assert len(loaded) == 90
