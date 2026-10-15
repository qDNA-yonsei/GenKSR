from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

from scripts.generate_data import _generate_kqd_hamiltonian


def _small_kqd_spec(output_dir: Path, seed: int):
    return (
        str(output_dir),
        0,          # Hamiltonian id
        2,          # qubits
        3,          # Krylov dimension (writes D0 and D1)
        1,          # Trotter steps
        64,         # shots
        seed,
        "1d",
        1,          # unused lattice side
        "j1j2",    # unused coupling model
        1.0,
        [0.0, 1.0],
        [0.5, 1.5],
    )


def test_kqd_shadow_files_are_byte_reproducible(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    _generate_kqd_hamiltonian(_small_kqd_spec(first_dir, seed=20260713))
    _generate_kqd_hamiltonian(_small_kqd_spec(second_dir, seed=20260713))

    first_files = sorted(first_dir.glob("data_D*.npy"))
    second_files = sorted(second_dir.glob("data_D*.npy"))
    assert [path.name for path in first_files] == [path.name for path in second_files]
    assert first_files

    for first, second in zip(first_files, second_files):
        assert first.read_bytes() == second.read_bytes()
        values = np.load(first)
        assert values.shape == (64, 3)
        assert np.all((0 <= values) & (values <= 5))
