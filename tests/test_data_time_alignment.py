import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).parents[1] / 'genksr' / 'data' / 'hamiltonian.py'
_SPEC = importlib.util.spec_from_file_location('genksr_data_hamiltonian_test', _MODULE_PATH)
data_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data_module)


class _HamiltonianStub:
    coeffs = np.array([1.0, 1.0])


@pytest.fixture
def tiny_dataset(tmp_path):
    with open(tmp_path / 'args.json', 'w') as f:
        json.dump({
            'protocol_version': 3,
            'task': 'kqd',
            'n_outcomes': 6,
            'lattice': '1d',
            'n_qubit': 2,
            'delta': 1.0,
            'h_z': 0.0,
            'sqkd_dt': None,
            'num_trotter_steps': 2,
        }, f)
    np.save(tmp_path / 'coupling_matrix_id0.npy', np.array([[0.0, 1.0], [1.0, 0.0]]))
    for data_index in (0, 1):
        np.save(tmp_path / f'data_D{data_index}_id0.npy', np.array([[data_index, 0]]))
    return tmp_path


def _load_times(monkeypatch, data_dir, task, sqkd_dt=None):
    monkeypatch.setattr(
        data_module, 'build_hamiltonian_from_couplings', lambda **_: _HamiltonianStub())
    monkeypatch.setattr(
        data_module, 'build_2d_xxz_hamiltonian', lambda **_: _HamiltonianStub())

    with open(data_dir / 'args.json', 'r') as f:
        metadata = json.load(f)
    metadata['task'] = task
    metadata['n_outcomes'] = 2 if task == 'sqkd' else 6
    metadata['lattice'] = '2d' if task == 'sqkd' else '1d'
    metadata['sqkd_dt'] = sqkd_dt
    with open(data_dir / 'args.json', 'w') as f:
        json.dump(metadata, f)

    _, _, _, condition_times = data_module.load_data_for_all_dims(
        data_dir=str(data_dir),
        num_samples_per=1,
        num_hamiltonians=None,
        rng=np.random.default_rng(0),
        krylov_dims=[0, 1],
        n_qubit=2,
        task=task,
        lattice=metadata['lattice'],
        sqkd_dt=sqkd_dt,
    )
    return condition_times


def test_kqd_loader_conditions_on_nonzero_physical_lags(monkeypatch, tiny_dataset):
    times = _load_times(monkeypatch, tiny_dataset, task='kqd')
    assert times == pytest.approx([np.pi / 2, np.pi])


def test_sqkd_loader_conditions_d0_on_time_zero(monkeypatch, tiny_dataset):
    times = _load_times(monkeypatch, tiny_dataset, task='sqkd', sqkd_dt=0.15)
    assert times == pytest.approx([0.0, 0.15])


def test_dataset_trotter_mismatch_is_rejected(tiny_dataset):
    with pytest.raises(ValueError, match='num_trotter_steps'):
        data_module.validate_protocol_metadata(
            str(tiny_dataset), task='kqd', lattice='1d', n_qubit=2,
            num_trotter_steps=6)
