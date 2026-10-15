import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import sample as sampling
from scripts.sample import binary_samples_uint8, measurement_samples_uint8
from genksr.utils.quantum import decode_pauli6_shadow


def test_binary_samples_uint8_compacts_valid_bitstrings():
    samples = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)

    compact = binary_samples_uint8(samples, expected_qubits=3)

    assert compact.dtype == np.uint8
    assert np.array_equal(compact, samples)


def test_binary_samples_uint8_rejects_wrong_shape():
    with pytest.raises(ValueError, match='expected'):
        binary_samples_uint8(np.zeros((2, 4)), expected_qubits=3)


def test_binary_samples_uint8_rejects_nonbinary_values():
    with pytest.raises(ValueError, match='non-binary'):
        binary_samples_uint8(np.array([[0, 1, 2]]), expected_qubits=3)


@pytest.mark.parametrize('invalid', [-1, 6, 256, 0.5, np.nan, np.inf])
def test_kqd_samples_reject_invalid_outcomes_before_uint8_conversion(invalid):
    with pytest.raises(ValueError, match='Pauli-6'):
        measurement_samples_uint8(np.array([[0, 1, invalid]]), n_qubit=2, task='kqd')


@pytest.mark.parametrize('shape', [(2, 2), (2, 4), (3,), (2, 3, 1)])
def test_kqd_samples_require_one_ancilla_column(shape):
    with pytest.raises(ValueError, match='expected'):
        measurement_samples_uint8(np.zeros(shape), n_qubit=2, task='kqd')


def test_measurement_samples_reject_unknown_task():
    with pytest.raises(ValueError, match='Unknown task'):
        measurement_samples_uint8(np.zeros((2, 3)), n_qubit=2, task='unknown')


def test_kqd_compact_samples_preserve_pauli6_measurement_signs():
    samples = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    compact = measurement_samples_uint8(samples, n_qubit=2, task='kqd')

    outcomes, bases = decode_pauli6_shadow(compact)

    np.testing.assert_array_equal(outcomes, [[-1, -1, -1], [1, 1, 1]])
    np.testing.assert_array_equal(bases, [[0, 1, 2], [0, 1, 2]])
    original_outcomes, original_bases = decode_pauli6_shadow(samples)
    np.testing.assert_array_equal(outcomes, original_outcomes)
    np.testing.assert_array_equal(bases, original_bases)


@pytest.mark.parametrize('task,lattice,samples', [
    ('kqd', '1d', [[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]),
    ('kqd', '2d', [[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]),
    ('sqkd', '2d', [[0, 1, 0, 1], [1, 0, 1, 0]]),
])
def test_sampling_saves_task_measurements_without_changing_values(
        tmp_path, monkeypatch, task, lattice, samples):
    """Exercise the actual save path, including KQD's ancilla and all six outcomes."""
    samples = np.asarray(samples, dtype=np.int64)
    data_dir = tmp_path / 'data' / 'tiny' / 'test'
    data_dir.mkdir(parents=True)
    (data_dir / 'args.json').write_text(json.dumps({
        'protocol_version': 3,
        'task': task,
        'lattice': lattice,
        'n_qubit': 4,
        'n_outcomes': 6 if task == 'kqd' else 2,
    }))
    coupling = np.eye(4)
    np.save(data_dir / 'coupling_matrix_id0.npy', coupling)

    def generate_samples(*args, **kwargs):
        assert kwargs['task'] == task
        return {0: samples.copy()}

    monkeypatch.setattr(sampling, 'generate_samples', generate_samples)
    hamiltonian = SimpleNamespace(coeffs=np.array([1.0]))
    for builder in ('build_hamiltonian_from_couplings', 'build_2d_xxz_hamiltonian'):
        monkeypatch.setattr(sampling, builder, lambda **_: hamiltonian)

    results_dir = tmp_path / 'results'
    sampling.run_sampling_for_dim(
        model=None, results_dir=str(results_dir), data_root=str(tmp_path / 'data'),
        data_subdir='tiny', snapshots=len(samples), rng=np.random.default_rng(0),
        gcn_features='one_hot', device='cpu', krylov_dim=0, n_qubit=4,
        task=task, lattice=lattice, splits=('test',),
    )

    output_dir = results_dir / 'samples' / 'test' / 'model'
    saved = np.load(output_dir / 'model_samples_D0_2_id0.npy', allow_pickle=False)
    assert saved.dtype == np.uint8
    np.testing.assert_array_equal(saved, samples)
    np.testing.assert_array_equal(
        np.load(output_dir / 'coupling_matrix_id0.npy', allow_pickle=False), coupling)
