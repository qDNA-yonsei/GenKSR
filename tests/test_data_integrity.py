import json

import numpy as np
import pytest

from genksr.data.integrity import (
    assert_disjoint_hamiltonian_splits,
    canonical_array_hash,
    write_dataset_manifest,
)


def _save_coupling(directory, identifier, matrix):
    directory.mkdir(exist_ok=True)
    np.save(directory / f'coupling_matrix_id{identifier}.npy', matrix)


def test_canonical_array_hash_includes_values_shape_and_dtype():
    array = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    assert canonical_array_hash(array) == canonical_array_hash(array.copy())
    assert canonical_array_hash(array) != canonical_array_hash(array.astype(np.float32))
    assert canonical_array_hash(array) != canonical_array_hash(array.reshape(4))


def test_train_test_duplicate_hamiltonian_is_rejected(tmp_path):
    train_dir = tmp_path / 'train'
    test_dir = tmp_path / 'test'
    matrix = np.array([[0.0, 0.7], [0.7, 0.0]])
    _save_coupling(train_dir, 0, matrix)
    _save_coupling(test_dir, 9, matrix.copy())

    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint_hamiltonian_splits(train_dir, test_dir)


def test_disjoint_splits_and_manifest(tmp_path):
    train_dir = tmp_path / 'train'
    test_dir = tmp_path / 'test'
    _save_coupling(train_dir, 0, np.array([[0.0, 0.7], [0.7, 0.0]]))
    _save_coupling(test_dir, 0, np.array([[0.0, 1.2], [1.2, 0.0]]))
    (train_dir / 'args.json').write_text('{"protocol_version": 3}')

    train_hashes, test_hashes = assert_disjoint_hamiltonian_splits(train_dir, test_dir)
    manifest_path = write_dataset_manifest(train_dir)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    assert set(train_hashes.values()).isdisjoint(test_hashes.values())
    assert manifest['protocol_version'] == 3
    assert set(manifest['files']) == {'args.json', 'coupling_matrix_id0.npy'}
    assert len(manifest['files']['coupling_matrix_id0.npy']['sha256']) == 64
