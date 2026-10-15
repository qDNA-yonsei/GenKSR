"""Dataset provenance and train/test split integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from genksr.protocol import PROTOCOL_VERSION


MANIFEST_FILENAME = 'manifest.json'


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hash(array):
    """Hash array values together with shape and dtype, independent of ``.npy`` headers."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def coupling_hashes(split_dir):
    hashes = {}
    for filename in sorted(os.listdir(split_dir)):
        if filename.startswith('coupling_matrix_id') and filename.endswith('.npy'):
            matrix = np.load(os.path.join(split_dir, filename), allow_pickle=False)
            hashes[filename] = canonical_array_hash(matrix)
    return hashes


def assert_disjoint_hamiltonian_splits(train_dir, test_dir):
    """Raise if any train and test coupling matrices describe the same Hamiltonian."""
    train_hashes = coupling_hashes(train_dir)
    test_hashes = coupling_hashes(test_dir)
    train_by_hash = {digest: name for name, digest in train_hashes.items()}
    duplicates = [
        (train_by_hash[digest], test_name)
        for test_name, digest in test_hashes.items()
        if digest in train_by_hash
    ]
    if duplicates:
        examples = ', '.join(f'{train} = {test}' for train, test in duplicates[:5])
        raise ValueError(
            f'Train/test Hamiltonian overlap detected ({len(duplicates)} duplicate(s)): '
            f'{examples}')
    return train_hashes, test_hashes


def write_dataset_manifest(split_dir):
    """Write SHA-256 provenance for every regular dataset file in one split."""
    files = {}
    for filename in sorted(os.listdir(split_dir)):
        path = os.path.join(split_dir, filename)
        if filename == MANIFEST_FILENAME or not os.path.isfile(path):
            continue
        files[filename] = {
            'bytes': os.path.getsize(path),
            'sha256': sha256_file(path),
        }

    manifest = {
        'protocol_version': PROTOCOL_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': files,
    }
    output_path = os.path.join(split_dir, MANIFEST_FILENAME)
    temporary_path = f'{output_path}.tmp'
    with open(temporary_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(temporary_path, output_path)
    return output_path
