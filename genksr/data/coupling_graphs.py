import itertools as it
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from genksr.constants import ONE_HOT_FEATURE, WDEGREE_FEATURE

__all__ = ['CouplingGraphsDataset']


class _ByIdSequence:
    """Indexable sequence that maps sample indices to per-Hamiltonian objects."""

    def __init__(self, sample_ids, objects_by_id):
        self._sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self._objects_by_id = objects_by_id

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        cid = int(self._sample_ids[int(item)])
        return self._objects_by_id[cid]


class CouplingGraphsDataset(Dataset):
    def __init__(self, coupling_matrices: np.ndarray, coupling_matrices_ids: List[int], rng: np.random.Generator,
                 device, feature_type, seed=None, sample_ids=None):
        self.rng = np.random.default_rng(seed=seed) if rng is None else rng
        coupling_matrices_ids = np.asarray(coupling_matrices_ids, dtype=np.int64)
        sample_ids = coupling_matrices_ids if sample_ids is None else np.asarray(sample_ids, dtype=np.int64)

        if len(coupling_matrices_ids) != coupling_matrices.shape[0]:
            raise ValueError("coupling_matrices_ids must align with coupling_matrices rows")

        self._coupling_matrices_dict = {
            int(k): v for k, v in zip(coupling_matrices_ids, coupling_matrices)
        }
        self._coupling_matrices_ids = sample_ids
        self._sample_ids = sample_ids
        self._coupling_matrices_tensor_by_id = {
            int(k): torch.as_tensor(v, dtype=torch.float32)
            for k, v in self._coupling_matrices_dict.items()
        }
        self._coupling_matrices_tensor = _ByIdSequence(
            self._sample_ids, self._coupling_matrices_tensor_by_id
        )

        self._device = device

        if len(np.unique(list(cm.shape[0] for cm in coupling_matrices))) == 1:
            self._num_qubits = coupling_matrices[0].shape[0]
        else:
            self._num_qubits = np.nan

        self._edge_indices, self._edge_weights, self._vertices_deg = self._build_graphs(coupling_matrices)

        if feature_type == ONE_HOT_FEATURE:
            assert not np.isnan(self._num_qubits), 'one_hot features only available for systems of equal size!'
            data_by_id = {
                int(cid): Data(x=torch.eye(self._num_qubits, dtype=torch.float32), edge_index=e, edge_weight=w)
                for cid, e, w in zip(coupling_matrices_ids, self._edge_indices, self._edge_weights)
            }
            self._node_dim = self._num_qubits

        elif feature_type == WDEGREE_FEATURE:
            data_by_id = {
                int(cid): Data(x=d, edge_index=e, edge_weight=w)
                for cid, e, w, d in zip(
                    coupling_matrices_ids, self._edge_indices, self._edge_weights, self._vertices_deg
                )
            }
            self._node_dim = 1
        else:
            raise ValueError(f'unknown graph node feature type {feature_type}')

        self._data_by_id = data_by_id
        self._data_list = _ByIdSequence(self._sample_ids, self._data_by_id)

    def _build_graphs(self, coupling_matrices):
        graphs_edge_indices = [
            [
                [(i, j), (j, i)] for i, j in it.combinations(range(self._num_qubits), 2) if mat[i, j] != 0.0
            ] for mat in coupling_matrices
        ]

        graphs_edge_weights = [
            torch.from_numpy(np.array([mat[i, j] for i, j in edges], dtype=np.float32).flatten())
            for edges, mat in zip(graphs_edge_indices, coupling_matrices)
        ]

        graphs_edge_indices = [
            torch.from_numpy(np.array([x for x in it.chain(*edge_indices)]).T)
            for edge_indices in graphs_edge_indices
        ]

        graphs_vertices_degrees = [
            torch.FloatTensor([
                torch.sum((edge_indices[0] == q) * edge_weights) for q in range(self._num_qubits)
            ]).reshape(-1, 1)
            for edge_indices, edge_weights in zip(graphs_edge_indices, graphs_edge_weights)
        ]

        return graphs_edge_indices, graphs_edge_weights, graphs_vertices_degrees

    def __getitem__(self, item) -> (Data, int, torch.Tensor):
        cid = int(self._sample_ids[item])
        return self._data_by_id[cid], cid, self._coupling_matrices_tensor_by_id[cid]

    def __len__(self):
        return len(self._sample_ids)

    @property
    def coupling_matrices_ids(self):
        return self._coupling_matrices_ids

    @property
    def coupling_matrices_dict(self):
        return self._coupling_matrices_dict

    @property
    def coupling_matrices_list(self):
        return self._coupling_matrices_tensor

    @property
    def data_list(self):
        return self._data_list

    @property
    def node_dim(self):
        return self._node_dim
