import numpy as np
from typing import List

from torch_geometric.data import Data as GraphData
import torch
from torch.utils.data import Dataset

from genksr.data.coupling_graphs import CouplingGraphsDataset

__all__ = [
    'normalize_coupling_matrices',
    'DatasetGCModel',
    'PER_INSTANCE_MINMAX',
    'NO_COUPLING_NORMALIZATION',
    'COUPLING_NORMALIZATION_MODES',
]

PER_INSTANCE_MINMAX = "per_instance_minmax"
NO_COUPLING_NORMALIZATION = "none"
COUPLING_NORMALIZATION_MODES = (
    PER_INSTANCE_MINMAX,
    NO_COUPLING_NORMALIZATION,
)


def map_to_sentence(data, start_token, end_token, token_shift):
    num_data = data.shape[0]
    data = data + token_shift
    data = np.concatenate([start_token * np.ones(shape=(num_data, 1)), data], axis=1)

    if end_token > 0:
        data = np.concatenate([data, end_token * np.ones(shape=(num_data, 1))], axis=1)

    data = data.astype(int)
    return data


def normalize_coupling_matrices(
        coupling_matrices: np.ndarray,
        mode: str = PER_INSTANCE_MINMAX) -> np.ndarray:
    """Apply the configured coupling-matrix normalization.

    Args:
        coupling_matrices: shape (nq, nq) or (nb, nq, nq)
        mode: ``per_instance_minmax`` (legacy default) or ``none``.

    Returns:
        Normalized coupling matrices.
    """
    coupling_matrices = np.asarray(coupling_matrices)
    if coupling_matrices.ndim == 2:
        coupling_matrices = np.expand_dims(coupling_matrices, axis=0)
    if coupling_matrices.ndim != 3:
        raise ValueError(
            "coupling_matrices must have shape (n,n) or (batch,n,n)")
    if mode not in COUPLING_NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown coupling normalization {mode!r}; expected one of "
            f"{COUPLING_NORMALIZATION_MODES}")
    if mode == NO_COUPLING_NORMALIZATION:
        return coupling_matrices.copy()

    cmax = np.max(coupling_matrices, axis=(1, 2)).reshape((-1, 1, 1))
    cmin = np.min(coupling_matrices, axis=(1, 2)).reshape((-1, 1, 1))
    scale = cmax - cmin
    if np.any(scale == 0):
        raise ValueError("Cannot min-max normalize a constant coupling matrix")
    coupling_matrices = (coupling_matrices - cmin) / scale

    return coupling_matrices


class DatasetGCModel(Dataset):
    """Task-specific measurement sequences prepared for the GCModel."""

    def __init__(
            self, measurements, coupling_matrices, hamiltonian_ids,
            noise_factors, rng: np.random.Generator, device,
            nodes_feature_type, start_token, end_token, token_shift,
            n_outcomes, coupling_normalization=PER_INSTANCE_MINMAX):
        self._coupling_matrices = coupling_matrices
        self._hamiltonian_ids = hamiltonian_ids.astype(np.int64)
        self._device = device
        self.noise_factors = torch.tensor(noise_factors, dtype=torch.float32)

        assert measurements.shape[0] == self._hamiltonian_ids.shape[0]
        if n_outcomes <= 0:
            raise ValueError(f'n_outcomes must be positive, got {n_outcomes}')
        if measurements.size and (
                np.min(measurements) < 0 or np.max(measurements) >= n_outcomes):
            raise ValueError(
                f'Measurements must be in [0, {n_outcomes - 1}], got range '
                f'[{np.min(measurements)}, {np.max(measurements)}]')

        self._num_data, self._num_qubits = measurements.shape

        self.rng = rng

        measurements = map_to_sentence(
            measurements, start_token=start_token, end_token=end_token, token_shift=token_shift
        )

        self._measurements_numpy = measurements
        self._measurements = torch.from_numpy(measurements)

        coupling_matrices = normalize_coupling_matrices(
            coupling_matrices, mode=coupling_normalization)
        if coupling_matrices.shape[0] == self._num_data:
            coupling_matrix_ids = self._hamiltonian_ids
            sample_ids = None
        else:
            coupling_matrix_ids = np.unique(self._hamiltonian_ids)
            if coupling_matrices.shape[0] != coupling_matrix_ids.shape[0]:
                raise ValueError(
                    "Compact coupling_matrices must contain one row per unique Hamiltonian id"
                )
            sample_ids = self._hamiltonian_ids

        self._coupling_graphs_dataset = CouplingGraphsDataset(
            coupling_matrices=coupling_matrices, coupling_matrices_ids=coupling_matrix_ids, rng=rng,
            device=device, feature_type=nodes_feature_type, sample_ids=sample_ids
        )

    def __getitem__(self, item):
        mmt_seq = self._measurements[item]
        coupling_graph = self._coupling_graphs_dataset[item]
        hid = self._hamiltonian_ids[item]
        noise_factor = self.noise_factors[item]
        return mmt_seq, coupling_graph, hid, noise_factor

    def __len__(self):
        return self._measurements_numpy.shape[0]

    @property
    def measurements(self) -> torch.Tensor:
        return self._measurements

    @property
    def measurements_numpy(self) -> np.ndarray:
        return self._measurements_numpy

    @property
    def hamiltonian_ids(self) -> np.ndarray:
        return self._hamiltonian_ids

    @property
    def coupling_matrices(self) -> np.ndarray:
        return self._coupling_matrices

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def coupling_graphs_data(self) -> List[GraphData]:
        return self._coupling_graphs_dataset.data_list

    @property
    def coupling_graphs_dataset(self) -> CouplingGraphsDataset:
        return self._coupling_graphs_dataset
