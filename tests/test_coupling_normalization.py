import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from genksr.data.dataset import (  # noqa: E402
    DatasetGCModel,
    normalize_coupling_matrices,
)


def _chain(n_qubit, weight):
    matrix = np.zeros((n_qubit, n_qubit), dtype=float)
    for site in range(n_qubit - 1):
        matrix[site, site + 1] = matrix[site + 1, site] = weight
    return matrix


def test_no_normalization_preserves_absolute_jxy_scale():
    couplings = np.stack([_chain(4, 0.3), _chain(4, 1.0)])
    raw = normalize_coupling_matrices(couplings, mode="none")
    np.testing.assert_allclose(raw, couplings)
    legacy = normalize_coupling_matrices(
        couplings, mode="per_instance_minmax")
    np.testing.assert_allclose(legacy[:, 0, 1], [1.0, 1.0])


def test_dataset_graph_edges_keep_distinct_raw_jxy_values():
    couplings = np.stack([_chain(4, 0.3), _chain(4, 1.0)])
    dataset = DatasetGCModel(
        measurements=np.asarray([[1, 0, 1, 0], [1, 0, 1, 0]]),
        coupling_matrices=couplings,
        hamiltonian_ids=np.asarray([0, 1]),
        noise_factors=np.asarray([0.0, 0.0]),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        nodes_feature_type="one_hot",
        start_token=0,
        end_token=-1,
        token_shift=1,
        n_outcomes=2,
        coupling_normalization="none",
    )
    graph0 = dataset.coupling_graphs_data[0]
    graph1 = dataset.coupling_graphs_data[1]
    np.testing.assert_allclose(graph0.edge_weight.numpy(), 0.3)
    np.testing.assert_allclose(graph1.edge_weight.numpy(), 1.0)


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="Unknown coupling normalization"):
        normalize_coupling_matrices(_chain(4, 0.3), mode="global_magic")
