import numpy as np
import torch
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm
from typing import List

from genksr.constants import measurement_outcomes_for_task
from genksr.data.coupling_graphs import CouplingGraphsDataset
from genksr.data.dataset import normalize_coupling_matrices
from genksr.nn.gc_model import GCModel


def generate_samples(
        model: GCModel,
        coupling_matrices: np.ndarray,
        coupling_matrices_ids: List[int],
        shots: int,
        device: torch.device,
        rng: np.random.Generator,
        graph_feature_type: str,
        noise_factor: float,
        task: str = "kqd",
        coupling_normalization: str = "per_instance_minmax",
):
    """Generate samples from the trained model for each coupling matrix.

    Args:
        model: Trained GCModel instance.
        coupling_matrices: Array of coupling matrices.
        coupling_matrices_ids: IDs for each coupling matrix.
        shots: Number of samples to generate per Hamiltonian.
        device: Torch device.
        rng: Random number generator.
        graph_feature_type: Node feature type for graphs.
        noise_factor: Noise/Krylov dimension factor.
        task: 'kqd' or 'sqkd' - determines qubit count in sampling.

    Returns:
        Dictionary mapping coupling matrix IDs to generated samples.
    """
    coupling_matrices = normalize_coupling_matrices(
        coupling_matrices=coupling_matrices,
        mode=coupling_normalization,
    )

    coupling_graphs_dataset = CouplingGraphsDataset(
        coupling_matrices, coupling_matrices_ids, rng, device=device,
        feature_type=graph_feature_type
    )

    generated_samples = {}
    expected_outcomes = measurement_outcomes_for_task(task)
    if model.n_outcomes != expected_outcomes:
        raise ValueError(
            f'Model has {model.n_outcomes} outcomes, but task {task!r} requires '
            f'{expected_outcomes}')

    pbar = tqdm(coupling_graphs_dataset, total=len(coupling_graphs_dataset))
    qubits = coupling_matrices.shape[-1]

    for coupling_graph, cm_id, _ in pbar:
        pbar.set_description_str(f'model sampling for hamiltonian {cm_id}')

        coupling_graph_batch = GraphBatch.from_data_list(data_list=[coupling_graph]).to(device)

        with torch.no_grad():
            if task == 'sqkd':
                samples = model.sample(
                    samples=shots,
                    coupling_graph=coupling_graph_batch,
                    noise_factor=noise_factor,
                    qubits=qubits,
                    print_progress=False
                )
            else:
                samples = model.sample(
                    samples=shots,
                    coupling_graph=coupling_graph_batch,
                    noise_factor=noise_factor,
                    qubits=qubits + 1,
                    print_progress=False
                )

        generated_samples[cm_id] = samples

    return generated_samples
