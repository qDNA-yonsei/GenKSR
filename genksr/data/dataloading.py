import itertools as it
from operator import itemgetter
from typing import List

import numpy as np
from torch_geometric.data import Batch as GraphBatch

from genksr.nn.common import make_std_mask


class TgtBatch:
    """Object for holding a batch of data with mask."""
    def __init__(self, tgt, pad):
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = make_std_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).data.sum()


def batch_generator_epoch_mixed(ds_list, batch_size, pad_token, rng, shuffle=False, device=None):
    dataset_sizes = [len(ds) for ds in ds_list]

    dataset_indices = [
        np.array_split(
            rng.choice(np.arange(sz), sz, replace=False) if shuffle else np.arange(sz),
            indices_or_sections=sz // batch_size
        ) for sz in dataset_sizes
    ]

    dataset_indices = [[(b, i) for b in idx_list] for i, idx_list in enumerate(dataset_indices)]
    dataset_indices = list(it.chain(*dataset_indices))

    if shuffle:
        rng.shuffle(dataset_indices)

    for indices, ds_id in dataset_indices:
        cgs = ds_list[ds_id].coupling_graphs_data
        mts = ds_list[ds_id].measurements
        hids = ds_list[ds_id].hamiltonian_ids
        nf = ds_list[ds_id].noise_factors

        graphs_batch = GraphBatch.from_data_list(data_list=list(itemgetter(*indices)(cgs)))
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]
        noise_factors_batch = nf[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch, noise_factors_batch


def batch_generator_epoch(ds, batch_size, pad_token, rng, device, shuffle=False):
    coupling_graphs = ds.coupling_graphs_data
    measurements_tensor = ds.measurements
    hamiltonian_ids = ds.hamiltonian_ids
    noise_factors = ds.noise_factors

    num_samples = len(ds)
    batch_indices = np.array_split(
        rng.permutation(num_samples) if shuffle else np.arange(num_samples),
        indices_or_sections=num_samples // batch_size
    )

    for indices in batch_indices:
        graphs_batch_list = itemgetter(*indices)(coupling_graphs)
        if not isinstance(graphs_batch_list, tuple):
            graphs_batch_list = (graphs_batch_list,)

        graphs_batch = GraphBatch.from_data_list(data_list=list(graphs_batch_list)).to(device)
        tgt_batch = TgtBatch(measurements_tensor[indices].to(device), pad=pad_token)
        hamiltonian_ids_batch = hamiltonian_ids[indices]
        noise_factors_batch = noise_factors[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch, noise_factors_batch


def batch_generator_iterations_mixed(ds_list, batch_size, pad_token, rng, iterations):
    for step in range(1, iterations + 1):
        ds = rng.choice(ds_list, 1)[0]

        cgs = ds.coupling_graphs_data
        mts = ds.measurements
        hids = ds.hamiltonian_ids
        nf = ds.noise_factors

        indices = ds.rng.choice(len(ds), batch_size, replace=True)
        graphs_batch = GraphBatch.from_data_list(data_list=[cgs[i] for i in indices])
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]
        noise_factors_batch = nf[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch, noise_factors_batch


def batch_generator_iterations(ds, batch_size, pad_token, rng, iterations):
    cgs = ds.coupling_graphs_data
    mts = ds.measurements
    hids = ds.hamiltonian_ids
    nf = ds.noise_factors

    for step in range(1, iterations + 1):
        indices = rng.choice(len(ds), batch_size, replace=True)
        graphs_batch = GraphBatch.from_data_list(data_list=[cgs[i] for i in indices])
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]
        noise_factors_batch = nf[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch, noise_factors_batch
