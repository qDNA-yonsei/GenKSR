from types import SimpleNamespace

import numpy as np
import pytest

from genksr.constants import measurement_outcomes_for_task


def test_measurement_alphabet_is_task_specific():
    assert measurement_outcomes_for_task('kqd') == 6
    assert measurement_outcomes_for_task('sqkd') == 2
    with pytest.raises(ValueError):
        measurement_outcomes_for_task('unknown')


def test_sampling_masks_special_tokens_and_returns_exact_batch_size():
    torch = pytest.importorskip('torch')
    nn = torch.nn

    from genksr.nn.gc_model import GCModel

    class Encoder(nn.Module):
        def forward(self, graph):
            return torch.zeros((1, 4, 4), device=graph.x.device)

    class Decoder(nn.Module):
        needs_mask = False

        def forward(self, embeddings, mask):
            return embeddings

    class InvalidTokenBiasedGenerator(nn.Module):
        def forward(self, hidden):
            logits = torch.full((hidden.shape[0], 3), -1000.0, device=hidden.device)
            logits[:, 0] = 1000.0  # start token: must never be sampled
            logits[:, 2] = 0.0     # physical outcome 1 after token_shift=1
            return logits

    model = GCModel(
        encoder=Encoder(),
        decoder=Decoder(),
        tgt_embed=nn.Embedding(3, 4),
        generator=InvalidTokenBiasedGenerator(),
        n_outcomes=2,
        d_model=4,
        pad_token=-1,
        start_token=0,
        end_token=-1,
        token_shift=1,
    )
    graph = SimpleNamespace(x=torch.zeros((4, 1)))

    samples = model.sample_batch(
        batch_size=32, coupling_graph=graph, noise_factor=0.2, qubits=4)

    assert samples.shape == (32, 4)
    assert np.all(samples == 1)
