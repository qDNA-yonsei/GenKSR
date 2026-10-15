from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from genksr.nn.common import (
    MultiHeadAttention, PositionwiseFeedForward,
    PositionalEncoding, Embeddings, Generator,
    make_std_mask,
)
from genksr.nn.decoder.transformer import DecoderLayer, TransformerDecoder

class GCModel(nn.Module):
    """Unified Graph-Conditional generative model.

    Supports both Transformer and Mamba decoder backends.
    The decoder's `needs_mask` property determines whether causal masks are used.
    """

    def __init__(
            self, encoder, decoder, tgt_embed, generator, n_outcomes, d_model,
            pad_token, start_token, end_token, token_shift
    ):
        super(GCModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.n_outcomes = n_outcomes
        self.d_model = d_model

        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.token_shift = token_shift
        self.noise_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )

    def forward(self, tgt, tgt_mask, coupling_graph, noise_factor):
        """Process masked target sequences, conditioned on coupling graph and noise factor."""
        tgt_embed = self.tgt_embed(tgt)

        graph_embed = self.encoder(coupling_graph)
        noise_embedding = self.noise_embed(noise_factor.unsqueeze(-1))

        combined_context = graph_embed + noise_embedding.unsqueeze(1)
        sliced_context = combined_context[:, :tgt_embed.shape[1], :]
        tgt_embed = tgt_embed + sliced_context

        mask = tgt_mask if self.decoder.needs_mask else None
        return graph_embed, self.decoder(tgt_embed, mask)

    def sample_batch(self, batch_size, coupling_graph, noise_factor, qubits):
        noise_factor_tensor = torch.full(
            (batch_size,),
            fill_value=noise_factor,
            device=coupling_graph.x.device,
            dtype=torch.float32
        )
        tgt = torch.full(
            (batch_size, 1),
            fill_value=self.start_token,
            device=coupling_graph.x.device,
            dtype=torch.long,
        )

        valid_start = self.token_shift
        valid_stop = self.token_shift + self.n_outcomes
        for _ in range(qubits):
            if self.decoder.needs_mask:
                mask = make_std_mask(tgt, pad=self.pad_token)
            else:
                mask = None

            _, out = self.forward(tgt, mask, coupling_graph, noise_factor_tensor)
            logits = self.generator(out[:, -1])[:, valid_start:valid_stop]
            probs = torch.softmax(logits, dim=-1).detach()
            next_outcomes = torch.multinomial(probs, 1, replacement=True) + valid_start
            tgt = torch.cat([tgt, next_outcomes], dim=1)

        return tgt[:, 1:].cpu().numpy() - self.token_shift

    def sample(self, samples, coupling_graph, noise_factor, qubits, batch_size=1000,
               print_progress=True) -> np.ndarray:
        batch_sizes = [batch_size] * int(samples // batch_size)
        batch_sizes = batch_sizes + (
            [] if samples % batch_size == 0 else [samples % batch_size])

        if print_progress:
            pbar = tqdm(batch_sizes,
                        postfix=f'generating {samples} samples from model...')
        else:
            pbar = batch_sizes

        batch_samples = [self.sample_batch(bs, coupling_graph, noise_factor, qubits) for bs in pbar]

        samples = np.concatenate(batch_samples, axis=0)
        return samples


def build_gc_model(
        decoder_type: str,
        n_outcomes: int,
        encoder,
        n_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        pad_token: int,
        start_token: int,
        end_token: int,
        token_shift: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
) -> GCModel:
    """Build a GCModel with the specified decoder backend.

    Args:
        decoder_type: 'transformer' or 'mamba'
        n_outcomes: number of measurement outcomes
        encoder: graph encoder module
        n_layers: number of decoder layers
        d_model: model dimension
        d_ff: feedforward dimension (transformer only)
        n_heads: number of attention heads (transformer only)
        dropout: dropout rate
        pad_token, start_token, end_token, token_shift: token configuration
        d_state, d_conv, expand: Mamba-specific parameters
    """
    if n_outcomes <= 0:
        raise ValueError(f'n_outcomes must be positive, got {n_outcomes}')
    if token_shift < 0:
        raise ValueError(f'token_shift must be non-negative, got {token_shift}')

    position = PositionalEncoding(d_model, dropout)
    vocab_size = n_outcomes + token_shift

    if decoder_type == 'transformer':
        attn = MultiHeadAttention(n_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        decoder = TransformerDecoder(
            DecoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout),
            n_layers=n_layers
        )
    elif decoder_type == 'mamba':
        from genksr.nn.decoder.mamba import MambaDecoderLayer, MambaDecoder
        mamba_layer = MambaDecoderLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        decoder = MambaDecoder(mamba_layer, n_layers=n_layers)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Use 'transformer' or 'mamba'.")

    model = GCModel(
        encoder=encoder,
        decoder=decoder,
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(position)),
        generator=Generator(d_model, vocab_size),
        n_outcomes=n_outcomes,
        d_model=d_model,
        pad_token=pad_token,
        start_token=start_token,
        end_token=end_token,
        token_shift=token_shift
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
