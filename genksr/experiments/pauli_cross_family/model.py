"""Permutation-invariant Pauli-set conditioned autoregressive model."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genksr.nn.common import (
    Embeddings,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    subsequent_mask,
)
from genksr.nn.decoder.transformer import DecoderLayer, TransformerDecoder


class PauliSetEncoder(nn.Module):
    """Encode an unordered, padded set of weighted Pauli strings.

    There is deliberately no positional encoding along the term-set axis.  The
    Transformer encoder is permutation equivariant and masked mean pooling makes
    the Hamiltonian representation permutation invariant.
    """

    def __init__(
        self,
        *,
        n_qubit: int,
        d_model: int,
        symbol_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        if n_qubit <= 0:
            raise ValueError("n_qubit must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by encoder n_heads")
        self.n_qubit = int(n_qubit)
        self.d_model = int(d_model)
        self.symbol_embedding = nn.Embedding(4, int(symbol_dim))
        self.term_projection = nn.Sequential(
            nn.Linear(self.n_qubit * int(symbol_dim) + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pauli_codes.ndim != 3 or pauli_codes.shape[-1] != self.n_qubit:
            raise ValueError(
                f"pauli_codes must have shape (B,L,{self.n_qubit}), got "
                f"{tuple(pauli_codes.shape)}"
            )
        if coefficients.shape != pauli_codes.shape[:2]:
            raise ValueError("coefficient shape must match Pauli batch and term axes")
        if term_mask.shape != pauli_codes.shape[:2] or term_mask.dtype != torch.bool:
            raise ValueError("term_mask must be boolean with shape (B,L)")
        if torch.any(term_mask.sum(dim=1) == 0):
            raise ValueError("every Hamiltonian must contain at least one term")
        if torch.any((pauli_codes < 0) | (pauli_codes > 3)):
            raise ValueError("Pauli codes must be in {0,1,2,3}")

        embedded = self.symbol_embedding(pauli_codes)
        embedded = embedded.reshape(
            pauli_codes.shape[0], pauli_codes.shape[1], -1
        )
        term_features = torch.cat([embedded, coefficients.unsqueeze(-1)], dim=-1)
        terms = self.term_projection(term_features)
        terms = self.set_transformer(terms, src_key_padding_mask=~term_mask)
        weights = term_mask.unsqueeze(-1).to(terms.dtype)
        pooled = (terms * weights).sum(dim=1) / weights.sum(dim=1)
        return self.output(pooled)


class CoefficientSumPauliEncoder(nn.Module):
    """Encode ``H = sum_l c_l P_l`` with an explicitly compositional sum.

    The signed sum retains the Hamiltonian coefficients, while the absolute
    coefficient sum separately records operator support without allowing
    opposite-signed terms to erase it.  Both summaries are permutation
    invariant and padding terms are masked exactly.
    """

    def __init__(
        self,
        *,
        n_qubit: int,
        d_model: int,
        symbol_dim: int,
        dropout: float,
    ):
        super().__init__()
        if n_qubit <= 0:
            raise ValueError("n_qubit must be positive")
        self.n_qubit = int(n_qubit)
        self.d_model = int(d_model)
        self.symbol_embedding = nn.Embedding(4, int(symbol_dim))
        self.term_projection = nn.Sequential(
            nn.Linear(self.n_qubit * int(symbol_dim), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pauli_codes.ndim != 3 or pauli_codes.shape[-1] != self.n_qubit:
            raise ValueError(
                f"pauli_codes must have shape (B,L,{self.n_qubit}), got "
                f"{tuple(pauli_codes.shape)}"
            )
        if coefficients.shape != pauli_codes.shape[:2]:
            raise ValueError("coefficient shape must match Pauli batch and term axes")
        if term_mask.shape != pauli_codes.shape[:2] or term_mask.dtype != torch.bool:
            raise ValueError("term_mask must be boolean with shape (B,L)")
        if torch.any(term_mask.sum(dim=1) == 0):
            raise ValueError("every Hamiltonian must contain at least one term")
        if torch.any((pauli_codes < 0) | (pauli_codes > 3)):
            raise ValueError("Pauli codes must be in {0,1,2,3}")

        embedded = self.symbol_embedding(pauli_codes).reshape(
            pauli_codes.shape[0], pauli_codes.shape[1], -1
        )
        terms = self.term_projection(embedded)
        mask = term_mask.to(dtype=terms.dtype)
        signed_weights = coefficients * mask
        absolute_weights = coefficients.abs() * mask
        signed = (terms * signed_weights.unsqueeze(-1)).sum(dim=1)
        support = (terms * absolute_weights.unsqueeze(-1)).sum(dim=1)
        return self.output(torch.cat([signed, support], dim=-1))


class PauliConditionalARModel(nn.Module):
    """Generate computational-basis bitstrings conditioned on ``(H, tau)``."""

    start_token = 0
    bit_token_shift = 1
    token_vocab_size = 3
    n_outcomes = 2

    def __init__(
        self,
        *,
        n_qubit: int,
        d_model: int = 128,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        symbol_dim: int = 16,
        dropout: float = 0.1,
        encoder_mode: str = "set_transformer",
    ):
        super().__init__()
        self.n_qubit = int(n_qubit)
        self.d_model = int(d_model)
        self.model_config = {
            "n_qubit": self.n_qubit,
            "d_model": self.d_model,
            "decoder_layers": int(decoder_layers),
            "decoder_heads": int(decoder_heads),
            "encoder_layers": int(encoder_layers),
            "encoder_heads": int(encoder_heads),
            "symbol_dim": int(symbol_dim),
            "dropout": float(dropout),
            "encoder_mode": str(encoder_mode),
        }
        if encoder_mode == "set_transformer":
            self.hamiltonian_encoder = PauliSetEncoder(
                n_qubit=self.n_qubit,
                d_model=self.d_model,
                symbol_dim=symbol_dim,
                n_layers=encoder_layers,
                n_heads=encoder_heads,
                dropout=dropout,
            )
        elif encoder_mode == "coefficient_sum":
            self.hamiltonian_encoder = CoefficientSumPauliEncoder(
                n_qubit=self.n_qubit,
                d_model=self.d_model,
                symbol_dim=symbol_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                "encoder_mode must be 'set_transformer' or 'coefficient_sum'"
            )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.token_embedding = nn.Sequential(
            Embeddings(self.d_model, self.token_vocab_size),
            PositionalEncoding(self.d_model, dropout, max_len=self.n_qubit + 1),
        )
        attention = MultiHeadAttention(decoder_heads, self.d_model, dropout)
        feed_forward = PositionwiseFeedForward(
            self.d_model, 4 * self.d_model, dropout
        )
        self.decoder = TransformerDecoder(
            DecoderLayer(
                self.d_model,
                deepcopy(attention),
                deepcopy(feed_forward),
                dropout,
            ),
            n_layers=decoder_layers,
        )
        self.output_head = nn.Linear(self.d_model, self.n_outcomes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode_condition(
        self,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        hamiltonian = self.hamiltonian_encoder(
            pauli_codes, coefficients, term_mask
        )
        tau = tau.reshape(-1, 1).to(dtype=hamiltonian.dtype)
        if tau.shape[0] not in (1, hamiltonian.shape[0]):
            raise ValueError("tau batch does not match Hamiltonian batch")
        if tau.shape[0] == 1 and hamiltonian.shape[0] > 1:
            tau = tau.expand(hamiltonian.shape[0], -1)
        return hamiltonian + self.time_embedding(tau)

    @staticmethod
    def _expand_context(context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context.shape[0] == batch_size:
            return context
        if context.shape[0] == 1:
            return context.expand(batch_size, -1)
        raise ValueError("condition batch must be one or match bitstring batch")

    def forward(
        self,
        bits: torch.Tensor,
        *,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.n_qubit:
            raise ValueError(
                f"bits must have shape (B,{self.n_qubit}), got {tuple(bits.shape)}"
            )
        if torch.any((bits < 0) | (bits > 1)):
            raise ValueError("bits must be binary")
        context = self.encode_condition(
            pauli_codes, coefficients, term_mask, tau
        )
        context = self._expand_context(context, bits.shape[0])
        start = torch.full(
            (bits.shape[0], 1),
            self.start_token,
            dtype=torch.long,
            device=bits.device,
        )
        teacher_tokens = torch.cat(
            [start, bits[:, :-1] + self.bit_token_shift], dim=1
        )
        embedded = self.token_embedding(teacher_tokens)
        embedded = embedded + context.unsqueeze(1)
        causal_mask = subsequent_mask(self.n_qubit).to(bits.device)
        decoded = self.decoder(embedded, causal_mask)
        return self.output_head(decoded)

    def log_prob(
        self,
        bits: torch.Tensor,
        *,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(
            bits,
            pauli_codes=pauli_codes,
            coefficients=coefficients,
            term_mask=term_mask,
            tau=tau,
        )
        token_log_prob = F.log_softmax(logits, dim=-1).gather(
            -1, bits.unsqueeze(-1)
        )
        return token_log_prob.squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
        tau: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        device = pauli_codes.device
        context = self.encode_condition(
            pauli_codes, coefficients, term_mask, tau
        )
        context = self._expand_context(context, num_samples)
        tokens = torch.full(
            (num_samples, 1),
            self.start_token,
            dtype=torch.long,
            device=device,
        )
        generated = []
        for _ in range(self.n_qubit):
            embedded = self.token_embedding(tokens) + context.unsqueeze(1)
            mask = subsequent_mask(tokens.shape[1]).to(device)
            decoded = self.decoder(embedded, mask)
            probabilities = torch.softmax(self.output_head(decoded[:, -1]), dim=-1)
            bit = torch.multinomial(
                probabilities, 1, replacement=True, generator=generator
            )
            generated.append(bit)
            tokens = torch.cat([tokens, bit + self.bit_token_shift], dim=1)
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def exact_distribution(
        self,
        *,
        pauli_codes: torch.Tensor,
        coefficients: torch.Tensor,
        term_mask: torch.Tensor,
        tau: torch.Tensor,
        batch_size: int = 512,
    ) -> np.ndarray:
        device = pauli_codes.device
        dimension = 1 << self.n_qubit
        indices = np.arange(dimension, dtype=np.int64)
        shifts = np.arange(self.n_qubit - 1, -1, -1, dtype=np.int64)
        bits_numpy = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int64)
        log_probabilities = []
        for start in range(0, dimension, batch_size):
            bits = torch.as_tensor(
                bits_numpy[start : start + batch_size],
                dtype=torch.long,
                device=device,
            )
            log_probabilities.append(
                self.log_prob(
                    bits,
                    pauli_codes=pauli_codes,
                    coefficients=coefficients,
                    term_mask=term_mask,
                    tau=tau,
                ).cpu()
            )
        probabilities = torch.exp(torch.cat(log_probabilities)).numpy()
        normalization = float(probabilities.sum())
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError("model produced an invalid exact distribution")
        probabilities /= normalization
        return probabilities.astype(np.float64)


def build_model(model_config: Mapping, *, n_qubit: int) -> PauliConditionalARModel:
    return PauliConditionalARModel(
        n_qubit=n_qubit,
        d_model=int(model_config["d_model"]),
        decoder_layers=int(model_config["decoder_layers"]),
        decoder_heads=int(model_config["decoder_heads"]),
        encoder_layers=int(model_config["encoder_layers"]),
        encoder_heads=int(model_config["encoder_heads"]),
        symbol_dim=int(model_config["symbol_dim"]),
        dropout=float(model_config["dropout"]),
        encoder_mode=str(model_config.get("encoder_mode", "set_transformer")),
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
