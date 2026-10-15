import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from genksr.nn.common import clones, SublayerConnection, LayerNorm


class MambaDecoderLayer(nn.Module):
    """Mamba-based decoder layer - handles causality inherently."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super(MambaDecoderLayer, self).__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for MambaDecoder. Install with: pip install mamba-ssm")
        self.d_model = d_model

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, tgt_mask=None):
        return self.sublayer(x, self.mamba)


class MambaDecoder(nn.Module):
    """Stack of Mamba decoder layers."""
    needs_mask = False

    def __init__(self, layer: MambaDecoderLayer, n_layers: int):
        super(MambaDecoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask=mask)
        return self.norm(x)
