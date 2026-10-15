import torch.nn as nn

from genksr.nn.common import (
    clones, SublayerConnection, MultiHeadAttention,
    PositionwiseFeedForward, LayerNorm,
)


class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(
            query=x, key=x, value=x, mask=tgt_mask
        ))
        x = self.sublayer[1](x, lambda x: self.self_attn(
            query=x, key=x, value=x, mask=tgt_mask
        ))
        return self.sublayer[2](x, self.feed_forward)


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    needs_mask = True

    def __init__(self, layer: DecoderLayer, n_layers: int):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask=mask)
        return self.norm(x)
