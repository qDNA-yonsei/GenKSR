from .transformer import DecoderLayer, TransformerDecoder

try:
    from .mamba import MambaDecoderLayer, MambaDecoder
except ImportError:
    pass  # mamba_ssm not installed; MambaDecoderLayer/MambaDecoder unavailable
