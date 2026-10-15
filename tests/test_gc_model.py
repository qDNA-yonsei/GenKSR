"""Smoke tests for GCModel."""
import torch
import torch.nn as nn


def test_build_gc_model_transformer():
    """Test that a transformer-based GCModel can be built and does a forward pass."""
    from genksr.nn.encoder.gcn_encoder import GCNEncoder
    from genksr.nn.gc_model import build_gc_model

    encoder = GCNEncoder(in_node_dim=4, out_node_dim=32, n_layers=2)

    model = build_gc_model(
        decoder_type='transformer',
        n_outcomes=6,
        encoder=encoder,
        n_layers=2,
        d_model=32,
        d_ff=64,
        n_heads=4,
        dropout=0.1,
        pad_token=-1,
        start_token=0,
        end_token=-1,
        token_shift=1,
    )

    assert isinstance(model, nn.Module)
    assert model.decoder.needs_mask is True
    print("Transformer GCModel build: OK")


def test_build_gc_model_mamba():
    """Test that a mamba-based GCModel can be built."""
    try:
        from genksr.nn.encoder.gcn_encoder import GCNEncoder
        from genksr.nn.gc_model import build_gc_model

        encoder = GCNEncoder(in_node_dim=4, out_node_dim=32, n_layers=2)

        model = build_gc_model(
            decoder_type='mamba',
            n_outcomes=6,
            encoder=encoder,
            n_layers=2,
            d_model=32,
            d_ff=64,
            n_heads=4,
            dropout=0.1,
            pad_token=-1,
            start_token=0,
            end_token=-1,
            token_shift=1,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        assert isinstance(model, nn.Module)
        assert model.decoder.needs_mask is False
        print("Mamba GCModel build: OK")
    except ImportError:
        print("Mamba GCModel build: SKIPPED (mamba_ssm not installed)")


def test_config_loading():
    """Test config loading."""
    from genksr.config import load_config, ExperimentConfig

    config = ExperimentConfig()
    assert config.decoder.type == 'transformer'
    assert config.data.task == 'kqd'
    print("Config loading: OK")


if __name__ == '__main__':
    test_build_gc_model_transformer()
    test_build_gc_model_mamba()
    test_config_loading()
    print("\nAll tests passed!")
