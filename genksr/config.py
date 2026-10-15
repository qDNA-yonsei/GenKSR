"""Configuration system using dataclasses with YAML loading and CLI overrides."""

from dataclasses import dataclass, field, fields, asdict
from typing import Literal, Optional

import yaml


@dataclass
class EncoderConfig:
    arch: str = "gcn_proj_3_128"
    gcn_layers: int = 3
    gcn_dim: int = 128
    features: str = "one_hot"  # 'one_hot' or 'weighted_degree'


@dataclass
class DecoderConfig:
    type: str = "transformer"  # 'transformer' or 'mamba'
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    # mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    final_lr: float = 1e-7
    weight_decay: float = 0.0
    batch_size: int = 256
    iterations: int = 200
    lr_scheduler: str = "warmup_cosine"
    warmup_frac: float = 0.05
    eval_every: int = 1
    checkpoint_every: int = 1
    epoch_mode: bool = True


@dataclass
class DataConfig:
    protocol_version: int = 3
    n_outcomes: Optional[int] = None  # derived from task: KQD=6, SQKD=2
    task: str = "kqd"       # 'kqd' or 'sqkd'
    lattice: str = "1d"     # '1d' or '2d'
    n_qubit: int = 16
    krylov_dim: int = 5
    train_samples: int = 10000
    data_root: str = "./data/heisenberg/"
    sample_structure: int = 2
    num_trotter_steps: int = 6
    hamiltonians: Optional[int] = None
    hamiltonian_subset_mode: str = "random"
    hamiltonian_subset_seed: Optional[int] = None
    measurement_subset_seed: Optional[int] = None
    eval_test: bool = False
    eval_validation: bool = False
    hamiltonian_family: str = "canonical"
    coupling_normalization: str = "per_instance_minmax"
    delta: float = 1.0       # XXZ anisotropy (1.0 = isotropic Heisenberg)
    h_z: float = 0.0         # external magnetic field along Z
    sqkd_dt: Optional[float] = None  # SQKD physical time per Krylov step override


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "./results"
    seed: int = 0
    debug: bool = False
    verbose: bool = True


def _update_dataclass(dc, updates: dict):
    """Recursively update a dataclass from a dict."""
    for key, value in updates.items():
        if hasattr(dc, key):
            current = getattr(dc, key)
            if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
                _update_dataclass(current, value)
            else:
                setattr(dc, key, value)


def load_config(yaml_path: str = None, cli_overrides: dict = None, base_yaml: str = None) -> ExperimentConfig:
    """Load config by layering: dataclass defaults -> base_yaml -> yaml_path -> CLI overrides.

    Args:
        yaml_path: Path to the experiment YAML config file.
        cli_overrides: Dict of dotted-key overrides, e.g. {'training.lr': 1e-4}.
        base_yaml: Optional shared-defaults YAML (e.g. configs/defaults.yaml) applied
            BEFORE the experiment YAML, so experiment configs only need to state what differs.

    Returns:
        ExperimentConfig instance. The fully-resolved values are what train.py dumps to
        results/<run>/config.json.
    """
    config = ExperimentConfig()

    for path in (base_yaml, yaml_path):
        if path is not None:
            with open(path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
            _update_dataclass(config, yaml_data)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return config


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert config to a plain dict for JSON serialization."""
    return asdict(config)
