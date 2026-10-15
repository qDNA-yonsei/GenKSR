"""Data package with lazy exports so lightweight integrity tools stay dependency-light."""

__all__ = [
    'DatasetGCModel',
    'normalize_coupling_matrices',
    'CouplingGraphsDataset',
    'SampleStructure',
    'get_sample_structure',
]


def __getattr__(name):
    if name in {'DatasetGCModel', 'normalize_coupling_matrices'}:
        from .dataset import DatasetGCModel, normalize_coupling_matrices
        return {
            'DatasetGCModel': DatasetGCModel,
            'normalize_coupling_matrices': normalize_coupling_matrices,
        }[name]
    if name == 'CouplingGraphsDataset':
        from .coupling_graphs import CouplingGraphsDataset
        return CouplingGraphsDataset
    if name in {'SampleStructure', 'get_sample_structure'}:
        from .sample_structure import SampleStructure, get_sample_structure
        return {
            'SampleStructure': SampleStructure,
            'get_sample_structure': get_sample_structure,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
