from .gcn_proj import GCNProjEncoder
from .gcn_encoder import GCNEncoder


def get_graph_encoder(arch: str, in_node_dim, gcn_dim, gcn_layers, d_model, qubits=None):
    if arch.startswith('gcn_proj'):
        return GCNProjEncoder(
            qubits, in_node_dim=in_node_dim, out_node_dim=gcn_dim, n_layers=gcn_layers, transformer_dim=d_model
        )

    if arch.startswith('gcn_encoder'):
        assert gcn_dim == d_model
        return GCNEncoder(
            in_node_dim=in_node_dim, out_node_dim=gcn_dim, n_layers=gcn_layers
        )

    raise ValueError(f'unknown graph encoder {arch}')
