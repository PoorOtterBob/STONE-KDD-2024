R"""
"""
#
import torch
import torch_geometric as thgeo
from typing import Dict, Callable, cast
from .embedding import glorot_embedding
from .linear import glorot_linear
from .gcn import glorot_gcn
from .gat import glorot_gat
from .cheb import glorot_cheb
from .gin import glorot_gine
from .gru import glorot_gru, glorot_gru_cell
from .lstm import glorot_lstm, glorot_lstm_cell
from .mha import glorot_mha
from .identity import glorot_identity
from ..snn import Linear, Static, MultiheadAttention


#
GLOROTS: Dict[type, Callable[[torch.nn.Module, torch.Generator], int]]


#
GLOROTS = {
    torch.nn.Embedding: glorot_embedding,
    torch.nn.Linear: glorot_linear,
    thgeo.nn.dense.linear.Linear: glorot_linear,
    Linear: (
        lambda module, rng: (
            glorot_linear(cast(torch.nn.Linear, module.lin), rng)
        )
    ),
    thgeo.nn.GCNConv: glorot_gcn,
    thgeo.nn.GATConv: glorot_gat,
    thgeo.nn.ChebConv: glorot_cheb,
    thgeo.nn.GINEConv: glorot_gine,
    torch.nn.GRU: glorot_gru,
    torch.nn.GRUCell: glorot_gru_cell,
    torch.nn.LSTM: glorot_lstm,
    torch.nn.LSTMCell: glorot_lstm_cell,
    MultiheadAttention: glorot_mha,
    torch.nn.Identity: glorot_identity,
    Static: glorot_identity,
}


def glorot(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Module initialization.
    """
    #
    return GLOROTS[type(module)](module, rng)