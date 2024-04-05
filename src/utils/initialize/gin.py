R"""
"""
#
import torch
import torch_geometric as thgeo
from typing import cast
from .linear import glorot_linear


def glorot_gine(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    GINE initialization.
    """
    #
    resetted = 0
    module = cast(thgeo.nn.GINEConv, module)
    mlp = getattr(module, "nn")
    lin = getattr(module, "lin")
    for layer in mlp:
        #
        if isinstance(layer, torch.nn.Linear):
            #
            resetted = resetted + glorot_linear(layer, rng)
    if isinstance(lin, thgeo.nn.dense.linear.Linear):
        #
        resetted = resetted + glorot_linear(lin, rng)
    return resetted