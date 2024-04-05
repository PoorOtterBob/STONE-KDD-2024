R"""
"""
#
import torch
import torch_geometric as thgeo
import numpy as onp
from typing import cast


def glorot_gcn(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    GCN initialization.
    """
    #
    weight: torch.nn.parameter.Parameter
    bias: torch.nn.parameter.Parameter

    #
    module = cast(thgeo.nn.GCNConv, module)
    weight = getattr(getattr(module, "lin"), "weight")
    bias = getattr(module, "bias")
    (fan_out, fan_in) = weight.shape
    a = onp.sqrt(6 / (fan_in + fan_out))
    weight.data.uniform_(-a, a, generator=rng)
    bias.data.uniform_(-a, a, generator=rng)
    return weight.numel() + bias.numel()