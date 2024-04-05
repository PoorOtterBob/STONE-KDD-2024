R"""
"""
#
import torch
import torch_geometric as thgeo
import numpy as onp
from typing import cast


def glorot_cheb(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Chebyshev spectral convolution initialization.
    """
    #
    resetted = 0
    module = cast(thgeo.nn.ChebConv, module)
    (fan_out, fan_in) = getattr(getattr(module, "lins")[0], "weight").shape
    a = onp.sqrt(6 / (fan_in + fan_out))
    for k in range(len(getattr(module, "lins"))):
        #
        weight = getattr(getattr(module, "lins")[k], "weight")
        weight.data.uniform_(-a, a, generator=rng)
        resetted = resetted + weight.numel()
    bias = getattr(module, "bias")
    bias.data.uniform_(-a, a, generator=rng)
    resetted = resetted + bias.numel()
    return resetted