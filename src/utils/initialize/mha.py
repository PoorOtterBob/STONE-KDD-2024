R"""
"""
#
import torch
import numpy as onp
from typing import cast
from ..snn import MultiheadAttention
from .linear import glorot_linear
from .identity import glorot_identity


def glorot_mha(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Multi-head attention initialization.
    """
    #
    module = cast(MultiheadAttention, module)
    (embed_size,) = module.mha.out_proj.bias.shape
    a = onp.sqrt(6 / (embed_size + embed_size))

    #
    resetted = 0
    module.mha.in_proj_weight.data.uniform_(-a, a, generator=rng)
    module.mha.in_proj_bias.data.uniform_(-a, a, generator=rng)
    module.mha.out_proj.weight.data.uniform_(-a, a, generator=rng)
    module.mha.out_proj.bias.data.uniform_(-a, a, generator=rng)
    resetted = resetted + module.mha.in_proj_weight.numel()
    resetted = resetted + module.mha.in_proj_bias.numel()
    resetted = resetted + module.mha.out_proj.weight.numel()
    resetted = resetted + module.mha.out_proj.bias.numel()
    if isinstance(module.transform, torch.nn.Linear):
        #
        resetted = resetted + glorot_linear(module.transform, rng)
    else:
        #
        resetted = resetted + glorot_identity(module.transform, rng)
    return resetted