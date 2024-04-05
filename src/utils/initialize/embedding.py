R"""
"""
#
import torch
import numpy as onp
from typing import cast


def glorot_embedding(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Embedding initialization.
    """
    #
    module = cast(torch.nn.Embedding, module)
    weight = module.weight
    (_, fan_out) = weight.shape
    a = onp.sqrt(6 / fan_out)

    #
    weight.data.uniform_(-a, a, generator=rng)
    return weight.numel()