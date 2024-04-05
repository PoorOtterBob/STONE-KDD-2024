R"""
"""
#
import torch
import numpy as onp
from typing import cast


def glorot_linear(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Linear initialization.
    """
    #
    module = cast(torch.nn.Linear, module)
    weight = module.weight
    bias = module.bias
    (fan_out, fan_in) = weight.shape
    a = onp.sqrt(6 / (fan_in + fan_out))

    #
    weight.data.uniform_(-a, a, generator=rng)
    resetted = weight.numel()

    # Pytorch always annotate bias as an existing object while it may be None.
    # Use attribute-checking to pass static typing.
    if hasattr(bias, "numel"):
        # May not have bias.
        bias.data.uniform_(-a, a, generator=rng)
        resetted = resetted + bias.numel()
    else:
        #
        resetted = resetted + 0
    return resetted