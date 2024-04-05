R"""
"""
#
import torch
import numpy as onp
from typing import cast


def glorot_gru(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    GRU initialization.
    """
    #
    module = cast(torch.nn.GRU, module)
    total = 0
    for l in range(module.num_layers):
        #
        weight_ih = getattr(module, "weight_ih_l{:d}".format(l))
        weight_hh = getattr(module, "weight_hh_l{:d}".format(l))
        bias_ih = getattr(module, "bias_ih_l{:d}".format(l))
        bias_hh = getattr(module, "bias_hh_l{:d}".format(l))
        (fan_out_ih, fan_in_ih) = weight_ih.shape
        (fan_out_hh, fan_in_hh) = weight_hh.shape

        # GRU will aggregate 3 groups of parameters on output dimension.
        fan_out_ih = fan_out_ih // 3
        fan_out_hh = fan_out_hh // 3

        #
        a_ih = onp.sqrt(6 / (fan_in_ih + fan_out_ih))
        a_hh = onp.sqrt(6 / (fan_in_hh + fan_out_hh))
        weight_ih.data.uniform_(-a_ih, a_ih, generator=rng)
        weight_hh.data.uniform_(-a_hh, a_hh, generator=rng)
        bias_ih.data.uniform_(-a_ih, a_ih, generator=rng)
        bias_hh.data.uniform_(-a_ih, a_ih, generator=rng)
        total = (
            total + weight_ih.numel() + weight_hh.numel() + bias_ih.numel()
            + bias_hh.numel()
        )
    return total


def glorot_gru_cell(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    GRUCell initialization.
    """
    #
    module = cast(torch.nn.GRUCell, module)
    total = 0

    #
    weight_ih = getattr(module, "weight_ih")
    weight_hh = getattr(module, "weight_hh")
    bias_ih = getattr(module, "bias_ih")
    bias_hh = getattr(module, "bias_hh")
    (fan_out_ih, fan_in_ih) = weight_ih.shape
    (fan_out_hh, fan_in_hh) = weight_hh.shape

    # GRU will aggregate 3 groups of parameters on output dimension.
    fan_out_ih = fan_out_ih // 3
    fan_out_hh = fan_out_hh // 3

    #
    a_ih = onp.sqrt(6 / (fan_in_ih + fan_out_ih))
    a_hh = onp.sqrt(6 / (fan_in_hh + fan_out_hh))
    weight_ih.data.uniform_(-a_ih, a_ih, generator=rng)
    weight_hh.data.uniform_(-a_hh, a_hh, generator=rng)
    bias_ih.data.uniform_(-a_ih, a_ih, generator=rng)
    bias_hh.data.uniform_(-a_ih, a_ih, generator=rng)
    total = (
        total + weight_ih.numel() + weight_hh.numel() + bias_ih.numel()
        + bias_hh.numel()
    )
    return total