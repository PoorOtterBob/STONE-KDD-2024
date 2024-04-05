R"""
"""
#
import torch
import torch_geometric as thgeo
import numpy as onp
from typing import cast


def glorot_gat(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    GAT initialization.
    """
    #
    module = cast(thgeo.nn.GATConv, module)
    lin_src = cast(thgeo.nn.dense.linear.Linear, module.lin_src)
    attn_src = cast(torch.nn.parameter.Parameter, module.att_src)
    attn_dst = cast(torch.nn.parameter.Parameter, module.att_dst)
    lin_edge = cast(thgeo.nn.dense.linear.Linear, module.lin_edge)
    attn_edge = cast(torch.nn.parameter.Parameter, module.att_edge)
    bias = cast(torch.nn.parameter.Parameter, module.bias)
    (fan_out, fan_in_node) = lin_src.weight.shape
    (_, fan_in_edge) = lin_edge.weight.shape
    a_node = onp.sqrt(6 / (fan_out + fan_in_node))
    a_edge = onp.sqrt(6 / (fan_out + fan_in_edge))

    #
    resetted = 0
    lin_src.weight.data.uniform_(-a_node, a_node, generator=rng)
    attn_src.data.uniform_(-a_node, a_node, generator=rng)
    attn_dst.data.uniform_(-a_node, a_node, generator=rng)
    lin_edge.weight.data.uniform_(-a_edge, a_edge, generator=rng)
    attn_edge.data.uniform_(-a_edge, a_edge, generator=rng)
    bias.data.uniform_(-a_node, a_node, generator=rng)
    resetted = resetted + lin_src.weight.numel()
    resetted = resetted + attn_src.numel()
    resetted = resetted + attn_dst.numel()
    resetted = resetted + lin_edge.weight.numel()
    resetted = resetted + attn_edge.numel()
    resetted = resetted + bias.numel()
    return resetted