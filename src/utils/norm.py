import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class LayerNorm(nn.Module):
    def __init__(self, hid_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hid_dim))
        self.bias = nn.Parameter(torch.zeros(hid_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class GroupNorm(nn.Module):
    def __init__(self, hid_dim, eps=1e-12):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hid_dim))
        self.bias = nn.Parameter(torch.zeros(hid_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias