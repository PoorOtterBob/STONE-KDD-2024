import torch
import numpy as np
from scipy.stats import wasserstein_distance as w_d

def similarity_hard_adj(S, e=0.5, order=0):
    S_adj = np.where(S > e, 1, 0)
    return S_adj

def similarity_matrix(X, order, p=2):
    b, t, n, f = X.shape
    X = X[:, :, :, 0]
    X = X.permute(0, 2, 1)
    S = torch.zeros((b, n, n, order+1), device=X.device) 
    for o in range(order+1):
        a = torch.diff(X, n=o, dim=-1)
        S[:, :, :, o] = 1. / (1 + torch.cdist(a, a, p=p))
    return S

def similarity_pred_matrix(Z, p=2):
    b, n, z, o = Z.shape
    S_pred = torch.zeros((b, n, n, o), device=Z.device)
    for oo in range(o):
        S_pred[:, :, :, oo] = torch.sigmoid(torch.matmul(Z[:, :, :, oo], Z.permute(0, 2, 1, 3)[:, :, :, oo]))
    return S_pred
