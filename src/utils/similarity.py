import torch
import numpy as np
from scipy.stats import wasserstein_distance as w_d

# X(b, t, n, f)
# n is sample node for training/val/test set

def similarity_hard_adj(S, e=0.5, order=0):
    S_adj = np.where(S > e, 1, 0)
    return S_adj

def similarity_matrix(X, order, p=2):
    b, t, n, f = X.shape
    X = X[:, :, :, 0]
    X = X.permute(0, 2, 1) # X: (b, n, t)
    S = torch.zeros((b, n, n, order+1), device=X.device)  # 在相同的设备上创建 S 张量
    for o in range(order+1):
        a = torch.diff(X, n=o, dim=-1)
        # S[:, :, :, o] = torch.cdist(a, a, p) # (b, n, n)
        S[:, :, :, o] = 1. / (1 + torch.cdist(a, a, p=p)) # (b, n, n)
    return S


# p为l_p距离的参数
def similarity_pred_matrix(Z, p=2):
    b, n, z, o = Z.shape
    # Z_T = Z.permute(0, 2, 1, 3) # (b, z, n, o)
    S_pred = torch.zeros((b, n, n, o), device=Z.device)
    for oo in range(o):
        # S_pred[:, :, :, oo] = torch.softmax(torch.matmul(Z[:, :, :, oo], Z.permute(0, 2, 1, 3)[:, :, :, oo]), dim=-1)
        S_pred[:, :, :, oo] = torch.sigmoid(torch.matmul(Z[:, :, :, oo], Z.permute(0, 2, 1, 3)[:, :, :, oo]))
        # S_pred[:, :, :, oo] = 1. / (1 + torch.matmul(Z[:, :, :, oo], Z_T[:, :, :, oo])) # (b, n, n, o)
        # a = Z[:, :, :, oo]
        # S_pred[:, :, :, oo] = 1. / (1 + torch.cdist(a, a, p=p))
    return S_pred


'''
# order 考虑了几阶相似度类似位移、速度、加速度
def similarity_matrix(X, order=0):
    b, t, n, f = X.shape
    X_cpu = X.cpu().detach().numpy()  # 将 X 移回 CPU 并转换为 NumPy 数组
    S = torch.zeros((b, n, n, order+1), device=X.device)  # 在相同的设备上创建 S 张量
    for i in range(b):
        for j in range(n):
            for k in range(n - j):
                if order == 0:
                    a = order_diff(X_cpu[i, :, j, 0], order)
                    b = order_diff(X_cpu[i, :, k, 0], order)
                    S[i, k, j, order] = w_d(a, b)
                    S[i, j, k, order] = S[i, k, j, order]
                else:
                    for o in range(order):
                        a = order_diff(X_cpu[i, :, j, 0], o)
                        b = order_diff(X_cpu[i, :, k, 0], o)
                        S[i, k, j, o] = w_d(a, b)
                        S[i, j, k, o] = S[i, k, j, o]
    return S


def order_diff(a, order):
    aa = a
    for o in range(order):
        aa = np.diff(aa)
    return aa
'''

'''
def similarity_matrix(X, order=0):
    b, t, n, f = X.shape
    S = torch.zeros((b, n, n, order+1), device=X.device)  # 在相同的设备上创建 S 张量
    for i in range(b):
        for j in range(n):
            for k in range(j+1):
                for o in range(order+1):
                    a = torch.diff(X[i, :, j, 0], n=o)
                    b = torch.diff(X[i, :, k, 0], n=o)
                    S[i, k, j, o] = torch.cdist(a, b, p=2)
                    S[i, j, k, o] = S[i, k, j, o]
    return S
'''