import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

def normalize_adj_mx(adj_mx, adj_type, lambda_max=None, return_type='dense'):
    if adj_type == 'normlap':
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx, lambda_max)]
    elif adj_type == 'symadj':
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'symadjmax':
        adj = [calculate_sym_adj_max(adj_mx)]
    elif adj_type == 'symadjmin':
        adj = [calculate_sym_adj_min(adj_mx)]
    elif adj_type == 'symadjaverage':
        adj = [calculate_sym_adj_average(adj_mx)]
    elif adj_type == 'origin':
        adj = [origin(adj_mx)]
    else:
        return []
    
    if return_type == 'dense':
        adj = [a.astype(np.float32).todense() for a in adj]
    elif return_type == 'coo':
        adj = [a.tocoo() for a in adj]
    return adj


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if len(adj_mx.shape)==2:
        if undirected:
            adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    elif len(adj_mx.shape)==4:
        if undirected:
            adj_mx = np.maximum.reduce([adj_mx, adj_mx.transpose(0,1,3,2)])
    else:
        if undirected:
            adj_mx = np.maximum.reduce([adj_mx, adj_mx.transpose(0,2,1)])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return res



def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adj_mx)
    return res



def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)



def calculate_sym_adj_max(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    adj_mx = sp.coo_matrix.maximum(adj_mx, adj_mx.transpose())
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(adj_mx.T).dot(d_mat_inv_sqrt)
    return res



def calculate_sym_adj_min(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    adj_mx = sp.coo_matrix.min(adj_mx, adj_mx.transpose())
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(adj_mx.T).dot(d_mat_inv_sqrt)
    return res



def calculate_sym_adj_average(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    adj_mx = (adj_mx + adj_mx.transpose()) / 2.
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(adj_mx.T).dot(d_mat_inv_sqrt)
    return res

def origin(adj_mx):
    res = sp.coo_matrix(adj_mx)
    return res