import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class Graph_Editer_Delete_Row(nn.Module):
    def __init__(self, K, node_num, num_sample, device):
        super(Graph_Editer_Delete_Row, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(K, node_num))
        
        self.B = nn.Parameter(torch.FloatTensor(node_num).unsqueeze(0).expand(K, -1))
        self.M = torch.ones(K, node_num, dtype=torch.float, requires_grad=False).to(device)
        self.node_num = node_num
        self.num_sample = num_sample

    def reset_parameters(self):
        nn.init.uniform_(self.B)
        # nn.init.uniform_(self.B2)

    def forward(self):
        n = self.node_num
        B = torch.softmax(self.B, dim=-1)

        S = (torch.multinomial(B, num_samples=int(self.num_sample*n)))  # [n, s]

    
        self.M[1:, S] = 0

        log_p = torch.sum(
            torch.sum(B[:, S], dim=-1) - torch.logsumexp(B, dim=-1)
        )
        return self.M, log_p
        # return torch.diag_embed(self.M), log_p

'''
class Graph_Editer_Delete_Row(nn.Module):
    def __init__(self, K, node_num, num_sample, device):
        super(Graph_Editer_Delete_Row, self).__init__()
        self.B1 = nn.Parameter(torch.randn(K, node_num))
        self.B2 = nn.Parameter(torch.randn(K, node_num))
        self.M = torch.ones(K, node_num, node_num, dtype=torch.float, requires_grad=F).to(device)
        self.node_num = node_num
        self.num_sample = num_sample

    def reset_parameters(self):
        nn.init.uniform_(self.B1)
        nn.init.uniform_(self.B2)

    def forward(self, adj):
        n = self.node_num
        B = torch.einsum('ki,kj->kij', self.B1, self.B2)
        B = torch.softmax(B, dim=-1)
        # print(B)
        # print(B.shape)
        S = []
        for i in range(B.shape[0]):
            S.append(torch.multinomial(B[i], num_samples=self.num_sample))  # [n, s]
        S = torch.stack(S, dim=0)
        # print(S.shape)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, self.num_sample)
        # print(col_idx.shape)
        self.M[:, S, col_idx] = 0
        # print(self.M)
        # print(self.M.shape)
        if len(adj.shape) == 2:
            k_adj = torch.einsum('kij,ij->kij', self.M, adj)
        else:
            k_adj = torch.einsum('kij,bij->kbij', self.M, adj)
        log_p = torch.sum(
            torch.sum(B[:, S, col_idx], dim=-1) - torch.logsumexp(B, dim=-1)
        )
        return k_adj, log_p
'''
'''
class Graph_Editer(nn.Module):
    def __init__(self, K, node_num, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, node_num, node_num))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p
'''