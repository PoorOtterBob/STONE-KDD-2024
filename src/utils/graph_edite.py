import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class Graph_Editer_Delete_Row(nn.Module):
    def __init__(self, K, node_num, num_sample, device):
        super(Graph_Editer_Delete_Row, self).__init__()
        
        self.B = nn.Parameter(torch.FloatTensor(K, node_num))
        self.M = torch.ones(K, node_num, dtype=torch.float, requires_grad=False).to(device)
        self.node_num = node_num
        self.num_sample = num_sample
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self):
        n = self.node_num
        B = torch.softmax(self.B, dim=-1)

        S = (torch.multinomial(B, num_samples=int(self.num_sample*n)))  # [n, s]

    
        self.M[1:, S] = 0

        log_p = torch.sum(
            torch.sum(B[:, S], dim=-1) - torch.logsumexp(B, dim=-1)
        )
        return self.M, log_p
