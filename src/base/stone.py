import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from src.base.model import BaseModel
from src.utils.graph_edite import Graph_Editer_Delete_Row as GE
EPS = np.finfo(np.float32).eps
torch.autograd.set_detect_anomaly(True)

'''class STONE(BaseModel):
    def __init__(self, SBlocks, TBlocks, node_num_un, node_num_ob, sem_dim, has_shallow_encode, 
                 Kt, Ks_s, Ks_t, dropout, adp_s_dim, adp_t_dim, 
                 x_output_dim, sem_output_dim, gate_output_dim, horizon, K, num_sample, device, **args):
        super(STONE, self).__init__(**args)
        self.ge = nn.ModuleList([GE(K, node_num_ob+node_num_un, num_sample, device), 
                                 GE(K, node_num_ob+node_num_un, num_sample, device)])
        self.module = STONE_Module(SBlocks, TBlocks, node_num_un, node_num_ob, sem_dim, has_shallow_encode, 
                                   Kt, Ks_s, Ks_t, dropout, adp_s_dim, adp_t_dim, 
                                   x_output_dim, sem_output_dim, gate_output_dim, horizon, **args)

    def forward(self, x, sem, adj=None, tadj=None, label=None):
        x1, x2, log_p = self.module(x, sem, self.ge)
        return x1, x2, log_p '''

class STONE(BaseModel):
    def __init__(self, SBlocks, TBlocks, node_num_un, node_num_ob, sem_dim, has_shallow_encode, 
                 Kt, Ks_s, Ks_t, dropout, adp_s_dim, adp_t_dim, 
                 x_output_dim, sem_output_dim, gate_output_dim, horizon, **args):
        super(STONE, self).__init__(**args)
        self.sstblocks = STBlock(SBlocks, TBlocks, node_num_ob, node_num_un, 
                                dropout, Kt, sem_dim, has_shallow_encode)
        self.staggblocks = STAggBlock(TBlocks[-1][-1], x_output_dim, 
                                      SBlocks[-1][-1], sem_output_dim, 
                                      node_num_ob, node_num_un, dropout, Ks_s, Ks_t, 
                                      adp_s_dim, adp_t_dim)
        self.gatefusion = GatedFusionBlock(sem_input_dim=sem_output_dim, 
                                           x_input_dim=x_output_dim, 
                                           gate_output_dim=gate_output_dim, 
                                           horizon=horizon)
        self.x1 = nn.Linear(31, gate_output_dim)
        self.relu = nn.ReLU()
        self.x2 = nn.Linear(gate_output_dim, 1)
        self.node_num_ob = node_num_ob
        
    def forward(self, x, sem, ge=None):  
        x, sem = self.sstblocks(x.permute(0, 3, 1, 2), sem)
        x = self.x1(x.transpose(2, 3))
        x = self.relu(x)
        x = self.x2(x).transpose(2, 3)
        # x, sem, x_adj, sem_adj = self.staggblocks(x.permute(0, 3, 1, 2).squeeze(-1), sem)
        x, sem, log_p = self.staggblocks(x.permute(0, 3, 1, 2).squeeze(-1), sem, ge)
        x = self.gatefusion(x, sem)
        x1 = x[..., :self.node_num_ob, :]
        x2 = x[..., self.node_num_ob:, :] 

        return x1, x2, log_p
        # return x1, x2, x_adj, sem_adj

class STBlock(nn.Module):
    def __init__(self, SBlocks, TBlocks, node_num_ob, node_num_un, dropout, Kt, sem_dim, has_shallow_encode):
        super(STBlock, self).__init__()
        self.s_mlp = nn.ModuleList()
        self.t_mlp = nn.ModuleList()
        
        for i in range(len(SBlocks)):
            self.s_mlp.append(Attention_MLP(node_num_ob+node_num_un, 
                                            SBlocks[i][0], 
                                            SBlocks[i][-1], 
                                            SBlocks[i][1], 
                                            dropout, 
                                            sem_dim, 
                                            has_shallow_encode[i])
                                            )
        for j in range(len(TBlocks)):
            dilation = 1
            self.t_mlp.append(Dilation_Gated_TemporalConvLayer(Kt, 
                                                               TBlocks[j][0], 
                                                               TBlocks[j][-1], 
                                                               dilation, 
                                                               node_num_ob+node_num_un)
                                                               )
        self.t_mlp.append(Dilation_Gated_TemporalConvLayer(2, 
                                                           TBlocks[j][0], 
                                                           TBlocks[j][-1], 
                                                           dilation, 
                                                           node_num_ob+node_num_un)
                                                           )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.SBlocks_len = len(SBlocks)
        self.TBlocks_len = len(TBlocks)
        self.node_num_ob = node_num_ob

    def forward(self, x, sem):
        for i in range(self.SBlocks_len):
            sem = self.s_mlp[i](sem)
        x_list = []
        for j in range(self.TBlocks_len + 1):
            x = self.t_mlp[j](x)
            x_list.append(x)
        x = torch.cat(x_list, dim=2)
        return x, sem


class STAggBlock(nn.Module):
    def __init__(self, x_input_dim, x_output_dim, sem_input_dim, sem_output_dim, 
                 node_num_ob, node_num_un, dropout, Ks_s, Ks_t, adp_s_dim, adp_t_dim):
        super(STAggBlock, self).__init__()
        # adp_adj
        self.t_diff2 = GraphConvLayer(sem_input_dim, sem_output_dim, Ks_t)
        self.s_conv2 = GraphConvLayer(x_input_dim, x_output_dim, Ks_s)
        self.t_adpadj = AdaptiveInteraction(x_input_dim, adp_s_dim, node_num_ob+node_num_un)
        self.s_adpadj = AdaptiveInteraction(sem_input_dim, adp_t_dim, node_num_ob+node_num_un)

        self.bn_sem = nn.BatchNorm2d(sem_output_dim)
        self.bn_x = nn.BatchNorm2d(x_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.node_num_ob = node_num_ob

    def forward(self, x, sem, ge):
        
        if self.training:
            m_sem, log_p1 = ge[0]()
            m_x, log_p2 = ge[1]()
            log_p = [log_p1, log_p2]
        else:
            m_sem = None
            m_x = None
            log_p = [0., 0.]

        sem_adpadj = self.t_adpadj(x, m_x)
        x_adpadj = self.s_adpadj(sem, m_sem)

        x = self.s_conv2(x, x_adpadj)
        sem = self.t_diff2(sem, sem_adpadj)

        x = self.relu(x)
        sem = self.relu(sem)

        x = self.bn_x(x.transpose(1, -1)).transpose(1, -1)
        sem = self.bn_sem(sem.transpose(1, -1)).transpose(1, -1)

        x = self.dropout(x)
        sem = self.dropout(sem)
        
        return x, sem, log_p
        # return x, sem, x_adpadj, sem_adpadj


class Attention_MLP(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, hidden_dim, dropout, sem_dim=None, has_shallow_encode=True, has_attention=True, num_layer=1):
        super(Attention_MLP, self).__init__()
        if has_shallow_encode:
            self.shallow_encode = nn.Sequential(nn.Linear(sem_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, input_dim)) 
        
        self.Q = nn.Linear(input_dim, output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim),
                                  nn.Sigmoid())
        self.dropout = nn.Dropout(p=dropout)
        self.has_attention = has_attention
        self.has_shallow_encode = has_shallow_encode
        self.num_layer = num_layer
        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, sem):
        if self.has_shallow_encode:
            sem = self.shallow_encode(sem)
        gate = self.gate(sem)
        residual = sem

        Q = self.Q(sem)
        K = self.K(sem)
        V = self.V(sem)

        att = torch.einsum('bid, bjd -> bij', Q, K)/math.sqrt(self.output_dim)
        att = torch.softmax(att, dim=-1)
        sem = torch.einsum('bjd, bij -> bid', V, att)
        sem = self.bn(sem.transpose(1, -1)).transpose(1, -1)
        sem = self.relu(sem)
        sem = gate*residual + (1-gate)*sem
        return sem


class Dilation_Gated_TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, dilation, node_num):
        super(Dilation_Gated_TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.node_num = node_num
        self.align = Align(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, \
                            kernel_size=(Kt, 1), enable_padding=False, dilation=dilation)
        self.dilation = dilation

    def forward(self, x):
        x_in = self.align(x)[:, :, (self.Kt - 1)*self.dilation:, :]
        x_causal_conv = self.causal_conv(x)

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
        return x


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)


    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result

class AdaptiveInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, **args):
        super(AdaptiveInteraction, self).__init__(**args)
        self.E_out1 = nn.Linear(input_dim, output_dim)
        self.E_in1 = nn.Linear(input_dim, output_dim)

        self.E_out2 = nn.Linear(input_dim, output_dim)
        self.E_in2 = nn.Linear(input_dim, output_dim)

        self.E_out3 = nn.Linear(input_dim, output_dim)
        self.E_in3 = nn.Linear(input_dim, output_dim)

        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.output_dim = output_dim
    
    def forward(self, input, m):
        E_out1 = self.E_out1(input)
        E_in1 = self.E_in1(input)

        E_out2 = self.E_out1(input)
        E_in2 = self.E_in1(input)

        E_out3 = self.E_out3(input)
        E_in3 = self.E_in3(input)
        
        E_out = torch.einsum('bid,bjd->bij', E_out1, E_out2)/math.sqrt(self.output_dim)
        E_in = torch.einsum('bid,bjd->bij', E_in1, E_in2)/math.sqrt(self.output_dim)
        
        E_out3 = torch.einsum('bij,bjd->bid', E_out, E_out2)
        E_in3 = torch.einsum('bij,bjd->bid', E_in, E_in2)

        E_out3 = self.bn1(E_out3.transpose(1, -1)).transpose(1, -1)
        E_out3 = self.bn2(E_in3.transpose(1, -1)).transpose(1, -1)

        if len(input.shape) == 2:
            adp_adj = torch.einsum('bik, bjk -> bij', E_out3, E_in3)
        else:
            adp_adj = torch.einsum('bik, bjk -> bij', E_out3, E_in3)

        adp_adj = self.relu(adp_adj)
        adp_adj = self.softmax(adp_adj)

        if self.training:
            adp_adj = torch.einsum('kj, bij -> kbij', m, adp_adj)
        else:
            adp_adj = adp_adj.unsqueeze(0)
        return adp_adj


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = nn.Linear(c_in, c_out)
        self.Ks = Ks
        self.tgraph_diff = GraphConv(c_out, c_out, Ks)

    
    def forward(self, x, adj):
        x_tgc_in = self.align(x)
        x_tgc = self.tgraph_diff(x_tgc_in, adj)
        x_tgc_out = torch.add(x_tgc, x_tgc_in.unsqueeze(0))
        return x_tgc_out

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks+1, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        x_list = [x]
        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        else:
            if self.training:
                x_list = [x.unsqueeze(0).expand(gso.shape[0], -1, -1, -1)] # (b, n, d) -> (o, M, b, n, d)
            else:
                x_list = [x.unsqueeze(0)] # (b, n, d) -> (o, M, b, n, d)
            adj = gso
            for i in range(self.Ks):
                x_list.append(torch.einsum('mbij, mbjd-> mbid', adj, x_list[i]))

        x = torch.stack(x_list, dim=0)
        cheb_graph_conv = torch.einsum('kmbit,kts -> mbis', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x): 
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, node_num = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, node_num]).to(x)], dim=1)
        else:
            x = x
        return x

class GatedFusionBlock(nn.Module):
    def __init__(self, sem_input_dim, x_input_dim, gate_output_dim, horizon):
        super(GatedFusionBlock, self).__init__()
        self.gate_sem = nn.Linear(sem_input_dim+x_input_dim, gate_output_dim)
        self.gate_x = nn.Linear(x_input_dim, gate_output_dim)
        self.out1 = nn.Linear(gate_output_dim, gate_output_dim)
        self.out2 = nn.Linear(gate_output_dim, horizon)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.gate_output_dim=gate_output_dim


    def forward(self, x, sem):
        gate_sem = self.gate_sem(torch.cat((sem, x), dim=-1))
        gate_sem = self.sigmoid(gate_sem)          
        output = self.gate_x(x)
        output = gate_sem * output
        output = self.out1(output)
        output = self.relu(output)
        output = self.out2(output)
        
        return output.transpose(-1, -2).unsqueeze(-1)
