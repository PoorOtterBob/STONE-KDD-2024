import numpy as np
import random
import torch
import numpy as np
import multiprocessing as mp
import math
import scipy.sparse as sp
from collections import defaultdict
from spatial_side_information import Spatial_Side_Information as SSI
from scipy.spatial.distance import cdist

class Spatial_Embedding():
    def __init__(self, num_nodes, adj, new_node_ratio, num_val, epsilon, c, seed, device, istest=False, test_ratio=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj = adj
        self.new_node_ratio = new_node_ratio
        self.num_val = num_val
        self.epsilon = epsilon
        self.c = c
        self.seed = seed
        self.device = device
        self.istest = istest
        self.test_ratio = test_ratio

    # new_node_ratio always be 1:10, and the input of it is 0.1
    # ex: 500 nodes for training set and 50 other nodes for addtion nodes in the val. and testing set. 
    # num_val is the number of different validation sets.
    # [1 + (num_val + 1) * new_node_ratio] * traning_set_node = num_nodes
    # 输出为三种集合的点
    # e.g. —— SD: num_nodes = 716, new_node_ratior = 0.1, num_val = 2
    def dataset_node_segment(self):
        np.random.seed(self.seed)
        # 生成节点索引
        node_indices = np.arange(self.num_nodes)
        # 随机打乱节点索引
        np.random.shuffle(node_indices)
        # 固定节点个数
        num_fixed_node = min(int(self.num_nodes / (1 + (self.num_val + 2) * self.new_node_ratio)), \
                             self.num_nodes - self.num_val - 2)
        # 新增节点数量
        num_additional_nodes_per = max(int(num_fixed_node * self.new_node_ratio), 1)
        # 固定节点索引
        fixed_indices = np.sort(node_indices[:num_fixed_node])
        # 划分训练集
        train_indices = [fixed_indices, np.sort(node_indices[num_fixed_node:num_fixed_node + num_additional_nodes_per])]
        # 划分验证集
        val_indices = {}
        for i in range(self.num_val):
            start_index = num_fixed_node + ((i+1) * num_additional_nodes_per)
            end_index = start_index + num_additional_nodes_per
        val_indices[i] = [fixed_indices, \
                          np.sort(node_indices[start_index:end_index])]
        # 划分测试集
        test_indices = [fixed_indices, \
                        np.sort(node_indices[num_fixed_node + ((self.num_val+1) * num_additional_nodes_per) : num_fixed_node + ((self.num_val+2) * num_additional_nodes_per)])]
        
        # dataset_node_segment_index = [fixed_indices, train_indices, val_indices, test_indices]
        
        indices = {
        'fixed': fixed_indices,
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
        }
        dataset_node_segment_index = {}
        for cat in ['fixed', 'train', 'val', 'test']:
            dataset_node_segment_index[cat + '_indices'] = indices[cat]
        
        return dataset_node_segment_index, num_fixed_node, num_additional_nodes_per

    def SSI_func(self, ob, dist):
        return SSI(ob, self.epsilon, self.c, self.seed, self.device).spatial_emb_matrix(adj=self.adj, dist=dist)

    ###以下为将num_node进行未知节点的划分###
    ###并进行SEM的计算###
    # Spatial Side Information
    # sem: Spatial Embedding Matrix (n, d), d为向量长度
    def load_node_index_and_segemnt(self):
        dataset_node_segment_index, num_ob, num_un = self.dataset_node_segment()
        node = {}
        sem = {}
        adj = {}
        node['fixed'] = dataset_node_segment_index['fixed' + '_indices']
        if self.istest == False:
            for cat in ['train', 'val', 'test']:
                if cat == 'val': 
                    if len(dataset_node_segment_index[cat + '_indices']) == 1:
                        node[cat] = dataset_node_segment_index[cat + '_indices'][0]
                        node[cat + '_observed_node'] = node[cat][0]
                        node[cat + '_unobserved_node'] = node[cat][-1]
                        node[cat + '_node'] = np.concatenate((node[cat + '_observed_node'], \
                                                    node[cat + '_unobserved_node']))
                        dist = cdist(self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                    self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                    metric='cityblock')
                        sem[cat], _ = self.SSI_func(node[cat], dist)
                        adj[cat] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                        adj[cat+'_observed'] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                        print('SEM for ' + cat + ' set has been calculated: ' + str(sem[cat].shape))
                    else:
                        for i in range(len(dataset_node_segment_index[cat])):
                            node[cat + '_' + str(i)] = dataset_node_segment_index[cat + '_indices'][i]
                            node[cat + '_observed_node_' + str(i)] = node[cat + '_' + str(i)][0]
                            node[cat + '_unobserved_node_' + str(i)] = node[cat + '_' + str(i)][-1]
                            node[cat + '_node_'] = np.concatenate((node[cat + '_observed_node'], \
                                                    node[cat + '_unobserved_node']))
                            dist = cdist(self.adj[node[cat + '_node' + str(i)], :][:, node[cat + '_node' + str(i)]], 
                                        self.adj[node[cat + '_node' + str(i)], :][:, node[cat + '_node' + str(i)]], 
                                        metric='cityblock')
                            adj[cat + '_' + str(i)] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                            adj[cat+'_observed' + '_' + str(i)] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                            sem[cat + '_' + str(i) + '_' + str(i)], _ = self.SSI_func(node[cat + '_' + str(i)], dist)
                else:
                    node[cat] = dataset_node_segment_index[cat + '_indices']
                    node[cat + '_observed_node'] = node[cat][0]
                    node[cat + '_unobserved_node'] = node[cat][-1]
                    node[cat + '_node'] = np.concatenate((node[cat + '_observed_node'], \
                                                        node[cat + '_unobserved_node']))
                    dist = cdist(self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                metric='cityblock')
                    sem[cat], _ = self.SSI_func(node[cat], dist)
                    adj[cat] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                    adj[cat+'_observed'] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                    print('SEM for ' + cat + ' set has been calculated: ' + str(sem[cat].shape))
        else: 
            # extra = dataset_node_segment_index['val_indices'][0]
            extra = dataset_node_segment_index['train_indices']
            if self.test_ratio == 0.05:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = node['test'][-1][:len(node['test'][-1])//2]
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un//2
            elif self.test_ratio == 0.1:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un
            elif self.test_ratio == 0.15:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = np.concatenate((node['test'][-1], extra[-1][:len(node['test'][-1])//2]))
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un + num_un//2
            elif self.test_ratio == 0.2:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = np.concatenate((node['test'][-1], extra[-1]))
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un*2
        return node, sem, adj, num_ob, num_un