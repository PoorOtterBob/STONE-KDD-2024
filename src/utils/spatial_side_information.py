import torch
import numpy as np
import multiprocessing as mp
import random
import math
import scipy.sparse as sp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

mp.set_start_method('spawn', True)

# set_node_index: set_node_index[0] is observed vertices, set_node_index[-1] is unobserved vertives.
class Spatial_Side_Information():
    def __init__(self, set_node_index, epsilon, c, seed, device):
        super(Spatial_Side_Information, self).__init__()
        self._anchor_node = set_node_index[0]
        self._node = np.concatenate((set_node_index[0], set_node_index[-1]))
        self._num_nodes = len(self._node)
        self._epsilon = epsilon
        self._c = c
        self._seed = seed
        self._device = device

    # n = len(self._anchor_node) is the num. of observed vertices, c is num. of repeation of sampling (hy.para.).
    def get_random_anchorset(self):
        np.random.seed(self._seed)
        c = self._c
        n = len(self._anchor_node)
        distortion = math.ceil(np.log2(n))
        sampling_rep_rounds = c
        anchorset_num = sampling_rep_rounds * distortion
        anchorset_id = [np.array([]) for _ in range(anchorset_num)]
        for i in range(distortion):
            anchor_size = int(math.ceil(n / np.exp2(i + 1)))
            for j in range(sampling_rep_rounds):
                anchorset_id[i*sampling_rep_rounds+j] = np.sort(self._anchor_node[np.random.choice(n, size=anchor_size, replace=False)])
        return anchorset_id, anchorset_num

    # node_range  
    # adj: sp.coo_matrix = self._adj
    # epsilon = self._epsilon: Endurance values that are considered to be the same as the string
    def nodes_dist_range(self, adj, node_range):
        dists_dict = defaultdict(dict)
        if False:
            # pseudo Hamming distance
            for node in node_range:
                for neighbor in self._node:
                    if neighbor not in dists_dict[node]:
                        dists_dict[node][neighbor] = 0
                    diff = (abs(self._adj[node, self._node] - self._adj[neighbor, self._node]) >= self._epsilon)
                    if isinstance(adj, sp.coo_matrix):
                        dists_dict[node][neighbor] = sp.coo_matrix.sum(diff)
                    elif isinstance(adj, np.ndarray):
                        dists_dict[node][neighbor] = np.sum(diff)
                    # no self-loop
                    if self._adj[node, neighbor] > 0:
                        dists_dict[node][neighbor] +=1
        else:
            # Manhattan distance
            for node in node_range:
                for neighbor in self._node:
                    if neighbor not in dists_dict[node]:
                        dists_dict[node][neighbor] = 0
                    diff = abs(adj[node, self._node] - adj[neighbor, self._node])
                    if isinstance(adj, sp.coo_matrix):
                        dists_dict[node][neighbor] = sp.coo_matrix.sum(diff)
                    elif isinstance(adj, np.ndarray):
                        dists_dict[node][neighbor] = np.sum(diff)
                    # no self-loop
                    if adj[node, neighbor] > 0:
                        dists_dict[node][neighbor] += 1
        return dists_dict

    # Fill in the dictionary
    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def all_pairs_dist_parallel(self, adj, num_workers=16):
        nodes = self._node
        if len(nodes) < 200:
            num_workers = int(num_workers/4)
        elif len(nodes) < 800:
            num_workers = int(num_workers/2)
        elif len(nodes) < 3000:
            num_workers = int(num_workers)
        slices = np.array_split(nodes, num_workers)
        pool = mp.Pool(processes = num_workers)
        results = [pool.apply_async(self.nodes_dist_range, args=(adj, slices[i], )) for i in range(num_workers)]
        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)

        pool.close()
        pool.join()
        return dists_dict


    # Calculate hamming distance dict
    def precompute_dist_data(self, adj):
        dists_array = np.zeros((self._num_nodes, self._num_nodes))
        # Parallel or not
        dists_dict = self.all_pairs_dist_parallel(adj)
        for i, node in enumerate(self._node):
            dists_array[i] = np.array(list(dists_dict[node].values()))
        return dists_array

    # Obtain raw spatial side information
    def get_dist_min(self, adj=None, dist=None):
        anchorset_id, anchorset_num = self.get_random_anchorset()
        if dist is None:
            dist = self.precompute_dist_data(adj)
        dist_min = torch.zeros(self._num_nodes, anchorset_num).to(self._device)
        dist_argmin = torch.zeros(self._num_nodes, anchorset_num).long().to(self._device)
        coefficient = torch.ones(self._num_nodes, anchorset_num).to(self._device)
        for j in enumerate(anchorset_id):
            jj = [np.where(self._node == element)[0][0] for element in j[-1]]
            for i, node_i in enumerate(self._node):
                dist_temp = dist[i, jj]
                dist_temp_tensor = torch.as_tensor(dist_temp, dtype=torch.float32)
                dist_min_temp, dist_argmin_temp = torch.min(dist_temp_tensor, dim=-1)
                dist_min[i, j[0]] = dist_min_temp
                dist_argmin[i, j[0]] = anchorset_id[j[0]][dist_argmin_temp]
        return dist_min, dist_argmin, coefficient

    def spatial_emb_matrix(self, adj=None, dist=None):
        dist_min, dist_argmin, coefficient = self.get_dist_min(adj, dist)
        spatial_emb_matrix = torch.mul(coefficient, dist_min)
        return spatial_emb_matrix, dist_argmin
    
    # dx: (b, o, n, n) -> dx: (n, n)
    def temporal_emb_matrix(self, adj=None, dist=None):
        dist_min, dist_argmin, coefficient = self.get_dist_min(adj, dist)
        temporal_emb_matrix = torch.mul(coefficient, dist_min)
        return temporal_emb_matrix, dist_argmin

