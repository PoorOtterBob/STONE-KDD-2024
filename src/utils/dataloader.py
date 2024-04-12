import os
import pickle
import torch
import math
import numpy as np
import random
import threading
import multiprocessing as mp
from scipy.spatial.distance import cdist
from src.utils.spatial_side_information import Spatial_Side_Information as SSI

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, order, node, seed, logger, c=None, epsilon=None, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.order = order
        self.node = node
                
        self.seed = seed
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.c = c
        self.epsilon = epsilon

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
    
    # shuffle data order in one batch
    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx
    
    # shuffle batch order in one epoch
    def shuffle_batch(self):
        blocks = [self.idx[i:i+self.bs] for i in range(0, self.size, self.bs)]
        random.shuffle(blocks)
        idx = np.concatenate(blocks)
        if self.size % self.bs != 0:
            print('Warning: In shuffle_batch, the batch size does not evenly divide the array size.')

    def write_to_shared_array(self, x, y, dx, idx_ind, start_idx, end_idx):
        r = 0.01
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]
            for o in range(-self.order, self.order+1): 
                a = np.diff(x[i][:, :, 0], n=abs(o), axis=0).transpose()
                dx[i, o] = cdist(a, (1 if o >= 0 else -1)*a, 'euclidean')

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                dx_shape = (len(idx_ind), 1+2*self.order, self.data.shape[1], self.data.shape[1])
                dx_shared = mp.RawArray('f', int(np.prod(dx_shape)))
                dx = np.frombuffer(dx_shared, dtype='f').reshape(dx_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, \
                                              args=(x, y, dx, idx_ind, start_index, end_index))

                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y, dx)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, node_segment, args, logger, istest=False):
    dataloader = {}
    scaler = {}
    if istest == True:
        cat = 'test'
        years = str(int(args.years) + 1)
        ptr = np.load(os.path.join(data_path, years, 'his.npz'))
        logger.info('Data shape in ' + cat +' set years:' + str(ptr['data'].shape))
        idx = np.load(os.path.join(data_path, years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(data=ptr['data'][..., :args.input_dim], 
                                                 idx=idx, 
                                                 seq_len=args.seq_len, 
                                                 horizon=args.horizon, 
                                                 bs=args.bs, 
                                                 order=args.order, 
                                                 node=node_segment[cat], 
                                                 seed=args.seed, 
                                                 logger=logger)
        scaler[cat] = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    else:
        for cat in ['train', 'val', 'test']:
            if cat == 'test':
                years = str(int(args.years) + 1)
            else: years = args.years
            ptr = np.load(os.path.join(data_path, years, 'his.npz'))
            logger.info('Data shape in ' + cat +' set years:' + str(ptr['data'].shape))
            idx = np.load(os.path.join(data_path, years, 'idx_' + cat + '.npy'))
            dataloader[cat + '_loader'] = DataLoader(data=ptr['data'][..., :args.input_dim], 
                                                    idx=idx, 
                                                    seq_len=args.seq_len, 
                                                    horizon=args.horizon, 
                                                    bs=args.bs, 
                                                    order=args.order, 
                                                    node=node_segment[cat], 
                                                    seed=args.seed, 
                                                    logger=logger)
            scaler[cat] = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)



def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
        }
    assert dataset in d.keys()
    return d[dataset]


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
    # The output is the nodes of the three sets
    # e.g. —— SD: num_nodes = 716, new_node_ratior = 0.1, num_val = 2
    def dataset_node_segment(self):
        np.random.seed(self.seed)
        node_indices = np.arange(self.num_nodes)
        np.random.shuffle(node_indices)
        num_fixed_node = min(int(self.num_nodes / (1 + (self.num_val + 2) * self.new_node_ratio)), \
                             self.num_nodes - self.num_val - 2)
        num_additional_nodes_per = max(int(num_fixed_node * self.new_node_ratio), 1)
        fixed_indices = np.sort(node_indices[:num_fixed_node])
        train_indices = [fixed_indices, np.sort(node_indices[num_fixed_node:num_fixed_node + num_additional_nodes_per])]
        val_indices = {}
        for i in range(self.num_val):
            start_index = num_fixed_node + ((i+1) * num_additional_nodes_per)
            end_index = start_index + num_additional_nodes_per
        val_indices[i] = [fixed_indices, \
                          np.sort(node_indices[start_index:end_index])]
        test_indices = [fixed_indices, \
                        np.sort(node_indices[num_fixed_node + ((self.num_val+1) * num_additional_nodes_per) : num_fixed_node + ((self.num_val+2) * num_additional_nodes_per)])]
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
            node['val'] = dataset_node_segment_index['val_indices'][0]
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
                node['test'][-1] = np.concatenate((node['test'][-1], node['val'][-1][:len(node['test'][-1])//2]))
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
                node['test'][-1] = np.concatenate((node['test'][-1], node['val'][-1]))
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
