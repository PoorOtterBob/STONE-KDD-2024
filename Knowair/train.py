import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub
from model.STONE import STONE
from frechet import Spatial_Embedding

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
if use_cuda: print('cuda is available')
device = torch.device('cuda:7' if use_cuda else 'cpu') # The num. of cuda need to rectify. 

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()


train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    # OOD segmental
    node_segment, sem_dist, adj_dist, node_num_ob, node_num_un = Spatial_Embedding(num_nodes=city_num, 
                                                                               adj=graph.attr_adj_dist, # sem 1
                                                                               new_node_ratio=config['train']['new_node_ratio'], 
                                                                               num_val=1, 
                                                                               epsilon=0.1, 
                                                                               c=config['train']['c'], # option: if args.c > args.horizon else args.horizon
                                                                               seed=3028, 
                                                                               device=device,
                                                                               istest=0,
                                                                               test_ratio=0).load_node_index_and_segemnt()
    _ , sem_direc, adj_direc, _ , _ = Spatial_Embedding(num_nodes=city_num, 
                                                            adj=graph.attr_adj_direc, # sem 2 
                                                            new_node_ratio=config['train']['new_node_ratio'], 
                                                            num_val=1, 
                                                            epsilon=0.1, 
                                                            c=config['train']['c'], # option: if args.c > args.horizon else args.horizon
                                                            seed=3028, 
                                                            device=device,
                                                            istest=0,
                                                            test_ratio=0).load_node_index_and_segemnt()
    
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device, graph.edge_index), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, node_num_ob+node_num_un, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    elif exp_model == 'STONE':
        # print(city_num)
        # print(graph.adj.shape) 184*184
        # print(graph.edge_attr.shape) # 3796*2
        # print(graph.edge_index.shape) # 2*3796
        # print(graph.attr_adj_dist)
        # print(graph.attr_adj_direc)
        # sys.exit(0)
        # For STONE
        blocks = []
        block = []
        has_shallow_encode = []
        for l in range(2):
            block.append([64, 16, 64])
            if l == 0:
                has_shallow_encode.append(True)
            else: 
                has_shallow_encode.append(False)
        for l in range(10):
            if l == 0:
                blocks.append([25, 128]) # 13 = input dim, which is the feature dim
            else:
                blocks.append([128, 128])
        # print(blocks)
        # print(block)
        # print(has_shallow_encode)
        input_dim=25, # 8 or 9 or sth. else.
        output_dim=1,
        node_num=node_num_ob+node_num_un,
        node_num_un=node_num_un, 
        node_num_ob=node_num_ob, 
        sem_dim=sem_dist['test'].shape[-1]*2, 
        SBlocks=block, 
        TBlocks=blocks, 
        has_shallow_encode=has_shallow_encode, 
        Kt=config['STONE']['Kt'],
        Ks_s=config['STONE']['Ks_s'], 
        Ks_t=config['STONE']['Ks_t'], 
        dropout=config['STONE']['dropout'], 
        adp_s_dim=config['STONE']['adp_s_dim'], 
        adp_t_dim=config['STONE']['adp_t_dim'], 
        x_output_dim=config['STONE']['x_output_dim'], 
        sem_output_dim=config['STONE']['sem_output_dim'], 
        gate_output_dim=config['STONE']['gate_output_dim'], 
        horizon=pred_len
        # print(sem_dim)
        # sys.exit(0)
        return STONE(input_dim=input_dim,
                     output_dim=output_dim,
                     node_num=node_num_ob+node_num_un,
                     SBlocks=block, 
                     TBlocks=blocks, 
                     node_num_un=node_num_un, 
                     node_num_ob=node_num_ob, 
                     sem_dim=sem_dim[0], 
                     has_shallow_encode=has_shallow_encode[0], 
                     Kt=Kt[0],
                     Ks_s=Ks_s[0], 
                     Ks_t=Ks_t[0], 
                     dropout=dropout[0], 
                     adp_s_dim=adp_s_dim[0], 
                     adp_t_dim=adp_t_dim[0], 
                     x_output_dim=x_output_dim[0], 
                     sem_output_dim=sem_output_dim[0], 
                     gate_output_dim=gate_output_dim[0], 
                     horizon=horizon
                     ), node_segment, sem_dist, sem_direc, node_num_ob, node_num_un, adj_dist, adj_direc
    else:
        raise Exception('Wrong model name!')

def recode(edge_index, edge_attr, sub_node_index):
    new_edge_index = []
    new_edge_attr = []
    for i in range(edge_index.shape[-1]):
        a = edge_index[0, i]
        b = edge_index[1, i]
        if np.logical_and(a in sub_node_index, b in sub_node_index):
            c = edge_attr[i, 0]
            d = edge_attr[i, 1]
            new_edge_index.append([np.argwhere(sub_node_index==a)[0], np.argwhere(sub_node_index==b)[0]])
            new_edge_attr.append([c, d])
    return torch.LongTensor(new_edge_index).squeeze(-1).transpose(0, 1),  torch.Tensor(np.float32(new_edge_attr))

def get_ood_graph(edge_index, edge_attr, node):
    edge_index_dict = {}
    edge_attr_dict = {}
    # edge_index = torch.LongTensor(edge_index)
    # edge_attr = torch.Tensor(np.float32(edge_attr))
    for cat in ['train', 'val', 'test']:
        edge_index_dict[cat], edge_attr_dict[cat] = recode(edge_index, edge_attr, node[cat+'_node'])
        # edge_index_dict[cat] = edge_index[(edge_index[:, 0].unsqueeze(1) == node[cat+'_node']) & 
        #                     (edge_index[:, 1].unsqueeze(1) == node[cat+'_node'])]
        # edge_attr_dict[cat] = edge_attr[(edge_index[:, 0].unsqueeze(1) == node[cat+'_node']) & 
        #                     (edge_index[:, 1].unsqueeze(1) == node[cat+'_node'])]
        print(cat, edge_index_dict[cat].shape, edge_attr_dict[cat].shape)
    return edge_index_dict, edge_attr_dict




def train(train_loader, node, sem, model, optimizer, edge_index=None, edge_attr=None):
    model.train()
    train_loss = 0
    # 加上S_delta: SEM的噪音
    spatial_noise = True
    sem = torch.stack([sem]*32, dim=0) # 32 be batch_size, maybe rectify. 
    for batch_idx, data in tqdm(enumerate(train_loader)):
        if spatial_noise:
            mean = 0  # 均值
            std = 0.1 # 标准差
            sem_noise = torch.add(sem, torch.normal(mean, std, sem.shape).to(sem.device))
            sem_noise = torch.clamp(sem_noise, min=0)
        optimizer.zero_grad()
        pm25, feature, time_arr = data
        pm25 = pm25[..., node, :].to(device) # (b, 2t, n, 1)
        feature = feature[..., node, :].to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        # print(pm25_label.shape) (b, t, n, 1) need to predict
        # print(feature.shape) (b, 2t, n, f=12)
        # print(pm25_hist.shape) (b, t, n, 1)
        # print(time_arr.shape) (b, 2t) sth. irrelvent to training/val.
        # sys.exit(0)
        if type(model).__name__ == 'STONE':
            pm25_pred = model(torch.cat((pm25_hist, feature[:, :hist_len, ...], feature[:, hist_len:, ...]), dim=-1), sem_noise)
        elif type(model).__name__ == 'GC_LSTM':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device))
        elif type(model).__name__ == 'PM25_GNN':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device), edge_attr.to(device))
        else:
            pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, node, sem, model, edge_index=None, edge_attr=None):
    model.eval()
    val_loss = 0
    sem = torch.stack([sem]*32, dim=0) # 32 be batch_size, maybe rectify.
    for batch_idx, data in tqdm(enumerate(val_loader)):      
        pm25, feature, time_arr = data
        pm25 = pm25[..., node, :].to(device)
        feature = feature[..., node, :].to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        if type(model).__name__ == 'STONE':
            pm25_pred = model(torch.cat((pm25_hist, feature[:, :hist_len, ...], feature[:, hist_len:, ...]), dim=-1), sem)
        elif type(model).__name__ == 'GC_LSTM':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device))
        elif type(model).__name__ == 'PM25_GNN':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device), edge_attr.to(device))
        else:
            pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, node, num_ob, sem, model, edge_index=None, edge_attr=None):
    model.eval()
    num_ob = num_ob[0]
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0

    predict_list1 = []
    label_list1 = []
    time_list1 = []
    test_loss1 = 0

    predict_list2 = []
    label_list2 = []
    time_list2 = []
    test_loss2 = 0

    sem = torch.stack([sem]*32, dim=0) # 32 be batch_size, maybe rectify.

    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25[..., node, :].to(device)
        feature = feature[..., node, :].to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        if type(model).__name__ == 'STONE':
            pm25_pred = model(torch.cat((pm25_hist, feature[:, :hist_len, ...], feature[:, hist_len:, ...]), dim=-1), sem)
        elif type(model).__name__ == 'GC_LSTM':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device))
        elif type(model).__name__ == 'PM25_GNN':
            pm25_pred = model(pm25_hist, feature, edge_index.to(device), edge_attr.to(device))
        else:
            pm25_pred = model(pm25_hist, feature)
        # All
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())
        # Ob
        loss1 = criterion(pm25_pred[..., :num_ob, :], pm25_label[..., :num_ob, :])
        test_loss1 += loss1.item()

        pm25_pred_val1 = np.concatenate([pm25_hist[..., :num_ob, :].cpu().detach().numpy(), pm25_pred[..., :num_ob, :].cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val1 = pm25[..., :num_ob, :].cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list1.append(pm25_pred_val1)
        label_list1.append(pm25_label_val1)
        time_list1.append(time_arr.cpu().detach().numpy())
        # Un
        loss2 = criterion(pm25_pred[..., num_ob:, :], pm25_label[..., num_ob:, :])
        test_loss2 += loss2.item()

        pm25_pred_val2 = np.concatenate([pm25_hist[..., num_ob:, :].cpu().detach().numpy(), pm25_pred[..., num_ob:, :].cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val2 = pm25[..., num_ob:, :].cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list2.append(pm25_pred_val2)
        label_list2.append(pm25_label_val2)
        time_list2.append(time_arr.cpu().detach().numpy())
    # All
    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0
    # Ob
    test_loss1 /= batch_idx + 1

    predict_epoch1 = np.concatenate(predict_list1, axis=0)
    label_epoch1 = np.concatenate(label_list1, axis=0)
    time_epoch1 = np.concatenate(time_list1, axis=0)
    predict_epoch1[predict_epoch1 < 0] = 0
    # Un
    test_loss2 /= batch_idx + 1

    predict_epoch2 = np.concatenate(predict_list2, axis=0)
    label_epoch2 = np.concatenate(label_list2, axis=0)
    time_epoch2 = np.concatenate(time_list2, axis=0)
    predict_epoch2[predict_epoch2 < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch, \
           test_loss1, predict_epoch1, label_epoch1, time_epoch1, \
           test_loss2, predict_epoch2, label_epoch2, time_epoch2


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list = [], []
    test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], []
    test_loss_list1, rmse_list1, mae_list1, csi_list1, pod_list1, far_list1 = [], [], [], [], [], []
    test_loss_list2, rmse_list2, mae_list2, csi_list2, pod_list2, far_list2 = [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model, node, sem_dist, sem_direc, node_num_ob, node_num_un, _ , _ = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        graph = Graph()
        edge_index, edge_attr = get_ood_graph(graph.edge_index, graph.edge_attr, node)

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, node['train_node'], 
                               torch.cat((sem_dist['train'], sem_direc['train']), dim=-1), 
                               model, 
                               optimizer, 
                               edge_index['train'], 
                               edge_attr['train'])
            val_loss = val(val_loader, node['val_node'], 
                           torch.cat((sem_dist['val'], sem_direc['val']), dim=-1), 
                           model, 
                           edge_index['val'], 
                           edge_attr['val'])

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch, \
                test_loss1, predict_epoch1, label_epoch1, time_epoch1, \
                test_loss2, predict_epoch2, label_epoch2, time_epoch2 = test(test_loader, 
                                                                             node['test_node'], 
                                                                             node_num_ob, 
                                                                             torch.cat((sem_dist['test'], sem_direc['test']), dim=-1), 
                                                                             model, 
                                                                             edge_index['test'], 
                                                                             edge_attr['test'])
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
                rmse1, mae1, csi1, pod1, far1 = get_metric(predict_epoch1, label_epoch1)
                rmse2, mae2, csi2, pod2, far2 = get_metric(predict_epoch2, label_epoch2)
                print('Train loss: %0.4f, Val loss: %0.4f' % (train_loss_, val_loss_))
                print('Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (test_loss, rmse, mae, csi, pod, far))
                print('Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (test_loss1, rmse1, mae1, csi1, pod1, far1))
                print('Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (test_loss2, rmse2, mae2, csi2, pod2, far2))
                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        test_loss_list.append(test_loss1)
        rmse_list.append(rmse1)
        mae_list.append(mae1)
        csi_list.append(csi1)
        pod_list.append(pod1)
        far_list.append(far1)
        test_loss_list.append(test_loss2)
        rmse_list.append(rmse2)
        mae_list.append(mae2)
        csi_list.append(csi2)
        pod_list.append(pod2)
        far_list.append(far2)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f' % (
            train_loss_, val_loss_))
        print(
            'Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            test_loss, rmse, mae, csi, pod, far))
        print(
            'Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            test_loss1, rmse1, mae1, csi1, pod1, far1))
        print(
            'Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            test_loss2, rmse2, mae2, csi2, pod2, far2))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list)) + \
                     'test_loss1 | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list1)) + \
                     'RMSE1      | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list1)) + \
                     'MAE1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list1)) + \
                     'CSI1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list1)) + \
                     'POD1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list1)) + \
                     'FAR1       | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list1)) + \
                     'test_loss2 | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list2)) + \
                     'RMSE2      | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list2)) + \
                     'MAE2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list2)) + \
                     'CSI2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list2)) + \
                     'POD2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list2)) + \
                     'FAR2       | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list2))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
