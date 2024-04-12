import os
import time
import torch
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)
from src.utils.graph_algo import normalize_adj_mx as nam
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics

class BaseEngine():
    def __init__(self, device, model, node, adj, num_node_ob, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, alpha, beta, beta0, year):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._node = node
        self.adj = adj
        self.num_node_ob = num_node_ob
        self._alpha = alpha
        self._beta = beta
        self._beta0 = beta0

        self.year = year

        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors, cat):
        def inv(tensor):
            return self._scaler[cat].inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_non-ood_s{}_{}.pt'.format(self._seed, self.year)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_non-ood_s{}_{}.pt'.format(self._seed, self.year)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   


    def train_batch(self):
        self.model.train()

        train_loss1 = []
        train_mape1 = []
        train_rmse1 = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            # X (b, t, n, f), label (b, t, n, 1)
            X = X[:, :, self._node['train_observed_node'], :]
            label = label[..., self._node['train_observed_node'], :]
            adj = self.adj['train_observed']
            if isinstance(adj, list):
                adj = self._to_device(self._to_tensor(adj))
            else: adj = self._to_device(self._to_tensor([adj]))
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, adj, label)
            pred, label = self._inverse_transform([pred, label], 'train')

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value ob', mask_value)

            loss1 = self._loss_fn(pred, label, mask_value)
            mape1 = masked_mape(pred, label, mask_value).item()
            rmse1 = masked_rmse(pred, label, mask_value).item()

            loss = loss1

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss1.append(loss1.item())
            train_mape1.append(mape1)
            train_rmse1.append(rmse1)

            self._iter_cnt += 1
        return np.mean(train_loss1), np.mean(train_mape1), np.mean(train_rmse1)


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss1, mtrain_mape1, mtrain_rmse1 = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss1, mvalid_mape1, mvalid_rmse1 = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss1, mtrain_rmse1, mtrain_mape1, \
                                             mvalid_loss1, mvalid_rmse1, mvalid_mape1, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            mvalid_loss = mvalid_loss1
            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds1 = []
        labels1 = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X = X[:, :, self._node[mode + '_observed_node'], :]
                label1 = label[..., self._node[mode + '_observed_node'], :]
                adj = self.adj[mode + '_observed']
                if isinstance(adj, list):
                    adj = self._to_device(self._to_tensor(adj))
                else: adj = self._to_device(self._to_tensor([adj]))
                X, label1 = self._to_device(self._to_tensor([X, label1]))
                pred = self.model(X, adj, label1)
                pred1 = pred[..., :self.num_node_ob, :]
                pred1, label1= self._inverse_transform([pred1, label1], mode)

                preds1.append(pred1.squeeze(-1).cpu())
                labels1.append(label1.squeeze(-1).cpu())

        preds1 = torch.cat(preds1, dim=0)
        labels1 = torch.cat(labels1, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value1 = torch.tensor(0)

        if labels1.min() < 1:
            mask_value1 = labels1.min()

        if mode == 'val':
            mae1 = self._loss_fn(preds1, labels1, mask_value1).item()
            mape1 = masked_mape(preds1, labels1, mask_value1).item()
            rmse1 = masked_rmse(preds1, labels1, mask_value1).item()
            return mae1, mape1, rmse1

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []

            print('Check mask value ob', mask_value1)

            for i in range(self.model.horizon):
                res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                log = 'Horizon {:d},Ob Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res1[0], res1[2], res1[1]))
                test_mae.append(res1[0])
                test_mape.append(res1[1])
                test_rmse.append(res1[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
