import os
import argparse
import numpy as np
import random

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)
from src.base.stone import STONE
from src.utils.args import get_public_config
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae_train
from src.utils.logging import get_logger
from src.utils.graph_edite import Graph_Editer_Delete_Row as GE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False



def get_config():
    parser = get_public_config()
    parser.add_argument('--TBlocks_num', type=int, default=5)
    parser.add_argument('--SBlocks_num', type=int, default=2)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks_s', type=int, default=1)
    parser.add_argument('--Ks_t', type=int, default=3)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--adj_type', type=str, default='transition')
    parser.add_argument('--adp_adj', type=int, default=1)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    args = parser.parse_args()

    folder_name = '{}-{}-{}'.format(args.dataset, args.adj_type, args.adp_adj)
    log_dir = './experiments/{}/{}/'.format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger



def main():
    args, log_dir, logger = get_config()
    if args.ood:
        from src.base.stone_engine import KrigingEngine
        from src.utils.dataloader_adp import load_dataset, load_adj_from_numpy, get_dataset_info, Spatial_Embedding
    else:
        from src.base.stone_engine_origin import KrigingEngine
        if args.tood:
            from src.utils.dataloader_adp import load_dataset, load_adj_from_numpy, get_dataset_info, Spatial_Embedding
        else:
            from src.utils.dataloader_adp_baseline_origin import load_dataset, load_adj_from_numpy, get_dataset_info, Spatial_Embedding

    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    
    
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    # adj is coo_matrix (no! dense!)
    adj = normalize_adj_mx(adj_mx, 'origin')[0]
    
    if args.mode == 'test':
        istest = True
        test_ratio = args.test_ratio # test what u want
    else:
        istest = False
        test_ratio = None
    
    # node segment
    node_segment, sem, adj, node_num_ob, node_num_un= Spatial_Embedding(num_nodes=node_num, 
                                                                        adj=adj, 
                                                                        new_node_ratio=args.new_node_ratio, 
                                                                        num_val=args.num_val, 
                                                                        epsilon=args.epsilon, 
                                                                        c=args.c, # option: if args.c > args.horizon else args.horizon
                                                                        seed=args.seed, 
                                                                        device=device,
                                                                        istest=istest,
                                                                        test_ratio=test_ratio).load_node_index_and_segemnt() 
    dataloader, scaler = load_dataset(data_path, node_segment, args, logger, istest)
    np.save('sem_test.npy', sem['test'].cpu().numpy())
    np.save('test_node_idx.npy', node_segment['test_node'])
    
    if args.mode == 'test':
        cat = 'test'
        adj[cat] = normalize_adj_mx(adj[cat], args.adj_type)
        adj[cat + '_observed'] = normalize_adj_mx(adj[cat + '_observed'], args.adj_type)
        adj[cat] = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj[cat]]
        adj[cat + '_observed'] = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj[cat + '_observed']]
        
    else:
        for cat in ['train', 'val', 'test']:
            adj[cat] = normalize_adj_mx(adj[cat], 'transition')[0]
            adj[cat + '_observed'] = normalize_adj_mx(adj[cat + '_observed'], 'transition')[0]
            adj[cat] = torch.tensor(adj[cat]).to(device)
            adj[cat + '_observed'] = torch.tensor(adj[cat + '_observed']).to(device)

    


    # blocks for STConv: [[12], [64, 16, 64]*b.n., [128] or [128, 128], [12]]
    # block for STDiff: [[[64, 64], [16, 16], [64, 64]]*b.n., [128, 128], [12]]
    blocks = []
    block = []
    has_shallow_encode = []
    for l in range(args.SBlocks_num):
        block.append([64, 16, 64])
        if l == 0:
            has_shallow_encode.append(True)
        else: 
            has_shallow_encode.append(False)

    for l in range(args.TBlocks_num):
        if l == 0:
            blocks.append([args.input_dim, 128])
        else:
            blocks.append([128, 128])
    x_output_dim = 128
    sem_output_dim = 64
    gate_output_dim = 128
    adp_s_dim = 20
    adp_t_dim = 20

    model = STONE(input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   node_num=node_num_ob+node_num_un if args.ood else node_num_ob,
                   SBlocks=block, 
                   TBlocks=blocks, 
                   node_num_un=node_num_un, 
                   node_num_ob=node_num_ob, 
                   sem_dim=sem['test'].shape[-1], 
                   has_shallow_encode=has_shallow_encode, 
                   Kt=args.Kt,
                   Ks_s=args.Ks_s, 
                   Ks_t=args.Ks_t, 
                   dropout=args.dropout, 
                   adp_s_dim=adp_s_dim, 
                   adp_t_dim=adp_t_dim, 
                   x_output_dim=x_output_dim, 
                   sem_output_dim=sem_output_dim, 
                   gate_output_dim=gate_output_dim, 
                   horizon=args.horizon,
                   )

    loss_fn = masked_mae_train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    ge_s = GE(args.KK, 
              node_num_ob+node_num_un, 
              args.num_sample,
              device)
    optimizer_ge_s = torch.optim.Adam(ge_s.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler_ge_s = torch.optim.lr_scheduler.StepLR(optimizer_ge_s, step_size=args.step_size, gamma=args.gamma)

    ge_t = GE(args.KK, 
              node_num_ob+node_num_un, 
              args.num_sample,
              device)
    optimizer_ge_t = torch.optim.Adam(ge_t.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler_ge_t = torch.optim.lr_scheduler.StepLR(optimizer_ge_t, step_size=args.step_size, gamma=args.gamma)


    engine = KrigingEngine(device=device,
                        model=model,
                        adj=adj,
                        node=node_segment,
                        sem=sem,
                        order=args.order,
                        horizon=args.horizon,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        mask_s=ge_s,
                        optimizer_ge_s=optimizer_ge_s,
                        scheduler_ge_s=scheduler_ge_s,
                        mask_t=ge_t,
                        optimizer_ge_t=optimizer_ge_t,
                        scheduler_ge_t=scheduler_ge_t,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed,
                        alpha=args.alpha,
                        beta=args.beta,
                        beta0=args.beta0,
                        year=int(args.years),
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
