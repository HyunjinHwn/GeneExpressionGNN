import os
import torch

from gnn.gnn_utils import *
from gnn.edgeweight import EdgeWeightGNN
from data import dataload, Dataset
from gnn.graph_generation import *

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN training for Gene Expression')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--select_method", type=str, default='order', help='random, order, greedy_forward')
    parser.add_argument("--select_gene_num", type=int, default=970, help='number of genes to select: 970, 108, ...')
    parser.add_argument("--graph", type=str, default='cos50_Lfull_Nposneg', help='sim_cand_strategy')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    # path (edit this)
    datapath = '../data/'
    graph_name = f'{args.graph}'
    config_name = f'{args.select_method}{args.select_gene_num}_lr{args.lr}'
        
    # load data
    l1000, rnaseq, _, RNAseq_ids_in_l1000, RNAseq_ids_nin_l1000 = dataload(root=datapath)
    data  = Dataset(l1000, rnaseq, RNAseq_ids_in_l1000, args.select_gene_num, 
                    select_method=args.select_method)
    print(f'Input: {data.X.shape[0]} * ({len(data.labeled_gene_ids)} out of {data.X.shape[1]}')
    print(f'Output: {data.Y.shape}')
        
    # load or generate graph
    adj = make_graph(args.graph, args.select_method+str(data.select_gene_num), 
                     RNAseq_ids_in_l1000, RNAseq_ids_nin_l1000, rnaseq[:,0:3000])
    data.l1000_features = torch.zeros(adj.shape[0], l1000.shape[-1]).cuda() 
    labeled_rna_idx = data.RNAseq_ids_in_l1000[data.labeled_gene_ids]
    data.l1000_features[labeled_rna_idx] = l1000[data.labeled_gene_ids].cuda()
    pos_encoding = torch.eye(adj.shape[0])        
    nhid = 64
    data.features = torch.hstack((pos_encoding, torch.zeros(adj.shape[0],1))).cuda()
    
    os.makedirs(f'checkpoints/{graph_name}/', exist_ok=True)
    os.makedirs(f'prediction/{graph_name}/', exist_ok=True)
    loss_func = torch.nn.L1Loss()
    model = EdgeWeightGNN(adj, nfeat=data.features.shape[1], nhid=nhid).cuda()
    best_model = train(model, data, config_name, graph_name, loss_func, 
                       epochs=args.epochs, lr=args.lr, wd=1e-6)
    