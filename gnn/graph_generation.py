import numpy as np
import torch.nn.functional as F
import torch
import os

def make_graph(graph_method, gene_method, ids_in_RNAseq, ids_nin_RNAseq, features):
    '''
    directed_bipartite_graph is default
    add high correlated nodes
    graph_method = sim_cand_strategy
    '''
    path = f'graphs/{graph_method}_{gene_method}.pt'
    if os.path.isfile(path): return torch.load(path)
    
    sim, L1000_method, NonL1000_method = graph_method.split('_')
    sim_metric, k = sim[:3], float(sim[3:])
    if k>=1 or k==0:
        sim_metric, k = sim[:3], int(sim[3:])
    assert(sim_metric in ['cos', 'pcc'])
    assert(('L' in L1000_method) and (L1000_method[1:] in ['all', 'full', 'pos', 'posneg', 'gcn']))
    assert(('N' in NonL1000_method) and (NonL1000_method[1:] in ['all', 'pos', 'posneg', 'gcn']))
    
    if sim_metric == 'cos':
        path = f'graphs/correlation/cosine_similarity.pt'
        if os.path.isfile(path): similarity = torch.load(path)
        else: similarity = cal_similarity(features, sim_metric, path)
    elif sim_metric == 'pcc':
        path = f'graphs/correlation/pearson_correlation.pt'
        if os.path.isfile(path): similarity = torch.load(path)
        else: similarity = cal_similarity(features, sim_metric, path)
        adj = torch.zeros(12320, 12320).cuda()
        adj[similarity>k] = 1
        torch.save(adj, path)
        return adj
    
    # Connect l1000 - non l1000
    adj = directed_bipartite_graph(ids_in_RNAseq, ids_nin_RNAseq)
    
    # Connect l1000 - l1000
    L1000_similarity = similarity[ids_in_RNAseq, np.array(ids_in_RNAseq).reshape(-1,1)]
    L1000_knn_graph = make_knn_graph(L1000_similarity, k, L1000_method[1:])
    adj[ids_in_RNAseq, np.array(ids_in_RNAseq).reshape(-1,1)] += L1000_knn_graph
    
    # Connect non l1000 - non l1000
    NonL1000_similarity = similarity[ids_nin_RNAseq, np.array(ids_nin_RNAseq).reshape(-1,1)]
    NonL1000_knn_graph = make_knn_graph(NonL1000_similarity, k, NonL1000_method[1:])
    adj[ids_nin_RNAseq, np.array(ids_nin_RNAseq).reshape(-1,1)] += NonL1000_knn_graph
   
    adj[adj>1]=1
    torch.save(adj, path)
    return adj

def directed_bipartite_graph(ids_in_RNAseq, ids_nin_RNAseq):
    '''
    connect all l1000 - nonl1000 pairs
    Since (W*A)xX, A[i,j] means  jth node feature passed to ith node in next layer
    '''
    nnodes = len(ids_in_RNAseq) + len(ids_nin_RNAseq) 
    adj = torch.zeros(nnodes, nnodes)
    for j in ids_in_RNAseq:
        adj[ids_nin_RNAseq, j] = 1 # for every node in l1000, connect all nodes in ids_nin_RNAseq
    return adj.cuda()

def make_knn_graph(similarity, k, method):
    '''
    Since (W*A)xX, A[i,j] means  jth node feature --> to ith node in next layer
    for ith node, nearest neighbors' feature --> ith node
    A[i, nearest] is needed
    '''
    nnodes = similarity.shape[0]
    if method == 'all' or method == 'full':
        adj = torch.ones((nnodes, nnodes)).cuda()
    elif k==0:
        adj =torch.zeros((nnodes, nnodes)).cuda()
    elif method == 'pos':
        adj = top_k(similarity, k + 1)
    elif method == 'posneg':
        adj = top_k(similarity, k)
        adj += top_k(-similarity, k)
    return adj      

def top_k(raw_graph, K):
    if K == raw_graph.shape[0]: return torch.ones_like(raw_graph).cuda()
    _, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = np.zeros(raw_graph.shape)
    indices_np = indices.detach().cpu().numpy()
    mask[np.arange(raw_graph.shape[0]).reshape(-1, 1), indices_np] = 1.
    mask = torch.Tensor(mask).cuda()
    return mask


def cal_similarity(node_embeddings, sim, path):
    if sim == 'cos':
        similarity = torch.zeros(node_embeddings.shape[0], node_embeddings.shape[0]).cuda()
        node_embeddings = node_embeddings.cuda()
        for i in range(node_embeddings.shape[0]):
            similarity[i, :] = F.cosine_similarity(node_embeddings[i][None,:], node_embeddings[:,:], dim=-1)
    elif sim == 'pcc':
        similarity = torch.corrcoef(node_embeddings)
    os.makedirs('graphs', exist_ok=True)
    os.makedirs('graphs/correlation', exist_ok=True)
    torch.save(similarity, path)
    return similarity
