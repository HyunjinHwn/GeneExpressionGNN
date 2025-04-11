from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import math

class EdgeWeightGNN(nn.Module):
    def __init__(self, adj, nfeat, nhid):
        super(EdgeWeightGNN, self).__init__()
        nnodes = adj.shape[0]
        self.mask = adj
        self.nhid = nhid
        
        # Learnable embedding
        self.learnable_embedding = True
        self.embedding = nn.Linear(nfeat, nhid)
        self.weight = Parameter(torch.ones(nnodes, nnodes))
        self.bias = Parameter(torch.zeros(nnodes, nhid))
        
        # 1layer
        stdv = 1. / math.sqrt(adj.mean())
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        
        # 2layer
        self.weight2 = Parameter(torch.ones(nnodes, nnodes))
        self.bias2 = Parameter(torch.zeros(nnodes, nhid))
        self.weight2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        
        # 3layer
        self.weight3 = Parameter(torch.ones(nnodes, nnodes))
        self.bias3 = Parameter(torch.zeros(nnodes, nhid))
        self.weight3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
        
        # 4layer
        self.weight4 = Parameter(torch.ones(nnodes, nnodes))
        self.bias4 = Parameter(torch.zeros(nnodes, nhid))
        self.weight4.data.uniform_(-stdv, stdv)
        self.bias4.data.uniform_(-stdv, stdv)

        # regressor    
        self.regressor4 = nn.Linear(nhid, 1)

    def forward(self, input):
        # self.weight: n*n, input: n*3 -> out: n*3
        # matmul ((masked & learnable A), X)
        if self.learnable_embedding:
            emb = self.embedding(input) # nhid
        else:
            emb = input
        out = torch.mm(self.weight*self.mask, emb)  + self.bias
        out = torch.relu(out)
        
        out = torch.mm(self.weight2*self.mask, out)  + self.bias2
        out = torch.relu(out)
        
        out = torch.mm(self.weight3*self.mask, out)  + self.bias3
        out = torch.relu(out)
        
        out = torch.mm(self.weight4*self.mask, out)  + self.bias4
        out = torch.relu(out)
        out = self.regressor4(out)
        return out.T
