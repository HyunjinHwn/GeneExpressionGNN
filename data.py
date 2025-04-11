import os
import numpy as np
import cmapPy
import pandas as pd
import torch
import random

def dataload( root):
    # Read data from files
    path1 = f'{root}/L1000_level_3_q2norm_n3176x970.xlsx'
    l1000_raw = pd.read_csv(path1, sep='\t')
    l1000_array = np.array(l1000_raw)           # shape: (970, 3176), min~max: 1.64900004863739 ~ 15.0040998458862
    l1000_array = l1000_array / np.max(l1000_array)

    # path2 = f'{root}/no_hdf5_compression_GTEx_RNAseq_Log2RPKM_q2norm_n3176x12320.gctx'
    path2 = f'{root}/GSE92743_Broad_GTEx_RNAseq_Log2RPKM_q2norm_n3176x12320.gctx'
    RNAseq_raw = cmapPy.pandasGEXpress.parse(path2)
    RNAseq_array = np.array(RNAseq_raw.data_df) # shape: (12320, 3176), min~max: 0.0 ~ 12.847019
    RNAseq_array = RNAseq_array / np.max(RNAseq_array)
    RNAseq_row_name = [int(i) for i in RNAseq_raw.row_metadata_df.index.tolist()] # list: len=12320
    RNAseq_row_name2idx = {k:i for i,k in enumerate(RNAseq_row_name)}

    path3 = f'{root}/lmid_970.txt'                   # L1000 names among RNAseq_row_name 
    with open(path3, 'r') as f:
        lmids = f.readlines()
    lmids = [int(l.strip()) for l in lmids]        # list: len=970
    RNAseq_ids_in_l1000 = [RNAseq_row_name2idx[k] for k in lmids]
    RNAseq_ids_nin_l1000 = [RNAseq_row_name2idx[k] for k in RNAseq_row_name if k not in lmids ]

    l1000 = torch.Tensor(l1000_array)
    rnaseq = torch.Tensor(RNAseq_array)
    
    return l1000, rnaseq, RNAseq_row_name2idx, RNAseq_ids_in_l1000, RNAseq_ids_nin_l1000

class Dataset():
    def __init__(self, l1000, rnaseq, RNAseq_ids_in_l1000, select_gene_num, select_method):
        self.idx_train = list(range(2500))
        self.idx_val = list(range(2500, 3000))
        self.idx_test = list(range(3000, 3176))
        self.X = l1000.T.cuda() # 3176*970
        self.Y = rnaseq.T.cuda() # 3176*12320
        self.select_gene_num = select_gene_num
        self.labeled_gene_ids = self.select_gene(select_method) 
        assert (self.select_gene_num == len(self.labeled_gene_ids))
        self.RNAseq_ids_in_l1000 = np.array(RNAseq_ids_in_l1000)
        self.gene_select_pool = set(range(len(RNAseq_ids_in_l1000)))
            
    def select_gene(self, select_method):
        if select_method == 'random':
            path = f'selected_genes/random_{self.select_gene_num}.pt'
            if os.path.isfile(path):
                gene_ids = torch.load(path)
            else:
                gene_ids = random.sample(range(len(self.gene_select_pool)), self.select_gene_num)
                gene_ids.sort()
                torch.save(gene_ids, path)
            return gene_ids
        
        elif select_method == 'order':
            gene_ids = list(range(0, self.select_gene_num))
            return gene_ids
        
        elif select_method == 'greedy_forward':
            path = f'selected_genes/greedy_forward.txt' # 108 genes in order
            if not os.path.isfile(path): 
                raise FileNotFoundError(f"File {path} not found.")
            
            # Read the gene ids from the file
            print(f"Loading gene ids from {path}")
            with open(path, 'r') as f:
                gene_ids = f.readlines()
            gene_ids = [int(g.strip()) for g in gene_ids]

            if self.select_gene_num > len(gene_ids):
                raise ValueError(f"gene_num({self.select_gene_num}) should be less than {len(gene_ids)}")

            print(f"Gene ids: {gene_ids}")
            return gene_ids[:self.select_gene_num]
