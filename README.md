# GeneExpressionGNN

GeneExpressionGNN is a project focused on leveraging graph neural networks (GNNs) and other models for inferring patterns in gene expression data. 
This repository provides tools and models to explore relationships between genes and their expression profiles.

## Features

- Code for gene expression prediction from L1000 data to RNAseq data.
- Implementation of graph neural network architectures.
- Training codes for GNN, MLP, and SwinIR.
- Evaluation metrics for gene expression prediction.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/HyunjinHwn/GeneExpressionGNN.git
    ```
2. Install the required dependencies:
    ```
    cmapPy==2.2.2
    numpy==1.26.4
    torch==2.0.0
    scipy==1.13.1
    pandas==2.2.3
    ```
3. Download the datasets in the following link:
    Please download RNAseq file through [this link](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92743&format=file&file=GSE92743%5FBroad%5FGTEx%5FRNAseq%5FLog2RPKM%5Fq2norm%5Fn3176x12320%2Egctx%2Egz)
    and unzip through following code.

    `gunzip GSE92743_Broad_GTEx_RNAseq_Log2RPKM_q2norm_n3176x12320.gctx.gz`

    Place this file in `data/` before you run the codes.


## Training the Models

### Training the GNN Models

You can run this project using either the Python script or the Jupyter Notebook:


#### Option 1: Jupyter Notebook

Open `gnn_training.ipynb` with Jupyter Notebook or JupyterLab, and run the cells in order.  

#### Option 2: Python Script

To run the training as a script, use:

```bash
python gnn_training.py --select_method order --select_gene_num 970 --graph cos50_Lfull_Nposneg --lr 0.0005 --gpu 0 --loss L1

python gnn_training.py --select_method greedy_forward --select_gene_num 108 --graph cos50_Lfull_Nposneg --lr 0.0005 --gpu 0 --loss L1
```

### Training the MLP Model

To train the MLP model, use:

Open `mlp_training.ipynb` with Jupyter Notebook or JupyterLab, and run the cells in order.  

### Training the SwinIR Model

To train the SwinIR model, move to the `swinir/` directory.

```bash
cd swinir
```

Then, follow the instructions(`readme.md`) in `swinir/`.


## Evaluating the Models

After running either `gnn_training.py`, `gnn_training.ipynb` or `mlp_training.ipynb`, the inferred gene expression profiles for the test set will be saved in the `prediction/` directory.

To evaluate the prediction performance, you can run the `evaluation.ipynb`.

### Evaluation with Intermediate Results

If you want to generate intermediate results(e.g. level4 data), 
place the inferred gene expression profiles files in `evaluation/level3` and run the cells in `metric_comparison.ipynb`.

Note that `metadata/model_summary.txt` should be properly edited, before runing `metric_comparison.ipynb`.

Please refer to the evaluation instructions(`evaluation/readme.md`) for the details.
