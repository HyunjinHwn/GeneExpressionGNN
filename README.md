# GeneExpressionGNN

GeneExpressionGNN is a project focused on leveraging graph neural networks (GNNs) for inferring patterns in gene expression data. 
This repository provides tools and models to explore relationships between genes and their expression profiles.

## Features

- maining code for gene expression prediction from L1000 data to RNAseq data.
- Implementation of graph neural network architectures.
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
    TBD

## Training the GNN Model

You can run this project using either the Python script or the Jupyter Notebook:


### Option 1: Jupyter Notebook

Open `gnn_training.ipynb` with Jupyter Notebook or JupyterLab, and run the cells in order.  

### Option 2: Python Script

To run the training as a script, use:

```bash
python gnn_training.py --select_method order --select_gene_num 970 --graph cos50_Lfull_Nposneg --lr 0.0005 --gpu 0 --loss L1

python gnn_training.py --select_method greedy_forward --select_gene_num 108 --graph cos50_Lfull_Nposneg --lr 0.0005 --gpu 0 --loss L1
```

## Training the MLP Model

To train the MLP model, use:

Open `mlp_training.ipynb` with Jupyter Notebook or JupyterLab, and run the cells in order.  

## Training the SwinIR Model

To train the SwinIR model, move to the `swinir/` directory.

```bash
cd swinir
```

Then, follow the instructions(`readme.md`) in `swinir/`.


## Evaluating the Models

After running either `gnn_training.py`, `gnn_training.ipynb` or `mlp_training.ipynb`, the inferred gene expression profiles for the test set will be saved in the `prediction/` directory.

To evaluate the prediction accuracy, you can run the evaluation script or notebook:

### Option 1: Jupyter Notebook

```bash
jupyter notebook evaluation.ipynb
```

### Option 2: Python Script
```bash
python evaluation.py --pred_path prediction/your_prediction_file.pt
```
