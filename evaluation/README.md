# Metric_comparison.ipynb

This notebook contains code for evaluating and comparing the performance of different models.

## Purpose
To assess model performance using standardized evaluation metrics across different prediction results.

## Recommandation
- It is recommended to convert the inferred results to a NumPy array and store them as a `.pkl` (pickle) file for compatibility with the evaluation script.

- Before running model evaluation, please ensure that the **model summary file** (which includes metadata for each inferred file) is properly configured.

  - The fields `"feature_selection_method"` and `"num_landmarks"` must follow the predefined format.
  - The filename must exactly match the corresponding `.pkl` file name.
  
| Filename                | Model  | Feature_Selection_Method | Num_of_Landmarks  |
|-------------------------|--------|---------------------------|--------------------|
| GT.pkl                  | GT     | all                       | 978                |
| MLP_random_324.pkl      | MLP    | random                    | 324                |
| KNR_LASSO_108.pkl       | KNR    | LASSO                     | 108                |
| LR_greedy_108.pkl       | LR     | greedy                    | 108                |
| GNN_greedy_10.pkl       | GNN    | greedy                    | 10                 |
| SwinIR_random_486.pkl   | SwinIR | random                    | 486                |


- Make sure that the inferred `.pkl` files to be evaluated are placed in the `level3/` directory under the working directory.

## Evaluation Metrics Included
- **Overall Error**  
  Measures absolute prediction accuracy (e.g., RMSE, MAE).

- **Spearman Correlation Coefficient (SCC)**  
  Assesses the rank correlation between predicted and true expression values.

- **Pearson Correlation Coefficient (PCC)**  
  Assesses the correlation between predicted and true expression values.

- **Recall**  
  Retrieves genes with correlation above the recall threshold derived from the null distribution

- **Gene ranking correlation**  
  Unlike the conventional sample-wise Spearman correlation, this metric evaluates gene-wise ranking consistency by comparing the predicted and ground truth rankings across genes.
  It is computed using the rank-transformed dataset obtained through the level3-to-level4 conversion process.

## Input
- Model prediction results and corresponding ground truth values (in `.pkl` format)

## Output
- Summary table of evaluation metrics for each model

## Usage
Run the notebook using Jupyter:
