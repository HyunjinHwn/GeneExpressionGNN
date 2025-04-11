## Code structure
This code is used for image restoration using SwinIR, with slight modifications based on the original cszn/KAIR repository, adapted for our regression task.
This code is optimized for use in Google Colab.

**Code list**

**1. Folder: `KAIR_L1000`**

Contents in this directory are based on a copy of the GitHub repository [cszn/KAIR](https://github.com/cszn/KAIR), modified for the L1000–RNA-seq inference task.

Key files and subfolders include:

- `main_train_psnr.py`: Saves test images (in our case, inferred RNAseq images using valid-set L1000 inputs) and logs individual stats (error, PCC, SCC)

- `main_train_psnr_no_outputs.py`: Omits saving test images and logging individual stats (only logs overall stats)

- `main_test_swinir_mod.py`: Generates test images using the pre-trained model on L1000 inputs

- Subdirectories:

  - `data/`: Defines and creates datasets

  - `utils/`: Helper functions (e.g., loading/saving images, calculating statistics, option management)

  - `models/`: Model architecture, network definition, loss functions



**2. Code: `SwinIR_image_processing.ipynb`**

This script converts the input value data into training images for the SwinIR model.

If you want to reproduce our experiment using the **GSE92743** dataset, simply provide the required files and the number of landmark features to the function below:

```python
def produce_training_images(L1000_gctx, RNAseq_gctx, outpath, lmids, 
                            train=2500, valid=500, test=176):
```

Please note that SwinIR was only used for the random (ordered) input type in the ablation study.

Therefore, methods like LASSO or greedy selection are not compatible with this code.




**3. Code: `option.json`**

This JSON script is essential for running the `SwinIR_train.ipynb` notebook. 
 
Please make sure to check and configure this file before execution.

Additional explanations are included in the script as inline comments.  

It defines:

- Hyperparameters for the SwinIR model  

- File paths for input data and output results

To reproduce the experiment, just update the model path and the paths to the training and validation data in the script.




**4. Code: `SwinIR_train.ipynb`**

training model with option file and images.

```CLI 

!torchrun --nproc_per_node=1 --master_port=1234 /main_train_psnr.py --opt /option.json  --dist False
```



**5. Code: `SwinIR_test.ipynb`**

Use this script along with the trained model to generate high-resolution images from low-resolution inputs.  

After generation, make sure to convert the output images into value format using the `SwinIR_image_processing.ipynb` script.




## Steps to Run SwinIR

1. Convert value data into image sets using [`SwinIR_image_processing.ipynb`](./SwinIR_image_processing.ipynb).

2. Train your images with [`SwinIR_train.ipynb`](./SwinIR_train.ipynb) using the configuration provided in `option.json`.

3. Obtain your own trained model.

4. Run inference using [`SwinIR_test.ipynb`](./SwinIR_test.ipynb).  

   Note: The image indexing step (as in `SwinIR_image_processing.ipynb`) is internally included in the test script and will be re-executed as part of the process.







Written on 2024.12.06 by Harim Kim

modified on 2025.04.10 by Taehyun Kim

Contents in this directory is made from a copy of github repository cszn/KAIR, and modified to be suitable for L1000-RNA-seq inference task.
