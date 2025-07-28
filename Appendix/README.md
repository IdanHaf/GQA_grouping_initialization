# Appendix Code

## Introduction
This folder contains the code we used to compare the training process of the mean pooling initialization and random initalization.

## Usage
For the PCA experiment we used the Google Colab environment.
For the CKA experiment we ran the code on the Lambda server, initially it requires running the `CKA_experiment.ipynb` file for generating the data for the plots, then the plots were created through the `cka_plots.ipynb`.

## Files
* `CKA_experiment.ipynb`- This notebook extract the outputs of the random initialized model and mean-pooled initialized model for the CKA plots.
* `cka_plots.ipynb` - This notebook uses the data from the `CKA_experiment.ipynb` file to generate the required plots.
* `GQA_utils.py` - This file contains the utilities functions for replacing the models with GQA and for generating the plots.
* `PCA_experiment.ipynb` - This notebook generates the PCA plots of the random initialized model and mean-pooled initialized.