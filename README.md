# GQA Grouping Initialization Via Head Similarity

## Abstract
Grouped-query attention (GQA) partitions attention heads into equally sized groups,
with each group sharing the same key-value head. This approach can drastically
speed up decoder inference with a minor quality degradation. To convert multi-
head attention (MHA) to GQA, attention heads are grouped sequentially, and each
group shares a mean-pooled key-value head. We propose an initialization strategy
for grouping the heads in a more informed manner. We show that our grouping
strategy achieves better results than the standard GQA grouping on decoder only
transformers.

## Install
Download the repository, and place the Transformers folder in "My Drive" folder on Goolgle drive.
All the packeges needed to install are part of the notebooks, and will be automaticlly installed when running the notebooks.

## Usage
> [!TIP]
> Recommended to use Google colab GPU runtime type.
Run the desired ipynb notebook in Google colab.

Note: some notebooks may need L4 GPU runtime type, which currently is not avialabe for free.

## Code structure
#### `./Pretrained_GQA_models`
   * Contain the code for creating figure 3 in the paper.
#### `./Project_OPT`
   * Most of the relevant code for the paper results.
#### `./Project_T5` 
   * Mostly Early code and expiriments to see the effect of the grouping on encoder-decoder type model.
