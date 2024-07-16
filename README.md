# RepCon
Implementation for paper Co-modeling the Sequential and Graphical Routes for Peptide Representation Learning

## Usage Overview

![Static Badge](https://img.shields.io/badge/CUDA-11.7-green)
![Static Badge](https://img.shields.io/badge/Python-3.7.4-red)
![Static Badge](https://img.shields.io/badge/PyTorch-1.13.1-blue)

## Training
To train a RepCon model on the example dataset AP, please use the following command:
```
python methods/co-modeling\ contrastive/main.py --dataset AP
```
Make sure the hyperparameter ```args.mode``` is set as 'train' before a trained model has been stored.

The predictive results can be found in the **'results'** folder in the root directory.
## Important hyperparameters
```args.seq_lr``` the learning rate of the sequential encoder & predictor.

```args.graph_lr``` the learning rate of the graphical encoder & predictor.

```args.nce_weight``` the weight which balance the supervised loss and the contrastive loss.

The other hyperparameters are relatively insensitive to downstream tasks, and the user can keep the default settings.
