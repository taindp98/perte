# Welcome to perte
A fast way to build lots of loss functions (```fonction de perte``` in French) in Deep Learning.

## Features

- Currently support:
    - Triplet loss and its variance
    - Angular margin penalty losses
    - Recall loss

## Installation

To install with pip, use: `pip install perte`. If you install with pip,
you should install PyTorch first by following the PyTorch [installation
instructions](https://pytorch.org/get-started/locally/).

## Usages

### Simple Triplet loss
```python
import torch
from perte import TripletLoss

## Initialize loss function
loss_fnc = TripletLoss(
    alpha=0.5, 
    reduction="mean", 
    device=torch.device("cpu")
)

## Compute the loss value
anchor_embd = torch.randn(1, 10)    ## features' dim = 10
positive_embd = torch.randn(1, 10)    ## features' dim = 10
negative_embd = torch.randn(1, 10)    ## features' dim = 10
loss_value = loss_fnc(anchor_embd, positive_embd, negative_embd)
```

### Hard Mining Triplet loss
```python
import torch
from perte import OnlineTripletLoss
from perte import AllTripletSelector
from perte import BatchHardTripletSelector
from perte import HardestNegativeTripletSelector
from perte import RandomNegativeTripletSelector
from perte import SemihardNegativeTripletSelector

## Initialize loss function
loss_fnc = OnlineTripletLoss(
    triplet_selector=AllTripletSelector,
    margin=0.5,
    reduction="mean", 
    device=torch.device("cpu")
)

## Compute the loss value
anchor_embd = torch.randn(1, 10)    ## features' dim = 10
positive_embd = torch.randn(1, 10)    ## features' dim = 10
negative_embd = torch.randn(1, 10)    ## features' dim = 10
loss_value = loss_fnc(anchor_embd, positive_embd, negative_embd)
```
