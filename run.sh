#!/bin/bash

# run.sh

# source activate ret_env

# DATASETS='oxford5k,paris6k,roxford5k,rparis6k'
DATASETS='oxford5k,paris6k'

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'resnet101-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale