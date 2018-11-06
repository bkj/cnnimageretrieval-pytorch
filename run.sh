#!/bin/bash

# run.sh

# source activate ret_env

# DATASETS='oxford5k,paris6k,roxford5k,rparis6k'
DATASETS='oxford5k,paris6k'

# --
# No finetuning

# CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
#     --network-offtheshelf 'resnet101-gem' \
#     --datasets $DATASETS \
#     --whitening 'retrieval-SfM-120k' \
#     --multiscale

# # >> oxford5k: mAP 49.94
# # >> oxford5k + whiten: mAP 76.18

# # >> paris6k: mAP 70.46
# # >> paris6k + whiten: mAP 89.33

# --
# Finetuning

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-resnet101-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> oxford5k: mAP 81.02
# >> oxford5k + whiten: mAP 88.15

# >> paris6k: mAP 87.77
# >> paris6k + whiten: mAP 92.53

# --
# VGG

# CUDA_VISIBLE_DEVICES=6 python -m cirtorch.examples.test \
#     --network-offtheshelf 'vgg16-rmac' \
#     --datasets $DATASETS \
#     --whitening 'retrieval-SfM-120k' \
#     --multiscale

# --
# TODO

# - query expansion
#  - nQE
#  - alphaQE
#  - manifold