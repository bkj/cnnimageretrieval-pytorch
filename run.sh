#!/bin/bash

# run.sh

# source activate ret_env

# DATASETS='oxford5k,paris6k,roxford5k,rparis6k'
DATASETS='oxford5k,paris6k'

# --
# FT VGG16 + GEM

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-vgg16-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> retrieval-SfM-120k: Whitening is precomputed, loading it...
# >> retrieval-SfM-120k: elapsed time: 0s
# >> loading oxford5k features
# >> oxford5k                     : mAP 82.49
# >> oxford5k + whiten            : mAP 87.21
# >> oxford5k + whiten + expansion: mAP 90.72
# >> oxford5k: elapsed time: 0s
# --
# >> loading paris6k features
# >> paris6k                     : mAP 82.21
# >> paris6k + whiten            : mAP 87.76
# >> paris6k + whiten + expansion: mAP 92.09
# >> paris6k: elapsed time: 1s
# --

# --
# FT ResNet + GEM

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-resnet101-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> retrieval-SfM-120k: Whitening is precomputed, loading it...
# >> retrieval-SfM-120k: elapsed time: 0s
# >> loading oxford5k features
# >> oxford5k                     : mAP 81.02
# >> oxford5k + whiten            : mAP 88.15
# >> oxford5k + whiten + expansion: mAP 90.29
# >> oxford5k: elapsed time: 1s
# --
# >> loading paris6k features
# >> paris6k                     : mAP 87.77
# >> paris6k + whiten            : mAP 92.53
# >> paris6k + whiten + expansion: mAP 95.41
# >> paris6k: elapsed time: 1s
# --

# --
# OTS VGG + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'vgg16-rmac' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> retrieval-SfM-120k: elapsed time: 7s
# >> loading oxford5k features
# >> oxford5k                     : mAP 52.15
# >> oxford5k + whiten            : mAP 66.65
# >> oxford5k + whiten + expansion: mAP 71.84
# >> oxford5k: elapsed time: 0s
# --
# >> loading paris6k features
# >> paris6k                     : mAP 74.94
# >> paris6k + whiten            : mAP 83.84
# >> paris6k + whiten + expansion: mAP 88.58
# >> paris6k: elapsed time: 1s
# --

# --
# OTS ResNet + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'resnet101-rmac' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> retrieval-SfM-120k: elapsed time: 36s
# >> loading oxford5k features
# >> oxford5k                     : mAP 56.15
# >> oxford5k + whiten            : mAP 78.68
# >> oxford5k + whiten + expansion: mAP 80.92
# >> oxford5k: elapsed time: 1s
# --
# >> loading paris6k features
# >> paris6k                     : mAP 76.17
# >> paris6k + whiten            : mAP 89.53
# >> paris6k + whiten + expansion: mAP 92.35
# >> paris6k: elapsed time: 1s
# --

# --
# OTS ResNet + GEM

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'resnet101-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --multiscale

# >> loading oxford5k features
# >> oxford5k                     : mAP 49.94
# >> oxford5k + whiten            : mAP 76.18
# >> oxford5k + whiten + expansion: mAP 77.70
# >> oxford5k: elapsed time: 1s
# --
# >> loading paris6k features
# >> paris6k                     : mAP 70.46
# >> paris6k + whiten            : mAP 89.33
# >> paris6k + whiten + expansion: mAP 92.12
# >> paris6k: elapsed time: 1s
# --

# --
# Notes:
# - All of these use supervised whitening, trained on a labeled
#   building landmark dataset