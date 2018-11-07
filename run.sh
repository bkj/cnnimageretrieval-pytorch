#!/bin/bash

# run.sh

# source activate ret_env

DATASETS='oxford5k,paris6k,roxford5k,rparis6k'
# DATASETS='oxford5k,paris6k'

# --
# FT VGG16 + GEM

CUDA_VISIBLE_DEVICES=6 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-vgg16-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --alpha-qe \
    --diffusion \
    --multiscale

# running queries
# >> oxford5k                     : mAP 82.49
# >> oxford5k + whiten            : mAP 87.21
# >> oxford5k + whiten + expansion: mAP 90.72
# >> oxford5k + whiten + diffusion: mAP 89.87
# --
# running queries
# >> paris6k                     : mAP 82.21
# >> paris6k + whiten            : mAP 87.76
# >> paris6k + whiten + expansion: mAP 92.09
# >> paris6k + whiten + diffusion: mAP 94.56
# --

# --
# FT ResNet + GEM

CUDA_VISIBLE_DEVICES=6 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-resnet101-gem' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --alpha-qe \
    --diffusion \
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
    --diffusion \
    --alpha-qe \
    --multiscale

# >> oxford5k                     : mAP 52.15
# >> oxford5k + whiten            : mAP 66.65
# >> oxford5k + whiten + expansion: mAP 71.84
# >> oxford5k + whiten + diffusion: mAP 79.08
# --
# running queries
# >> paris6k                     : mAP 74.94
# >> paris6k + whiten            : mAP 83.84
# >> paris6k + whiten + expansion: mAP 88.58
# >> paris6k + whiten + diffusion: mAP 89.88
# --

# --
# OTS ResNet + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'resnet101-rmac' \
    --datasets $DATASETS \
    --whitening 'retrieval-SfM-120k' \
    --diffusion \
    --multiscale

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
# OTS VGG + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'vgg16-rmac' \
    --datasets instre \
    --whitening 'retrieval-SfM-120k' \
    --diffusion \
    --alpha-qe \
    --multiscale

# >> instre                     : mAP 41.86
# >> instre + whiten            : mAP 51.07
# >> instre + whiten + expansion: mAP 58.18
# >> instre + whiten + diffusion: mAP 70.46
# --

# --
# FT VGG + GEM

CUDA_VISIBLE_DEVICES=6 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-vgg16-gem' \
    --datasets instre \
    --whitening 'retrieval-SfM-120k' \
    --alpha-qe \
    --diffusion \
    --multiscale

# >> instre                     : mAP 42.97
# >> instre + whiten            : mAP 54.72
# >> instre + whiten + expansion: mAP 61.17
# >> instre + whiten + diffusion: mAP 73.96
# --

# --
# OTS ResNet + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-offtheshelf 'resnet101-rmac' \
    --datasets instre \
    --whitening 'retrieval-SfM-120k' \
    --alpha-qe \
    --diffusion \
    --multiscale

# >> instre                     : mAP 42.44
# >> instre + whiten            : mAP 62.59
# >> instre + whiten + expansion: mAP 68.24
# >> instre + whiten + diffusion: mAP 78.08

# --
# FT ResNet + RMAC

CUDA_VISIBLE_DEVICES=7 python -m cirtorch.examples.test \
    --network-path 'retrievalSfM120k-resnet101-gem' \
    --datasets instre \
    --whitening 'retrieval-SfM-120k' \
    --alpha-qe \
    --diffusion \
    --multiscale

# >> instre                     : mAP 49.72
# >> instre + whiten            : mAP 67.83
# >> instre + whiten + expansion: mAP 72.64
# >> instre + whiten + diffusion: mAP 80.08

# !! Kindof strange, since this network is finetuned for out-of-domain data

# --
# Notes:
# - All of these use supervised whitening, trained on a labeled
#   building landmark dataset.  For INSTRE that's a bad idea.
