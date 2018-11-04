#!/bin/bash

# run.sh

python3 -m cirtorch.examples.test \
    --gpu-id '0' \
    --network-offtheshelf 'resnet101-gem' \
    --datasets 'oxford5k,paris6k,roxford5k,rparis6k' \
    --whitening 'retrieval-SfM-120k' \
    --multiscale