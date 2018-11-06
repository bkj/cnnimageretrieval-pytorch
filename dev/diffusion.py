#!/usr/bin/env python

"""
    test.py
"""

import argparse
import os
import sys
import math
import pickle
from time import time

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

network_path = 'retrievalSfM120k-vgg16-gem'
whitening    = 'retrieval-SfM-120k'
net_name     = network_path + '_pre'

state = load_url(PRETRAINED[network_path], model_dir=os.path.join(get_data_root(), 'networks'))

net = init_network(
    model      = state['meta']['architecture'],
    pooling    = state['meta']['pooling'],
    whitening  = state['meta']['whitening'],
    mean       = state['meta']['mean'],
    std        = state['meta']['std'],
    pretrained = False,
)

net.load_state_dict(state['state_dict'])
net.meta['Lw'] = state['meta']['Lw']

Lw = net.meta['Lw'][whitening]
Lw = Lw['ms']

dataset    = 'oxford5k'
multiscale = True

cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))

vecs_path  = f'./results/vecs/{dataset}__{net_name}__ms{multiscale}.npy'
qvecs_path = f'./results/qvecs/{dataset}__{net_name}__ms{multiscale}.npy'

vecs  = np.load(vecs_path)
qvecs = np.load(qvecs_path)

# --
# Performance w/o whitening

scores = np.dot(vecs.T, qvecs)
ranks  = np.argsort(-scores, axis=0)
compute_map_and_print(dataset + '                     ', ranks, cfg['gnd'])

# --
# Performance w/ whitening

vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
scores   = np.dot(vecs_lw.T, qvecs_lw)
ranks    = np.argsort(-scores, axis=0)
compute_map_and_print(dataset + ' + whiten            ', ranks, cfg['gnd'])

# --
# Performance w/ whitening + alphaQE

n     = 50
alpha = 3
scores      = np.dot(vecs_lw.T, qvecs_lw)
ranks       = np.argsort(-scores, axis=0)
score_ranks = -np.sort(-scores, axis=0)

exp_vecs = vecs_lw[:,ranks[:n]].copy()
exp_vecs *= np.expand_dims(score_ranks[:n], 0) ** alpha
exp_vecs = exp_vecs.sum(axis=1)

qexp_vecs    = (qvecs_lw + exp_vecs) / (score_ranks[:n].sum(axis=0) + 1)
scores       = np.dot(vecs_lw.T, qexp_vecs)
ranks        = np.argsort(-scores, axis=0)
compute_map_and_print(dataset + ' + whiten + expansion', ranks, cfg['gnd'])

# --
# fast spectral ranking

from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import fbpca

n_neighbors  = 50
qn_neighbors = 10
dim          = 1024
alpha        = 0.99
gamma        = 1

# Pairwise similarity matrix
sim = vecs_lw.T.dot(vecs_lw)

# Set values < 0 to 0
sim = sim.clip(min=0)

# Remove 1s on diagonal (to avoid self-loops)
np.fill_diagonal(sim, 0)

# Raise similarity to power of gamma
sim = sim ** gamma

# Find `n_neighbors` largest entry in sim
thresh = np.sort(sim, axis=0)[-n_neighbors].reshape(1, -1)

# Set values less than threshold to 0
sim[sim < thresh] = 0

# Make symmetric
W = np.minimum(sim, sim.T)

# Normalize W by node weights
D = W.sum(axis=1)
D[D == 0] = 1e-6           #  No division by zero
D = np.diag(D ** -0.5)

S = D.dot(W).dot(D)
S = (S + S.T) / 2          # Fix numerical precision issues

eigval, eigvec = fbpca.eigens(S, k=dim, n_iter=20)
h_eigval = 1 / (1 - alpha * eigval)
Q        = eigvec.dot(np.diag(h_eigval)).dot(eigvec.T)

# Make query
ysim    = vecs_lw.T.dot(qvecs_lw)
ythresh = np.sort(ysim, axis=0)[-qn_neighbors].reshape(1, -1)
ysim[ysim < ythresh] = 0
ysim = ysim ** gamma

scores = Q.dot(ysim)
ranks  = np.argsort(-scores, axis=0)
compute_map_and_print(dataset + ' + whiten + diffusion', ranks, cfg['gnd'])

