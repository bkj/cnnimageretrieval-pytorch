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

from cirtorch.utils.expansion import run_query_simple, run_query_alpha_qe, run_query_diffusion

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names  = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

def parse_args():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument('--network-path', '-npath',
                        help='network path, destination where network is saved')
    
    group.add_argument('--network-offtheshelf', '-noff',
                        help='network off-the-shelf, in the format ARCHITECTURE-POOLING or ARCHITECTURE-POOLING-whiten,' + 
                        ' examples: resnet101-gem | resnet101-gem-whiten')
    
    parser.add_argument('--datasets', '-d', default='oxford5k,paris6k',
                       help='comma separated list of test datasets: ' + 
                            ' | '.join(datasets_names) + 
                            ' (default: oxford5k,paris6k)')
    
    parser.add_argument('--image-size', '-imsize', default=1024, type=int,
                        help='maximum size of longer image side used for testing (default: 1024)')
    
    parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true',
                        help='use multiscale vectors for testing')
    
    parser.add_argument('--whitening', '-w', default=None, choices=whitening_names,
                        help='dataset used to learn whitening for testing: ' + 
                            ' | '.join(whitening_names) + 
                            ' (default: None)')
    
    parser.add_argument('--diffusion', action="store_true")
    parser.add_argument('--alpha-qe', action="store_true")
    
    return parser.parse_args()

# >>
def load_landmark(dataset, dir_main):
    cfg     = configdataset(dataset, 'data/test')
    images  = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
    bbxs    = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
    
    return {
        "images"  : images,
        "qimages" : qimages,
        "bbxs"    : bbxs,
        "gnd"     : cfg['gnd'],
    }

from scipy.io import loadmat
def load_instre(root='~/data/instre/'):
    x = loadmat('data/test/instre/gnd_instre.mat')
    
    images  = [xx[0] for xx in list(x['imlist'].squeeze())]
    images  = [os.path.expanduser(os.path.join(root, p)) for p in images]
    
    qimages = [xx[0] for xx in list(x['qimlist'].squeeze())]
    qimages = [os.path.expanduser(os.path.join(root, p)) for p in qimages]
    
    gnd = [{
        "bbx"  : tuple(g[1].squeeze()),
        "ok"   : g[0].squeeze(),
        # "junk" : [],
    } for g in x['gnd'][0]]
    
    return {
        "images"  : images,
        "qimages" : qimages,
        "bbxs"    : [g['bbx'] for g in gnd],
        "gnd"     : gnd,
    }
# <<


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # Download datasets
    
    download_train('./data')
    download_test('./data')
    
    # --
    # Load network
    
    if args.network_path is not None:
        net_name = args.network_path + '_pre'
        
        if args.network_path in PRETRAINED:
            state = load_url(PRETRAINED[args.network_path], model_dir='./data/networks')
        else:
            state = torch.load(args.network_path)
        
        net = init_network(
            model      = state['meta']['architecture'],
            pooling    = state['meta']['pooling'],
            whitening  = state['meta']['whitening'],
            mean       = state['meta']['mean'],
            std        = state['meta']['std'],
            pretrained = False,
        )
        
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
    
    elif args.network_offtheshelf is not None:
        net_name = args.network_offtheshelf + '_ots'
        
        offtheshelf = args.network_offtheshelf.split('-')
        if len(offtheshelf) == 3:
            if offtheshelf[2] == 'whiten':
                offtheshelf_whiten = True
            else:
                raise(RuntimeError("Incorrect format of the off-the-shelf network. Examples: resnet101-gem | resnet101-gem-whiten"))
        else:
            offtheshelf_whiten = False
        
        net = init_network(
            model     = offtheshelf[0],
            pooling   = offtheshelf[1],
            whitening = offtheshelf_whiten,
        )
    
    print(net.meta_repr(), file=sys.stderr)
    
    # --
    # Set params
    
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]
    
    net.cuda()
    net.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = net.meta['mean'],
            std  = net.meta['std'],
        )
    ])
    
    # --
    # Compute or load whitening
    
    Lw = None
    if args.whitening is not None:
        start = time()
        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            Lw = net.meta['Lw'][args.whitening]
            Lw = Lw['ms'] if args.multiscale else Lw['ss']
            
        else:
            print('compute whitening', file=sys.stderr)
            
            # loading db
            db_root  = os.path.join('data/train', args.whitening)
            ims_root = os.path.join(db_root, 'ims')
            db_fn    = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
            
            db = pickle.load(open(db_fn, 'rb'))
            images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
            
            whitening_path = f'./results/whitening/{net_name}__{args.whitening}__ms{args.multiscale}.npy'
            if os.path.exists(whitening_path):
                wvecs = np.load(whitening_path)
            else:
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp).numpy()
                np.save(whitening_path, wvecs)
            
            t = time()
            m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
            Lw = {'m': m, 'P': P}
    
    # --
    # Run on datasets
    
    for dataset in args.datasets.split(','): 
        if dataset == 'instre':
            cfg = load_instre()
        else:
            cfg = load_landmark(dataset, 'data/test')
        
        vecs_path  = f'./results/vecs/{dataset}__{net_name}__ms{args.multiscale}.npy'
        qvecs_path = f'./results/qvecs/{dataset}__{net_name}__ms{args.multiscale}.npy'
        if os.path.exists(vecs_path):
            vecs  = np.load(vecs_path)
            qvecs = np.load(qvecs_path)
        else:
            print('compute features', file=sys.stderr)
            vecs  = extract_vectors(net, cfg['images'], args.image_size, transform, ms=ms, msp=msp).numpy()
            np.save(vecs_path, vecs)
            
            qvecs = extract_vectors(net, cfg['qimages'], args.image_size, transform, bbxs=cfg['bbxs'], ms=ms, msp=msp).numpy()
            np.save(qvecs_path, qvecs)
        
        print('running queries')
        
        # Performance w/o whitening
        simple_ranks = run_query_simple(vecs, qvecs)
        compute_map_and_print(dataset + '                     ', simple_ranks, cfg['gnd'])
        
        # Performance w/ whitening
        if args.whitening:
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            
            whitened_ranks  = run_query_simple(vecs_lw, qvecs_lw)
            compute_map_and_print(dataset + ' + whiten            ', whitened_ranks, cfg['gnd'])
            
            if args.alpha_qe:
                alpha_qe_ranks = run_query_alpha_qe(vecs_lw, qvecs_lw)
                compute_map_and_print(dataset + ' + whiten + expansion', alpha_qe_ranks, cfg['gnd'])
            
            if args.diffusion:
                diffusion_ranks = run_query_diffusion(vecs_lw, qvecs_lw)
                compute_map_and_print(dataset + ' + whiten + diffusion', diffusion_ranks, cfg['gnd'])
        
        print('--')
