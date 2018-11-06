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
    ('vgg16-f', 'gem', 'retrievalSfM120k') : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    ('resnet101-f', 'gem', 'retrievalSfM120k') : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names  = ['oxford5k,paris6k', 'roxford5k,rparis6k', 'oxford5k,paris6k,roxford5k,rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str)
    parser.add_argument('--pooling',   type=str)
    parser.add_argument('--whitener',  type=str)
    parser.add_argument('--expansion', type=str)
    
    parser.add_argument('--db', type=str)
    parser.add_argument('--query', type=str)
    
    parser.add_argument('--image-size', default=1024, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # Download datasets
    
    download_train('./data/')
    download_test('./data/')
    
    # --
    # Load network
    
    net_desc = (args.model, args.pooling, args.whitener)
    
    net = init_network(
        model = args.
    )
    
    
    if args.network_path is not None:
        net_name = args.network_path + '_pre'
        
        print(f">> Loading network: {args.network_path}", file=sys.stderr)
        if args.network_path in PRETRAINED:
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
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
        
        print(net.meta_repr(), file=sys.stderr)
    
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
        
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(
            model     = offtheshelf[0],
            pooling   = offtheshelf[1],
            whitening = offtheshelf_whiten,
        )
        print(">>>> loaded network: ")
        print(net.meta_repr())
    
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
            std  = net.meta['std']
        )
    ])
    
    # --
    # Compute or load whitening
    
    Lw = None
    if args.whitening is not None:
        start = time()
        
        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            Lw = net.meta['Lw'][args.whitening]
            Lw = Lw['ms'] if args.multiscale else Lw['ss']
        else:
            print('>> {}: Learning whitening...'.format(args.whitening))
            
            # loading db
            db_root  = os.path.join(get_data_root(), 'train', args.whitening)
            ims_root = os.path.join(db_root, 'ims')
            db_fn    = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
            
            db = pickle.load(open(db_fn, 'rb'))
            images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
            
            whitening_path = f'./results/whitening/{net_name}__{args.whitening}__ms{args.multiscale}.npy'
            # !! This is going to be wrong if we change any params to `extract_vectors`
            if os.path.exists(whitening_path):
                print('\t>> {}: Loading...'.format(args.whitening))
                wvecs = np.load(whitening_path)
            else:
                print('\t>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp).numpy()
                np.save(whitening_path, wvecs)
            
            print('\t>> {}: Learning...'.format(args.whitening))
            t = time()
            m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
            Lw = {'m': m, 'P': P}
            print('\t\t elapsed=%f' % (time() - t))
            
            
        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time()-start)))
    
    # --
    # Run on datasets
    
    for dataset in args.datasets.split(','): 
        start = time()
        
        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        
        vecs_path  = f'./results/vecs/{dataset}__{net_name}__ms{args.multiscale}.npy'
        qvecs_path = f'./results/qvecs/{dataset}__{net_name}__ms{args.multiscale}.npy'
        if os.path.exists(vecs_path):
            print(f'>> loading {dataset} features')
            vecs  = np.load(vecs_path)
            qvecs = np.load(qvecs_path)
        else:
            print(f'>> computing {dataset} features')
            images  = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs    = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
            
            vecs  = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp).numpy()
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp).numpy()
            np.save(vecs_path, vecs)
            np.save(qvecs_path, qvecs)
        
        # Performance w/o whitening
        scores = np.dot(vecs.T, qvecs)
        ranks  = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset + '                     ', ranks, cfg['gnd'])
        
        # Performance w/ whitening
        if args.whitening:
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks  = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten            ', ranks, cfg['gnd'])
        
            # Query expansion (alpha_qe)
            # !! Check that this is right
            n     = 50
            alpha = 3
            score_ranks = -np.sort(-scores, axis=0)
            
            exp_vecs = vecs_lw[:,ranks[:n]]
            exp_vecs *= np.expand_dims(score_ranks[:n], 0) ** alpha
            exp_vecs = exp_vecs.sum(axis=1)
            
            qexp_vecs    = (qvecs_lw + exp_vecs) / (score_ranks[:n].sum(axis=0) + 1)
            scores       = np.dot(vecs_lw.T, qexp_vecs)
            ranks        = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten + expansion', ranks, cfg['gnd'])
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time() - start)))
        print('--')
