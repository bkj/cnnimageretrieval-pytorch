import os
import sys

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from tqdm import tqdm
from torch.autograd import Variable

import torchvision

from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC
from cirtorch.layers.normalization import L2N
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

POOLING = {
    'mac'  : MAC,
    'spoc' : SPoC,
    'gem'  : GeM,
    'rmac' : RMAC,
}

WHITENING = {
    'alexnet-gem'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'vgg16-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'resnet101-gem' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
}

OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
}


class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, pool, whiten, meta):
        super().__init__()
        
        self.features = nn.Sequential(*features)
        self.pool     = pool
        self.whiten   = whiten
        self.norm     = L2N()
        self.meta     = meta
    
    def forward(self, x):
        # features -> pool -> norm
        x = self.features(x)
        x = self.pool(x)
        x = self.norm(x)
        
        x = x.squeeze(-1).squeeze(-1)
        
        # if whiten exist: whiten -> norm
        if self.whiten is not None:
            x = self.norm(self.whiten(x))
            
        # permute so that it is Dx1 column vector per image (DxN if many images)
        x = x.permute(1, 0)
        return x

    def __repr__(self):
        tmpstr = super().__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(
        model='resnet101',
        pooling='gem',
        whitening=False, 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        pretrained=True,
    ):
    
    # --
    # Get feature extractor
    
    if pretrained and (model not in FEATURES):
        net_in = getattr(torchvision.models, model)(pretrained=True) # pretrained on imagenet
    else:
        net_in = getattr(torchvision.models, model)(pretrained=False) # will load weights later
    
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if model.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif model.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif model.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif model.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif model.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown model: {}!'.format(model))
    
    pool = POOLING[pooling]()
    dim  = OUTPUT_DIM[model]
    
    # initialize whitening
    whiten = None
    if whitening:
        w = '{}-{}'.format(model, pooling)
        whiten = nn.Linear(dim, dim, bias=True)
        if w in WHITENING:
            print(">> {}: for '{}' custom computed whitening '{}' is used"
                .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
            whiten_dir = os.path.join(get_data_root(), 'whiten')
            whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
        else:
            print(">> {}: for '{}' there is no whitening computed, random weights are used"
                .format(os.path.basename(__file__), w))
    
    # create a generic image retrieval network
    net = ImageRetrievalNet(
        features=features,
        pool=pool, 
        whiten=whiten, 
        meta={
            'architecture' : model,
            'pooling'      : pooling,
            'whitening'    : whitening,
            'outputdim'    : dim,
            'mean'         : mean,
            'std'          : std,
        }
    )
    
    # initialize features with custom pretrained network if needed
    if pretrained and model in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), model, os.path.basename(FEATURES[model])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[model], model_dir=model_dir))
        
    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    _ = net.cuda()
    _ = net.eval()
    
    # creating dataset loader
    dataset    = ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # Why?
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    num_regions = 15
    wrong_size = 0
    
    vecs = torch.zeros(net.meta['outputdim'], num_regions * len(images))
    for i, img in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = Variable(img.cuda())
        
        if len(ms) == 1:
            tmp = extract_ss(net, img)
            if tmp.shape[1] != num_regions:
                wrong_size += 1
            
            _num_regions = min(num_regions, tmp.shape[1])
            vecs[:, (num_regions * i):(num_regions * i + _num_regions)] = tmp[:,:_num_regions]
        else:
            raise NotImplemented
            # vecs[:, i] = extract_ms(net, img, ms, msp)
    
    print('wrong_size=%d' % wrong_size, file=sys.stderr)
    return vecs


def extract_ss(net, img):
    return net(img).cpu().data.squeeze()


def extract_ms(net, img, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    
    for s in ms: 
        if s == 1:
            img_t = img.clone()
        else:
            size = (int(img.size(-2) * s), int(img.size(-1) * s))
            img_t = nn.functional.upsample(img, size=size, mode='bilinear')
        
        v += net(img_t).pow(msp).cpu().data.squeeze()
    
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()
    
    return v