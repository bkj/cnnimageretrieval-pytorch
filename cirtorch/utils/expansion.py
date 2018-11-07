import fbpca
import numpy as np
from cirtorch.utils.evaluate import compute_map_and_print

def run_query_simple(vecs, qvecs):
    scores = np.dot(vecs.T, qvecs)
    ranks  = np.argsort(-scores, axis=0)
    return ranks

def run_query_diffusion(vecs, qvecs, n_neighbors=50, qn_neighbors=10, dim=1024, alpha=0.99, gamma=1):
    # Make DB matrix
    sim = vecs.T.dot(vecs)
    sim = sim.clip(min=0)
    np.fill_diagonal(sim, 0)
    sim = sim ** gamma
    
    thresh = np.sort(sim, axis=0)[-n_neighbors].reshape(1, -1)
    sim[sim < thresh] = 0
    
    W = np.minimum(sim, sim.T)
    D = W.sum(axis=1)
    D[D == 0] = 1e-6
    D = np.diag(D ** -0.5)
    S = D.dot(W).dot(D)
    
    S = (S + S.T) / 2 # Fix numerical precision issues
    eigval, eigvec = fbpca.eigens(S, k=dim, n_iter=20)
    h_eigval = 1 / (1 - alpha * eigval)
    Q        = eigvec.dot(np.diag(h_eigval)).dot(eigvec.T)
    
    # Make query
    ysim    = vecs.T.dot(qvecs)
    ythresh = np.sort(ysim, axis=0)[-qn_neighbors].reshape(1, -1)
    ysim[ysim < ythresh] = 0
    ysim = ysim ** gamma
    
    # Run search
    scores = Q.dot(ysim)
    ranks  = np.argsort(-scores, axis=0)
    
    return ranks


def run_query_alpha_qe(vecs, qvecs, n=50, alpha=3):
    
    # !! Probably redundant
    oscores      = np.dot(vecs.T, qvecs)
    oranks       = np.argsort(-oscores, axis=0)
    score_oranks = -np.sort(-oscores, axis=0)
    
    exp_vecs = vecs[:,oranks[:n]]
    exp_vecs *= np.expand_dims(score_oranks[:n], 0) ** alpha
    exp_vecs = exp_vecs.sum(axis=1)
    
    qexp_vecs    = (qvecs + exp_vecs) / (score_oranks[:n].sum(axis=0) + 1)
    scores       = np.dot(vecs.T, qexp_vecs)
    ranks        = np.argsort(-scores, axis=0)
    
    return ranks

