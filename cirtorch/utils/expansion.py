import sys
import fbpca
import numpy as np
from cirtorch.utils.evaluate import compute_map_and_print

try:
    import faiss
    FAISS_ENABLED = True
except:
    print('could not import faiss')
    FAISS_ENABLED = False

def run_query_simple(vecs, qvecs):
    scores = np.dot(vecs.T, qvecs)
    
    if num_regions > 1:
        scores = agg_regions(
            scores,
            num_queries = int(qvecs.shape[1] / num_regions),
            num_images  = int(vecs.shape[1] / num_regions),
            num_regions = num_regions,
        )
    
    ranks  = np.argsort(-scores, axis=0)
    return ranks


def _numpy_make_graph(vecs, n_neighbors, gamma, symmetric=True):
    print('_numpy_make_graph', file=sys.stderr)
    
    sim = vecs.T.dot(vecs)
    sim = sim.clip(min=0)
    np.fill_diagonal(sim, 0)
    thresh = np.sort(sim, axis=0)[-n_neighbors].reshape(1, -1)
    sim[sim < thresh] = 0
    
    sim = sim ** gamma
    if symmetric:
        sim = np.minimum(sim, sim.T)
    
    return sim

def _faiss_make_graph(vecs, n_neighbors, gamma, symmetric=True):
    print('_faiss_make_graph', file=sys.stderr)
    
    """
        Should be substantially faster than numpy version above
        Still brute-force, but multithreaded
    """
    assert symmetric == True
    
    tmp = vecs.T.astype(np.float32)
    tmp = np.ascontiguousarray(tmp)
    
    findex = faiss.IndexFlatIP(tmp.shape[1])
    findex.add(tmp)
    
    D, I = findex.search(tmp, n_neighbors + 1)
    D, I = D[:,1:], I[:,1:]
    
    rows = np.repeat(np.arange(tmp.shape[0]), n_neighbors)
    cols = np.hstack(I)
    vals = np.hstack(D)
    
    sim = np.zeros((tmp.shape[0], tmp.shape[0]))
    sim[(rows, cols)] = vals
    if symmetric:
        sim = np.minimum(sim, sim.T)
    
    return sim


def run_query_diffusion(vecs, qvecs, n_neighbors=50, qn_neighbors=10, dim=1024, alpha=0.99, gamma=1):
    
    if FAISS_ENABLED:
        W = _faiss_make_graph(vecs, n_neighbors, gamma, symmetric=True)
    else:
        W = _numpy_make_graph(vecs, n_neighbors, gamma, symmetric=True)
    
    D = W.sum(axis=1)
    D[D == 0] = 1e-6
    D = np.diag(D ** -0.5)
    S = D.dot(W).dot(D)
    
    print('eigens')
    S = (S + S.T) / 2 # Ensure symmetric (fix numerical precision issues)
    eigval, eigvec = fbpca.eigens(S, k=dim, n_iter=20)
    h_eigval = 1 / (1 - alpha * eigval)
    Q        = eigvec.dot(np.diag(h_eigval)).dot(eigvec.T)
    
    # Make query
    print('sort')
    ysim    = vecs.T.dot(qvecs)
    ythresh = np.sort(ysim, axis=0)[-qn_neighbors].reshape(1, -1)
    ysim[ysim < ythresh] = 0
    ysim = ysim ** gamma
    
    # Run search
    print('Q.dot')
    scores = Q.dot(ysim)
    # <<
    if num_regions > 1:
        scores = agg_regions(
            scores,
            num_queries = int(qvecs.shape[1] / num_regions),
            num_images  = int(vecs.shape[1] / num_regions),
            num_regions = num_regions,
        )
    # >>
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

