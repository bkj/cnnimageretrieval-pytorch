import sys
import fbpca
import numpy as np
from scipy import sparse
from cirtorch.utils.evaluate import compute_map_and_print

try:
    import faiss
    FAISS_ENABLED = True
except:
    print('could not import faiss')
    FAISS_ENABLED = False


def agg_regions(scores, num_queries, num_images, num_regions):
    scores = scores.T
    scores = scores.reshape(num_queries * num_regions, num_images, num_regions).max(axis=2)
    scores = scores.reshape(num_queries, num_regions, num_images).sum(axis=1)
    scores = scores.T
    return scores


def run_query_simple(vecs, qvecs, num_regions=1):
    scores = np.dot(vecs.T, qvecs)
    
    if num_regions > 1:
        scores = agg_regions(
            scores,
            num_queries = int(qvecs.shape[1] / num_regions),
            num_images  = int(vecs.shape[1] / num_regions),
            num_regions = num_regions,
        )
    
    ranks = np.argsort(-scores, axis=0)
    return ranks


def _numpy_make_graph(vecs, n_neighbors, gamma):
    sim = vecs.T.dot(vecs)
    sim = sim.clip(min=0)
    np.fill_diagonal(sim, 0)
    thresh = np.sort(sim, axis=0)[-n_neighbors].reshape(1, -1)
    sim[sim < thresh] = 0
    
    sim = sim ** gamma
    
    # make mutual knn graph
    sim = np.minimum(sim, sim.T)
    
    # symmetric normalization
    d = W.sum(axis=1)
    d[d == 0] = 1e-6
    d = d ** -0.5
    
    D = np.diag(d)
    S = D.dot(W).dot(D)
    S = (S + S.T) / 2
    
    return sim


def _faiss_make_graph(vecs, n_neighbors, gamma):
    num_vecs = vecs.shape[1]
    
    tmp = vecs.T.astype(np.float32)
    tmp = np.ascontiguousarray(tmp)
    
    findex = faiss.IndexFlatIP(tmp.shape[1])
    findex.add(tmp)
    
    D, I = findex.search(tmp, n_neighbors + 1)
    D, I = D[:,1:], I[:,1:]
    
    rows = np.repeat(np.arange(num_vecs), n_neighbors)
    cols = np.hstack(I)
    vals = np.hstack(D)
    sim  = sparse.csr_matrix((vals, (rows, cols)), shape=(num_vecs, num_vecs))
    
    # make mutual knn graph
    sim = sim.minimum(sim.T)
    
    # Symmetric normalization
    d = np.asarray(W.sum(axis=1)).squeeze()
    d[d == 0] = 1e-6
    d = d ** -0.5
    
    D = sparse.eye(vecs.shape[1]).tocsr()
    D.data *= d
    S = D.dot(W).dot(D)
    S = (S + S.T) / 2
    
    return S


def run_query_diffusion(vecs, qvecs, n_neighbors=50, qn_neighbors=10, dim=1024, 
    alpha=0.99, gamma=1, num_regions=1, n_iter=20):
    
    # n_iter is important
    
    print("FAISS_ENABLED=%d" % FAISS_ENABLED)
    print("num_regions=%d" % num_regions)
    
    _make_graph = _faiss_make_graph if FAISS_ENABLED else _numpy_make_graph
    print('construct knn graph')
    S = _make_graph(vecs, n_neighbors=n_neighbors, gamma=gamma)
    
    print('compute eigenvalues')
    eigval, eigvec = fbpca.eigens(S, k=dim, n_iter=n_iter)
    h_eigval = 1 / (1 - alpha * eigval)
    
    print('precompute U_bar')
    U_bar = eigvec.dot(np.diag(h_eigval)) # Very big dense matrix.  In paper, they make this sparse.
    
    # Make query
    print('L2 search queries')
    ysim    = vecs.T.dot(qvecs)
    ythresh = np.sort(ysim, axis=0)[-qn_neighbors].reshape(1, -1)
    ysim[ysim < ythresh] = 0
    ysim = ysim ** gamma
    
    if num_regions > 1:
        print('aggregate ysim')
        num_queries = int(qvecs.shape[1] / num_regions)
        num_images  = int(vecs.shape[1] / num_regions)
        ysim = ysim.reshape(num_images * num_regions, num_queries, num_regions).sum(axis=-1)
    
    # Run search
    print('diffusion query')
    scores = U_bar.dot(eigvec.T.dot(ysim))
    
    if num_regions > 1:
        print('aggregate results')
        scores = scores.reshape(num_images, num_regions, num_queries).sum(axis=1)
    
    ranks = np.argsort(-scores, axis=0)
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
    
    ranks = np.argsort(-scores, axis=0)
    return ranks

