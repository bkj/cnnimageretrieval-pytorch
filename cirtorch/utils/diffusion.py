import fbpca
import numpy as np
from cirtorch.utils.evaluate import compute_map_and_print

def run_diffusion(vecs, qvecs, n_neighbors=50, qn_neighbors=10, dim=1024, alpha=0.99, gamma=1):
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

