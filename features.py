from numpy import *
from scipy.sparse import dia_matrix


def ThreePropFeatures(W, y, sym=0):
    """
    Algorithm to generate 3Prop features (see paper).

    Parameters
    ----------
    W: sparse matrix, size = (n_nodes, n_nodes)
        Input netowrk/graph
    y:  array of shape = (n_nodes, )
        Target labels
    sym: 0/1
        Symmetric or asymmetric version of the algorithm

    Returns
    -------
    3Prop features, array of shape = (n_nodes, 3)
    """

    y[where(y != 1)[0]] = 0
    m = W.shape[0]
    d = array(W.sum(0))

    # asymmetric
    if sym == 0:
        D1 = dia_matrix((d ** -1., array([0])), shape=(m, m))
        P = D1 * W;

    # symmetric
    if sym == 1:
        D1 = dia_matrix((d ** -0.5, array([0])), shape=(m, m))
        P = (D1 * W) * D1;

    f1 = P * y
    f2 = P * f1
    f3 = P * f2
    feat = vstack((f1, f2, f3)).T
    return feat
