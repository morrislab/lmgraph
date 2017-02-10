from numpy import *
from numpy.random import *
from scipy.sparse import dia_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.io import mmread
import sys
import constants
import util


def learn(NW, y):
    """
    Run label propagation on the inputs.

    Parameters
    ----------
    NW: list of sparse matrices, each of size = (n_nodes, n_nodes)
        List of networks/graphs
    y:  array of shape = (n_nodes, )
        Target labels

    Returns
    -------
    Average AUC computed from the test set splits.
    """

    # no. of networks
    K = len(NW)

    # combine networks, uniform weighting
    nu = ones(K) * 1. / K
    W = nu[0] * NW[0]
    for i in arange(1, K): W = W + nu[i] * NW[i]
    del NW

    auc = zeros(constants.NTRIALS)

    # regularization parameters
    cparam = 2 ** array([-14., -12., -10., -8., -6., -4. - 2., -1., 0., 1., 2., 4., 6., 8.])

    for trial in arange(constants.NTRIALS):
        seed(trial)
        m = W.shape[0]

        # split data set
        pids = where(y == 1)[0]
        npids = len(pids)
        pids = pids[permutation(npids)]
        nids = where(y != 1)[0]
        nnids = len(nids)
        nids = nids[permutation(nnids)]
        tr_pids, val_pids, te_pids = pids[0:3 * npids / 5], pids[3 * npids / 5:4 * npids / 5], pids[4 * npids / 5:]
        tr_nids, val_nids, te_nids = nids[0:3 * nnids / 5], nids[3 * nnids / 5:4 * nnids / 5], nids[4 * nnids / 5:]

        trids = hstack((tr_pids, tr_nids))
        valids = hstack((val_pids, val_nids))
        teids = hstack((te_pids, te_nids))

        tr_y = zeros(m)
        tr_y[trids] = y[trids]
        pids = where(tr_y == 1)[0]
        npids = len(pids)
        nids = where(tr_y == -1)[0]
        nnids = len(nids)
        tr_y[valids] = (npids - nnids) * 1. / (npids + nnids)
        tr_y[teids] = (npids - nnids) * 1. / (npids + nnids)

        rmse = []
        for c in cparam:
            f = lprop(W, tr_y, c)
            rmse.append(sum((f[valids] - y[valids]) ** 2))
        bparam = cparam[argmin(rmse)]

        # retrain with training + validation set
        tr_y = zeros(m)
        trids = hstack((tr_pids, tr_nids, val_pids, val_nids))
        (tr_y)[trids] = (y)[trids]
        pids = where(tr_y == 1)[0];
        npids = len(pids);
        nids = where(tr_y == -1)[0];
        nnids = len(nids);
        (tr_y)[teids] = (npids - nnids) * 1. / (npids + nnids)
        f = lprop(W, tr_y, bparam)

        auc[trial] = util.auc((f)[teids], (y)[teids])

    return auc


def lprop(W, y, c):
    """
    Label propagation subroutine.

    Parameters
    ----------
    W: sparse matrix, size = (n_nodes, n_nodes)
        Input netowrk/graph
    y:  array of shape = (n_nodes, )
        Target labels

    Returns
    -------
    Predicted label, array of shape = (n_nodes, )
    """

    m = W.shape[0]
    d = array(W.sum(0))
    D = dia_matrix((d, array([0])), shape=(m, m))
    L = D - W

    I = sp.eye(m, m)
    A = I + m * c * L
    f = cg(A, y)[0]

    return f


if __name__ == "__main__":

    # load networks and labels
    dname = sys.argv[1]  # name of the data files in data folder
    NW = load('./data/' + dname + '_X.npy')  # list of networks
    Y = mmread('./data/' + dname + '_y')  # target labels

    # check if the target function has at least MIN_LBLS positive labels
    lbl_idx = int(sys.argv[2])
    Y = array(Y.todense())
    if sum(Y, 0)[lbl_idx] < constants.MIN_LBLS:
        auc = -1
    else:
        y = Y[:, lbl_idx] * 2 - 1
        auc = learn(NW, y)

    print 'Mean AUC: {}'.format(mean(auc))
