from numpy.random import *
from scipy.sparse import csc_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.io import mmread
from features import *
import sys
import constants
import util


def learn(NW, y, attr=None):
    """
    Run lmgraph algorithm (see paper) on the inputs.

    Parameters
    ----------
    NW: list of sparse matrices, each of size = (n_nodes, n_nodes)
        List of networks/graphs
    y:  array of shape = (n_nodes, )
        Target labels

    Returns
    -------
    Average AUC computed on the test set splits.
    """

    K = len(NW)  # no. of networks
    auc = zeros(constants.NTRIALS)

    # regularization parameters
    cparam = 2 ** array([-14., -12., -10., -8., -6., -4. - 2., -1., 0., 1., 2., 4., 6., 8.])

    for trial in arange(constants.NTRIALS):
        seed(trial)

        # split data set into training, validation and test sets
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

        # 'extracting features...'
        m = len(y)
        X = []  # empty((K,m,n))
        if attr is not None:
            X.append(attr)  # to integrate features

        for i in arange(K + 1):
            if i == K:  # combine all data
                W = NW[0]
                for j in arange(1, K): W = W + NW[j]
                W = (1. / K) * W
            else:
                W = NW[i]

            zids = where(array(W.sum(0)) == 0)[1]
            vids = setdiff1d(arange(m), zids)
            W = (W[vids])[:, vids]

            yy = zeros(m)
            vvids = hstack((trids, valids))
            yy[vvids] = y[vvids]

            feat = zeros((m, 3))
            feat[vids] = ThreePropFeatures(W, yy[vids], 0)

            X.append(feat)

        nzids = arange(m)  # where(sum(sum(X,0),1)!=0)[0]

        # reset tr, val, te ids
        trids = intersect1d(trids, nzids)
        valids = intersect1d(valids, nzids)
        teids = intersect1d(teids, nzids)

        # create ensemble
        Xtr = [(X[kk])[trids] for kk in arange(len(X))]
        Xval = [(X[kk])[valids] for kk in arange(len(X))]
        W = ensemble(Xtr, y[trids], Xval, y[valids], cparam)

        # learn network wts.
        # use predictions as features
        feat = empty((len(valids), len(X)))
        for i in arange(len(X)):
            _X = (X[i])[valids]
            if sp.issparse(_X):
                pred = array(dot(_X, csc_matrix(W[i]).T).todense())
                pred.shape = len(pred),
                feat[:, i] = pred
            else:
                feat[:, i] = dot(_X, W[i])
        nu = array([util.auc(feat[:, i], y[valids]) for i in arange(len(X))])

        # test directly
        f = zeros(len(teids))
        for i in arange(len(X)):
            _X = (X[i])[teids]
            if sp.issparse(_X):
                o = array(dot(_X, csc_matrix(W[i]).T).todense())
                o.shape = len(o),
                f = f + nu[i] * o
            else:
                f = f + nu[i] * dot(_X, W[i])
        auc[trial] = util.auc(f, y[teids])

    return auc


def ensemble(Xtr, ytr, Xval, yval, cparam):
    K = len(Xtr)  # no. of data sources
    W = []
    for i in arange(K):
        _Xtr = Xtr[i]
        _Xval = Xval[i]
        _sp = sp.issparse(_Xtr)  # sparse or not flag

        rmse = []
        for c in cparam:
            if _sp:
                w = rlsr_sparse(_Xtr, ytr, c)
                w = csc_matrix(w).T
                f = dot(_Xval, w).todense()
                w = w.todense()
                w.shape = len(w)
            else:
                w = rlsr(_Xtr, ytr, c)
                f = dot(_Xval, w)
            rmse.append(sum((f - yval) ** 2))
        bparam = cparam[argmin(rmse)]

        # retrain with training + validation set
        if _sp:
            w = rlsr_sparse(sp.vstack([_Xtr, _Xval]), hstack((ytr, yval)), bparam)
            ww = w  # abs(w)
            sids = argsort(ww)[::-1]
            w[sids[len(ww) - len(where(cumsum(ww[sids]) >= 0.95 * sum(ww))[0]):]] = 0
        else:
            w = rlsr(vstack((_Xtr, _Xval)), hstack((ytr, yval)), bparam)
        W.append(w)

    return W


def rlsr(X, y, c):
    m, n = X.shape
    w = dot(dot(linalg.inv(dot(X.T, X) + m * c * eye(n)), X.T), y)
    return w


def rlsr_sparse(X, y, c):
    m, n = X.shape
    y = csc_matrix(y).T
    A = dot(X.T, X) + m * c * sp.eye(n, n)
    b = dot(X.T, y).todense()
    w = cg(A, b)[0]
    return w


if __name__ == "__main__":

    # load networks and labels
    dname = sys.argv[1]  # name of the data files in data folder
    NW = load('./data/' + dname + '_X.npy')  # list of networks
    Y = mmread('./data/' + dname + '_y')  # target labels

    # load attributes/features
    attr = None
    try:
        attr = csc_matrix(mmread('./data/' + dname + '_attr.mtx'))
    except:
        # could not load attributes (or missing file)
        pass

    # check if the target function has at least MIN_LBLS positive labels
    lbl_idx = int(sys.argv[2])
    Y = array(Y.todense())
    if sum(Y, 0)[lbl_idx] < constants.MIN_LBLS:
        auc = -1
    else:
        y = Y[:, lbl_idx]
        auc = learn(NW, y, attr)

    print 'Mean AUC: {}'.format(mean(auc))
