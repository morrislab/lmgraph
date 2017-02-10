from numpy import *


def auc(yp, yt):
    pids = where(yt == 1)[0]
    nids = where(yt != 1)[0]
    auc = sum([int(yp[i] > yp[j]) for i in pids for j in nids])
    ties = sum([0.5 * int(yp[i] == yp[j]) for i in pids for j in nids])
    return (ties + auc) * 1. / (len(pids) * len(nids))


if __name__ == "__main__":
    pass
