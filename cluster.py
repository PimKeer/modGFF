import numpy as np
from numba import jit
import scipy

def find(i, labels):
    while labels[i] != i:
        i = labels[i]
    return i

def union(i, j, labels):
    labels[max(find(i, labels),find(j, labels))] = min(find(i, labels),find(j, labels))

def union3(i, j, k, labels):
    [mi, me, ma] = sorted([find(i, labels),find(j, labels),find(k, labels)])
    labels[ma] = mi
    labels[me] = mi


def cluster(x, N):
    """Hoshen-Kopelman routine to find the clusters for a given 0-1 array."""
    x = np.append(x, 0)

    labels = np.arange(N ** 3+1)

    for i in range(N ** 3):
        if x[i] == 0:
            pass
        else:
            if np.mod(i, N) == 0:
                ix = -1
            else:
                ix = i - 1
            if np.mod(i, N ** 2) <= N - 1:
                iy = -1
            else:
                iy = i - N
            if np.mod(i, N ** 3) <= N ** 2 - 1:
                iz = -1
            else:
                iz = i - N ** 2

            if x[ix] == 0 and x[iy] == 0 and x[iz] == 0:
                pass
            elif x[ix] == 1 and x[iy] == 0 and x[iz] == 0:
                labels[i] = find(ix, labels)
            elif x[ix] == 0 and x[iy] == 1 and x[iz] == 0:
                labels[i] = find(iy, labels)
            elif x[ix] == 0 and x[iy] == 0 and x[iz] == 1:
                labels[i] = find(iz, labels)
            elif x[ix] == 0 and x[iy] == 1 and x[iz] == 1:
                union(iy, iz, labels)
                labels[i] = min(find(iy, labels),find(iz, labels))
            elif x[ix] == 1 and x[iy] == 0 and x[iz] == 1:
                union(ix, iz, labels)
                labels[i] = min(find(ix, labels),find(iz, labels))
            elif x[ix] == 1 and x[iy] == 1 and x[iz] == 0:
                union(ix, iy, labels)
                labels[i] = min(find(ix, labels),find(iy, labels))
            else:
                union3(ix, iy, iz, labels)
                labels[i] = min(find(ix, labels),find(iy, labels),find(iz, labels))

            labels[-1] = N**3

    for i in range(N ** 3):
        if x[i] == 0:
            pass
        else:
            labels[i] = find(i, labels)

    labels[x == 0] = N ** 3
    return labels[:-1]