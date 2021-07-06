import numpy as np
from numba import jit
import scipy
from matplotlib import pyplot as plt

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
    """Hoshen-Kopelman routine to find the clusters for a given 0-1 array (in 3D)."""
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
    # print(np.all(test==labels))
    return labels[:-1]

def cluster2(x, N):
    """Alternative Hoshen-Kopelman routine to find the clusters for a given 0-1 array (in 3D)."""
    x = np.append(x, 0)

    labels = np.arange(N ** 3 + 1)

    for i in range(N ** 3):
        if x[i] == 0:
            pass
        else:
            if np.mod(i, N) == 0:
                if np.mod(i, N ** 2) <= N - 1:
                    if np.mod(i, N ** 3) <= N ** 2 - 1:
                        ix, iy, iz = -1, -1, -1
                    else:
                        ix, iy, iz = -1, -1, i - N ** 2
                else:
                    if np.mod(i, N ** 3) <= N ** 2 - 1:
                        ix, iy, iz = -1, i - N, -1
                    else:
                        ix, iy, iz = -1, i - N, i - N ** 2
            else:
                if np.mod(i, N ** 2) <= N - 1:
                    if np.mod(i, N ** 3) <= N ** 2 - 1:
                        ix, iy, iz = i - 1, -1, -1
                    else:
                        ix, iy, iz = i - 1, -1, i - N ** 2
                else:
                    if np.mod(i, N ** 3) <= N ** 2 - 1:
                        ix, iy, iz = i - 1, i - N, -1
                    else:
                        ix, iy, iz = i - 1, i - N, i - N ** 2

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
                labels[i] = min(find(iy, labels), find(iz, labels))
            elif x[ix] == 1 and x[iy] == 0 and x[iz] == 1:
                union(ix, iz, labels)
                labels[i] = min(find(ix, labels), find(iz, labels))
            elif x[ix] == 1 and x[iy] == 1 and x[iz] == 0:
                union(ix, iy, labels)
                labels[i] = min(find(ix, labels), find(iy, labels))
            else:
                union3(ix, iy, iz, labels)
                labels[i] = min(find(ix, labels), find(iy, labels), find(iz, labels))

    for i in range(N ** 3):
        if x[i] == 0:
            pass
        else:
            labels[i] = find(i, labels)

    labels[x == 0] = N ** 3
    return labels[:-1]

def cluster2D(x, N):
    """Hoshen-Kopelman routine to find the clusters for a given 0-1 array (in 2D)."""
    x = np.append(x, 0)

    labels = np.arange(N ** 2 + 1)

    for i in range(N ** 2):
        if x[i] == 0:
            pass
        else:
            if np.mod(i, N) == 0:
                if np.mod(i, N ** 2) <= N - 1:
                    ix, iy, = -1, -1
                else:
                    ix, iy = -1, i - N
            else:
                if np.mod(i, N ** 2) <= N - 1:
                    ix, iy, = i - 1, -1
                else:
                    ix, iy = i - 1, i - N

            if x[ix] == 0 and x[iy] == 0:
                pass
            elif x[ix] == 1 and x[iy] == 0:
                labels[i] = find(ix, labels)
            elif x[ix] == 0 and x[iy] == 1:
                labels[i] = find(iy, labels)
            elif x[ix] == 1 and x[iy] == 1:
                union(ix, iy, labels)
                labels[i] = min(find(ix, labels), find(iy, labels))

    for i in range(N ** 2):
        if x[i] == 0:
            pass
        else:
            labels[i] = find(i, labels)

    labels[x == 0] = N ** 2
    # print(np.all(test==labels))
    return labels[:-1]

# TESTING AREA

if __name__ == '__main__':

    def gamma(x):
        clusterarray = np.bincount(np.bincount(x)[:-1])
        print(np.bincount(x))
        print(np.bincount(np.bincount(x)))
        print(clusterarray)
        # print(np.bincount(x))
        # print(clusterarray)
        print(np.arange(len(clusterarray), dtype='int64'))
        gamma = (clusterarray * np.arange(len(clusterarray), dtype='int64') ** 2).sum()
        return gamma

    np.random.seed(2)
    N = 10
    p = 0.5
    x = np.random.random(N**2)
    for i in range(N**2):
        if x[i] <= p:
            x[i] = 1
        else:
            x[i] = 0
    print(x)
    y = cluster2D(x, N)
    print(x.reshape(N,N))
    print(y.reshape(N,N))
    print(y.reshape(N,N).T)
    print(gamma(y))
    #y2 = cluster2(x, N)