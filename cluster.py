import numpy as np
import matplotlib.pyplot as plt
import cg

def levelset(x, h):
    return np.where(x > h, 1, 0)

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

def gamma(x, h, N):
    y = levelset(x, h)
    z = cluster(y, N)
    clusterarray = np.bincount(np.bincount(z)[:-1])
    gamma = (clusterarray * np.arange(len(clusterarray))**2).sum()
    return gamma

plt.close()

k_max = 7
Nk = 200 ## Amount of samples made for the average of gamma.
Nh = 7 ## Amount of cuts h we try.
Nn = 1 ## Amount of different N's.

RhN = np.zeros(3*Nh)

for i in range(Nn):
    N = 5 * (i+1)
    Rh = np.zeros(Nh)
    for j in range(Nh):
        h = 0.13+0.005*j
        g1 = 0
        g2 = 0
        for k in range(Nk):
            print("i = ", i, "j = ", j, "k = ", k)
            x1 = cg.cgnorm(cg.A, N, k_max)
            x2 = cg.cgnorm(cg.A, 2*N, k_max)

            g1 += gamma(x1, h, N)
            g2 += gamma(x2, h, 2*N)
        Rh[j] = g2/g1
    RhN[i*Nh:(i+1)*Nh] = Rh

harray = 0.13+0.005*np.arange(Nh)
for l in range(Nn):
    plt.plot(harray, RhN[l*Nh:(l+1)*Nh])
plt.show()






"""
clustercount = np.zeros(100)
for k in range(100):
    print(k)
    N = int(25)
    x = np.random.random(N ** 3)
    h = 0.01*k
    y = levelset(x, h)
    z = cluster(y,N)
    countz = np.bincount(z)
    clustercount[k] = np.count_nonzero(countz)
plt.scatter(1-0.01*np.arange(1,101),clustercount)
plt.show()
N = 3
x = np.random.random(N ** 3)
h = 0.5
y = levelset(x, h)
z = cluster(y,N)
g = gamma(z,N)
print(y)
print(z)
print(np.bincount(z))
print(g)
"""