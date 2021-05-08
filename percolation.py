import numpy as np
import matplotlib.pyplot as plt
import cg
from cluster import *

def levelset(x, h):
    return np.where(x > h, 1, 0)

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
Narr = np.array([20]) ## N's to use.

RhN = np.zeros(3*Nh)

for i in range(len(Narr)):
    N = Narr[i]
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
        print(Rh)
    RhN[i*Nh:(i+1)*Nh] = Rh

harray = 0.13+0.005*np.arange(Nh)
for l in range(len(Narr)):
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