import numpy as np
import matplotlib.pyplot as plt
import cg
from cluster import *
import scipy.special as ss
import genGFFSheffield as ggff

def levelset(x, h):
    """Gives an array of 0's and 1's, where the sites with a 1 are in the level set above h."""
    return np.where(x >= h, 1, 0)

def gamma(x, h, N):
    """Computes the second moment of the cluster size for a given DGFF sample and threshold h."""
    y = levelset(x, h)
    z = cluster(y, N)
    clusterarray = np.bincount(np.bincount(z)[:-1])
    gamma = (clusterarray * np.arange(len(clusterarray))**2).sum()
    return gamma

plt.close()

k_max = 7
Nk = 200 # Amount of samples made for the average of gamma.
Narr = np.array([10,20,40]) # N's to use.
parr = np.array([0.13,0.145,0.16,0.175]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.

RhN = np.zeros(len(Narr)*Nh)

for i in range(len(Narr)):
    N = Narr[i]
    Rh = np.zeros(Nh)
    for j in range(Nh):
        g1 = 0
        g2 = 0
        for k in range(Nk):
            #x1 = cg.cg(cg.A, N, k_max)
            #x2 = cg.cg(cg.A, 2*N, k_max)

            x1 = ggff.gen3DGFFSheffield(N,N,N).reshape(N**3)
            x2 = ggff.gen3DGFFSheffield(2 * N, 2 * N, 2 * N).reshape((2 * N) ** 3)

            h1 = np.quantile(x1,1-parr[j])
            h2 = np.quantile(x2,1-parr[j])

            g1 += gamma(x1, h1, N)
            g2 += gamma(x2, h2, 2*N)
            if np.mod(k,1) == 0:
                print("i = ", i, "j = ", j, "k = ", k, "  :  ", g2/g1)
        Rh[j] = g2/g1
        print(Rh)
    RhN[i*Nh:(i+1)*Nh] = Rh

for l in range(len(Narr)):
    plt.plot(parr, RhN[l*Nh:(l+1)*Nh])
plt.show()
