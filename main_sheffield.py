import numpy as np
import cg
from cluster import *

import genGFFSheffield as ggff

def levelset(x, h):
    """Gives an array of 0's and 1's, where the sites with a 1 are in the level set above h."""
    return np.where(x > h, 1, 0)

def gamma(x, h, N):
    """Computes the second moment of the cluster size for a given DGFF sample and threshold h."""
    y = levelset(x, h)
    z = cluster(y, N)
    clusterarray = np.bincount(np.bincount(z)[:-1])
    gamma = (clusterarray * np.arange(len(clusterarray),dtype='int64')**2).sum()
    return gamma


epsilon1 = np.array([3e-160])
epsilon2 = epsilon1
M = 1
Neps = 1
Nk = 10000 # Amount of samples made for the average of gamma.
Narr = np.array([10,20]) # N's to use.
parr = np.array([0.13,0.135,0.14,0.145,0.15,0.155,0.16]) # Cuts we try

Nh = len(parr) # Amount of cuts h we try.
carr = np.array([0])
barr = np.array([1])

RhN = np.zeros(len(Narr)*Nh)

for b in barr:
    for k1 in epsilon1:
        for i in range(len(Narr)):
            N = Narr[i]

            Rh = np.zeros(Nh)
            for j in range(Nh):
                g1 = 0
                g2 = 0
                k = 0
                while k < Nk:

                    x1 = ggff.gen3DGFFSheffield(N,N,N).reshape(N**3)
                    x2 = ggff.gen3DGFFSheffield(2 * N, 2 * N, 2 * N).reshape((2 * N) ** 3)

                    h1 = np.quantile(x1,1-parr[j], interpolation='midpoint')
                    h2 = np.quantile(x2,1-parr[j], interpolation='midpoint')

                    y1 = gamma(x1,h1,N-1)
                    y2 = gamma(x2,h2,2*N-2)

                    g1 += y1
                    g2 += y2

                    if np.mod(k,100) == 0:
                        print("b = ", b, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2/g1, y1, y2)
                    k += 1

                Rh[j] = g2/g1
                print(Rh)

            RhN[i*Nh:(i+1)*Nh] = Rh
            np.save('RN_sheffield' + str(N) + '.npy', RhN)


