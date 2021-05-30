import numpy as np
import cg
from cluster import *

#from matplotlib import pyplot as plt
import genGFFSheffield as ggff
# import genGFFChafai as ggch

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

epsilon = 1e-162
Neps = 1
Nk = 100 # Amount of samples made for the average of gamma.
Narr = np.array([5,6,7,8,9,10,11,12]) # N's to use.
parr = np.array([0.16]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.

RhN = np.zeros(len(Narr)*Nh)

for e in range(Neps):
    for i in range(len(Narr)):
        N = Narr[i]

        # sG1 = np.load('gff_sG_' + str(N) + '.npy')
        # sG2 = np.load('gff_sG_' + str(2*N-1) + '.npy')

        Rh = np.zeros(Nh)
        for j in range(Nh):
            g1 = 0
            g2 = 0
            k = 0
            while k < Nk:
                x1 = cg.cgpf0(cg.C, N, epsilon)
                # print(np.shape(x1))
                # ggff.plotGFF(x1.reshape((N,N,N))[2],N,N)
                # plt.show()
                x2 = cg.cgpf0(cg.C, 2*N, epsilon)
                # ggff.plotGFF(x2.reshape((2*N, 2*N, 2*N))[2], 2*N, 2*N)
                # plt.show()

                # x1 = ggff.gen3DGFFSheffield(N,N,N).reshape(N**3)
                # x2 = ggff.gen3DGFFSheffield(2 * N, 2 * N, 2 * N).reshape((2 * N) ** 3)

                # x1 = ggch.genGFFChafai2(N,sG1).reshape((N - 1)**3)
                # x2 = ggch.genGFFChafai2(2 * N-1,sG2).reshape((2 * N - 2) ** 3)

                if x1[1] == 0 and x2[1] == 0:
                    x1 = cg.cut(x1[0],N+2,1)
                    x2 = cg.cut(x2[0],2*N+2,1)

                    h1 = np.quantile(x1,1-parr[j])
                    h2 = np.quantile(x2,1-parr[j])

                    g1 += gamma(x1, h1, N)#-1)
                    g2 += gamma(x2, h2, 2*N)# -2)
                    if np.mod(k,1) == 0:
                        print("e = ", e, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2/g1)
                    k += 1
                else:
                    pass

            Rh[j] = g2/g1
            print(Rh)
        RhN[i*Nh:(i+1)*Nh] = Rh
        np.save('Rh' + str(N) + '.npy', RhN)


