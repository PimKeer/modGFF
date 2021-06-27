import numpy as np
import cg
from cluster import *
import conductances as cond
import gibbs as gb
import direct_sampling as ds

#from matplotlib import pyplot as plt
import genGFFSheffield as ggff
import genGFFChafai as ggch

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
Nk = 500 # Amount of samples made for the average of gamma.
Narr = np.array([11]) # N's to use.
parr = np.array([0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,
                 0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,
                 0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,
                 0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,
                 0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,
                 0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,
                 0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3]) # Cuts we try

Nh = len(parr) # Amount of cuts h we try.
carr = np.array([0])
barr = np.array([1])

RhN = np.zeros(len(Narr)*Nh)

for b in barr:
    for k1 in epsilon1:
        for i in range(len(Narr)):
            N = Narr[i]

            sG1 = np.load('2_gff_sG_11.txt.npy')
            sG2 = np.load('njit_gff_sG_21.txt.npy')

            Rh = np.zeros(Nh)
            for j in range(Nh):
                g1 = 0
                g2 = 0
                k = 0
                while k < Nk:

                    # x1 = cg.cut(cg.cgpf0(cg.C, wx1, wy1, wz1, N, epsilon=epsilon1, kmax=N**3, M=M)[0],N+2,1)
                    # x2 = cg.cut(cg.cgpf0(cg.C, wx2, wy2, wz2, 2*N, epsilon=epsilon2, kmax=(2*N)**3, M=M)[0],2*N+2,1)

                    # x1 = ds.sampler_CG(np.zeros(N**3),M1,K=k1,init=np.zeros(N ** 3))
                    # x2 = ds.sampler_CG(np.zeros((2 * N) ** 3), M2, K=2 * k1, init=np.zeros((2 * N) ** 3))

                    # x1 = ds.sampler_CG(np.zeros(N**3),M1,K=k1,init=np.random.normal(size=N**3))
                    # x2 = ds.sampler_CG(np.zeros((2*N)**3),M2,K=2*k1,init=np.random.normal(size=(2*N)**3))

                    # x1 = ggff.gen3DGFFSheffield(N,N,N).reshape(N**3)
                    # x2 = ggff.gen3DGFFSheffield(2 * N, 2 * N, 2 * N).reshape((2 * N) ** 3)

                    x1 = ggch.genGFFChafai2(N,sG1).reshape((N - 1)**3)
                    x2 = ggch.genGFFChafai2(2 * N-1,sG2).reshape((2 * N - 2) ** 3)

                    # kmax = 1000000
                    # x1 = np.zeros((N+2)**3)
                    # x2 = np.zeros((2*N+2)**3)
                    # x1 = gb.gibbs0(x1, N, kmax)
                    # x2 = gb.gibbs0(x2, 2*N, kmax)
                    #
                    # x1 = cg.cut(x1, N + 2, 1)
                    # x2 = cg.cut(x2,2*N+2,1)
                    # h1 = np.quantile(x1, 1 - parr[j])
                    # h2 = np.quantile(x2,1-parr[j])
                    #
                    # g1 += gamma(x1, h1, N)#-1)
                    # g2 += gamma(x2, h2, 2*N)# -2)
                    # if np.mod(k,1) == 0:
                    #     print("e = ", e, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2/g1)

                    # h1 = np.quantile(x1, 1 - parr[j])
                    # h2 = np.quantile(x2, 1 - parr[j])
                    #
                    # y1 = gamma(x1, h1, N-1)
                    # y2 = gamma(x2, h2, 2 * N-2)
                    #
                    # g1 += y1
                    # g2 += y2
                    # print("e = ", e, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2 / g1, y1, y2)
                    # k += 1

                    if True: #x1[1] == 0 and x2[1] == 0:
                        #k1 = x1[2]
                        #k2 = x2[2]

                        # x1 = cg.cut(x1[0],N+2,1)
                        # x2 = cg.cut(x2[0],2*N+2,1)

                        # x1 = x1[0]
                        # x2 = x2[0]

                        h1 = np.quantile(x1,1-parr[j], interpolation='midpoint')
                        h2 = np.quantile(x2,1-parr[j], interpolation='midpoint')
                        y1 = gamma(x1,h1,N-1)
                        y2 = gamma(x2,h2,2*N-2)
                        g1 += y1
                        g2 += y2
                        if np.mod(k,1) == 0:
                            print("b = ", b, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2/g1, y1, y2)
                        k += 1
                    else:
                        pass
                Rh[j] = g2/g1
                print(Rh)
            RhN[i*Nh:(i+1)*Nh] = Rh
            np.save('Rh' + str(N) + '.npy', RhN)


