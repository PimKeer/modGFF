import numpy as np
import cg
from cluster import *
import conductances as cond
import gibbs as gb
import direct_sampling as ds
from matrices import *

from matplotlib import pyplot as plt
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

# def genBernoulli(N,p):
#     x = np.random.random(size=N**3)
#     for i in range(N**3):
#         if x[i] <= p:
#             x[i] = 1
#         else:
#             x[i] = 0
#     return x

epsilon1 = np.array([1e-160])
epsilon2 = epsilon1
M = 1
Neps = 1
Nk = 500 # Amount of samples made for the average of gamma.
Narr = np.array([10]) # N's to use.
parr = np.array([0.23]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.
carr = np.array([0])
barr = np.array([1])

RhN = np.zeros(len(Narr)*Nh)

for b in barr:
    for e in epsilon1:
        for i in range(len(Narr)):
            N = Narr[i]

            wx1, wy1, wz1 = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
            wx2, wy2, wz2 = np.ones((2 * N + 2) ** 3), np.ones((2 * N + 2) ** 3), np.ones((2 * N + 2) ** 3)

            def M1(x):
                N = 10
                wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
                x = x.reshape(N ** 3)
                return C2(x, wx, wy, wz, N).reshape((N ** 3, 1))

            def M2(x):
                N = 20
                wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
                x = x.reshape(N ** 3)
                return C2(x, wx, wy, wz, N).reshape((N ** 3, 1))

            a = b
            # wx1, wy1, wz1 = cond.wxyz(N, a, b)
            # wx2, wy2, wz2 = cond.wxyz(2*N, a, b)


            # sG1 = np.load('test_gff_sG_5.txt.npy')
            # sG2 = np.load('test_gff_sG_9.txt.npy')

            Rh = np.zeros(Nh)
            for j in range(Nh):
                g1 = 0
                g2 = 0
                k = 0
                while k < Nk:

                    # x1 = ds.sampler_CG(np.zeros(N**3),M1,K=k1,init=np.zeros(N ** 3))
                    # x2 = ds.sampler_CG(np.zeros((2 * N) ** 3), M2, K=2 * k1, init=np.zeros((2 * N) ** 3))

                    # x1 = ds.sampler_CG(np.zeros(N**3),M1,K=N**3,init=np.random.normal(size=N**3),tol=1e-40)
                    # x2 = ds.sampler_CG(np.zeros((2*N)**3),M2,K=(2*N)**3,init=np.random.normal(size=(2*N)**3),tol=1e-40)

                    # x1 = ds.sampler_CG(np.zeros(N ** 3), M1, K=N ** 3, init=np.zeros(N ** 3), tol=e)
                    # x2 = ds.sampler_CG(np.zeros((2 * N) ** 3), M2, K=(2 * N) ** 3,init=np.zeros((2 * N) ** 3), tol=e)

                    # x1 = np.random.random(N**3)
                    # x2 = np.random.random((2*N)**3)

                    # x1 = ggff.gen3DGFFSheffield(N+2,N+2,N+2).reshape((N+2)**3)
                    # x2 = ggff.gen3DGFFSheffield(2 * N+2, 2 * N+2, 2 * N+2).reshape((2 * N+2) ** 3)
                    try:
                        x1 = cg.cgpf02(cg.C2, wx1, wy1, wz1, N, x0=np.random.normal(size=N ** 3), epsilon=epsilon1,
                                      M=M)  # np.random.normal(size=(N+2)**3)
                        x2 = cg.cgpf02(cg.C2, wx2, wy2, wz2, 2 * N, x0=np.random.normal(size=(2 * N) ** 3), epsilon=epsilon2,
                                      M=M)  # np.random.normal(size=(2*N+2)**3)

                        if x1[1] == 0 and x2[1] == 0:
                            # k1 = x1[2]
                            # k2 = x2[2]

                            x1 = x1[0]
                            x2 = x2[0]

                            h1 = np.quantile(x1, 1 - parr[j], interpolation='midpoint')
                            h2 = np.quantile(x2, 1 - parr[j], interpolation='midpoint')

                            y1 = gamma(x1, h1, N)  # -1)
                            y2 = gamma(x2, h2, 2 * N)  # -2)
                            g1 += y1
                            g2 += y2
                            if np.mod(k, 1) == 0:
                                print("b = ", b, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2 / g1, y1, y2)
                            k += 1
                        else:
                            pass
                    except ZeroDivisionError:
                        pass

                    # x1 = ds.sampler_CG(np.zeros(N ** 3), M1, K=N ** 3, init=x1, tol=e)
                    # x2 = ds.sampler_CG(np.zeros((2 * N) ** 3), M2, K=(2 * N) ** 3, init=x2, tol=e)

                    # x1 = cg.cut(ggch.genGFFChafai2(N,sG1).reshape((N - 1)**3),N-1,0)
                    # x2 = cg.cut(ggch.genGFFChafai2(2 * N-1,sG2).reshape((2 * N - 2) ** 3),2*N-2,0)

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


                Rh[j] = g2/g1
                print(Rh)
            RhN[i*Nh:(i+1)*Nh] = Rh
            np.save('Rh' + str(N) + '.npy', RhN)


