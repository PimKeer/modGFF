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

epsilon1 = np.array([1e-160,1e-140,1e-120,1e-100,1e-80,1e-70,1e-60,1e-40,1e-20])
epsilon2 = epsilon1
M = 1
Neps = 1
Nk = 500 # Amount of samples made for the average of gamma.
Narr = np.array([10]) # N's to use.
parr = np.array([0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17]) # Cuts we try
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

            Rh = np.zeros(Nh)
            for j in range(Nh):
                g1 = 0
                g2 = 0
                k = 0
                while k < Nk:
                    try:
                        x1 = cg.cgpf0(cg.C, wx1, wy1, wz1, N, x0=np.random.normal(size=(N+2)**3), epsilon=e, M=M) #np.random.normal(size=(N+2)**3)
                        x2 = cg.cgpf0(cg.C, wx2, wy2, wz2, 2 * N, x0 = np.random.normal(size=(2*N+2)**3), epsilon=e, M=M) #np.random.normal(size=(2*N+2)**3)

                        if x1[1] == 0 and x2[1] == 0:
                            # k1 = x1[2]
                            # k2 = x2[2]

                            x1 = cg.cut(x1[0], N + 2, 1)
                            x2 = cg.cut(x2[0], 2 * N + 2, 1)

                            h1 = np.quantile(x1, 1 - parr[j], interpolation='midpoint')
                            h2 = np.quantile(x2, 1 - parr[j], interpolation='midpoint')

                            # l1 = levelset(x1, h1)
                            # l2 = levelset(x2, h2)
                            # for z in np.array([4,9,14]):
                            #     plt.imshow(l1.reshape(N,N,N)[z], cmap="Greys", alpha=1)
                            #     plt.show()
                            #     plt.imshow(l2.reshape(2*N, 2*N, 2*N)[z], cmap="Greys", alpha=1)
                            #     plt.show()

                            y1 = gamma(x1, h1, N)  # -1)
                            y2 = gamma(x2, h2, 2 * N)  # -2)
                            g1 += y1
                            g2 += y2
                            if np.mod(k, 10) == 0:
                                print("b = ", b, "i = ", i, "j = ", j, "k = ", k, "  :  ", g2 / g1, y1, y2)
                            k += 1
                        else:
                            pass
                    except ZeroDivisionError:
                        pass

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


