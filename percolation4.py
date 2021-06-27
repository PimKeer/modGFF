import numpy as np
from cluster import *
import genGFFSheffield as ggff
from matplotlib import pyplot as plt

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

Nk = 5000 # Amount of samples made for the average of gamma.
Narr = np.array([10]) # N's to use.
parr = np.array([0.13,0.135,0.14,0.145,0.15,0.155,0.16]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.
R = []
E = []

for i in range(len(Narr)):
    N = Narr[i]
    Rh = np.zeros(Nh)
    Eh = np.zeros(Nh)

    for j in range(Nh):
        y1arr = np.zeros(Nk)
        y2arr = np.zeros(Nk)
        g1 = 0
        g2 = 0
        k = 0
        while k < Nk:
            x1 = ggff.gen3DGFFSheffield(N,N,N).reshape(N**3)
            x2 = ggff.gen3DGFFSheffield(2 * N, 2 * N, 2 * N).reshape((2 * N) ** 3)

            h1 = np.quantile(x1,1-parr[j])#, interpolation='midpoint')
            h2 = np.quantile(x2,1-parr[j])#, interpolation='midpoint')

            y1 = gamma(x1,h1,N)
            y2 = gamma(x2,h2,2*N)

            g1 += y1
            g2 += y2

            y1arr[k] = y1
            y2arr[k] = y2

            if np.mod(k,1) == 0:
                print("N = ", N, "p = ", parr[j], "k = ", k, "  :  ", g2/g1, y1, y2)
            k += 1
        Rh[j] = g2/g1

        z1 = np.array_split(y1arr,10)
        z2 = np.array_split(y2arr,10)

        av = np.zeros(10)

        for z in range(10):
            av[z] = np.average(z2[z])/np.average(z1[z])

        print(av)
        Eh[j] = np.std(av)

    R.append(Rh)
    E.append(Eh)

    print(Rh)
    print(Eh)

for i in range(len(Narr)):
    qm = np.poly1d(np.polyfit(parr, R[i], 2))
    x = np.linspace(0.13, 0.16, 10000)
    plt.plot(x,qm(x), linestyle='--')
    plt.errorbar(parr,R[i],yerr=E[i],capsize=5,linestyle='None')
    plt.xlabel("$p$")
    plt.ylabel("$R_N$")
plt.show()