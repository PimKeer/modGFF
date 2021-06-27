import numpy as np
from cluster import *
import genGFFSheffield as ggff
import pandas as pd
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

def gamma2(x, h, N):
    """Computes the second moment of the cluster size for a given DGFF sample and threshold h."""
    y = levelset(x, h)
    z = cluster(y, N)
    clusterarray = np.bincount(np.bincount(z)[:-1])
    gamma = (clusterarray * np.arange(len(clusterarray))**2).sum()
    return gamma

N = 10
x1 = ggff.gen3DGFFSheffield(N, N, N).reshape(N ** 3)
print(gamma(x1,0,N))
print(gamma2(x1,0,N))

Nk = 5000 # Amount of samples made for the average of gamma.
Narr = np.array([10,20,40]) # N's to use.
parr = np.array([0.13,0.135,0.14,0.145,0.15,0.155,0.16]) # Cuts we try
Nh = len(parr) # Amount of cuts h we try.


R = [np.array([20.64388545,21.1511516,22.59138739,23.37893987,24.8844019,25.10520398,27.05304619])]
R.append(np.array([20.21384727,21.18986604,22.20661928,23.12598579,25.320884,27.1979515,28.62694308]))
R.append(np.array([18.44732418,19.64954528,21.87738331,23.88929185,25.17581676,28.04525445,29.98856354]))

E = [np.array([0.51791915,0.70315389,0.77278936,1.30246325,1.0829812,0.73156913,1.11234073])]
E.append(np.array([0.52736901,0.91623963,0.77008539,0.71437123,0.8516745,0.88460973,0.9024745]))
E.append(np.array([0.5899398,0.67496123,0.58256453,1.24159805,1.27367734,0.73931034,1.19331514]))

col = ['r','g','b']
line = ['dashed', 'dashdot', 'dotted']

for i in range(len(Narr)):
    qm = np.poly1d(np.polyfit(parr, R[i], 2))
    x = np.linspace(0.13, 0.16, 10000)
    plt.plot(x,qm(x), linestyle=line[i], color=col[i], label="N = "+str(Narr[i]))
    plt.errorbar(parr,R[i],yerr=E[i],capsize=5,linestyle='None',ecolor=col[i])
    plt.xlabel("$p$")
    plt.ylabel("$R_N$")
    plt.legend()
plt.show()