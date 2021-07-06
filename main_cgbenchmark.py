import numpy as np
from cluster import *
from matplotlib import pyplot as plt
from numba import njit
from clustertest import *
import cg
from matrices import *

def levelset(x, h):
    """Gives an array of 0's and 1's, where the sites with a 1 are in the level set above h."""
    return np.where(x >= h, 1, 0)

def gamma(x, h, N):
    """Computes the second moment of the cluster size for a given DGFF sample and threshold h."""
    y = levelset(x, h)
    z = cluster(y, N)
    clusterarray = np.bincount(np.bincount(z)[:-1])
    gamma = (clusterarray * np.arange(len(clusterarray),dtype='int64')**2).sum()
    return gamma


Nk = 5000 # Amount of samples made for the average of gamma.
Narr = np.array([10]) # N's to use.
parr = np.array([0.08,0.09,0.1]) # Cuts we try
Np = len(parr) # Amount of cuts h we try.
M = 1  # Scaling constant
R = [] # List containing arrays of computed R_N values, one array for each N
E = [] # List containing arrays of uncertainties in the R_N values, one array for each N

for i in range(len(Narr)):
    N = Narr[i] # Set lattice size
    Rh = np.zeros(Np) # This array will contain the R_N values for each given p, for the current N
    Eh = np.zeros(Np) # This array will contain the uncertainties in the R_N values for each given p, for the current N

    wx1, wy1, wz1 = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3) # Set conductances
    wx2, wy2, wz2 = np.ones((2 * N + 2) ** 3), np.ones((2 * N + 2) ** 3), np.ones((2 * N + 2) ** 3) # Set conductances

    if i == 0:
        e = 1e-80
    else:
        e = 1e-80


    for j in range(Np):
        p = parr[j] # Set occupation probability
        y1arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_N
        y2arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_2N

        g1 = 0 # Running average of y1arr
        g2 = 0 # Running average of y2arr
        k = 0 # Iteration counter

        while k < Nk:
            try:
                x1 = cg.cgpf0(cg.C, wx1, wy1, wz1, N, x0=np.random.normal(size=(N + 2) ** 3), epsilon=e, M=M)  # Generate a ZBC CG DGFF sample (lattice size N)
                x2 = cg.cgpf0(cg.C, wx2, wy2, wz2, 2 * N, x0=np.random.normal(size=(2 * N + 2) ** 3), epsilon=e, M=M)  # Generate a ZBC CG DGFF sample (lattice size 2N)

                x1c = cg.cut(x1[0], N + 2, 1)  # Cut away zero boundary layer
                x2c = cg.cut(x2[0], 2 * N + 2, 1)  # Cut away zero boundary layer

                h1 = np.quantile(x1c, 1 - parr[j], interpolation='midpoint')  # Determine the threshold height given p
                h2 = np.quantile(x2c, 1 - parr[j], interpolation='midpoint')  # Determine the threshold height given p

                y1 = gamma(x1c, h1, N)  # Compute the second moment for cluster size
                y2 = gamma(x2c, h2, 2 * N)  # Compute the second moment for cluster size

                g1 += y1  # Add this moment to the running average
                g2 += y2  # Add this moment to the running average

                y1arr[k] = y1  # Add this moment to the moment list
                y2arr[k] = y2  # Add this moment to the moment list

                if np.mod(k, 1) == 0:
                    print("N = ", N, "p = ", parr[j], "k = ", k, "  :  ", g2 / g1, y1,
                          y2)  # Print out the values as a progress update


                k += 1 # Next iteration

            except ZeroDivisionError:
                pass

        np.save('cgbenchmark1_'+str(p)+str(N)+'.npy', y1arr)
        np.save('cgbenchmark2_'+str(p)+str(N)+'.npy', y2arr)

        Rh[j] = g2/g1 # Add latest running average to list of R_N

        # The code beneath is to divide the data into 10 equal subsets to find the standard deviation as error bar.
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

# Plot the results.
col = ['r', 'g', 'b', 'y', 'k']
line = ['-', '--', '-.', ':', '-']

size = 14

plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)

for i in range(len(Narr)):
    qm = np.poly1d(np.polyfit(parr, R[i], 2))
    x = np.linspace(parr[0], parr[-1], 10000)
    plt.plot(x,qm(x), linestyle=line[i], color=col[i], label="N = "+str(Narr[i]))
    plt.errorbar(parr,R[i],yerr=E[i],capsize=5,linestyle='None',ecolor=col[i])
    # plt.xticks([0.585,0.59,0.595,0.60])
    plt.xlabel("$p$")
    plt.ylabel("$R_N$")
    plt.legend()
plt.show()