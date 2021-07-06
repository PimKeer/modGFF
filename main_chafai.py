import numpy as np
from cluster import *
from matplotlib import pyplot as plt
from numba import njit
import genGFFChafai as ggch

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


Nk = 10000 # Amount of samples made for the average of gamma.
Narr = np.array([11]) # N's to use.
parr = np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13]) # Cuts we try
Np = len(parr) # Amount of cuts h we try.
R = [] # List containing arrays of computed R_N values, one array for each N
E = [] # List containing arrays of uncertainties in the R_N values, one array for each N

for i in range(len(Narr)):
    N = Narr[i] # Set lattice size
    Rh = np.zeros(Np) # This array will contain the R_N values for each given p, for the current N
    Eh = np.zeros(Np) # This array will contain the uncertainties in the R_N values for each given p, for the current N

    if N == 6:
        sG1 = np.load('Chafaï matrices/2_gff_sG_6.txt.npy')
        sG2 = np.load('Chafaï matrices/2_gff_sG_11.txt.npy')
    elif N == 11:
        sG1 = np.load('Chafaï matrices/2_gff_sG_11.txt.npy')
        sG2 = np.load('Chafaï matrices/njit_gff_sG_21.txt.npy')

    for j in range(Np):
        p = parr[j] # Set occupation probability
        y1arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_N
        y2arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_2N
        g1 = 0 # Running average of y1arr
        g2 = 0 # Running average of y2arr
        k = 0 # Iteration counter
        while k < Nk:
            x1 = ggch.genGFFChafai2(N, sG1).reshape((N - 1) ** 3) # Generate a Chafai DGFF sample (lattice size N)
            x2 = ggch.genGFFChafai2(2 * N - 1, sG2).reshape((2 * N - 2) ** 3)  # Generate a Chafai DGFF sample (lattice size 2N)

            h1 = np.quantile(x1, 1 - parr[j], interpolation='midpoint') # Determine the threshold height given p
            h2 = np.quantile(x2, 1 - parr[j], interpolation='midpoint') # Determine the threshold height given p

            y1 = gamma(x1, h1, N - 1) # Compute the second moment for cluster size
            y2 = gamma(x2, h2, 2 * N - 2) # Compute the second moment for cluster size

            g1 += y1 # Add this moment to the running average
            g2 += y2 # Add this moment to the running average

            y1arr[k] = y1 # Add this moment to the moment list
            y2arr[k] = y2 # Add this moment to the moment list

            if np.mod(k,200) == 0:
                print("N = ", N, "p = ", parr[j], "k = ", k, "  :  ", g2/g1, y1, y2) # Print out the values as a progress update
            k += 1 # Next iteration
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

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIG_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title

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