import numpy as np
from cluster import *
from matplotlib import pyplot as plt
from numba import njit

Nk = 10000 # Amount of samples made for the average of gamma.
Narr = np.array([4,8]) # N's to use.
parr = np.array([0.585,0.59,0.595,0.60]) # Cuts we try
Np = len(parr) # Amount of cuts h we try.
R = [] # List containing arrays of computed R_N values, one array for each N
E = [] # List containing arrays of uncertainties in the R_N values, one array for each N

# Example values for R and E from an earlier run
# R = [np.array([8.89631918,9.1670137,9.53575105,9.84963358,10.06718746,10.3696186,10.76419234,11.01120787,11.3975067,11.82351483,12.04534423,12.42346034,12.70638498,13.07932479]),
#      np.array([7.77676307,7.97773821,8.58174696,8.9472646,9.51400522,9.95568955,10.60196538,11.1210405,11.64148907,12.2220517,12.73830269,13.30097147,13.88834218,14.34721706]),
#      np.array([6.25965392,6.58888766,6.98374173,7.3647663,8.07547141,8.79307094,9.56302866,10.57019139,11.39262464,12.61297261,13.65297763,14.70882732,15.34905849,16.2732407])]
# E = [np.array([0.11943006,0.08433428,0.14098583,0.11902372,0.11599812,0.10669372,0.13566709,0.12605093,0.1409752,0.11341701,0.07418248,0.13719741,0.11544317,0.14293746]),
#      np.array([0.26310007,0.23105276,0.23043899,0.26088746,0.25148759,0.24457272,0.29336611,0.26809933,0.2446679,0.26841714,0.25030237,0.24790672,0.32812166,0.23355936]),
#      np.array([0.17847925,0.17843752,0.20990995,0.16623804,0.19310972,0.12460153,0.26448589,0.2816964,0.29852619,0.26902651,0.2577366,0.31365023,0.33875275,0.21697951])]

@njit
def siteperc(N,p):
    """Generate a Bernoulli site percolation sample on an NxN lattice, with occupation probability p."""
    x = np.random.random(N**2)
    for i in range(N**2):
        if x[i] <= p:
            x[i] = 1
        else:
            x[i] = 0
    return x

def gamma(x):
    """Computes the second moment of the cluster size for a given sample of 1's and 0's."""
    clusterarray = np.bincount(np.bincount(x)[:-1])
    #print(np.bincount(x))
    #print(clusterarray)
    gamma = (clusterarray * np.arange(len(clusterarray),dtype='int64')**2).sum()
    return gamma

for i in range(len(Narr)):
    N = Narr[i] # Set lattice size
    Rh = np.zeros(Np) # This array will contain the R_N values for each given p, for the current N
    Eh = np.zeros(Np) # This array will contain the uncertainties in the R_N values for each given p, for the current N

    for j in range(Np):
        p = parr[j] # Set occupation probability
        y1arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_N
        y2arr = np.zeros(Nk) # This array will contain all cluster size second moments, we average over these to find Gamma_2N
        g1 = 0 # Running average of y1arr
        g2 = 0 # Running average of y2arr
        k = 0 # Iteration counter
        while k < Nk:
            x1 = siteperc(N,p) # Generate a Bernoulli site percolation sample (lattice size N)
            x2 = siteperc(2*N,p) # Generate a Bernoulli site percolation sample (lattice size 2N)

            y1 = gamma(cluster2D(x1,N)) # Compute the second moment for cluster size
            y2 = gamma(cluster2D(x2,2*N)) # Compute the second moment for cluster size

            g1 += y1 # Add this moment to the running average
            g2 += y2 # Add this moment to the running average

            y1arr[k] = y1 # Add this moment to the moment list
            y2arr[k] = y2 # Add this moment to the moment list

            if np.mod(k,100) == 0:
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

# Plot the results
for i in range(len(Narr)):
    qm = np.poly1d(np.polyfit(parr, R[i], 2))
    x = np.linspace(parr[0], parr[-1], 10000)
    plt.plot(x,qm(x), linestyle='--')
    plt.errorbar(parr,R[i],yerr=E[i],capsize=5,linestyle='None')
    plt.xlabel("$p$")
    plt.ylabel("$R_N$")
plt.show()