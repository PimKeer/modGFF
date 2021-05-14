import numpy as np
import time
from numba import jit
import h5py
import genGFFSheffield as ggff
import matplotlib.pyplot as plt

"""
def xpos(i, N):
    if np.mod(i, N) == N-1:
        a = i - N + 1
    else:
        a = i + 1
    return a
vxpos = np.vectorize(xpos, excluded=['N'])

def xneg(i, N):
    if np.mod(i, N) == 0:
        a = i + N - 1
    else:
        a = i - 1
    return a
vxneg = np.vectorize(xneg, excluded=['N'])

def ypos(i, N):
    if np.mod(i, N ** 2) >= N ** 2 - N:
        b = i - N ** 2 + N
    else:
        b = i + N
    return b
vypos = np.vectorize(ypos, excluded=['N'])

def yneg(i, N):
    if np.mod(i, N ** 2) <= N - 1:
        b = i + N ** 2 - N
    else:
        b = i - N
    return b
vyneg = np.vectorize(yneg, excluded=['N'])

def zpos(i, N):
    if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
        c = i - N ** 3 + N ** 2
    else:
        c = i + N ** 2
    return c
vzpos = np.vectorize(zpos, excluded=['N'])

def zneg(i, N):
    if np.mod(i, N ** 3) <= N ** 2 - 1:
        c = i + N ** 3 - N ** 2
    else:
        c = i - N ** 2
    return c
vzneg = np.vectorize(zneg, excluded=['N'])

with h5py.File('y.hdf5', 'r') as hy:
    y = hy['y'][:]
"""
N = 100
y = np.zeros(N**3)

#@jit(nopython=True)
def gibbs(y, N, k_max):
    """Returns a Gaussian PBC Gibbs sample."""
    wx = 1 #- np.random.random(N ** 3).astype(np.float64)
    wy = 1 #- np.random.random(N ** 3).astype(np.float64)
    wz = 1 #- np.random.random(N ** 3).astype(np.float64)

    for k in range(k_max):
        print(k)
        for i in range(N**3):
            y[i] = np.random.normal()

            if np.mod(i, N) == N - 1:
                pass #y[i] += 1/6 * y[i - N + 1]  #* wx[i]
            else:
                y[i] += 1/6 * y[i + 1] #* wx[i]

            if np.mod(i, N) == 0:
                pass #y[i] += 1 / 6 * y[i + N - 1]  #* wx[i + N - 1]
            else:
                y[i] += 1 / 6 * y[i - 1] # * wx[i - 1]

            if np.mod(i, N ** 2) >= N ** 2 - N:
                pass #y[i] += 1 / 6 * y[i - N ** 2 + N]  #* wy[i]
            else:
                y[i] += 1 / 6 * y[i + N] #* wy[i]

            if np.mod(i, N ** 2) <= N - 1:
                pass #y[i] += 1 / 6 * y[i + N ** 2 - N]  #* wy[i + N ** 2 - N]
            else:
                y[i] += 1 / 6 * y[i - N] #* wy[i - N]

            if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
                pass #y[i] += 1 / 6 * y[i - N ** 3 + N ** 2]  #* wz[i]
            else:
                y[i] += 1 / 6 * y[i + N ** 2] #* wz[i]

            if np.mod(i, N ** 3) <= N ** 2 - 1:
               pass # y[i] += 1 / 6 * y[i + N ** 3 - N ** 2] #* wz[i + N ** 3 - N ** 2]
            else:
                y[i] += 1 / 6 * y[i - N ** 2] #* wz[i - N ** 2]

        if np.mod(k,10) == 0:
            hy = h5py.File(str(k)+'a.h5', 'w')
            hy.create_dataset('y', data=y)
    return y

z = np.zeros((N + 2) ** 3)

def gibbs0(z, N, k_max):
    """Returns a Gaussian ZBC Gibbs sample."""
    wx = 1 #- np.random.random(N ** 3).astype(np.float64)
    wy = 1 #- np.random.random(N ** 3).astype(np.float64)
    wz = 1 #- np.random.random(N ** 3).astype(np.float64)

    for k in range(k_max):
        print(k)
        for i in range((N+2)**3):
            if np.mod(i,N+2) > 0 \
                    and np.mod(i,N+2) < N+1 \
                    and np.mod(i,(N+2)**2) > (N+2) - 1 \
                    and np.mod(i,(N+2)**2) < (N+2) ** 2 - (N+2) \
                    and np.mod(i, (N+2) ** 3) > (N+2) ** 2 - 1 \
                    and np.mod(i, (N+2) ** 3) < (N+2) ** 3 - (N+2) ** 2:
                z[i] = np.random.normal()

                z[i] += 1 / 6 * z[i + 1]  # * wx[i]

                z[i] += 1 / 6 * z[i - 1]  # * wx[i - 1]

                z[i] += 1 / 6 * z[i + N]  # * wy[i]

                z[i] += 1 / 6 * z[i - N]  # * wy[i - N]

                z[i] += 1 / 6 * z[i + N ** 2]  # * wz[i]

                z[i] += 1 / 6 * z[i - N ** 2]  # * wz[i - N ** 2]
            else:
                z[i] = 0

        if np.mod(k,10) == 0:
            hz = h5py.File(str(k)+'b.h5', 'w')
            hz.create_dataset('z', data=z)
    return z


z = gibbs0(z, N, 50)

for i in range(10):
    filename = str(10*i)+"b.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = np.array(f[a_group_key]).reshape((N+2,N+2,N+2))

    ggff.plotGFF(data[1],N+2,N+2)
    plt.show()
