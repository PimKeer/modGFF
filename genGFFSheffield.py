import numpy as np
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

def l(j,k,m,n):
    if j+k==2:
        return 0
    else:
        return 1/np.sqrt((np.sin((j-1)*np.pi/m))**2+(np.sin((k-1)*np.pi/n))**2)

def gen2DGFFSheffield(m,n):
    #table = np.stack([i for i in np.ndindex(m,n)]).reshape(m,n,2)
    table = np.indices((m,n))
    lvec = np.vectorize(l)
    table = lvec(table[0],table[1],m,n)
    Z = np.random.normal(0,1,m*n)+1j*np.random.normal(0,1,m*n)
    Z = Z.reshape(m,n)
    table = np.multiply(table,Z)
    table = np.fft.fft2(table)
    return np.real(table)

def plotGFF(GZ,m,n):
    X = np.arange(1,m)
    Y = np.arange(1,n)
    Z = GZ
    fig = plt.figure()
    ax = Axes3D(fig)
    #surf = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
    df = pd.DataFrame({'x': np.repeat(np.arange(1,m+1),n), 'y': np.tile(np.arange(1,n+1),m), 'z': GZ.flatten()})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
    #surf = ax.plot_trisurf(np.repeat(np.arange(1,L),L-1), np.tile(np.arange(1,L),L-1), GZ.flatten(),  cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
"""
m = 50
n = 50

G = gen2DGFFSheffield(m,n)
print(G)
plotGFF(G,m,n)
plt.show()

H = np.random.normal(0,100,m*n).reshape(m,n)
plotGFF(H,m,n)
plt.show()

def ber(GZ,L,d,h):
    GZ = GZ.flatten()
    perc = np.zeros(len(GZ))
    for i in range(len(GZ)):
        if GZ[i] >= h:
            perc[i] = 1
    perc = np.resize(perc, (m,n))
    return perc

for h in range(10):
    perc = ber(G,15,2,50*(h-5))
    plt.subplot(2,5,h+1)
    plt.imshow(perc, alpha=0.8)
    #plt.xticks(np.arange(m))
    #plt.yticks(np.arange(n))
    plt.title('h =' + str((h-5)*100))

plt.show()

"""
