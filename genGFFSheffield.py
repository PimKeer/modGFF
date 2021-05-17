import numpy as np
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import scipy.special as ss

def l2(j,k,m,n):
    if j+k==0:
        return 0
    else:
        return 1/np.sqrt((np.sin(j*np.pi/m))**2+(np.sin(k*np.pi/n))**2)

def l3(j,k,l,m,n,o):
    if j+k+l==0:
        return 0
    else:
        return 1/np.sqrt((np.sin(j*np.pi/m))**2+(np.sin(k*np.pi/n))**2+(np.sin(l*np.pi/o))**2)

def gen2DGFFSheffield(m,n):
    #table = np.stack([i for i in np.ndindex(m,n)]).reshape(m,n,2)
    table = np.indices((m,n))
    l2vec = np.vectorize(l2)
    table = l2vec(table[0],table[1],m,n)
    Z = np.random.normal(0,1,m*n)+1j*np.random.normal(0,1,m*n)
    Z = Z.reshape(m,n)
    table = np.multiply(table,Z)
    table = np.fft.ifft2(table)*np.sqrt(m*n)
    return np.real(table)

def gen3DGFFSheffield(m,n,o):
    table = np.indices((m,n,o))
    l3vec = np.vectorize(l3)
    table = l3vec(table[0],table[1],table[2],m,n,o)
    Z = np.random.normal(0,1,m*n*o)+1j*np.random.normal(0,1,m*n*o)
    #Z = np.vectorize(complex)(ss.erfinv(2*np.random.uniform(size=m*n*o)-1),ss.erfinv(2*np.random.uniform(size=m*n*o)-1))
    Z = Z.reshape((m,n,o))
    table = np.multiply(table,Z)
    table = np.fft.ifftn(table)*np.sqrt(m*n*o)
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
o = 50

G2 = gen2DGFFSheffield(m,n)
print(G2)
plotGFF(G2,m,n)
plt.show()

G3 = gen3DGFFSheffield(m,n,o)
print(G3)
for i in range(o):
    plotGFF(G3[i],m,n)
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
