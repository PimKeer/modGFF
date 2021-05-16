import numpy as np
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

def e(n,L,k,d):
    """Eigenvectors of discrete Laplace operator."""
    return 2**(d/2)*np.prod(np.sin(np.pi*np.multiply(n,k/L)))

def l(n,L):
    """Eigenvalues of discrete Laplace operator."""
    return -2*np.sum((np.sin(np.pi*n/(2*L)))**2)

def sqrtG(L,d,x,y):
    """Compute a square root of G_A for given x,y in A."""
    s = 0
    for n in product(np.arange(1,L), repeat=d):
        n = np.array(list(n))
        s += e(n,L,x,d)*e(n,L,y,d)/np.sqrt(-l(n,L))
    return s

def genGFFChafai(L,d):
    """Generate a d-dimensional Gaussian Free Field on the square {1,...,L-1}^d."""
    GZ = np.zeros(0)
    for x in product(np.arange(1,L), repeat=d):
        s = 0
        for y in product(np.arange(1,L), repeat=d):
            s += sqrtG(L,d,np.array(list(x)),np.array(list(y)))*np.random.normal(size=1)
        GZ = np.append(GZ,s)
        print(x)
    GZ = np.resize(GZ, d*(L-1,))
    return GZ

def plotGFF(GZ,L,d):
    """Plot a given 1D or 2D Gaussian Free Field."""
    if d == 1:
        plt.plot(np.arange(1,L),GZ)
    elif d == 2:
        fig = plt.figure()
        ax = Axes3D(fig)
        #surf = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
        df = pd.DataFrame({'x': np.repeat(np.arange(1,L),L-1), 'y': np.tile(np.arange(1,L),L-1), 'z': GZ.flatten()})
        surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
        #surf = ax.plot_trisurf(np.repeat(np.arange(1,L),L-1), np.tile(np.arange(1,L),L-1), GZ.flatten(),  cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        print('Please enter a 1D or 2D GFF.')

L = 15

y = genGFFChafai(10,3)
print(y)
"""
g = genGFFChafai(L,2)
print(g)
print(np.size(g))
print(np.repeat(np.arange(1,L),L-1))
print(np.tile(np.arange(1,L),L-1))
print(g.flatten())
print(len(np.repeat(np.arange(1,L),L-1)))
print(len(np.tile(np.arange(1,L),L-1)))
print(len(g.flatten()))
print()
plotGFF(g,L,2)
plt.show()
"""