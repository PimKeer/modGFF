import numpy as np
#from itertools import product
import time

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import pandas as pd
# from numba import jit

def e(L,n1,n2,n3,k1,k2,k3):
    """Eigenvectors of discrete Laplace operator."""
    return 2 ** (3/2) * np.sin(np.pi * n1 * k1 / L) * np.sin(np.pi * n2 * k2 / L) * np.sin(np.pi * n3 * k3 / L)

def sl(L,n1,n2,n3):
    """Eigenvalues of discrete Laplace operator."""
    lnL = 2 * ((np.sin((np.pi * n1) / (2 * L))) ** 2 + (np.sin((np.pi * n2) / (2 * L))) ** 2 + (np.sin((np.pi * n3) / (2 * L))) ** 2)
    return 1/np.sqrt(lnL)
"""
L = 5
evec = np.vectorize(e)
n = np.indices((L-1,L-1,L-1))+1
print('hallo', e(L,n[0],n[1],n[2],1,2,3))
"""

def sqrtG(L,x1,x2,x3,y1,y2,y3):
    G = 0
    n = np.indices((L-1,L-1,L-1))+1
    G = (sl(L,n[0],n[1],n[2]) * e(L,n[0],n[1],n[2],x1,x2,x3) * e(L,n[0],n[1],n[2],y1,y2,y3)).sum()
    return G

def sqrtG2(L,x1,x2,x3,y1,y2,y3):
    G = 0
    for n1 in range(1,L):
        for n2 in range(1,L):
            for n3 in range(1,L):
                G += sl(L,n1,n2,n3) * e(L,n1,n2,n3,x1,x2,x3) * e(L,n1,n2,n3,y1,y2,y3)
    return G

def savesqrtG(L):
    sqrtGvec = np.vectorize(sqrtG)
    xy = np.indices((L - 1, L - 1, L - 1, L - 1, L - 1, L - 1)) + 1
    sG = sqrtGvec(L,xy[0],xy[1],xy[2],xy[3],xy[4],xy[5])
    np.save('test_gff_sG_'+str(L)+'.txt', sG)

for L in range(2,21):
    start = time.time()
    savesqrtG(L)
    end = time.time()
    print(end-start)
"""
L=100
start1 = time.time()
print(sqrtG(L,1,3,2,1,2,3))
end1 = time.time()
start2 = time.time()
print(sqrtG2(L,1,3,2,1,2,3))
end2 = time.time()
print(end1-start1)
print(end2-start2)


L = 5
y = np.indices((L-1,L-1,L-1))+1
sqrtGvec = np.vectorize(sqrtG)
print('test', sqrtGvec(L,2,3,1,y[0],y[1],y[2]))
"""

def genGFFChafai1(L):
    """Generate a GFF directly, in the case that no sqrtG file is available."""
    sqrtGvec = np.vectorize(sqrtG)
    GZ = np.zeros((L-1,L-1,L-1))
    Z = np.random.normal(0,1,size=(L-1,L-1,L-1))
    x = np.indices((L-1,L-1,L-1))+1
    y = np.indices((L-1,L-1,L-1))+1
    for x1 in range(1,L):
        for x2 in range(1,L):
            for x3 in range(1,L):
                # print(x1,x2,x3)
                GZ[x1-1][x2-1][x3-1] = (np.multiply(sqrtGvec(L,x1,x2,x3,y[0],y[1],y[2]),Z)).sum()
                # print(np.multiply(sqrtG(L,x1,x2,x3,y[0],y[1],y[2]),Z).sum())
    """
    for x1 in range(1,L):
        for x2 in range(1,L):
            for x3 in range(1,L):
                GZ[x1-1][x2-1][x3-1] = 0
                for y1 in range(1,L):
                    for y2 in range(1,L):
                        for y3 in range(1,L):
                            print(x1,x2,x3,y1,y2,y3)
                            GZ[x1-1][x2-1][x3-1] += sqrtG(L,x1,x2,x3,y1,y2,y3) * Z[y1-1][y2-1][y3-1]
    """
    return GZ

def genGFFChafai2(L,sG):
    # sG = np.load('gff_sG_'+str(L)+'.npy')
    GZ = np.zeros((L - 1, L - 1, L - 1))
    Z = np.random.normal(0, 1, size=(L - 1, L - 1, L - 1))
    for x1 in range(1, L):
        for x2 in range(1, L):
            for x3 in range(1, L):
                GZ[x1 - 1][x2 - 1][x3 - 1] = np.sum(np.multiply(sG[x1 - 1][x2 - 1][x3 - 1], Z))
    return GZ
"""
def plotGFF(GZ,L,d):
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


L = 10

y = genGFFChafai2(L)
print(y)
print(np.shape(y))
for l in range(L-1):
    plotGFF(y[l],L,2)
    plt.show()


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