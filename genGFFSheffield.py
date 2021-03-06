import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from numba import njit


def l2(j,k,m,n):
    if j+k==0:
        return 0
    else:
        return 1/np.sqrt((np.sin(j*np.pi/(2*m)))**2+(np.sin(k*np.pi/(2*n)))**2)

@njit
def l3(j,k,l,m,n,o):
    if j+k+l == 0:
        return 0
    else:
        return 1/np.sqrt((np.sin(j*np.pi/(m)))**2+(np.sin(k*np.pi/(n)))**2+(np.sin(l*np.pi/(o)))**2)

def gen2DGFFSheffield(m,n):
    table = np.zeros((m,n))
    # l2vec = np.vectorize(l2)
    # table = l2vec(table[0],table[1],m,n)
    for i in range(m):
        for j in range(n):
            table[i][j] = l2(i,j,m,n)
    Z = np.random.normal(0,1/2,m*n)+1j*np.random.normal(0,1/2,m*n)
    Z = Z.reshape(m,n)
    table = np.multiply(table,Z)
    table = np.fft.ifft2(table)*np.sqrt(m*n)
    return np.real(table)

@njit
def tab3(m,n,o):
    table = np.zeros((m,n,o))
    for i in range(m):
        for j in range(n):
            for k in range(o):
                table[i][j][k] = l3(i,j,k,m,n,o)
    return table

def gen3DGFFSheffield(m,n,o):
    table = tab3(m,n,o)
    Z = np.random.normal(0,1/2,size=m*n*o)+1j*np.random.normal(0,1/2,size=m*n*o)
    Z = Z.reshape(m,n,o)
    table = np.multiply(table,Z)
    table = np.fft.ifftn(table)*np.sqrt(m*n*o)
    return np.real(table)

def plotGFF(GZ,m,n):
    X = np.arange(1,m)
    Y = np.arange(1,n)
    Z = GZ
    fig = plt.figure()
    ax = Axes3D(fig)
    # surf = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
    df = pd.DataFrame({'x': np.repeat(np.arange(1,m+1),n), 'y': np.tile(np.arange(1,n+1),m), 'z': GZ.flatten()})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
    # surf = ax.plot_trisurf(np.repeat(np.arange(1,L),L-1), np.tile(np.arange(1,L),L-1), GZ.flatten(),  cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)


if __name__ == "__main__":
    m = 10
    n = 10
    o = 10

    G2 = gen2DGFFSheffield(m,n)
    plotGFF(G2,m,n)
    plt.show()
    #
    # G3 = gen3DGFFSheffield(m,n,o)
    # print(G3.reshape((m*n*o)).sum())
    # for i in range(o):
    #     plotGFF(G3[i],m,n)
    #     plt.show()

    for i in range(100):
        print(gen3DGFFSheffield(m,n,o).reshape(m*n*o).sum())

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
        perc = ber(G3,15,2,50*(h-5))
        plt.subplot(2,5,h+1)
        plt.imshow(perc, alpha=0.8)
        #plt.xticks(np.arange(m))
        #plt.yticks(np.arange(n))
        plt.title('h =' + str((h-5)*100))

    plt.show()