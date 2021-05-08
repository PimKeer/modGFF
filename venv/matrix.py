import tables as tb
import numpy as np
import PyGauss.pygauss.direct_sampling as pg
from scipy import sparse
import time as time
import h5py as h

"""
ndim = 10e9
h5file = tb.File('test.h5', mode='w', title="Test Array")
root = h5file.root
x = h5file.create_carray(root,'x',tb.Float64Atom(),shape=(ndim,ndim))
x[:100,:100] = np.random.random(size=(100,100)) # Now put in some data
h5file.close()
"""

def a(i, N):
    if np.mod(i, N) == N-1:
        a = i - N + 1
    else:
        a = i + 1
    return a
va = np.vectorize(a, excluded=['N'])

def b(i, N):
    if np.mod(i, N ** 2) >= N ** 2 - N:
        b = i - N ** 2 + N
    else:
        b = i + N
    return b
vb = np.vectorize(b, excluded=['N'])

def c(i, N):
    if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
        c = i - N ** 3 + N ** 2
    else:
        c = i + N ** 2
    return c
vc = np.vectorize(c, excluded=['N'])





def Matrix(N):
    N = int(N)

    M = sparse.dok_matrix((N ** 3, N ** 3), dtype=np.float32)

    for i in range(N**3):
        M[i, i] = -3
        if np.mod(i, N) == N - 1:
            M[i , i - N + 1] = 1 - np.random.random(size = 1)
        else:
            M[i, i + 1] = 1 - np.random.random(size = 1)
        if np.mod(i, N**2) >= N**2 - N:
            M[i, i - N**2 + N] = 1 - np.random.random(size = 1)
        else:
            M[i, i + N] = 1 - np.random.random(size = 1)
        if np.mod(i, N**3) >= N**3 - N**2:
            M[i, i - N**3 + N**2] = 1 - np.random.random(size = 1)
        else:
            M[i, i + N**2] = 1 - np.random.random(size = 1)

    return M



def anotherMatrix(N):
    N = int(N)

    M = sparse.dok_matrix((N ** 3, N ** 3), dtype=np.float32)

    for i in range(N ** 3):
        M[i, i] = -3
        if np.mod(i, N) == N - 1:
            M[i, i - N + 1] = 1 - np.random.random(size=1)
        if np.mod(i, N ** 2) >= N ** 2 - N:
            M[i, i - N ** 2 + N] = 1 - np.random.random(size=1)
        if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
            M[i, i - N ** 3 + N ** 2] = 1 - np.random.random(size=1)
        else:
            M[i, i + 1] = 1 - np.random.random(size=1)
            M[i, i + N] = 1 - np.random.random(size=1)
            M[i, i + N ** 2] = 1 - np.random.random(size=1)
    return M

def genMatrix(N):
    """Runtime: O(n^3)"""
    N = int(N)

    d = {}
    for i in range(N):
        print(i)
        for j in range(N):
            for k in range(N):
                ref = i+j*N+k*N**2
                d[ref] = [ref,
                          np.mod(i + 1, N) + j * N + k * N ** 2,
                          i + np.mod(j + 1, N) * N + k * N ** 2,
                          i + j * N + np.mod(k + 1, N) * N ** 2]
    return d
    """
    M = sparse.dok_matrix((N**3,N**3), dtype = np.float32)

    M.update(d)
    
    for x,y in d.items():
        M[x, y[0]] = -3
        M[x, y[1:4]] = 1 - np.random.random(size = 3)
    
    return M
    """
N = 1000
"""
start = time.time()
M = genMatrix(N)
end = time.time()
print(end-start)

start = time.time()
M = Matrix(N)
end = time.time()
print(end-start)

start = time.time()
M = anotherMatrix(N)
end = time.time()
print(end-start)

M = M.tocsr()
M = M + M.transpose()

L = int(1e9)
K = int(1e6)

A = sparse.dok_matrix((L,L), dtype = np.float32)

for x in range(int(L/K)):
    rows = np.repeat(np.arange(start = x*K, stop = (x+1)*K), 3)
    cols = np.tile(np.arange(3),K)
    A[rows, cols] = 1 - np.random.random(size = int(3*K))
    print(x)
"""
start = time.time()
x = np.arange(N**3)
A = va(x, N)
B = vb(x, N)
C = vc(x, N)
"""
print(x)
print(A)
print(B)
print(C)
"""
p = np.dstack((x,A,B,C))
#print(p)

s = -6*np.ones(N**3)
wa = 1 - np.random.random(N**3)
wb = 1 - np.random.random(N**3)
wc = 1 - np.random.random(N**3)

v = np.dstack((s,wa,wb,wc))
#print(v)
end = time.time()
print(end-start)
