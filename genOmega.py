import matplotlib.pyplot as plt
import time
from scipy import sparse
import numpy as np

N = int(10)
n, m = N, N
density = 0.1
size = int(n * m * density)

rows = np.random.randint(0, n, size=size)
cols = np.random.randint(0, m, size=size)
data = np.random.rand(size)

arr = sparse.csr_matrix((data, (rows, cols)), shape=(n, m))
print(arr.A)

Z = np.random.multivariate_normal(np.zeros(N),arr.A*arr.A.T)
time.time()
print(Z)
plt.plot(Z)
plt.show()
