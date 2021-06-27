import numpy as np
from matplotlib import pyplot as plt

N = 30
parr = np.array([0.1])
plen = 10
for p in range(plen):
    x = np.random.random(N**2)
    for i in range(N**2):
        if x[i] <= parr[0]:
            x[i] = 1
        else:
            x[i] = 0
    plt.imshow(x.reshape(N,N), cmap="Greys", alpha=1)
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.show()