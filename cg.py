import numpy as np
import time
from matrices import *
import math
import scipy
from numba import jit, njit
import genGFFSheffield as ggff
from matplotlib import pyplot as plt
import conductances as cond

@njit
def cgpf0(C, wx, wy, wz, N, x0, epsilon = 0, kmax = 1e324, M=1):
    """CG sampler with ZBC."""

    b = np.zeros((N + 2) ** 3)
    x_old = x0

    for i in range((N + 2) ** 3):
        b[i] = M*np.random.normal(0,1)
        # x_old[i] = M*np.random.normal(0,1)


    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            b[i] = 0
            # x_old[i] = 0

    r_old = b - C(x_old, wx, wy, wz, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, wx, wy, wz, N))
    y_old = x_old

    k = 1
    conjloss = 0

    while np.linalg.norm(r_old) >= epsilon and k <= kmax:
        gamma = np.dot(r_old, r_old) / (d_old)
        x_new = x_old + gamma * p_old
        z = np.random.normal(0, 1)
        y_new = y_old + z / np.sqrt(d_old) * p_old
        r_new = r_old - gamma * C(p_old, wx, wy, wz, N)
        beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
        p_new = r_new - beta * p_old
        # print(k,np.linalg.norm(b-C(x_old, wx, wy, wz, N)),np.linalg.norm(r_old), np.linalg.norm(p_old), gamma, y_new, beta, p_new, d_old)
        d_new = np.dot(p_new, C(p_new, wx, wy, wz, N))

        if np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))) >= M**2*1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            l = k
            conjloss = 1
            break

        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

        if k >= 250:
            conjloss = 0

    return y_old, conjloss, k # y70, y80, y90, y100, y110, conjloss, k

#@njit
def cgpf02(C, wx, wy, wz, N, x0, epsilon = 0, kmax = 1e324, M=1):
    """CG sampler with ZBC (alternative version)."""

    b = np.zeros(N ** 3)
    for i in range(N ** 3):
        b[i] = M*np.random.normal(0,1)

    x_old = x0

    r_old = b - C(x_old, wx, wy, wz, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, wx, wy, wz, N))
    y_old = x_old

    k = 1
    conjloss = 0

    while np.linalg.norm(r_old) >= epsilon and k <= kmax:
        gamma = np.dot(r_old,r_old)/(d_old)
        x_new = x_old + gamma * p_old
        z = np.random.normal(0,1)
        y_new = y_old + z / np.sqrt(d_old) * p_old
        r_new = r_old - gamma*C(p_old, wx, wy, wz, N)
        beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
        p_new = r_new-beta*p_old
        # print(k,np.linalg.norm(b - C(x_old, wx, wy, wz, N)),np.linalg.norm(r_old), np.linalg.norm(r_new), y_new, beta, p_new, d_old)
        d_new = np.dot(p_new,C(p_new, wx, wy, wz, N))

        if np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))) >= M**2*1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            l = k
            conjloss = 1
            break

        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    if k >= kmax:
        conjloss = 1

    return y_old/M, conjloss

def cut(x, N, n):
    """Cut the outer n layers from a NxNxN GFF realisation x."""
    x_cut = []

    for i in range(N**3):
        if np.mod(i, N) < n \
                or np.mod(i, N) > N - n - 1 \
                or np.mod(i, N ** 2) < N * n \
                or np.mod(i, N ** 2) > N ** 2 - N * n - 1 \
                or np.mod(i, N ** 3) < N ** 2 * n \
                or np.mod(i, N ** 3) > N ** 3 - N ** 2 * n - 1:
            pass
        else:
            x_cut.append(x[i])

    return np.array(x_cut)


# TESTING AREA
if __name__ == '__main__':
    N = 20
    wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)

    def M(x):
        N = 20
        wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
        x  = x.reshape(N**3)
        return C2(x, wx, wy, wz, N).reshape((N**3,1))

    y=cgpf0(C, wx, wy, wz, N, x0 = np.random.normal(size=(N+2)**3), epsilon=1e-90)[0] #np.zeros((N+2)**3), epsilon = 1e-90)[0]
    for i in range(N):
        ggff.plotGFF(y.reshape(N+2, N+2, N+2)[i], N+2, N+2)
        plt.show()

