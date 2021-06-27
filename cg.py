import numpy as np
import time
from matrices import *
import math
import scipy
from numba import jit, njit
import genGFFSheffield as ggff
from matplotlib import pyplot as plt
import conductances as cond
import direct_sampling as ds

@njit
def cgpf0(C, wx, wy, wz, N, x0, epsilon = 0, kmax = 1e324, M=1):
    """CG sampler with ZBC (variation 2)."""

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
            x_old[i] = 0



    # Kd = 1e300
    # Kr = 1e81

    r_old = b - C(x_old, wx, wy, wz, N)
    # print(np.linalg.norm(r_old))
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, wx, wy, wz, N))
    y_old = x_old

    k = 1
    conjloss = 0

    while np.linalg.norm(r_old) >= epsilon and k <= kmax: # kmax is N-dependent, for N=5 1000 gives enough margin.
    #while np.linalg.norm(b-C(x_old, wx, wy, wz, N)) >= 3e-13 and k <= 250:  # kmax is N-dependent, for N=5 1000 gives enough margin.
        gamma = np.dot(r_old, r_old) / (d_old)
        x_new = x_old + gamma * p_old
        z = np.random.normal(0, 1)
        y_new = y_old + z / np.sqrt(d_old) * p_old
        r_new = r_old - gamma * C(p_old, wx, wy, wz, N)
        beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
        p_new = r_new - beta * p_old
        # print(k, np.linalg.norm(b-C(x_old, wx, wy, wz, N)))
        # print(k,np.linalg.norm(b-C(x_old, wx, wy, wz, N)),np.linalg.norm(r_old), np.linalg.norm(p_old), gamma, y_new, beta, p_new, d_old)
        d_new = np.dot(p_new, C(p_new, wx, wy, wz, N))
        # print(k,np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))))
        """
        if np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))) >= M**2*1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            l = k
            conjloss = 1
            break
        """
        # print(k,
        #       np.dot(Kr*r_old,Kr*r_old),
        #       np.linalg.norm(p_old),
        #       d_old,
        #       gamma,
        #       # np.linalg.norm(y_new),
        #       np.linalg.norm(r_new),
        #       beta,
        #       np.abs(np.dot(p_new, C(p_old, N))),
        #       np.linalg.norm(p_new),
        #       d_old)
        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

        if k >= 250:
            conjloss = 0

    return y_old, conjloss, k

#@njit
def cgpf02(C, wx, wy, wz, N, x0, epsilon = 0, kmax = 1e324, M=1):
    """CG sampler with ZBC (variation 2)."""

    b = np.zeros(N ** 3)
    for i in range(N ** 3):
        b[i] = M*np.random.normal(0,1)

    x_old = x0

    # Kd = 1e300
    # Kr = 1e81

    r_old = b - C(x_old, wx, wy, wz, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, wx, wy, wz, N))
    y_old = x_old

    k = 1
    conjloss = 0

    while np.linalg.norm(r_old) >= epsilon and k <= kmax: # kmax is N-dependent, for N=5 1000 gives enough margin.
    #while np.linalg.norm(b - C(x_old, wx, wy, wz, N)) >= 3e-13 and k <= 250:
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

        # print(k,
        #       np.dot(Kr*r_old,Kr*r_old),
        #       np.linalg.norm(p_old),
        #       d_old,
        #       gamma,
        #       # np.linalg.norm(y_new),
        #       np.linalg.norm(r_new),
        #       beta,
        #       np.abs(np.dot(p_new, C(p_old, N))),
        #       np.linalg.norm(p_new),
        #       d_old)
        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    # gamma = np.dot(r_old, r_old) / (d_old/1e20)
    # print(gamma)
    # z = np.random.normal()
    # print(z)
    # y_new = y_old + z * p_old / np.sqrt(d_old/1e20)
    # print(y_new)
    # r_new = r_old - gamma * C(p_old, N)
    # print(r_new)
    # beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
    # print(beta)
    # p_new = r_new - beta * p_old
    # print(p_new)
    # d_new = np.dot(p_new, C(p_new, N))
    # print(d_new)

    if k >= kmax:
        conjloss = 1

    return y_old/M, conjloss

@njit
def cgpf0acc(C, wx, wy, wz, N, epsilon = 0, kmax = 1e324, M = 1e150):
    """CG sampler with PBC (variation 2)."""
    b = np.zeros((N + 2) ** 3)
    x_old = np.zeros((N + 2) ** 3)

    for i in range((N + 2) ** 3):
        b[i] = M * np.random.normal(0, 1)
        x_old[i] = M * np.random.normal(0, 1)

    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            b[i] = 0
            x_old[i] = 0

    # Kd = 1e300
    # Kr = 1e81

    r_old = b - C(x_old, wx, wy, wz, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, wx, wy, wz, N))
    y_old = x_old

    k = 1
    conjloss = 0

    gammalist = []
    dlist = [d_old]
    rlist = [np.linalg.norm(r_old)]

    while np.linalg.norm(r_old) >= epsilon and k <= kmax:  # kmax is N-dependent, for N=5 1000 gives enough margin.
        gamma = np.dot(r_old, r_old) / (d_old)
        z = np.random.normal(0, 1)
        y_new = y_old + z / np.sqrt(d_old) * p_old
        r_new = r_old - gamma * C(p_old, wx, wy, wz, N)
        beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
        p_new = r_new - beta * p_old
        print(k,np.linalg.norm(r_old), np.linalg.norm(r_new), y_new, beta, p_new, d_old, np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))))
        d_new = np.dot(p_new, C(p_new, wx, wy, wz, N))
        # print(np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))))

        if np.abs(np.dot(p_new, C(p_old, wx, wy, wz, N))) >= M**2*1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            l = k
            conjloss = 1
            break

        # print(k,
        #       np.dot(Kr*r_old,Kr*r_old),
        #       np.linalg.norm(p_old),
        #       d_old,
        #       gamma,
        #       # np.linalg.norm(y_new),
        #       np.linalg.norm(r_new),
        #       beta,
        #       np.abs(np.dot(p_new, C(p_old, N))),
        #       np.linalg.norm(p_new),
        #       d_old)
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

        gammalist.append(gamma)
        dlist.append(d_new)
        rlist.append(np.linalg.norm(r_new))

    if conjloss == 0:
        l = k

    # print(gammalist)
    # print(dlist)
    # print(rlist)
    traceT = 0
    for j in range(0,l-1):
        for i in range(j,l-1):
            # print('T  ', gammalist[i] * rlist[i] ** 2 / (rlist[j] ** 2))
            traceT += gammalist[i] * rlist[i] ** 2 / (rlist[j] ** 2)
    traceA = 0
    for i in range(0,l-1):
        # print('A  ', gammalist[i] * rlist[i] ** 2)
        traceA += gammalist[i] * rlist[i] ** 2
    return y_old/M, x_old/M , traceT, traceA/M**2, b/M, l, conjloss, np.linalg.norm(r_new)

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

if __name__ == '__main__':
    # a = 0.001
    # b = 1

    # wx, wy, wz = cond.wxyz(N, 0, 0)

    # for N in range(2,100):
    #     start = time.time()
    #     wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
    #
    #     e1 = np.zeros(N**3)
    #     e1[0] = 1
    #
    #     A = C2(e1,wx,wy,wz,N)
    #     for c in range(N**3-1):
    #         ec = np.zeros(N**3)
    #         ec[c+1] = 1
    #
    #         Ac = C2(ec,wx,wy,wz,N)
    #         A = np.append(A,Ac)
    #     A = A.reshape(N**3,N**3)
    #     end = time.time()
    #
    #     print(np.linalg.cond(A),end-start)

    N = 10
    wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)

    def M(x):
        N = 10
        wx, wy, wz = np.ones((N + 2) ** 3), np.ones((N + 2) ** 3), np.ones((N + 2) ** 3)
        x  = x.reshape(N**3)
        return C2(x, wx, wy, wz, N).reshape((N**3,1))

    y=cgpf0(C, wx, wy, wz, N, x0 = np.zeros((N+2)**3), kmax=12**3)[0]
    # z=ds.sampler_CG(np.zeros(N**3),M,tol=1e-4,K=1000,init=np.random.normal(size=N**3))
    # print(y)
    # print(cut(y,N+2,1).reshape(N,N,N))
    for i in range(N):
        # ggff.plotGFF(y.reshape(N,N,N)[i],N,N)
        # plt.show()
        ggff.plotGFF(y.reshape(N+2, N+2, N+2)[i], N+2, N+2)
        plt.show()
        # ggff.plotGFF(cut(y,N+2,1).reshape(N, N, N)[i], N, N)
        # plt.show()
"""
if __name__ == '__main__':
    N = 40
    epsilon = np.array([1e-160]) # N = 5: 1e75 good; N=10: 1e-28 workable, 1e-30 best; N=20: 1e-300 soms fout (0.7 max score)
    u = np.arange(N**3)#len(epsilon))
    mmax = 1
    M = 1

    mtma = np.zeros(N**3)#len(epsilon))
    larr = np.zeros(N**3)#len(epsilon))

    for i in range(N**3-1,N**3):
        mT = 0
        mA = 0
        lav = 0
        m = 0

        wx = np.ones((N + 2) ** 3)
        wy = np.ones((N + 2) ** 3)
        wz = np.ones((N + 2) ** 3)

        #y, x, t, a, b, l, c = cgpf0acc(C, wx, wy, wz, N, epsilon[i], M)
        #for j in range(N**3):
        #    print(cut(C(x, wx, wy, wz, N), N+2, 1)[j], cut(b, N+2, 1)[j], cut(C(x, wx, wy, wz, N), N+2, 1)[j]-cut(b, N+2, 1)[j])

        while m < mmax:
            try:
                y,x,t,a,b,l,c,r = cgpf0acc(C, wx, wy, wz, N, 1e-160, kmax=i, M=1)
                if c == 0:
                    mT += t
                    mA += a
                    lav += l
                    # print(cut(C(x,wx,wy,wz,N),N,1),cut(b,N,1))
                    if m == mmax-1:
                        print(i, l, mT/mA, t, a, r)
                    m += 1
            except ZeroDivisionError:
                print("fout")
                pass
        mtma[i] = mT/mA
        larr[i] = lav/mmax

    plt.plot(u,larr)
    plt.show()
    plt.plot(u,mtma)
    plt.show()
"""