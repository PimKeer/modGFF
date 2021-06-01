import numpy as np
import time
from matrices import *
import math
import scipy
from numba import jit, njit
import genGFFSheffield as ggff
from matplotlib import pyplot as plt

#np.random.seed(1)

#@jit(nopython=True)
def cg(A, N, k_max = 10,epsilon=False):
    """CG sampler with PBC (variation 1)."""
    c = np.random.normal(size = N**3)

    r_old = c
    p_old = r_old
    d_old = np.dot(p_old, A(p_old, N))
    y_old = np.zeros(N**3)

    conjloss = 0

    gammas = np.zeros(k_max)
    ds = np.zeros(k_max+1)
    ds[0] = d_old
    rolds = np.zeros(k_max+1)
    rolds[0] = np.dot(r_old,r_old)

    #r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    #r_array = r0 * np.zeros(k_max+1)
    #t = np.zeros(k_max+1)
    if epsilon:
        pass
    else:
        for k in range(1,k_max+1):
            #start = time.time()
            gamma = np.dot(r_old,r_old)/d_old
            r_new = r_old - gamma * A(p_old, N)
            beta = - np.dot(r_new,r_new)/(gamma * d_old)
            p_new = r_new - beta * p_old

            if np.abs(np.dot(p_new, A(p_old, N))) >= 1e-4 and conjloss == 0:
                print("Loss of conjugacy at iteration ", k)
                conjloss = 1

            d_new = np.dot(p_new, A(p_new, N))
            z = np.random.normal(size = 1)
            y_new = y_old + z/np.sqrt(d_old) * p_old

            gammas[k-1] = gamma
            ds[k] = d_new
            rolds[k] = np.dot(r_new,r_new)

            r_old = r_new
            p_old = p_new
            d_old = d_new
            y_old = y_new

            #end = time.time()
            #t[k] = end-start

            #r_array[k] = np.sqrt(np.dot(r_old,r_old))

    return y_old#, r_array, t

def cgpf(A, N, epsilon):
    """CG sampler with PBC (variation 2)."""
    x_old = np.zeros(N ** 3)
    #b = np.random.binomial(1,0.5,size=N ** 3)*2-1
    #b = b / np.sqrt(np.dot(b,b))
    b = np.random.normal(size=N ** 3)
    #b = b/np.linalg.norm(b)

    r_old = b - A(x_old, N)
    p_old = r_old
    d_old = np.dot(p_old, A(p_old, N))
    # print(d_old)
    y_old = x_old

    k = 1
    conjloss = 0

    gammalist = []
    dlist = [d_old]
    rlist = [np.sqrt(np.dot(r_old,r_old))]

    while np.dot(r_old,r_old) >= epsilon:
        print(k, np.dot(r_old,r_old))
        gamma = np.dot(r_old,r_old)/d_old
        x_new = x_old + gamma*p_old
        z = np.random.normal(0,1,size=1)
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*A(p_old,N)
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,A(p_new,N))

        if np.abs(np.dot(p_new, A(p_old, N))) >= 1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            conjloss = 1
            l = k

        gammalist.append(gamma)
        dlist.append(d_new)
        rlist.append(np.linalg.norm(r_new))

        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    if conjloss == 0:
        l = k

    # print(gammalist)
    # print(dlist)
    # print(rlist)
    traceT = 0
    for j in range(l-1):
        for i in range(j,l-1):
            traceT += gammalist[i] * rlist[i] ** 2 / (rlist[j]) ** 2
    traceA = 0
    for i in range(l-1):
        traceA += gammalist[i] * rlist[i] ** 2
    return y_old, x_old , traceT, traceA, b
"""
N = 4
epsilon = np.array([1e-4])
M = 10000

mtma = np.zeros(len(epsilon))
for i in range(len(epsilon)):
    mT = 0
    mA = 0
    for m in range(M):
        y,x,t,a,b = cgpf(C, N, epsilon[i])
        mT += t
        mA += a
        print('trace', m, mT/mA)
    mtma[i] = mT/mA
print(mtma)
for n in range(N):
   ggff.plotGFF(y.reshape((N,N,N))[n],N,N)
   plt.show()
"""
@njit
def C(x, N):
    """Returns the matrix vector product Cx, where C is the zero boundary precision matrix."""
    y = np.zeros((N+2) ** 3)

    wx = np.ones((N+2) ** 3)
    wy = np.ones((N+2) ** 3)
    wz = np.ones((N+2) ** 3)

    for i in range((N+2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            y[i] = 0
        else:
            y[i] = x[i]

            y[i] -= 1/6 * wx[i] * x[i + 1]

            y[i] -= 1/6 * wx[i - 1] * x[i - 1]

            y[i] -= 1/6 * wy[i] * x[i + (N + 2)]

            y[i] -= 1/6 * wy[i - (N + 2)] * x[i - (N + 2)]

            y[i] -= 1/6 * wz[i] * x[i + (N + 2) ** 2]

            y[i] -= 1/6 * wz[i - (N + 2) ** 2] * x[i - (N + 2) ** 2]

    return y

@njit
def cgpf0(C, N, epsilon = 0, kmax = 1e324):
    """CG sampler with PBC (variation 2)."""

    b = np.zeros((N + 2) ** 3)
    for i in range((N + 2) ** 3):
        b[i] = np.random.normal()

    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            b[i] = 0
    x_old = np.zeros((N + 2) ** 3)

    r_old = b # - C(x_old, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, N))
    # print(d_old)
    y_old = x_old

    k = 1
    conjloss = 0

    #if N <= 10:
    #    kmax = 550
    #elif N > 1:

    while np.linalg.norm(r_old) > epsilon and k <= kmax: # kmax is N-dependent, for N=5 1000 gives enough margin.
    # while k <= epsilon:
        if d_old == 0:
            print()
            break
        gamma = np.dot(r_old,r_old)/d_old
        z = np.random.normal()
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*C(p_old,N)
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,C(p_new,N))
        # print(np.abs(np.dot(p_new, C(p_old, N))))

        # if math.isnan(beta):
        #     conjloss = 1
        #     break

        if np.abs(np.dot(p_new, C(p_old, N))) >= 1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            l = k
            conjloss = 1
            break

        print(k,
              np.linalg.norm(r_old),
              np.linalg.norm(p_old),
              d_old,
              gamma,
              np.linalg.norm(y_new),
              np.linalg.norm(r_new),
              beta,
              np.abs(np.dot(p_new, C(p_old, N))),
              np.linalg.norm(p_new),
              d_old)
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    gamma = np.dot(r_old, r_old) / d_old
    # print(gamma)
    z = np.random.normal()
    # print(z)
    y_new = y_old + z * p_old / np.sqrt(d_old)
    # print(y_new)
    r_new = r_old - gamma * C(p_old, N)
    # print(r_new)
    beta = - np.dot(r_new, r_new) / np.dot(r_old, r_old)
    # print(beta)
    p_new = r_new - beta * p_old
    # print(p_new)
    d_new = np.dot(p_new, C(p_new, N))
    # print(d_new)

    if k >= kmax:
        conjloss = 1

    return y_old, conjloss

@njit
def cgpf0acc(C, N, epsilon = 0, kmax = 1e324):
    """CG sampler with PBC (variation 2)."""

    b = np.zeros((N + 2) ** 3)
    for i in range((N + 2) ** 3):
        b[i] = np.random.normal()

    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            b[i] = 0
    x_old = np.zeros((N + 2) ** 3)

    r_old = b - C(x_old, N)
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, N))
    # print(d_old)
    y_old = x_old

    k = 1
    conjloss = 0

    if N <= 10:
        klim = 550
    elif N > 10 and N <= 20:
        klim = 2000

    gammalist = []
    dlist = [d_old]
    rlist = [np.linalg.norm(r_old)]

    while np.linalg.norm(r_old) > epsilon and k <= kmax:
    # while k <= epsilon:
        # print(np.linalg.norm(r_old))
        if d_old == 0:
            print(k)
            break
        gamma = np.dot(r_old,r_old)/d_old
        x_new = x_old + gamma*p_old
        z = np.random.normal(0,1,size=1)
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*C(p_old,N)
        # print(r_new[2],r_new[N+2],r_new[(N+2)**2+2],r_new[(N+2)**3-N])
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,C(p_new,N))

        if np.abs(np.dot(p_new, C(p_old, N))) >= 1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            conjloss = 1
            l = k
            break

        if k >= klim:
            l=k
            conjloss = 1
            break

        gammalist.append(gamma)
        dlist.append(d_new)
        rlist.append(np.linalg.norm(r_new))

        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

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
    return y_old, x_old , traceT, traceA, b, l, conjloss

def cut(x, L, l):
    """Cut the outer l layers from a LxLxL GFF realisation x."""
    x_cut = []

    for i in range(L**3):
        if np.mod(i, L) < l \
                or np.mod(i, L) > L - l - 1 \
                or np.mod(i, L ** 2) < L * l \
                or np.mod(i, L ** 2) > L ** 2 - L * l - 1 \
                or np.mod(i, L ** 3) < L ** 2 * l \
                or np.mod(i, L ** 3) > L ** 3 - L ** 2 * l - 1:
            pass
        else:
            x_cut.append(x[i])

    return np.array(x_cut)
"""
if __name__ == '__main__':
    N = 20
    tick = time.time()
    y=cgpf0(C, N, epsilon = 4e-161)[0]
    tock = time.time()
    print(tock-tick)
    # print(y)
    # print(cut(y,N+2,1))
    # for i in range(N):
    #     ggff.plotGFF(y.reshape(N+2,N+2,N+2)[i],N+2,N+2)
    #     plt.show()
    #     ggff.plotGFF(cut(y,N+2,1).reshape(N, N, N)[i], N, N)
    #     plt.show()
"""
if __name__ == '__main__':
    N = 10
    epsilon = np.array([1e-162])
    u = np.arange(len(epsilon))
    M = 20

    mtma = np.zeros(len(epsilon))
    larr = np.zeros(len(epsilon))

    for i in range(len(epsilon)):
        mT = 0
        mA = 0
        lav = 0
        m = 0
        while m < M:
            try:
                y,x,t,a,b,l,c = cgpf0acc(C, N, epsilon[i])
                if c == 0:
                    mT += t
                    mA += a
                    lav += l
                    print(i, l, mT/mA, t, a)
                    m += 1
            except ZeroDivisionError:
                pass
        mtma[i] = mT/mA
        larr[i] = lav/M

    plt.plot(u,larr)
    plt.show()
    plt.plot(u,mtma)
    plt.show()
