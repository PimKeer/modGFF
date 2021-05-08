import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import genGFFSheffield as ggff
from matrices import *

#np.random.seed(1)

#@jit(nopython=True)
def cg(A, N, k_max = 10,epsilon=False):
    c = np.random.normal(size = N**3)

    r_old = c
    p_old = r_old
    d_old = np.dot(p_old, A(p_old, N))
    y_old = np.zeros(N**3)

    conjloss = 0

    gammas = np.zeros(k_max)
    ds = np.zeros(k_max)

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
            ds[k-1] = d_old

            r_old = r_new
            p_old = p_new
            d_old = d_new
            y_old = y_new

            #end = time.time()
            #t[k] = end-start

            #r_array[k] = np.sqrt(np.dot(r_old,r_old))

    return y_old#, r_array, t

def cg0(C, N, k_max):
    c = np.random.normal(size = (N+2)**3)
    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            c[i] = 0

    r_old = c
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, N))
    y_old = np.zeros((N+2)**3)

    conjloss = 0

    #r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    #r_array = r0 * np.zeros(k_max+1)
    #t = np.zeros(k_max+1)

    for k in range(1,k_max+1):
        print(k)
        #start = time.time()
        gamma = np.dot(r_old,r_old)/d_old
        r_new = r_old - gamma * C(p_old, N)
        beta = - np.dot(r_new,r_new)/(gamma * d_old)
        p_new = r_new - beta * p_old

        if np.abs(np.dot(p_new, C(p_old, N))) >= 1e-4 and conjloss == 0:
            print("Loss of conjugacy at iteration ", k)
            conjloss = 1

        d_new = np.dot(p_new, C(p_new, N))
        z = np.random.normal(size = 1)
        y_new = y_old + z/np.sqrt(d_old) * p_old

        r_old = r_new
        p_old = p_new
        d_old = d_new
        y_old = y_new

        #end = time.time()
        #t[k] = end-start

        #r_array[k] = np.sqrt(np.dot(r_old,r_old))

    return y_old#, r_array, t

def cgacc(A, N, k_max = 10, random = False):
    if random:
        c = np.random.normal(size = N**3)
    else:
        c = np.random.binomial(1,0.5,size=N ** 3)*2-1

    r_old = c
    p_old = r_old
    d_old = np.dot(p_old, A(p_old, N))
    y_old = np.zeros(N ** 3)

    conjloss = 0

    gammas = np.zeros(k_max)
    ds = np.zeros(k_max)

    # r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    # r_array = r0 * np.zeros(k_max+1)
    # t = np.zeros(k_max+1)

    for k in range(1, k_max + 1):
        # start = time.time()
        gamma = np.dot(r_old, r_old) / d_old
        r_new = r_old - gamma * A(p_old, N)
        beta = - np.dot(r_new, r_new) / (gamma * d_old)
        p_new = r_new - beta * p_old

        if np.abs(np.dot(p_new, A(p_old, N))) >= 1e-4 and conjloss == 0:
            print("Loss of conjugacy at iteration ", k)
            conjloss = 1

        d_new = np.dot(p_new, A(p_new, N))
        z = np.random.normal(size=1)
        y_new = y_old + z / np.sqrt(d_old) * p_old

        gammas[k - 1] = gamma
        ds[k - 1] = d_old

        r_old = r_new
        p_old = p_new
        d_old = d_new
        y_old = y_new

        # end = time.time()
        # t[k] = end-start

        # r_array[k] = np.sqrt(np.dot(r_old,r_old))

    traceT = 0
    for j in range(k_max):
        for i in range(j, k_max):
            traceT += (gammas[i] ** 2 * ds[i]) / (gammas[j] * ds[j])

    traceA = 0
    for i in range(k_max):
        traceA += gammas[i] ** 2 * ds[i]

    return traceT/traceA, traceT, traceA

def cgnorm(A, N, k_max):
    y = cg(A, N, k_max)
    y_max = np.max(np.abs(y))
    return y/y_max

N = 50
k_max = 10

theta = cg0(C,N,k_max)

for i in range(N):
    ggff.plotGFF(theta.reshape((N+2,N+2,N+2))[i],N+2,N+2)
    plt.show()
"""
print(np.shape(theta[0]))

plt.plot(theta[1])
plt.show()

plt.plot(theta[2])
plt.show()
print(np.mean(theta[2][1:]))

ggff.plotGFF(theta[0].reshape((N,N,N))[0],N,N)
plt.show()

mu = np.zeros(N**3)
d = len(mu)
size = 1
init = np.zeros(N**3)
print(np.shape(np.random.normal(0,1,size=d)))

r_old = np.random.normal(0,1,size=(d,size)) - A(init).reshape((1000,1))
print(np.shape(A(init)))
print(np.shape(r_old))

p_old = r_old
print(p_old[0])
d_old = (p_old * A(p_old)).sum(axis=0)
print(type(y))
print(type(z))

def I(x):
    return x

s = pg.sampler_CG(np.zeros(N**3),A,5,np.zeros(N**3))
print(s)
"""