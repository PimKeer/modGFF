import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import genGFFSheffield as ggff
from matrices import *

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
    b = np.random.normal(size=N ** 3)
    x_old = np.zeros(N ** 3)

    r_old = b - A(x_old, N)
    p_old = r_old
    d_old = np.dot(p_old, A(p_old, N))
    y_old = x_old

    k = 1
    conjloss = 0

    gammalist = []
    dlist = [d_old]

    while np.dot(r_old,r_old) >= epsilon:
        gamma = np.dot(r_old,r_old)/d_old
        x_new = x_old + gamma*p_old
        z = np.random.normal(0,1,size=1)
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*A(p_old,N)
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,A(p_new,N))

        if np.abs(np.dot(p_new, A(p_old, N))) >= 1e-4 and conjloss == 0:
            #print("loss of conjugacy at iteration: ", k)
            conjloss = 1
            l = k

        gammalist.append(gamma)
        dlist.append(d_new)

        x_old = x_new
        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    traceT = 0
    for j in range(min(l,k)):
        for i in range(j,l):
            traceT += gammalist[i] ** 2 * dlist[i] / (gammalist[j] * dlist[j])

    traceA = 0
    for i in range(min(l,k)):
        traceA += gammalist[i] ** 2 * dlist[i]

    return y_old, x_old, traceT, traceA
"""
N = 10
epsilon = 1e-2
M = 10000

mT = 0
mA = 0

for m in range(M):
    y, x, t, a = cgpf(A, N, epsilon)
    mT += t
    mA += a
    #print(t/a)
    print(m, mT/mA)

z = y - np.mean(y)
print(z)
print(x)
print(t)

for i in range(N):
    ggff.plotGFF(z.reshape((N,N,N))[i],N,N)
    plt.show()
"""
# N=10
#
# theta = cg(A, N,k_max=1000)
#
# for i in range(N):
#     ggff.plotGFF(theta.reshape((N,N,N))[i],N,N)
#     plt.show()

def cg0(C, N, k_max, epsilon):
    """CG sampler with ZBC."""
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
    k=1

    gammalist = []
    dlist = [d_old]

    #r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    #r_array = r0 * np.zeros(k_max+1)
    #t = np.zeros(k_max+1)

    while np.dot(r_old,r_old) >= epsilon or k<=k_max:
        print(k)
        gamma = np.dot(r_old,r_old)/d_old
        r_new = r_old - gamma * C(p_old, N)
        beta = - np.dot(r_new,r_new)/(gamma * d_old)
        p_new = r_new - beta * p_old

        if np.abs(np.dot(p_new, C(p_old, N))) >= 1e-4 and conjloss == 0:
            print("Loss of conjugacy at iteration ", k)
            conjloss = 1
            l=k

        d_new = np.dot(p_new, C(p_new, N))
        z = np.random.normal(size = 1)
        y_new = y_old + z/np.sqrt(d_old) * p_old

        r_old = r_new
        p_old = p_new
        d_old = d_new
        y_old = y_new
        k += 1

        gammalist.append(gamma)
        dlist.append(d_new)

    if conjloss == 0:
        l = k

    traceT = 0
    for j in range(l-1):
        for i in range(j, l-1):
            traceT += gammalist[i] ** 2 * dlist[i] / (gammalist[j] * dlist[j])

    traceA = 0
    for i in range(l-1):
        traceA += gammalist[i] ** 2 * dlist[i]

    return y_old, traceT, traceA

N = 10
epsilon = 1e-4
m = 540

mT = 0
mA = 0


y, t, a = cg0(C, N, m, epsilon)
print(m,t/a)

def cgnorm(A, N, k_max):
    y = cg(A, N, k_max)
    mu = np.mean(y)
    sd = np.std(y)
    return (y-mu)/sd

def cgacc(A, N, k_max = 10, random = False):
    """Trace ratio accuracy test."""
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
    ds = np.zeros(k_max + 1)
    ds[0] = d_old
    rolds = np.zeros(k_max + 1)
    rolds[0] = np.dot(r_old, r_old)

    # r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    # r_array = r0 * np.zeros(k_max+1)
    # t = np.zeros(k_max+1)

    k = 1

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
    ds[k] = d_new
    rolds[k] = np.dot(r_new, r_new)

    r_old = r_new
    p_old = p_new
    d_old = d_new
    y_old = y_new

    print(np.abs(np.dot(p_new, A(p_old, N))))

    k = 2

    while np.abs(np.dot(p_new, A(p_old, N))) < 1e-4 or conjloss == 0:
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
        ds[k] = d_new
        rolds[k] = np.dot(r_new, r_new)

        r_old = r_new
        p_old = p_new
        d_old = d_new
        y_old = y_new

        k += 1

        # end = time.time()
        # t[k] = end-start

        # r_array[k] = np.sqrt(np.dot(r_old,r_old))

    # print(gammas)
    # print(ds)
    # print(rolds)

    traceT = 0
    for j in range(k):
        for i in range(j, k):
            #traceT += (gammas[i] ** 2 * ds[i]) / (gammas[j] * ds[j])
            traceT += (gammas[i] * rolds[i]) / (rolds[j])

    traceA = 0
    for i in range(k):
        #traceA += gammas[i] ** 2 * ds[i]
        traceA += gammas[i] * rolds[i]

    return traceT/traceA, traceT, traceA

""" 
#np.random.seed(1)
moy = 0
moyA = 0
moyT = 0
M = 10000
for m in range(1,M+1):
    x = cgacc(A, 10, k_max=1000 , random=False)
    moy += x[0]
    moyT += x[1]
    moyA += x[2]
    print(m, "  :  ", moyT/moyA)

print(moy/M)

N = 50
k_max = 10

theta = cg0(C,N,k_max)

for i in range(N):
    ggff.plotGFF(theta.reshape((N+2,N+2,N+2))[i],N+2,N+2)
    plt.show()

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