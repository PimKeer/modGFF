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
def cgpf0(C, N, epsilon = 1e-4):
    """CG sampler with PBC (variation 2)."""

    b = np.random.normal(size=(N + 2) ** 3)
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

    while np.dot(r_old,r_old) >= epsilon:
        # print(np.dot(r_old,r_old))
        gamma = np.dot(r_old,r_old)/d_old
        z = np.random.normal(0,1,size=1)
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*C(p_old,N)
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,C(p_new,N))

        if np.abs(np.dot(p_new, C(p_old, N))) >= 1e-4 and conjloss == 0:
            print("loss of conjugacy at iteration: ", k)
            conjloss = 1
            l = k

        y_old = y_new
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k += 1

    return y_old

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

def cgpf0acc(C, N, epsilon):
    """CG sampler with PBC (variation 2)."""

    b = np.random.normal(size = (N+2)**3)
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

    gammalist = []
    dlist = [d_old]
    rlist = [np.linalg.norm(r_old)]

    # while np.linalg.norm(r_old) >= epsilon:
    while k <= epsilon:
        # print(np.linalg.norm(r_old))
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
    return y_old, x_old , traceT, traceA, b, l

def cgpfacc(S, N, epsilon):
    """CG sampler with PBC (variation 2)."""

    b = np.random.normal(size = N**2)
    x_old = np.zeros(N**2)

    r_old = b - S(x_old, N)
    p_old = r_old
    d_old = np.dot(p_old, S(p_old, N))
    # print(d_old)
    y_old = x_old

    k = 1
    conjloss = 0

    gammalist = []
    dlist = [d_old]
    rlist = [np.linalg.norm(r_old)]

    # while np.linalg.norm(r_old) >= epsilon:
    while k <= epsilon:
        # print(np.dot(r_old,r_old))
        gamma = np.dot(r_old,r_old)/d_old
        x_new = x_old + gamma*p_old
        z = np.random.normal(0,1,size=1)
        y_new = y_old + z*p_old/np.sqrt(d_old)
        r_new = r_old - gamma*S(p_old,N)
        beta = - np.dot(r_new,r_new)/np.dot(r_old,r_old)
        p_new = r_new-beta*p_old
        d_new = np.dot(p_new,S(p_new,N))
        # print(d_new)

        if np.abs(np.dot(p_new, S(p_old, N))) >= 1e-4 and conjloss == 0:
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
    for j in range(0,l-1):
        for i in range(j,l-1):
            # print('T  ', gammalist[i] * rlist[i] ** 2 / (rlist[j] ** 2))
            traceT += gammalist[i] * rlist[i] ** 2 / (rlist[j] ** 2)
    traceA = 0
    for i in range(0,l-1):
        # print('A  ', gammalist[i] * rlist[i] ** 2)
        traceA += gammalist[i] * rlist[i] ** 2
    return y_old, x_old , traceT, traceA, b, l

N = 10
u = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
epsilon = u
M = 20

mtma = np.zeros(len(epsilon))
larr = np.zeros(len(epsilon))

for i in range(len(epsilon)):
    mT = 0
    mA = 0
    lav = 0
    for m in range(M):
        y,x,t,a,b,l = cgpf0acc(C, N, epsilon[i])
        mT += t
        mA += a
        lav += l
        print(i, l, mT/mA, t, a)
    mtma[i] = mT/mA
    larr[i] = lav/M

plt.plot(u,larr)
plt.show()
plt.plot(u,mtma)
plt.show()

for n in range(N+2):
    ggff.plotGFF(y.reshape(N,N),N,N)
    plt.show()

# print(I(x,N))
# print(b)
#
# print(y)
# print(x)
# print(t)
# print(t/a)

# N=10
#
# theta = cg(A, N,k_max=1000)
#
# for i in range(N):
#     ggff.plotGFF(theta.reshape((N,N,N))[i],N,N)
#     plt.show()

def cg0(C, N, k_max, epsilon):
    """CG sampler with ZBC."""
    #c = np.random.normal(size = (N+2)**3)
    c = np.random.normal(size=N ** 3)
    # for i in range((N + 2) ** 3):
    #     if np.mod(i, N + 2) <= 0 \
    #             or np.mod(i, N + 2) >= N + 1 \
    #             or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
    #             or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
    #             or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
    #             or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
    #         c[i] = 0

    r_old = c
    p_old = r_old
    d_old = np.dot(p_old, C(p_old, N))
    #y_old = np.zeros((N+2)**3)
    y_old = np.zeros(N** 3)

    conjloss = 0
    k=1

    gammalist = []
    dlist = [d_old]

    #r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    #r_array = r0 * np.zeros(k_max+1)
    #t = np.zeros(k_max+1)

    while np.dot(r_old,r_old) >= epsilon and k<=k_max:
        gamma = np.dot(r_old,r_old)/d_old
        r_new = r_old - gamma * C(p_old, N)
        beta = - np.dot(r_new,r_new)/(gamma * d_old)
        p_new = r_new - beta * p_old

        print(k, np.linalg.norm(C(p_old, N)))

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
"""
N = 10
epsilon = 1e-4
m = 50

mT = 0
mA = 0

y, t, a = cg0(I, N, m, epsilon)
print(m,t,a,t/a)
"""
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