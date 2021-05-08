import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import genGFFSheffield as ggff

#np.random.seed(1)

#@jit(nopython=True)
def A(x, N):
    """Returns the matrix vector product Ax, where A is the periodic boundary precision matrix."""
    y = np.zeros(N ** 3)

    wx = np.ones(N ** 3)
    wy = np.ones(N ** 3)
    wz = np.ones(N ** 3)

    for i in range(N ** 3):
        y[i] = x[i]

        if np.mod(i, N) == N - 1:
            y[i] -= 1/6 * wx[i] * x[i - N + 1]
        else:
            y[i] -= 1/6 * wx[i] * x[i + 1]

        if np.mod(i, N) == 0:
            y[i] -= 1/6 * wx[i + N - 1] * x[i + N - 1]
        else:
            y[i] -= 1/6 * wx[i - 1] * x[i - 1]

        if np.mod(i, N ** 2) >= N ** 2 - N:
            y[i] -= 1/6 * wy[i] * x[i - N ** 2 + N]
        else:
            y[i] -= 1/6 * wy[i] * x[i + N]

        if np.mod(i, N ** 2) <= N - 1:
            y[i] -= 1/6 * wy[i + N ** 2 - N] * x[i + N ** 2 - N]
        else:
            y[i] -= 1/6 * wy[i - N] * x[i - N]

        if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
            y[i] -= 1/6 * wz[i] * x[i - N ** 3 + N ** 2]
        else:
            y[i] -= 1/6 * wz[i] * x[i + N ** 2]

        if np.mod(i, N ** 3) <= N ** 2 - 1:
            y[i] -= 1/6 * wz[i + N ** 3 - N ** 2] * x[i + N ** 3 - N ** 2]
        else:
            y[i] -= 1/6 * wz[i - N ** 2] * x[i - N ** 2]
    return y

#@jit(nopython=True)
def B(x, N):
    """Returns the matrix vector product Ax, where A is the free boundary precision matrix."""
    y = np.zeros(N ** 3)

    for i in range(N ** 3):
        y[i] = x[i]

        if np.mod(i, N) == N - 1:
            pass
        else:
            y[i] -= 1/6 * wx[i] * x[i + 1]

        if np.mod(i, N) == 0:
            pass
        else:
            y[i] -= 1/6 * wx[i - 1] * x[i - 1]

        if np.mod(i, N ** 2) >= N ** 2 - N:
            pass
        else:
            y[i] -= 1/6 * wy[i] * x[i + N]

        if np.mod(i, N ** 2) <= N - 1:
            pass
        else:
            y[i] -= 1/6 * wy[i - N] * x[i - N]

        if np.mod(i, N ** 3) >= N ** 3 - N ** 2:
            pass
        else:
            y[i] -= 1/6 * wz[i] * x[i + N ** 2]

        if np.mod(i, N ** 3) <= N ** 2 - 1:
            pass
        else:
            y[i] -= 1/6 * wz[i - N ** 2] * x[i - N ** 2]
    return y

#@jit(nopython=True)
def C(x, N):
    """Returns the matrix vector product Ax, where A is the zero boundary precision matrix."""
    y = np.zeros(N ** 3)

    for i in range(N ** 3):
        y[i] = x[i]

        y[i] -= 1/6 * wx[i] * x[i + 1]

        y[i] -= 1/6 * wx[i - 1] * x[i - 1]

        y[i] -= 1/6 * wy[i] * x[i + N]

        y[i] -= 1/6 * wy[i - N] * x[i - N]

        y[i] -= 1/6 * wz[i] * x[i + N ** 2]

        y[i] -= 1/6 * wz[i - N ** 2] * x[i - N ** 2]

    return y

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

x = cgacc(A, 5, random=False)
print(x)

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
    d_old = np.dot(p_old, C(p_old, N+2))
    y_old = np.zeros((N+2)**3)

    conjloss = 0

    #r0 = np.sqrt(np.sqrt(np.dot(r_old,r_old)))
    #r_array = r0 * np.zeros(k_max+1)
    #t = np.zeros(k_max+1)

    for k in range(1,k_max+1):
        print(k)
        #start = time.time()
        gamma = np.dot(r_old,r_old)/d_old
        r_new = r_old - gamma * C(p_old, N+2)
        beta = - np.dot(r_new,r_new)/(gamma * d_old)
        p_new = r_new - beta * p_old

        if np.abs(np.dot(p_new, A(p_old, N+2))) >= 1e-4 and conjloss == 0:
            print("Loss of conjugacy at iteration ", k)
            conjloss = 1

        d_new = np.dot(p_new, C(p_new, N+2))
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

def cgnorm(A, N, k_max):
    y = cg(A, N, k_max)
    y_max = np.max(np.abs(y))
    return y/y_max

"""   
def cgtest:
    d = len(mu)
    init = np.reshape(init, (d, 1))
    mu = np.reshape(mu, (d, 1))
    loss_conj = 0

    theta = np.zeros((d, size))
    # Initialization
    k = 1
    r_old = np.random.normal(0, 1, size=(d, size)) - A(init).reshape((d, size))
    # rd = np.random.randint(0, 2, (d, size))
    # rd[rd == 0] = -1
    # r_old = rd - B(init)
    p_old = r_old
    d_old = (p_old * A(p_old)).sum(axis=0)
    y = init
    r_new = np.ones((d, size))
    loss_conj = 0
    while (col_vector_norms(r_new, 2) >= tol).any() and k <= K:
        gam = (r_old * r_old).sum(axis=0) / d_old
        z = np.random.normal(0, 1, size=size)
        y = y + z / np.sqrt(d_old) * p_old
        r_new = r_old - gam * A(p_old)
        beta = - (r_new * r_new).sum(axis=0) / (r_old * r_old).sum(axis=0)

        p_new = r_new - beta * p_old
        if (np.abs((p_new * A(p_old)).sum(axis=0)) >= 1e-4).any() and loss_conj == 0 and info == True:
            print('Loss of conjugacy happened at iteration k = %i.' % k)
            loss_conj = 1
            k_loss = k
        d_new = (p_new * A(p_new)).sum(axis=0)
        r_old = r_new
        p_old = p_new
        d_old = d_new
        k = k + 1
    theta = mu + y

    if info == True and loss_conj == 1:
        return (theta, k - 1, loss_conj, k_loss)
    elif info == True and loss_conj == 0:
        return (theta, k - 1, loss_conj)
    else:
        return theta

N = 10
k_max = 10

theta = cg0(A,N,k_max)

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