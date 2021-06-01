import numpy as np

def I(x, N):
    # y = np.zeros(N**3)
    # for i in range(N ** 3):
    #     if np.mod(i,2) == 0:
    #         y[i] = x[i]
    #     else:
    #         y[i] = x[i]
    return x

def test(x,N):
    D = 2*np.eye(N**3)+np.eye(N**3, k=1)+np.eye(N**3, k=-1)+3*np.eye(N**3, k=2)+3*np.eye(N**3, k=-2)
    return D.dot(x)

# N = 2
# x = np.array([1,2,5,0,3,1,2,0])
# y =test(x,N)
# print(y)

print('HPC test')

def testpf(x,N):
    S = 0.001*np.eye(N**2)
    for i in range(N**2):
        if i == 0 or i == 9 or i == 90 or i == 99:
            S[i][i] += 3
        elif (i>=1 and i<=8) or (i>=91 and i<=98) or (np.mod(i,10)==0 and i != 0 and i != 90) or (np.mod(i,10)==9 and i != 9 and i != 99):
            S[i][i] += 5
        else:
            S[i][i] += 8
    for i in range(N**2):
        for j in range(N**2):
            if i == 0:
                if (j == i+1) or (j == i+10) or (j == i+11):
                    S[i][j] = -1
            elif i == 9:
                if (j == i-1) or (j == i+10) or (j == i+9):
                    S[i][j] = -1
            elif i == 90:
                if (j == i+1) or (j == i-10) or (j == i-9):
                    S[i][j] = -1
            elif i == 99:
                if (j == i-1) or (j == i-10) or (j == i-11):
                    S[i][j] = -1
            elif (i>=1 and i<=8):
                if (j == i-1) or (j == i+1) or (j == i+9) or (j == i+10) or (j == i+11):
                    S[i][j] = -1
            elif (i>=91 and i<=98):
                if (j == i-1) or (j == i+1) or (j == i-9) or (j == i-10) or (j == i-11):
                    S[i][j] = -1
            elif (np.mod(i,10)==0 and i != 0 and i != 90):
                if (j == i-10) or (j == i-9) or (j == i+1) or (j == i+10) or (j == i+11):
                    S[i][j] = -1
            elif (np.mod(i,10)==9 and i != 9 and i != 99):
                if (j == i-11) or (j == i-10) or (j == i-1) or (j == i+9) or (j == i+10):
                    S[i][j] = -1
            else:
                if (i!=j) and (j == i+1 or j == i-1 or j == i+10 or j == i-10 or j == i+11 or j == i+9 or j == i-9 or j == i-11):
                    S[i][j] = -1
    return np.dot(S,x)
"""
N = 10
x = np.ones(N**2)
print(testpf(x,N))
"""
#@jit(nopython=True)
def A(x, N):
    """Returns the matrix vector product Ax, where A is a NxN periodic boundary precision matrix."""
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
    """Returns the matrix vector product Bx, where B is the free boundary precision matrix."""
    y = np.zeros(N ** 3)

    wx = np.ones(N ** 3)
    wy = np.ones(N ** 3)
    wz = np.ones(N ** 3)


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

    return y/2

def C2(x, N):
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
            y[i] = 10*x[i] + 2*x[i-1] + 2*x[i+1] + x[i+N] + x[i-N]

    return y
