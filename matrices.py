import numpy as np
from numba import jit

def I(x, N):
    y = np.zeros(N**3)
    for i in range(N ** 3):
        if np.mod(i,2) == 0:
            y[i] = x[i]
        else:
            y[i] = x[i]
    return x

def test(x,N):
    y = np.zeros(8)
    y[0] = x[0] + 2 * x[1]
    y[1] = 2 * x[0] + x[1]
    y[2] = x[0] + 2 * x[2]
    y[3] = 2 * x[3] + x[1]
    y[4] = x[4] + 2 * x[1]
    y[5] = 2 * x[0] + x[5]
    y[6] = x[7]
    y[7] = x[6]
    return y

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

    return y
