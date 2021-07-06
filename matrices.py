import numpy as np
from numba import njit

def Iold(x, N):
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
def Cold(x, N):
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

def C2old(x, N):
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


@njit
def I(x, wx, wy, wz, N):
    y = np.zeros((N+2) ** 3)

    for i in range((N + 2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            y[i] = 0

        else:
            y[i] = x[i]

    return y

@njit
def NtoN2(i,N):
    z = i//N ** 2
    y = (i - z * N ** 2)//N
    x = i - z * N ** 2 - y * N
    return (x + 1) + (y + 1) * (N + 2) + (z + 1) * (N + 2) ** 2

@njit
def C2(x,wx,wy,wz,N):
    y = np.zeros(N ** 3)

    for i in range(N ** 3):
        y[i] = 1 / 6 * (wx[NtoN2(i - 1, N)]
                        + wx[NtoN2(i, N)]
                        + wy[NtoN2(i - N, N)]
                        + wy[NtoN2(i, N)]
                        + wz[NtoN2(i - N ** 2, N)]
                        + wz[NtoN2(i, N)]) * x[i]

        if np.mod(i, N) == 0: # Left plane
            if np.mod(i, N ** 2) <= N - 1: # Left-front edge
                if i == 0: # Left-front-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 3 - N ** 2: # Left-front-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of the left-front edge
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 2) >= N ** 2 - N: # Left-back edge
                if i == N ** 2 - N: # Left-back-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 3 - N: # Left-back-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of the left-back edge
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) <= N ** 2 - 1: # Left-bottom edge
                if i == 0: # Left-front-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 2 - N: # Left-back-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                else: # Rest of the left-bottom edge
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) >= N ** 3 - N ** 2: # Left-top edge
                if i == N ** 3 - N ** 2: # Left-front-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                elif i == N ** 3 - N: # Left-back-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else:  # Rest of the left-top edge
                    y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

            else: # Rest of left plane
                y[i] -= 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                        + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                        + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                        + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                        + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

        elif np.mod(i, N) == N - 1: # Right plane
            if np.mod(i, N ** 2) <= N - 1: # Right-front edge
                if i == N - 1: # Right-front-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 3 - N ** 2 + N - 1: # Right-front-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of the right-front edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 2) >= N ** 2 - N: # Right-back edge
                if i == N ** 2 - 1: # Right-back-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 3 - 1: # Right-back-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of the right-back edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) <= N ** 2 - 1: # Left-bottom edge
                if i == N - 1: # Right-front-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif i == N ** 2 - 1: # Right-back-bottom vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                else: # Rest of the right-bottom edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) >= N ** 3 - N ** 2: # Left-top edge
                if i == N ** 3 - N ** 2 + N - 1: # Right-front-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                elif i == N ** 3 - 1: # Right-back-top vertex
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of the right-top edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

            else: # Rest of right plane
                y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                        + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                        + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                        + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                        + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

        else: # Rest of cube
            if np.mod(i, N ** 2) <= N - 1: # Front plane
                if np.mod(i, N ** 3) <= N ** 2 - 1: # Rest of front-bottom edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif np.mod(i, N ** 3) >= N ** 3 - N ** 2: # Rest of front-top edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else: # Rest of front plane
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 2) >= N ** 2 - N: # Back plane
                if np.mod(i, N ** 3) <= N ** 2 - 1:  # Rest of back-bottom edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

                elif np.mod(i, N ** 3) >= N ** 3 - N ** 2:  # Rest of back-top edge
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

                else:  # Rest of back plane
                    y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                            + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                            + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                            + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                            + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) <= N ** 2 - 1: # Rest of bottom plane
                y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                        + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                        + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                        + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                        + 1 / 6 * wz[NtoN2(i, N)] * x[i + N ** 2]

            elif np.mod(i, N ** 3) >= N ** 3 - N ** 2: # Rest of top plane
                y[i] -= 1 / 6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                        + 1 / 6 * wx[NtoN2(i, N)] * x[i + 1] \
                        + 1 / 6 * wy[NtoN2(i - N, N)] * x[i - N] \
                        + 1 / 6 * wy[NtoN2(i, N)] * x[i + N] \
                        + 1 / 6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2]

            else: # Inner cube
                y[i] -= 1/6 * wx[NtoN2(i - 1, N)] * x[i - 1] \
                        + 1/6 * wx[NtoN2(i, N)] * x[i + 1] \
                        + 1/6 * wy[NtoN2(i - N, N)] * x[i - N] \
                        + 1/6 * wy[NtoN2(i, N)] * x[i + N] \
                        + 1/6 * wz[NtoN2(i - N ** 2, N)] * x[i - N ** 2] \
                        + 1/6 * wz[NtoN2(i, N)] * x[i + N ** 2]

    return y

@njit
def C(u, wx, wy, wz, N):
    """Returns the matrix vector product Cu, where C is the zero boundary precision matrix."""
    v = np.zeros((N+2) ** 3)

    for i in range((N+2) ** 3):
        if np.mod(i, N + 2) <= 0 \
                or np.mod(i, N + 2) >= N + 1 \
                or np.mod(i, (N + 2) ** 2) <= (N + 2) - 1 \
                or np.mod(i, (N + 2) ** 2) >= (N + 2) ** 2 - (N + 2) \
                or np.mod(i, (N + 2) ** 3) <= (N + 2) ** 2 - 1 \
                or np.mod(i, (N + 2) ** 3) >= (N + 2) ** 3 - (N + 2) ** 2:
            v[i] = u[i]
        else:
            v[i] = u[i] * 1/6 * (wx[i] + wx[i-1] + wy[i] + wy[i - (N + 2)] + wz[i] + wz[i - (N + 2) ** 2])

            v[i] -= 1/6 * wx[i] * u[i + 1]

            v[i] -= 1/6 * wx[i - 1] * u[i - 1]

            v[i] -= 1/6 * wy[i] * u[i + (N + 2)]

            v[i] -= 1/6 * wy[i - (N + 2)] * u[i - (N + 2)]

            v[i] -= 1/6 * wz[i] * u[i + (N + 2) ** 2]

            v[i] -= 1/6 * wz[i - (N + 2) ** 2] * u[i - (N + 2) ** 2]

    return v
