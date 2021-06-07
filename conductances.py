import numpy as np

def wxyz(N, a, b):
    """Generate the conductances for an (N+2)**3 zero boundary checkerboard pattern with pixel size 5 (note that N should be a multiple of 10."""
    try:
        n = int(N / 10)

        wx0 = a * np.zeros((N + 2) ** 2)
        wx15 = np.tile(
            np.append(
                a * np.ones(N + 2),
                np.append(
                    np.tile(
                        np.append(
                            np.tile(
                                np.append(
                                    np.append(
                                        np.array([b, b, b, b, b, a, a, a, a, a, a]),
                                        np.tile(np.array([b, b, b, b, a, a, a, a, a, a]), n - 1)
                                    ),
                                    np.array([a])
                                ),
                                5
                            ),
                            np.tile(
                                np.append(
                                    np.tile(np.array([a, a, a, a, a, a, b, b, b, b]), n - 1),
                                    np.array([a, a, a, a, a, a, b, b, b, b, b, a])
                                ),
                                5
                            ),
                        ),
                        n
                    ),
                    a * np.ones(N + 2)
                )
            ),
            5
        )
        wx610 = np.tile(
            np.append(
                a * np.ones(N + 2),
                np.append(
                    np.tile(
                        np.append(
                            np.tile(
                                np.append(
                                    np.tile(np.array([a, a, a, a, a, a, b, b, b, b]), n - 1),
                                    np.array([a, a, a, a, a, a, b, b, b, b, b, a])
                                ),
                                5
                            ),
                            np.tile(
                                np.append(
                                    np.append(
                                        np.array([b, b, b, b, b, a, a, a, a, a, a]),
                                        np.tile(np.array([b, b, b, b, a, a, a, a, a, a]), n - 1)
                                    ),
                                    np.array([a])
                                ),
                                5
                            )
                        ),
                        n
                    ),
                    a * np.ones(N + 2)
                )
            ),
            5
        )
        wx11 = a * np.zeros((N + 2) ** 2)
        wx = np.append(
            wx0,
            np.append(
                np.tile(
                    np.append(
                        wx15,
                        wx610
                    ),
                    n
                ),
                wx11
            )
        )

        wy0 = a * np.zeros((N + 2) ** 2)
        wy15 = np.tile(
            np.append(
                a * np.ones(N + 2),
                np.append(
                    np.tile(
                        np.append(
                            np.tile(
                                np.append(
                                    np.append(
                                        np.array([b, b, b, b, b, a, a, a, a, a, a]),
                                        np.tile(np.array([b, b, b, b, a, a, a, a, a, a]), n - 1)
                                    ),
                                    np.array([a])
                                ),
                                5
                            ),
                            np.tile(
                                np.append(
                                    np.tile(np.array([a, a, a, a, a, a, b, b, b, b]), n - 1),
                                    np.array([a, a, a, a, a, a, b, b, b, b, b, a])
                                ),
                                5
                            ),
                        ),
                        n
                    ),
                    a * np.ones(N + 2)
                )
            ),
            5
        )
        wy610 = np.tile(
            np.append(
                a * np.ones(N + 2),
                np.append(
                    np.tile(
                        np.append(
                            np.tile(
                                np.append(
                                    np.append(
                                        np.array([b, b, b, b, b, a, a, a, a, a, a]),
                                        np.tile(np.array([b, b, b, b, a, a, a, a, a, a]), n - 1)
                                    ),
                                    np.array([a])
                                ),
                                5
                            ),
                            np.tile(
                                np.append(
                                    np.tile(np.array([a, a, a, a, a, a, b, b, b, b]), n - 1),
                                    np.array([a, a, a, a, a, a, b, b, b, b, b, a])
                                ),
                                5
                            ),
                        ),
                        n
                    ),
                    a * np.ones(N + 2)
                )
            ),
            5
        )

        wy = wx.reshape(N + 2, N + 2, N + 2).swapaxes(1,2).reshape((N + 2) ** 3)

        wz = wx.reshape(N + 2, N + 2, N + 2).swapaxes(0,2).reshape((N + 2) ** 3)

        return wx, wy, wz

    except:
        print('N must be a multiple of 10.')
        return np.zeros((N + 2) ** 3), np.zeros((N + 2) ** 3), np.zeros((N + 2) ** 3)


"""
N = 10
wx = wxyz(N,0,1)[0].reshape(N+2,N+2,N+2)
wy = wxyz(N,0,1)[1].reshape(N+2,N+2,N+2)
wz = wxyz(N,0,1)[2].reshape(N+2,N+2,N+2)

for i in range(len(wx)):
    print(wz[i])
    x = input('next: \n')
"""

# wx0 = np.zeros(12 ** 2)
# wx15 = np.tile(
#     np.append(np.append(np.zeros(12), np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), 5)),
#               np.append(np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5), np.zeros(12))), 5)
# wx610 = np.tile(
#     np.append(np.append(np.zeros(12), np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5)),
#               np.append(np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), 5), np.zeros(12))), 5)
# wx11 = np.zeros(12 ** 2)
# wx = np.append(np.append(wx0, wx15), np.append(wx610, wx11))
#
# wy0 = np.zeros(12 ** 2)
# wy15 = np.tile(
#     np.append(np.append(np.tile(np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 5), np.zeros(12)),
#               np.append(np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5), np.zeros(12))), 5)
# wy610 = np.tile(
#     np.append(np.append(np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5), np.zeros(12)),
#               np.append(np.tile(np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 5), np.zeros(12))), 5)
# wy11 = np.zeros(12 ** 2)
# wy = np.append(np.append(wy0, wy15), np.append(wy610, wy11))
#
# wz04 = np.tile(
#     np.append(np.append(np.zeros(12), np.tile(np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 5)),
#               np.append(np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5), np.zeros(12))), 5)
# wz5 = np.zeros(12 ** 2)
# wz610 = np.tile(
#     np.append(np.append(np.zeros(12), np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), 5)),
#               np.append(np.tile(np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 5), np.zeros(12))), 5)
# wz11 = np.zeros(12 ** 2)
# wz = np.append(np.append(wz04, wz5), np.append(wz610, wz11))