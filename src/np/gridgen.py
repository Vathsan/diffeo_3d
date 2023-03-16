import scipy.signal

import numpy as np


def mygriddata(nx, ny, v):
    """Linear interpolation"""
    m, n = nx.shape
    rx = np.clip(nx, 0, m - 1.0)
    ry = np.clip(ny, 0, n - 1.0)
    sw = np.floor(rx).astype(np.int)
    nw = np.floor(ry).astype(np.int)

    sw[np.equal(sw, rx)] = sw[np.equal(sw, rx)] - 1
    nw[np.equal(nw, ry)] = nw[np.equal(nw, ry)] - 1

    se = np.clip(sw + 1, 0, m - 1.0).astype(np.int)
    ne = np.clip(nw + 1, 0, n - 1.0).astype(np.int)

    res = (
        v[sw, nw] * (se - rx) * (ne - ry)
        + v[sw, ne] * (se - rx) * (ry - nw)
        + v[se, nw] * (rx - sw) * (ne - ry)
        + v[se, ne] * (rx - sw) * (ry - nw)
    )

    return res


def euler_2d(v1, v2, f, tstep):
    """Euler ODE solver"""
    yy, xx = np.meshgrid(range(f.shape[1]), range(f.shape[0]))

    h = 1 / tstep
    for t in np.arange(0, 1, h):
        if t == 0:
            k1_1 = v1 / f
            k1_2 = v2 / f
        else:
            k1_1 = mygriddata(xx, yy, v1) / (t + (1 - t) * mygriddata(xx, yy, f))
            k1_2 = mygriddata(xx, yy, v2) / (t + (1 - t) * mygriddata(xx, yy, f))

        xx = xx + k1_1 * h
        yy = yy + k1_2 * h

    return xx, yy


def fast_sine_transform_y(v):
    """Perform fast sine transform in y direction"""
    n, m = v.shape
    v = np.hstack([np.zeros((n, 1)), v, np.zeros((n, m + 1))])
    v = np.fft.fft(v, axis=-1)
    v = v.imag
    return v[:, 1 : m + 1]


def fast_sine_transform_x(v):
    """Perform fast sine transform in x direction"""
    n, m = v.shape
    v = np.vstack([np.zeros((1, m)), v, np.zeros((n + 1, m))])
    v = np.fft.fft(v, axis=0)
    np.set_printoptions(suppress=True)
    v = v.imag
    return v[1 : n + 1, :]


def poisson_solver_2d_fft(f):
    """Poisson solver"""
    n, m = f.shape
    yy, xx = np.meshgrid(range(1, m + 1), range(1, n + 1))
    lx, ly = (
        2.0 * (1.0 - np.cos(xx * np.pi / (n + 1.0))),
        2.0 * (1.0 - np.cos(yy * np.pi / (m + 1.0))),
    )
    lxy = 1.0 / (lx + ly)

    res = (
        4.0
        / ((n + 1.0) * (m + 1.0))
        * lxy
        * fast_sine_transform_y(fast_sine_transform_x(f))
    )

    return -1.0 * fast_sine_transform_y(fast_sine_transform_x(res))


def div_curl_solver_2d(f1, f2, inv):
    """Perform div curl solver"""
    if not inv:
        ker = np.array([[1, 4, 1], [0, 0, 0], [-1, -4, -1]])
    else:
        ker = np.array([[-1, -4, -1], [0, 0, 0], [1, 4, 1]])

    df1dx = scipy.signal.convolve2d(f1, ker, mode="same")
    df2dx = scipy.signal.convolve2d(f2, ker, mode="same")
    df1dy = scipy.signal.convolve2d(f1, ker.T, mode="same")
    df2dy = scipy.signal.convolve2d(f2, ker.T, mode="same")

    F1 = (df1dx + df2dy) / 12.0
    F2 = (df1dy - df2dx) / 12.0

    return poisson_solver_2d_fft(F1), poisson_solver_2d_fft(F2)


def gridgen(f1, f2, j_lb=0.4, j_ub=4.0, n_euler=20, inv=False):
    """Grid generation"""
    f1 = f1 / np.mean(f1)

    ker = np.array(
        [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    )

    # print ('min f1: ' + str(np.min(f1)))
    # print ('max f1: ' + str(np.max(f1)))

    # while np.max(f1) > j_ub and np.min(f1) < j_lb:
    while np.max(f1) > j_ub or np.min(f1) < j_lb:
        f1 = scipy.signal.convolve2d(f1, ker, mode="same", boundary="wrap")
        # print ('in loop')
        # print('min f1: ' + str(np.min(f1)))
        # print('max f1: ' + str(np.max(f1)))

    v1, v2 = div_curl_solver_2d(f1 - 1.0, f2, inv)
    posx, posy = euler_2d(v1, v2, f1, n_euler)

    return posx, posy
