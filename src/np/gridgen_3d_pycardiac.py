# gridgen_3d_pycardiac.py
#
# These functions perform the grid generation using numpy. Jit is used to speed up the computations for all functions
# except for gridgen.
#
# Notes:
#   1. I subtracted 1 from nx, ny, nz in mygriddata_3d, because this matches the original numba implementation from
#      Kumar's 2D code. Therefore when scaling the pytorch version to compare to numpy, 1 has to be added.
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################

import scipy.signal
import scipy.ndimage
import numpy as np
from numba import jit

@jit
def mygriddata_3d(nx, ny, nz, v):
    """Linear interpolation"""

    sz = nx.shape
    nx, ny, nz = nx - 1, ny - 1, nz - 1 # See note above.
    rx = np.clip(nx, 0, sz[0] - 1.0)
    ry = np.clip(ny, 0, sz[1] - 1.0)
    rz = np.clip(nz, 0, sz[2] - 1.0)

    frx = np.floor(rx).astype(np.int)
    fry = np.floor(ry).astype(np.int)
    frz = np.floor(rz).astype(np.int)

    frx[np.equal(frx, rx)] = frx[np.equal(frx, rx)] - 1
    fry[np.equal(fry, ry)] = fry[np.equal(fry, ry)] - 1
    frz[np.equal(frz, rz)] = frz[np.equal(frz, rz)] - 1

    crx = np.clip(frx + 1, 0, sz[0] - 1.0).astype(np.int)
    cry = np.clip(fry + 1, 0, sz[1] - 1.0).astype(np.int)
    crz = np.clip(frz + 1, 0, sz[2] - 1.0).astype(np.int)

    res = (
          v[frx, fry, frz] * (crx - rx) * (cry - ry) * (crz - rz)
        + v[frx, fry, crz] * (crx - rx) * (cry - ry) * (rz - frz)
        + v[frx, cry, frz] * (crx - rx) * (ry - fry) * (crz - rz)
        + v[frx, cry, crz] * (crx - rx) * (ry - fry) * (rz - frz)
        + v[crx, fry, frz] * (rx - frx) * (cry - ry) * (crz - rz)
        + v[crx, fry, crz] * (rx - frx) * (cry - ry) * (rz - frz)
        + v[crx, cry, frz] * (rx - frx) * (ry - fry) * (crz - rz)
        + v[crx, cry, crz] * (rx - frx) * (ry - fry) * (rz - frz)
    )

    return res

@jit
def euler_3d(v1, v2, v3, f, n_euler):
# def euler_3d(v1, v2, v3, f, tstep):
    """Euler ODE solver"""

    yy, xx, zz = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), range(1,f.shape[2]+1))
    # xx, yy, zz = scipy.mgrid[1:f.shape[0]+1, 1:f.shape[1]+1, 1:f.shape[2]+1] # This is the same as above.

    # h = 1.0 / tstep + 0.0
    h = 1.0 / n_euler + 0.0
    temp = f + 0.0
    res = np.zeros_like(f, dtype=np.double)

    for t in np.arange(0, 1, h):
        if t == 0:
            k1_1 = v1 / temp
            k1_2 = v2 / temp
            k1_3 = v3 / temp
        else:
            temp = t + (1 - t) * mygriddata_3d(xx, yy, zz, f)
            k1_1 = mygriddata_3d(xx, yy, zz, v1) / temp
            k1_2 = mygriddata_3d(xx, yy, zz, v2) / temp
            k1_3 = mygriddata_3d(xx, yy, zz, v3) / temp

        xx = xx + k1_1 * h
        yy = yy + k1_2 * h
        zz = zz + k1_3 * h

    return xx, yy, zz

@jit
def m_norm(x):
    return x * np.size(x) / np.sum(x)

@jit
def fast_sine_transform_x_3d(v):
    '''Computes the 1D fast sine transform in 3D in the X direction'''

    n, m, s = v.shape
    # v = np.vstack([np.zeros((1, m, s), dtype=np.float32), v, np.zeros((n + 1, m, s), dtype=np.float32)])
    # v = np.vstack([np.zeros((1, m, s)), v, np.zeros((n + 1, m, s))])
    v = np.vstack((np.zeros((1, m, s)), v, np.zeros((n + 1, m, s))))
    v = np.fft.fft(v, axis=0)
    np.set_printoptions(suppress=True)
    v = v.imag
    # v = np.float32(v)

    return v[1:n + 1, :, :]

@jit
def fast_sine_transform_y_3d(v):
    '''Computes the 1D fast sine transform in 3D in the Y direction'''

    n, m, s = v.shape
    # v = np.hstack([np.zeros((n, 1, s), dtype=np.float32), v, np.zeros((n, m + 1, s), dtype=np.float32)])
    # v = np.hstack([np.zeros((n, 1, s)), v, np.zeros((n, m + 1, s))])
    v = np.hstack((np.zeros((n, 1, s)), v, np.zeros((n, m + 1, s))))
    v = np.fft.fft(v, axis=1)
    v = v.imag
    # v = np.float32(v)

    return v[:, 1:m + 1, :]

@jit
def fast_sine_transform_z_3d(v):
    '''Computes the 1D fast sine transform in 3D in the Z direction'''

    n, m, s = v.shape
    # v = np.dstack([np.zeros((n, m, 1), dtype=np.float32), v, np.zeros((n, m, s + 1), dtype=np.float32)])
    # v = np.dstack([np.zeros((n, m, 1)), v, np.zeros((n, m, s + 1))])
    v = np.dstack((np.zeros((n, m, 1)), v, np.zeros((n, m, s + 1))))
    v = np.fft.fft(v, axis=2)
    v = v.imag
    # v = np.float32(v)

    return v[:, :, 1:s + 1]

@jit
def poisson_solver_3d_fft(f):
    """Poisson solver"""
    n, m, s = f.shape

    yy, xx, zz = np.meshgrid(range(1, m + 1), range(1, n + 1), range(1, s + 1))
    # xx, yy, zz = scipy.mgrid[1:n + 1, 1:m + 1, 1:s + 1] # Same as above.

    lx, ly, lz = (
        2.0 * (1.0 - np.cos(xx * np.pi / (n + 1.0))),
        2.0 * (1.0 - np.cos(yy * np.pi / (m + 1.0))),
        2.0 * (1.0 - np.cos(zz * np.pi / (s + 1.0)))
    )
    lxyz = 1.0 / (lx + ly + lz)

    res = (
        6.0
        / ((n + 1.0) * (m + 1.0) * (s + 1.0))
        * lxyz
        * fast_sine_transform_z_3d(fast_sine_transform_y_3d(fast_sine_transform_x_3d(f)))
    )


    return -1.0 * fast_sine_transform_z_3d(fast_sine_transform_y_3d(fast_sine_transform_x_3d(res)))

@jit
def div_curl_solver_3d(f1, f2, f3, f4, inv=False):
    """Perform div curl solver"""

    ### Now using Sobel kernels ###

    if not inv:
        dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dx_kernel[0, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        dx_kernel[2, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

        dy_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dy_kernel[:, 0, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        dy_kernel[:, 2, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

        dz_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dz_kernel[:, :, 0] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        dz_kernel[:, :, 2] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

    else:
        dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dx_kernel[0, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        dx_kernel[2, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        dy_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dy_kernel[:, 0, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        dy_kernel[:, 2, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

        dz_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        dz_kernel[:, :, 0] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        dz_kernel[:, :, 2] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]




    ### F1 term ###
    # df1/dx - df4/dy + df3/dz
    # each row, col, slice convol = n x (n+2) x (n+2)
    # each df term is n x n x n

    ### F2 term ###
    # df4/dx + df1/dy - df2/dz

    ### F3 term ###
    # -df3/dx + df2/dy + df1/dz

    # df1dx = scipy.ndimage.convolve(f1, dx_kernel, mode='constant', cval=0.0)
    # df1dy = scipy.ndimage.convolve(f1, dy_kernel, mode='constant', cval=0.0)
    # df1dz = scipy.ndimage.convolve(f1, dz_kernel, mode='constant', cval=0.0)
    #
    # df2dy = scipy.ndimage.convolve(f2, dy_kernel, mode='constant', cval=0.0)
    # df2dz = scipy.ndimage.convolve(f2, dz_kernel, mode='constant', cval=0.0)
    #
    # df3dx = scipy.ndimage.convolve(f3, dx_kernel, mode='constant', cval=0.0)
    # df3dz = scipy.ndimage.convolve(f3, dz_kernel, mode='constant', cval=0.0)
    #
    # df4dx = scipy.ndimage.convolve(f4, dx_kernel, mode='constant', cval=0.0)
    # df4dy = scipy.ndimage.convolve(f4, dy_kernel, mode='constant', cval=0.0)

    # df1dx = scipy.signal.convolve(f1, dx_kernel, mode='same')
    # df1dy = scipy.signal.convolve(f1, dy_kernel, mode='same')
    # df1dz = scipy.signal.convolve(f1, dz_kernel, mode='same')
    #
    # df2dy = scipy.signal.convolve(f2, dy_kernel, mode='same')
    # df2dz = scipy.signal.convolve(f2, dz_kernel, mode='same')
    #
    # df3dx = scipy.signal.convolve(f3, dx_kernel, mode='same')
    # df3dz = scipy.signal.convolve(f3, dz_kernel, mode='same')
    #
    # df4dx = scipy.signal.convolve(f4, dx_kernel, mode='same')
    # df4dy = scipy.signal.convolve(f4, dy_kernel, mode='same')

    df1dx = scipy.ndimage.convolve(f1, dx_kernel, mode='wrap')
    df1dy = scipy.ndimage.convolve(f1, dy_kernel, mode='wrap')
    df1dz = scipy.ndimage.convolve(f1, dz_kernel, mode='wrap')

    df2dy = scipy.ndimage.convolve(f2, dy_kernel, mode='wrap')
    df2dz = scipy.ndimage.convolve(f2, dz_kernel, mode='wrap')

    df3dx = scipy.ndimage.convolve(f3, dx_kernel, mode='wrap')
    df3dz = scipy.ndimage.convolve(f3, dz_kernel, mode='wrap')

    df4dx = scipy.ndimage.convolve(f4, dx_kernel, mode='wrap')
    df4dy = scipy.ndimage.convolve(f4, dy_kernel, mode='wrap')

    ### F1, F2, F3 terms ###

    F1 = poisson_solver_3d_fft((df1dx - df4dy + df3dz) / 32.0)  # adding up all values of kernel = 32
    F2 = poisson_solver_3d_fft((df4dx + df1dy - df2dz) / 32.0)
    F3 = poisson_solver_3d_fft((-df3dx + df2dy + df1dz) / 32.0)

    return F1, F2, F3

### Do not use jit for this one!!! because of while loop ###
def gridgen_3d(f1c, f2c, f3c, f4c, j_lb=0.1, j_ub=6.0, n_euler=20, inv=False):
# def gridgen_3d(f1, f2, f3, f4, j_lb=0.4, j_ub=4.0, n_euler=20, inv=False):
    """Grid generation"""

    # added
    while_iter = 0

    f1c = f1c / np.mean(f1c)

    ker = np.ones((3, 3, 3)) / 27.0

    # added
    print('np.min(f1c): ' + str(np.min(f1c)))
    print('np.max(f1c): ' + str(np.max(f1c)))

    num_pixels_j_lb = len(np.where(f1c < j_lb)[0])
    num_pixels_j_ub = len(np.where(f1c > j_ub)[0])
    print('num pixels < j_lb: ' + str(num_pixels_j_lb))
    print('num pixels > j_ub: ' + str(num_pixels_j_ub))

    ### Feel free to change to the one above ###
    while (np.max(f1c) > j_ub or np.min(f1c) < j_lb):
        f1c = scipy.ndimage.convolve(f1c, ker, mode='wrap')

        # added
        while_iter += 1
        print("while iter loop: " + str(while_iter))


    print("while_iter: " + str(while_iter))


    # added temp
    f1c = m_norm(f1c)

    f1 = f1c
    f2 = f2c
    f3 = f3c
    f4 = f4c

    v1, v2, v3 = div_curl_solver_3d(f1 - 1.0, f2, f3, f4, inv)
    posx, posy, posz = euler_3d(v1, v2, v3, f1, n_euler)

    # We return f1-f4 along with pos, it is used during the optimization.
    return posx, posy, posz, f1, f2, f3, f4