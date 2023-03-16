# gridgen_3d.py
#
# This code implements the 3D-to-3D grid generation in numpy. The radial component, or divergence, is represented by f1.
# The rotational component, or curl, is represented by the three terms, f2, f3, and f4. These represent the curl around
# each of the x, y, and z axes.
#
# Notes:
# 1. Optional - use jit from numba to reduce the run time. This was used in Kumar's original 2D numba version.
# 2. One major difference is that mygriddata_3d subtracts 1 from nx, ny, and nz. This matches the original 2D numba
#    implementation from Kumar. Therefore, when rescaling the values in torch (-1 to 1) to compare to numpy, be sure
#    add 1 to pos.
# 3. Padding is performed in the div_curl_solver to remain consistent with the original 2D numba version. The padded
#    images are then passed into scipy.ndimage.convolve. One could instead perform no padding and use scipy.signal.
#    convolve, but the edges would have to be checked.
# 4. The original kernels were incorrect. They have been changed to 3D Sobel filters.
# 5. Divide by 16 and not 12 in the div_curl_solver. This is because the elements of one side of the filter add to 16.
# 6. There was an issue in the euler_3d code where f was being updated in each loop. This was incorrect.
#
# Resources:
# https://stackoverflow.com/questions/62173972/numpy-hstack-alternative-for-numba-njit
#
# Deepa Krishnaswamy
# University of Alberta
# June 2021
#
# 06-14-21 - Made some changes to the euler function. Instead of using a 3x3x3 kernel, using a 11x11x11 kernel. This
#            smooths out the f1 radial component more when it's below j_lb or above j_ub. Before was getting stuck in
#            the while loop as the number of pixels < j_lb was increasing. Feel free to change this.
# 06-15-21 - In euler_3d changed the tstep=20 to n_euler=20 to remain consistent with the call for gridgen_3d function
#          - In gridgen_3d changed the default j_lb and j_ub to j_lb=0.1 and j_ub=6.0 as these values seem to work
#            better for my ACDC MRI datasets and US datasets from the Mazankowski
#          - In the div_curl_solver function, changed F1 to division by 32
# 06-18-21 - Removed all uses of jit. Was causing problems in the euler function
########################################################################################################################

import scipy.signal
import scipy.ndimage
import numpy as np

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
            # k1_1 = v1 / f
            # k1_2 = v2 / f
            # k1_3 = v3 / f
            k1_1 = v1 / temp
            k1_2 = v2 / temp
            k1_3 = v3 / temp
        else:
            ### Old version -- this was incorrect because f was calculated each time ###
            # f = t + (1 - t) * mygriddata_3d(xx, yy, zz, f)
            # k1_1 = mygriddata_3d(xx, yy, zz, v1) / f
            # k1_2 = mygriddata_3d(xx, yy, zz, v2) / f
            # k1_3 = mygriddata_3d(xx, yy, zz, v3) / f

            # res = mygriddata_3d(xx, yy, zz, f)
            # temp = t + (1 - t) * res
            # res = mygriddata_3d(xx, yy, zz, v1)
            # k1_1 = res / temp
            # res = mygriddata_3d(xx, yy, zz, v2)
            # k1_2 = res / temp
            # res = mygriddata_3d(xx, yy, zz, v3)
            # k1_3 = res / temp

            temp = t + (1 - t) * mygriddata_3d(xx, yy, zz, f)
            k1_1 = mygriddata_3d(xx, yy, zz, v1) / temp
            k1_2 = mygriddata_3d(xx, yy, zz, v2) / temp
            k1_3 = mygriddata_3d(xx, yy, zz, v3) / temp

        xx = xx + k1_1 * h
        yy = yy + k1_2 * h
        zz = zz + k1_3 * h

    return xx, yy, zz

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


def div_curl_solver_3d(f1, f2, f3, f4, inv=False):
    """Perform div curl solver"""

    """Perform div curl solver in 3D"""
    ### These are the old kernels ###
    # if not inv:
    #     # From my pycardiac 3D functions
    #     dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dx_kernel = np.zeros((3, 3, 3))
    #     dx_kernel[0, :, :] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]
    #     dx_kernel[2, :, :] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #
    #     dy_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dy_kernel = np.zeros((3, 3, 3))
    #     dy_kernel[:, 0, :] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]
    #     dy_kernel[:, 2, :] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #
    #     dz_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dz_kernel = np.zeros((3, 3, 3))
    #     dz_kernel[:, :, 0] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]
    #     dz_kernel[:, :, 2] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #
    # else:
    #
    #     dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dx_kernel = np.zeros((3, 3, 3))
    #     dx_kernel[0, :, :] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #     dx_kernel[2, :, :] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]
    #
    #     dy_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dy_kernel = np.zeros((3, 3, 3))
    #     dy_kernel[:, 0, :] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #     dy_kernel[:, 2, :] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]
    #
    #     dz_kernel = np.zeros((3, 3, 3), dtype=np.float32)
    #     # dz_kernel = np.zeros((3, 3, 3))
    #     dz_kernel[:, :, 0] = [[-1, -6, -1], [-1, -6, -1], [-1, -6, -1]]
    #     dz_kernel[:, :, 2] = [[1, 6, 1], [1, 6, 1], [1, 6, 1]]

    ### Now using Sobel kernels ###
    # if not inv:
    if (inv==0):
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

    ### In the original 2D numba version of Kumar, this is present to pad the images. See note above. ###

    f1_t = np.zeros((f1.shape[0] + 2, f1.shape[1] + 2, f1.shape[2] + 2), dtype=np.float32)
    # f1_t = np.zeros((f1.shape[0] + 2, f1.shape[1] + 2, f1.shape[2] + 2))
    f1_t[1:f1_t.shape[0] - 1, 1:f1_t.shape[1] - 1, 1:f1_t.shape[2] - 1] = f1
    f1_t[0, :, :] = 2 * f1_t[1, :, :] - f1_t[2, :, :]
    f1_t[f1_t.shape[0] - 1, :, :] = 2 * f1_t[f1_t.shape[0] - 2, :, :] - f1_t[f1_t.shape[0] - 3, :, :]
    f1_t[:, 0, :] = 2 * f1_t[:, 1, :] - f1_t[:, 2, :]
    f1_t[:, f1_t.shape[1] - 1, :] = 2 * f1_t[:, f1_t.shape[1] - 2, :] - f1_t[:, f1_t.shape[1] - 3, :]
    f1_t[:, :, 0] = 2 * f1_t[:, :, 1] - f1_t[:, :, 2]
    f1_t[:, :, f1_t.shape[2] - 1] = 2 * f1_t[:, :, f1_t.shape[2] - 2] - f1_t[:, :, f1_t.shape[2] - 3]

    f2_t = np.zeros((f2.shape[0] + 2, f2.shape[1] + 2, f2.shape[2] + 2), dtype=np.float32)
    # f2_t = np.zeros((f2.shape[0] + 2, f2.shape[1] + 2, f2.shape[2] + 2))
    f2_t[1:f2_t.shape[0] - 1, 1:f2_t.shape[1] - 1, 1:f2_t.shape[2] - 1] = f2
    f2_t[0, :, :] = 2 * f2_t[1, :, :] - f2_t[2, :, :]
    f2_t[f2_t.shape[0] - 1, :, :] = 2 * f2_t[f2_t.shape[0] - 2, :, :] - f2_t[f2_t.shape[0] - 3, :, :]
    f2_t[:, 0, :] = 2 * f2_t[:, 1, :] - f2_t[:, 2, :]
    f2_t[:, f2_t.shape[1] - 1, :] = 2 * f2_t[:, f2_t.shape[1] - 2, :] - f2_t[:, f2_t.shape[1] - 3, :]
    f2_t[:, :, 0] = 2 * f2_t[:, :, 1] - f2_t[:, :, 2]
    f2_t[:, :, f2_t.shape[2] - 1] = 2 * f2_t[:, :, f2_t.shape[2] - 2] - f2_t[:, :, f2_t.shape[2] - 3]

    f3_t = np.zeros((f3.shape[0] + 2, f3.shape[1] + 2, f3.shape[2] + 2), dtype=np.float32)
    # f3_t = np.zeros((f3.shape[0] + 2, f3.shape[1] + 2, f3.shape[2] + 2))
    f3_t[1:f3_t.shape[0] - 1, 1:f3_t.shape[1] - 1, 1:f3_t.shape[2] - 1] = f3
    f3_t[0, :, :] = 2 * f3_t[1, :, :] - f3_t[2, :, :]
    f3_t[f3_t.shape[0] - 1, :, :] = 2 * f3_t[f3_t.shape[0] - 2, :, :] - f3_t[f3_t.shape[0] - 3, :, :]
    f3_t[:, 0, :] = 2 * f3_t[:, 1, :] - f3_t[:, 2, :]
    f3_t[:, f3_t.shape[1] - 1, :] = 2 * f3_t[:, f3_t.shape[1] - 2, :] - f3_t[:, f3_t.shape[1] - 3, :]
    f3_t[:, :, 0] = 2 * f3_t[:, :, 1] - f3_t[:, :, 2]
    f3_t[:, :, f3_t.shape[2] - 1] = 2 * f3_t[:, :, f3_t.shape[2] - 2] - f3_t[:, :, f3_t.shape[2] - 3]

    f4_t = np.zeros((f4.shape[0] + 2, f4.shape[1] + 2, f4.shape[2] + 2), dtype=np.float32)
    # f4_t = np.zeros((f4.shape[0] + 2, f4.shape[1] + 2, f4.shape[2] + 2))
    f4_t[1:f2_t.shape[0] - 1, 1:f4_t.shape[1] - 1, 1:f4_t.shape[2] - 1] = f4
    f4_t[0, :, :] = 2 * f4_t[1, :, :] - f4_t[2, :, :]
    f4_t[f4_t.shape[0] - 1, :, :] = 2 * f4_t[f4_t.shape[0] - 2, :, :] - f4_t[f4_t.shape[0] - 3, :, :]
    f4_t[:, 0, :] = 2 * f4_t[:, 1, :] - f4_t[:, 2, :]
    f4_t[:, f4_t.shape[1] - 1, :] = 2 * f4_t[:, f4_t.shape[1] - 2, :] - f4_t[:, f4_t.shape[1] - 3, :]
    f4_t[:, :, 0] = 2 * f4_t[:, :, 1] - f4_t[:, :, 2]
    f4_t[:, :, f4_t.shape[2] - 1] = 2 * f4_t[:, :, f4_t.shape[2] - 2] - f4_t[:, :, f4_t.shape[2] - 3]


    ### F1 term ###
    # df1/dx - df4/dy + df3/dz
    # each row, col, slice convol = n x (n+2) x (n+2)
    # each df term is n x n x n

    ### F2 term ###
    # df4/dx + df1/dy - df2/dz

    ### F3 term ###
    # -df3/dx + df2/dy + df1/dz

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

    ### Instead of using the above scipy.signal.convolve without padding, we use ###
    ### scipy.ndimage.convolve using the padded images ###
    ### We could remove this from both the pytorch and numpy version if wanted. See note above. ###

    df1dx = scipy.ndimage.convolve(f1_t, dx_kernel, mode='constant', cval=0.0)
    df1dy = scipy.ndimage.convolve(f1_t, dy_kernel, mode='constant', cval=0.0)
    df1dz = scipy.ndimage.convolve(f1_t, dz_kernel, mode='constant', cval=0.0)
    df1dx = df1dx[1:-1, 1:-1, 1:-1]
    df1dy = df1dy[1:-1, 1:-1, 1:-1]
    df1dz = df1dz[1:-1, 1:-1, 1:-1]

    df2dy = scipy.ndimage.convolve(f2_t, dy_kernel, mode='constant', cval=0.0)
    df2dz = scipy.ndimage.convolve(f2_t, dz_kernel, mode='constant', cval=0.0)
    df2dy = df2dy[1:-1, 1:-1, 1:-1]
    df2dz = df2dz[1:-1, 1:-1, 1:-1]

    df3dx = scipy.ndimage.convolve(f3_t, dx_kernel, mode='constant', cval=0.0)
    df3dz = scipy.ndimage.convolve(f3_t, dz_kernel, mode='constant', cval=0.0)
    df3dx = df3dx[1:-1, 1:-1, 1:-1]
    df3dz = df3dz[1:-1, 1:-1, 1:-1]

    df4dx = scipy.ndimage.convolve(f4_t, dx_kernel, mode='constant', cval=0.0)
    df4dy = scipy.ndimage.convolve(f4_t, dy_kernel, mode='constant', cval=0.0)
    df4dx = df4dx[1:-1, 1:-1, 1:-1]
    df4dy = df4dy[1:-1, 1:-1, 1:-1]

    ### F1, F2, F3 terms ###

    # F1 = (df1dx - df4dy + df3dz) / 12.0
    # F2 = (df4dx + df1dy - df2dz) / 12.0
    # F3 = (-df3dx + df2dy + df1dz) / 12.0

    # # If you add up the terms in one part of the Sobel filter, it adds to 16, hence we divide here.
    # F1 = (df1dx - df4dy + df3dz) / 16.0
    # F2 = (df4dx + df1dy - df2dz) / 16.0
    # F3 = (-df3dx + df2dy + df1dz) / 16.0

    # If you add up the terms in the Sobel filter it adds to 32, hence we normalize here.
    F1 = (df1dx - df4dy + df3dz) / 32.0
    F2 = (df4dx + df1dy - df2dz) / 32.0
    F3 = (-df3dx + df2dy + df1dz) / 32.0


    return poisson_solver_3d_fft(F1), poisson_solver_3d_fft(F2), poisson_solver_3d_fft(F3)

def gridgen_3d(f1c, f2c, f3c, f4c, j_lb=0.1, j_ub=6.0, n_euler=20, inv=False):
# def gridgen_3d(f1, f2, f3, f4, j_lb=0.4, j_ub=4.0, n_euler=20, inv=False):
    """Grid generation"""

    # added
    while_iter = 0

    f1c = f1c / np.mean(f1c)

    # ker = np.array(
    #                 [ [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]] , \
    #                   [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]] , \
    #                   [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]] ])

    # Changed 6-14-21 - This kernel provides more smoothing
    ker = np.ones((11, 11, 11)) / 1331.0

    # added
    print('np.min(f1c): ' + str(np.min(f1c)))
    print('np.max(f1c): ' + str(np.max(f1c)))

    num_pixels_j_lb = len(np.where(f1c < j_lb)[0])
    num_pixels_j_ub = len(np.where(f1c > j_ub)[0])
    print('num pixels < j_lb: ' + str(num_pixels_j_lb))
    print('num pixels > j_ub: ' + str(num_pixels_j_ub))

    # while (np.max(f1) > j_ub or np.min(f1) < j_lb):
    #     # f1 = scipy.signal.convolve(f1, ker, mode="same", boundary="wrap")
    #     f1 = scipy.signal.convolve(f1, ker, mode="same") # boundary="wrap" doesn't exist?

    ### Feel free to change to the one above ###
    while ((np.max(f1c) > j_ub) or \
           (np.min(f1c) < j_lb) or \
           (np.max(f1c) > j_ub and np.min(f1c) < j_lb)):
        # f1 = scipy.signal.convolve(f1, ker, mode="same", boundary="wrap")
        f1c = scipy.signal.convolve(f1c, ker, mode="same") # boundary="wrap" doesn't exist?

        # added
        while_iter += 1
        print("while iter loop: " + str(while_iter))


    print("while_iter: " + str(while_iter))

    f1 = f1c
    f2 = f2c
    f3 = f3c
    f4 = f4c

    v1, v2, v3 = div_curl_solver_3d(f1 - 1.0, f2, f3, f4, inv)
    posx, posy, posz = euler_3d(v1, v2, v3, f1, n_euler)

    # We return f1-f4 along with pos, it is used during the optimization.
    return posx, posy, posz, f1, f2, f3, f4