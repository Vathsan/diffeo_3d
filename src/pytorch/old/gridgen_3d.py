# gridgen_3d.py
#
# This code implements the 3D-to-3D grid generation in pytorch. The radial component, or divergence, is represented by f1.
# The rotational component, or curl, is represented by the three terms, f2, f3, and f4. These represent the curl around
# each of the x, y, and z axes.
#
# Notes:
# 1. When rescaling to compare to numpy, be sure to add 1 to pos (slightly different from the scaling that Kumar did).
#    This is because the mygriddata_3d in numpy subtracts 1 from nx, ny, and nz. This matches the original 2D numba
#    implementationi from Kumar.
# 2. Padding is performed in the div_curl_solver to remain consistent with the original 2D numba version. The padded
#    images are then passed into nnf.conv3d. The padding could be removed from both the numpy and torch versions if
#    there is a significant slow down. Since the difference would just be on the edge, it might not make much of a
#    difference in the final displacement fields.
# 3. The original kernels were incorrect. They have been changed to 3D Sobel filters.
# 4. Divide by 16 and not 12 in the div_curl_solver. This is because the elements of one side of the filter add to 16.
# 5. There was an issue in the euler_3d code where f was being updated in each loop. This was incorrect.
#
# Deepa Krishnaswamy
# University of Alberta
# June 2021
#
# 06-14-21 - Made some changes to the euler function. Instead of using a 3x3x3 kernel, using a 11x11x11 kernel. This
#            smooths out the f1 radial component more when it's below j_lb or above j_ub. Before was getting stuck in
#            the while loop as the number of pixels < j_lb was increasing. Feel free to change this.
# 06-15-21 - In euler_3d changed the nsteps=20 to n_euler=20 to remain consistent with the call for gridgen_3d function
#          - In gridgen_3d changed the default j_lb and j_ub to j_lb=0.1 and j_ub=6.0 as these values seem to work
#            better for my ACDC MRI datasets and US datasets from the Mazankowski
#          - In the div_curl_solver function, changed F1 to division by 32
########################################################################################################################

import torch
import torch.nn.functional as nnf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def mygriddata_3d(tgrid, val):
    """Linear interpolation"""
    tgrid = tgrid.permute(0, 2, 3, 4, 1)
    tgrid = tgrid[..., [2, 1, 0]]
    # Kumar's version uses the padding with zeros, I use border. This is because I do different padding in the
    # div_curl solver.
    # tres = nnf.grid_sample(
    #     val, tgrid, align_corners=True, mode="bilinear", padding_mode="zeros"
    # )
    tres = nnf.grid_sample(
        val, tgrid, align_corners=True, mode="bilinear", padding_mode="border"
    )
    return tres

# def euler_3d(v, f1, nsteps=20):
def euler_3d(v, f1, n_euler = 20):
    """Euler ODE solver"""
    nx, ny, nz = torch.meshgrid(
        torch.linspace(-1, 1, f1.shape[-3], device=device),
        torch.linspace(-1, 1, f1.shape[-2], device=device),
        torch.linspace(-1, 1, f1.shape[-1], device=device),
    )

    # ########
    # # added - copied from poisson
    # nx = torch.transpose(nx, 0, 1)
    # ny = torch.transpose(ny, 0, 1)
    # nz = torch.transpose(nz, 0, 1)
    # #######

    tgrid = torch.stack([nx, ny, nz], dim=3)
    tgrid = tgrid.repeat(f1.shape[0], 1, 1, 1, 1).permute(0, 4, 1, 2, 3)

    f1 = f1.repeat(1, 3, 1, 1, 1)

    # h = 1.0 / nsteps
    h = 1.0 / n_euler
    f1_orig = f1

    for t in torch.arange(0, 1, h):
        if t == 0:
            k = v / f1
        else:
            # This line was in the 2D version from Kumar. It was incorrect, because f1 should not be updated each time.
            # f1 = t + (1 - t) * mygriddata_3d(tgrid, f1)
            # The line below is correct.
            f1 = t + (1 - t) * mygriddata_3d(tgrid, f1_orig)
            k = mygriddata_3d(tgrid, v) / f1

        temp = torch.tensor(
                            [[[[f1.shape[-3] - 1]], [[f1.shape[-2] - 1]], [[f1.shape[-1] - 1]]]], device=device,
        )
        temp = torch.unsqueeze(temp,4)
        tgrid = tgrid + 2 * k * h / temp


    return tgrid


def fast_sine_transform_z_3d(v):
    """Performs the fast sine transform in the z direction"""
    n, m, s = v.shape[-4], v.shape[-3], v.shape[-2]
    v = nnf.pad(v, (0, 0, 1, s + 1, 0, 0, 0, 0, 0, 0, 0, 0)) # last dim to first dim
    v = torch.fft(v, signal_ndim=1)
    # v = torch.fft.fft(v, dim=2) # If we use pytorch 1.8 later, this line will need to be checked.
    # return imaginary value only
    v[..., 0] = v[..., 1]
    v[..., 1] = 0
    return v[..., 1: s + 1, :]

def fast_sine_transform_y_3d(v):
    """Performs the fast sine transform in the y direction"""
    n, m, s = v.shape[-4], v.shape[-3], v.shape[-2]
    v = nnf.pad(v, (0, 0, 0, 0, 1, m + 1, 0, 0, 0, 0, 0, 0)) # last dim to first dim
    v = torch.fft(v.permute(0, 1, 2, 4, 3, 5), signal_ndim=1).permute(0, 1, 2, 4, 3, 5)
    # v = torch.fft.fft(v, dim=1) # If we use pytorch 1.8 later, this line will need to be checked.
    # return imaginary value only
    v[..., 0] = v[..., 1]
    v[..., 1] = 0
    return v[..., 1: m + 1, :, :]

def fast_sine_transform_x_3d(v):
    """Performs the fast sine transform in the x direction"""
    n, m, s = v.shape[-4], v.shape[-3], v.shape[-2]
    v = nnf.pad(v, (0, 0, 0, 0, 0, 0, 1, n + 1, 0, 0, 0, 0)) # last dim to first dim
    v = torch.fft(v.permute(0, 1, 4, 3, 2, 5), signal_ndim=1).permute(0, 1, 4, 3, 2, 5)
    # v = torch.fft.fft(v, dim=0) # If we use pytorch 1.8 later, this line will need to be checked.
    # return imaginary value only
    v[..., 0] = v[..., 1]
    v[..., 1] = 0
    return v[..., 1 : n + 1, :, :, :]

def poisson_solver_3d_fft(f):
    """Poisson solver in 3d"""
    n, m, s = f.shape[-3], f.shape[-2], f.shape[-1]
    pi = 3.141592653589793

    ##############################
    ### The below is for testing ###
    # XI, YI, ZI = torch.meshgrid(
    #                     torch.arange(1, m + 1, device=device, dtype=torch.float64), \
    #                     torch.arange(1, n + 1, device=device, dtype=torch.float64), \
    #                     torch.arange(1, s + 1, device=device, dtype=torch.float64)
    # )

    ##############################
    ##### I had this before ######
    # XI, YI, ZI = torch.meshgrid(
    #                     torch.arange(1, m + 1, device=device), \
    #                     torch.arange(1, n + 1, device=device), \
    #                     torch.arange(1, s + 1, device=device)
    # )

    # # need to swap axes -- in 1.7 transpose is the same as 1.8 swapaxes
    # XI = torch.transpose(XI, 0, 1)
    # YI = torch.transpose(YI, 0, 1)
    # ZI = torch.transpose(ZI, 0, 1)
    ################################

    XI, YI, ZI = torch.meshgrid(
                        torch.arange(1, n + 1, device=device), \
                        torch.arange(1, m + 1, device=device), \
                        torch.arange(1, s + 1, device=device)
    )


    # LL = torch.zeros((XI.shape + (2,)), device=device, dtype=torch.float64)
    LL = torch.zeros((XI.shape + (2,)), device=device)

    # LL[:, :, :, 0] = 1.0 / (
    #         6.0 - 2.0 * torch.cos(XI * pi / (o + 1)) - 2.0 * torch.cos(YI * pi / (n + 1)) - 2.0 * torch.cos(ZI * pi / (m + 1))
    # )

    LL[:, :, :, 0] = 1.0 / (
            2.0 * (1.0 - torch.cos(XI * pi / (n + 1.0))) +
            2.0 * (1.0 - torch.cos(YI * pi / (m + 1.0))) +
            2.0 * (1.0 - torch.cos(ZI * pi / (s + 1.0)))
    )

    # FF = torch.zeros((f.shape + (2,)), device=device, dtype=torch.float64)
    FF = torch.zeros((f.shape + (2,)), device=device)
    FF[..., 0] = f

    LL = LL.repeat(f.shape[0], f.shape[1], 1, 1, 1, 1)

    X = (
        6.0
        / ((n + 1.0) * (m + 1.0) * (s + 1.0))
        * LL
        * fast_sine_transform_z_3d(fast_sine_transform_y_3d(fast_sine_transform_x_3d(FF)))
    )

    return -1.0 * fast_sine_transform_z_3d(fast_sine_transform_y_3d(fast_sine_transform_x_3d(X)))


def div_curl_solver_3d(f, inv=False):

    """Perform div curl solver in 3D"""
    ### Old kernels ###
    # # if not inv:
    # # why do I have to switch? Same in Kumar's code.
    # if inv:
    #     dx_kernel = torch.tensor([ [[1,6,1], [1,6,1], [1,6,1]], \
    #                                [[0,0,0], [0,0,0], [0,0,0]], \
    #                                [[-1,-6,-1], [-1,-6,-1], [-1,-6,-1]] ], dtype=torch.float, device=device)
    #
    #     dy_kernel = torch.tensor([ [[1,6,1], [0,0,0], [-1,-6,-1]], \
    #                                [[1,6,1], [0,0,0], [-1,-6,-1]], \
    #                                [[1,6,1], [0,0,0], [-1,-6,-1]] ], dtype=torch.float, device=device)
    #
    #     dz_kernel = torch.tensor([ [[1,0,-1], [6,0,-6], [1,0,-1]], \
    #                                [[1,0,-1], [6,0,-6], [1,0,-1]], \
    #                                [[1,0,-1], [6,0,-6], [1,0,-1]] ], dtype=torch.float, device=device)
    #
    # else:
    #
    #     dx_kernel = torch.tensor([ [[-1,-6,-1], [-1,-6,-1], [-1,-6,-1]], \
    #                                [[0,0,0], [0,0,0], [0,0,0]], \
    #                                [[1,6,1], [1,6,1], [1,6,1]] ], dtype=torch.float, device=device)
    #
    #     dy_kernel = torch.tensor([ [[-1,-6,-1], [0,0,0], [1,6,1]], \
    #                                [[-1,-6,-1], [0,0,0], [1,6,1]], \
    #                                [[-1,-6,-1], [0,0,0], [1,6,1]] ], dtype=torch.float, device=device)
    #
    #     dz_kernel = torch.tensor([ [[-1,0,1], [-6,0,6], [-1,0,1]], \
    #                                [[-1,0,1], [-6,0,6], [-1,0,1]], \
    #                                [[-1,0,1], [-6,0,6], [-1,0,1]] ], dtype=torch.float, device=device)

    ### New kernels - 3D Sobel ###
    # if not inv:
    # why do I have to switch? Same in Kumar's code.
    if inv:
        dx_kernel = torch.tensor([ [[1,2,1], [2,4,2], [1,2,1]], \
                                   [[0,0,0], [0,0,0], [0,0,0]], \
                                   [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]] ], dtype=torch.float, device=device)

        dy_kernel = torch.tensor([ [[1,2,1], [0,0,0], [-1,-2,-1]], \
                                   [[2,4,2], [0,0,0], [-2,-4,-2]], \
                                   [[1,2,1], [0,0,0], [-1,-2,-1]] ], dtype=torch.float, device=device)

        dz_kernel = torch.tensor([ [[1,0,-1], [2,0,-2], [1,0,-1]], \
                                   [[2,0,-2], [4,0,-4], [2,0,-2]], \
                                   [[1,0,-1], [2,0,-2], [1,0,-1]] ], dtype=torch.float, device=device)

    else:

        dx_kernel = torch.tensor([ [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]], \
                                   [[0,0,0], [0,0,0], [0,0,0]], \
                                   [[1,2,1], [2,4,2], [1,2,1]] ], dtype=torch.float, device=device)

        dy_kernel = torch.tensor([ [[-1,-2,-1], [0,0,0], [1,2,1]], \
                                   [[-2,-4,-2], [0,0,0], [2,4,2]], \
                                   [[-1,-2,-1], [0,0,0], [1,2,1]] ], dtype=torch.float, device=device)

        dz_kernel = torch.tensor([ [[-1,0,1], [-2,0,2], [-1,0,1]], \
                                   [[-2,0,2], [-4,0,4], [-2,0,2]], \
                                   [[-1,0,1], [-2,0,2], [-1,0,1]] ], dtype=torch.float, device=device)

    dx_kernel = torch.unsqueeze(torch.unsqueeze(dx_kernel,0),0)
    dy_kernel = torch.unsqueeze(torch.unsqueeze(dy_kernel,0),0)
    dz_kernel = torch.unsqueeze(torch.unsqueeze(dz_kernel,0),0)

    dx_kernel = dx_kernel.repeat(4, 1, 1, 1, 1)
    dy_kernel = dy_kernel.repeat(4, 1, 1, 1, 1)
    dz_kernel = dz_kernel.repeat(4, 1, 1, 1, 1)

    # f input: torch.Size([bsz, 4, 15, 17, 19])
    ### pad each f1, f2, f3, f4 ###
    ### This is to remain consistent with the original 2D numba version ###
    f_t = torch.zeros((f.shape[0],f.shape[1],f.shape[2]+2,f.shape[3]+2,f.shape[4]+2))
    # for each of the f1, f2, f3, f4. over all bsz
    for i in torch.arange(0,4):
        f_t[:, i, 1:f_t.shape[2] - 1, 1:f_t.shape[3] - 1, 1:f_t.shape[4] -1] = f[:, i, :, :, :]
        f_t[:, i, 0, :, :] = 2 * f_t[:, i, 1, :, :] - f_t[:, i, 2, :, :]
        f_t[:, i, f_t.shape[2] - 1, :, :] = 2 * f_t[:, i, f_t.shape[2] - 2, :, :] - f_t[:, i, f_t.shape[2] - 3, :, :]
        f_t[:, i, :, 0, :] = 2 * f_t[:, i, :, 1, :] - f_t[:, i, :, 2, :]
        f_t[:, i, :, f_t.shape[3] - 1, :] = 2 * f_t[:, i, :, f_t.shape[3] - 2, :] - f_t[:, i, :, f_t.shape[3] - 3, :]
        f_t[:, i, :, :, 0] = 2 * f_t[:, i, :, :, 1] - f_t[:, i, :, :, 2]
        f_t[:, i, :, :, f_t.shape[4] - 1] = 2 * f_t[:, i, :, :, f_t.shape[4] - 2] - f_t[:, i, :, :, f_t.shape[4] - 3]


    # dfdx = nnf.conv3d(f, dx_kernel, padding=1, groups=4) # check groups
    # dfdy = nnf.conv3d(f, dy_kernel, padding=1, groups=4) # check groups
    # dfdz = nnf.conv3d(f, dz_kernel, padding=1, groups=4) # check groups

    dfdx = nnf.conv3d(f_t, dx_kernel, padding=0, groups=4) # check groups
    dfdy = nnf.conv3d(f_t, dy_kernel, padding=0, groups=4) # check groups
    dfdz = nnf.conv3d(f_t, dz_kernel, padding=0, groups=4) # check groups

    # This worked before because 1 div, 1 curl input, and v1, v2 output
    # But now, 1 div, 3 curl input, and v1, v2, v3 output
    # F = torch.empty_like(f)
    F = torch.zeros((f.shape[0], 3, f.shape[2], f.shape[3], f.shape[4]))

    # F[:, 0, :, :, :] = ( dfdx[:, 0, :, :, :] - dfdy[:, 3, :, :, :] + dfdz[:, 2, :, :, :]) / 12.0
    # F[:, 1, :, :, :] = ( dfdx[:, 3, :, :, :] + dfdy[:, 0, :, :, :] - dfdz[:, 1, :, :, :]) / 12.0
    # F[:, 2, :, :, :] = (-dfdx[:, 2, :, :, :] + dfdy[:, 1, :, :, :] + dfdz[:, 0, :, :, :]) / 12.0

    # # If you add up the terms in one part of the Sobel filter, it adds to 16, hence we divide here.
    # F[:, 0, :, :, :] = ( dfdx[:, 0, :, :, :] - dfdy[:, 3, :, :, :] + dfdz[:, 2, :, :, :]) / 16.0
    # F[:, 1, :, :, :] = ( dfdx[:, 3, :, :, :] + dfdy[:, 0, :, :, :] - dfdz[:, 1, :, :, :]) / 16.0
    # F[:, 2, :, :, :] = (-dfdx[:, 2, :, :, :] + dfdy[:, 1, :, :, :] + dfdz[:, 0, :, :, :]) / 16.0

    # If you add up the terms in the Sobel filter it adds to 32, hence we normalize here.
    F[:, 0, :, :, :] = ( dfdx[:, 0, :, :, :] - dfdy[:, 3, :, :, :] + dfdz[:, 2, :, :, :]) / 32.0
    F[:, 1, :, :, :] = ( dfdx[:, 3, :, :, :] + dfdy[:, 0, :, :, :] - dfdz[:, 1, :, :, :]) / 32.0
    F[:, 2, :, :, :] = (-dfdx[:, 2, :, :, :] + dfdy[:, 1, :, :, :] + dfdz[:, 0, :, :, :]) / 32.0

    v = poisson_solver_3d_fft(F)

    return v[...,0]


def gridgen_3d(f, j_lb=0.1, j_ub=6.0, n_euler=20, inv=False):
# def gridgen_3d(f, j_lb=0.4, j_ub=4.0, n_euler=20, inv=False):

    """Grid generation -- equivalent to fc2pos"""
    with torch.no_grad():
        f[:, 0:1, :, :, :] = f[:, 0:1, :, :, :] / torch.mean(
            f[:, 0:1, :, :, :], dim=(-1, -2, -3), keepdims=True
        )

        # k = torch.tensor([ [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]], \
        #                    [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]], \
        #                    [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]] ], dtype=torch.float, device=device)

        # Changed 6-14-21 - This kernel provides more smoothing
        # In numpy - ker = np.ones((11, 11, 11)) / 1331.0
        k = torch.ones((11, 11, 11), dtype=torch.float, device=device) / 1331.0
        k = torch.unsqueeze(k,0)
        k = torch.unsqueeze(k,0)

        # while torch.max(f[:, 0, :, :, :]) > j_ub or torch.min(f[:, 0, :, :, :]) < j_lb:
        #     f[:, 0:1, :, :, :] = nnf.conv3d(
        #         nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1), "replicate"), k, padding=0
        #     )

        ### This is what I have in my numpy version. Feel free to change to the one above ###
        # while (torch.max(f[:, 0:1, :, :, :]) > j_ub) or \
        #       (torch.min(f[:, 0:1, :, :, :]) < j_lb) or \
        #       (torch.max(f[:, 0:1, :, :, :]) > j_ub and torch.min(f[:, 0:1, :, :, :]) < j_lb):
        while (torch.max(f[:, 0:1, :, :, :]) > j_ub or torch.min(f[:, 0:1, :, :, :]) < j_lb):
            # f[:, 0:1, :, :, :] = nnf.conv3d(
            #     nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1), "replicate"), k, padding=0
            # )
            #
            # f[:, 0:1, :, :, :] = nnf.conv3d(
            #     nnf.pad(f[:, 0:1, :, :, :], (5,5,5,5,5,5), "replicate"), weight=k, stride=1, padding=0
            # )

            f[:, 0:1, :, :, :] = nnf.conv3d(f[:,0:1,:,:,:], weight=k, stride=1, padding=5)



        # d = torch.zeros_like(f)
        # This worked before because 1 div, 1 curl input, and v1, v2 output
        # But now, 1 div, 3 curl input, and v1, v2, v3 output
        # d = torch.zeros(f.shape[0], 4, f.shape[2], f.shape[3], f.shape[4], dtype=torch.float, device=device)
        d = torch.zeros(f.shape[0], 4, f.shape[2], f.shape[3], f.shape[4], dtype=f.dtype, device=device)
        # subtract 1 from just f1
        d[:, 0, :, :, :] = 1

        v = div_curl_solver_3d(f - d, inv)
        pos = euler_3d(v, f[:, 0:1, :, :, :], n_euler)

        # I return pos and f instead of just pos
        return pos, f