# gridgen_3d_pycardiac.py
#
# These functions perform the grid generation using pytorch.
#
# Notes:
#   1. I used padding_mode="border" instead of "zeros". This matches the griddata from numpy.
#   2. The convolutions in div_curl_solver match what's done in numpy, but are different from Kumar's 2D version.
#   3. The convolution in the while loop in gridgen matches what's done in numpy, but is different from Kumar's 2D version.
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################


import torch
import torch.nn.functional as nnf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def mygriddata_3d(tgrid, val):
    """Linear interpolation"""
    tgrid = tgrid.permute(0, 2, 3, 4, 1)
    tgrid = tgrid[..., [2, 1, 0]]
    # tres = nnf.grid_sample(
    #     val, tgrid, align_corners=True, mode="bilinear", padding_mode="zeros"
    # )
    tres = nnf.grid_sample(
        val, tgrid, align_corners=True, mode="bilinear", padding_mode="border"
    )

    return tres

def euler_3d(v, f1, n_euler = 20):
    """Euler ODE solver"""
    nx, ny, nz = torch.meshgrid(
        torch.linspace(-1, 1, f1.shape[-3], device=device),
        torch.linspace(-1, 1, f1.shape[-2], device=device),
        torch.linspace(-1, 1, f1.shape[-1], device=device),
    )

    tgrid = torch.stack([nx, ny, nz], dim=3)
    tgrid = tgrid.repeat(f1.shape[0], 1, 1, 1, 1).permute(0, 4, 1, 2, 3)

    f1 = f1.repeat(1, 3, 1, 1, 1)

    h = 1.0 / n_euler
    # f1_orig = f1

    for t in torch.arange(0, 1, h):
        if t == 0:
            k = v / f1
        else:
            # f1 = t + (1 - t) * mygriddata_3d(tgrid, f1_orig)
            # k = mygriddata_3d(tgrid, v) / f1
            temp0 = t + (1 - t) * mygriddata_3d(tgrid, f1)
            k = mygriddata_3d(tgrid, v) / temp0

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

    # dfdx = nnf.conv3d(f, dx_kernel, padding=1, groups=4)
    # dfdy = nnf.conv3d(f, dy_kernel, padding=1, groups=4)
    # dfdz = nnf.conv3d(f, dz_kernel, padding=1, groups=4)

    # dfdx = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "replicate"), dx_kernel, padding=0, groups=4) # THIS IS ONE I THOUGHT WORKS.
    # dfdy = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "replicate"), dy_kernel, padding=0, groups=4)
    # dfdz = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "replicate"), dz_kernel, padding=0, groups=4)

    dfdx = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "circular"), dx_kernel, padding=0, groups=4) # when I did my test in test_conv_torch_vs_numpy2.py this matched the numpy conv I used.
    dfdy = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "circular"), dy_kernel, padding=0, groups=4)
    dfdz = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "circular"), dz_kernel, padding=0, groups=4)

    # dfdx = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "constant", value=0), dx_kernel, padding=0, groups=4)
    # dfdy = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "constant", value=0), dy_kernel, padding=0, groups=4)
    # dfdz = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "constant", value=0), dz_kernel, padding=0, groups=4)

    F = torch.zeros((f.shape[0], 3, f.shape[2], f.shape[3], f.shape[4]))

    # If you add up the terms in the Sobel filter it adds to 32, hence we normalize here.
    F[:, 0, :, :, :] = ( dfdx[:, 0, :, :, :] - dfdy[:, 3, :, :, :] + dfdz[:, 2, :, :, :]) / 32.0
    F[:, 1, :, :, :] = ( dfdx[:, 3, :, :, :] + dfdy[:, 0, :, :, :] - dfdz[:, 1, :, :, :]) / 32.0
    F[:, 2, :, :, :] = (-dfdx[:, 2, :, :, :] + dfdy[:, 1, :, :, :] + dfdz[:, 0, :, :, :]) / 32.0

    v = poisson_solver_3d_fft(F)

    return v[...,0]


def gridgen_3d(f, j_lb=0.1, j_ub=6.0, n_euler=20, inv=False):

    """Grid generation -- equivalent to fc2pos"""
    with torch.no_grad():

        f[:, 0:1, :, :, :] = f[:, 0:1, :, :, :] / torch.mean(
            f[:, 0:1, :, :, :], dim=(-1, -2, -3), keepdims=True
        )

        k = torch.ones((3, 3, 3), dtype=torch.float, device=device) / 27.0
        k = torch.unsqueeze(k,0)
        k = torch.unsqueeze(k,0)

        # added
        while_iter = 0

        # added
        print('torch min(f1c): ' + str(torch.min(f[:,0:1,:,:,:])))
        print('torch max(f1c): ' + str(torch.max(f[:,0:1,:,:,:])))

        while (torch.max(f[:, 0:1, :, :, :]) > j_ub or torch.min(f[:, 0:1, :, :, :]) < j_lb):
            # f[:, 0:1, :, :, :] = nnf.conv3d(
            #     nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), k, padding=0
            # )
            f[:, 0:1, :, :, :] = nnf.conv3d(nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "circular"), k, padding=0)


            # added
            while_iter += 1
            print("while iter loop: " + str(while_iter))

        print("while_iter: " + str(while_iter))

        # add this in temporarily
        f[:, 0:1, :, :, :] = f[:, 0:1, :, :, :] / torch.mean(
            f[:, 0:1, :, :, :], dim=(-1, -2, -3), keepdims=True
        )

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