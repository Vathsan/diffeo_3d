import torch
import torch.nn.functional as nnf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mygriddata(tgrid, val):
    """Linear interpolation"""
    tgrid = tgrid.permute(0, 2, 3, 1)
    tgrid = tgrid[..., [1, 0]]
    tres = nnf.grid_sample(
        val, tgrid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )
    return tres


def euler_2d(v, f1, nsteps=20):
    """Euler ODE solver"""
    nx, ny = torch.meshgrid(
        torch.linspace(-1, 1, f1.shape[-2], device=device),
        torch.linspace(-1, 1, f1.shape[-1], device=device),
    )
    tgrid = torch.stack([nx, ny], dim=2)
    tgrid = tgrid.repeat(f1.shape[0], 1, 1, 1).permute(0, 3, 1, 2)

    f1 = f1.repeat(1, 2, 1, 1)

    h = 1.0 / nsteps
    for t in torch.arange(0, 1, h):
        if t == 0:
            k = v / f1
        else:
            k = mygriddata(tgrid, v) / (t + (1 - t) * mygriddata(tgrid, f1))

        tgrid = tgrid + 2 * k * h / torch.tensor(
            [[[[f1.shape[-2] - 1]], [[f1.shape[-1] - 1]]]], device=device,
        )

    return tgrid


def fast_sine_transform_y(v):
    """Perform fast sine transform in y direction"""
    n, m = v.shape[-3], v.shape[-2]
    v = nnf.pad(v, (0, 0, 1, m + 1, 0, 0, 0, 0, 0, 0))
    v = torch.fft(v, signal_ndim=1)
    # return imaginary value only
    v[..., 0] = v[..., 1]
    v[..., 1] = 0
    return v[..., 1 : m + 1, :]


def fast_sine_transform_x(v):
    """Perform fast sine transform in x direction"""
    n, m = v.shape[-3], v.shape[-2]
    v = nnf.pad(v, (0, 0, 0, 0, 1, n + 1, 0, 0, 0, 0))
    v = torch.fft(v.permute(0, 1, 3, 2, 4), signal_ndim=1).permute(0, 1, 3, 2, 4)
    # return imaginary value only
    v[..., 0] = v[..., 1]
    v[..., 1] = 0
    return v[..., 1 : n + 1, :, :]


def poisson_solver_2d_fft(f):
    """Poisson solver"""
    n, m = f.shape[-2], f.shape[-1]
    pi = 3.141592653589793
    XI, YI = torch.meshgrid(
        torch.arange(1, n + 1, device=device), torch.arange(1, m + 1, device=device)
    )
    LL = torch.zeros((XI.shape + (2,)), device=device)
    LL[:, :, 0] = 1.0 / (
        4.0 - 2.0 * torch.cos(XI * pi / (n + 1)) - 2.0 * torch.cos(YI * pi / (m + 1))
    )

    FF = torch.zeros((f.shape + (2,)), device=device)
    FF[..., 0] = f

    LL = LL.repeat(f.shape[0], f.shape[1], 1, 1, 1)

    X = (
        4.0
        / ((n + 1.0) * (m + 1.0))
        * LL
        * fast_sine_transform_y(fast_sine_transform_x(FF))
    )

    return -1.0 * fast_sine_transform_y(fast_sine_transform_x(X))


def div_curl_solver_2d(f, inv):
    """Perform div curl solver"""
    if not inv:
        weights = torch.tensor(
            [[[[-1, -4, -1], [0, 0, 0], [1, 4, 1]]]], dtype=torch.float, device=device
        )
    else:
        weights = torch.tensor(
            [[[[1, 4, 1], [0, 0, 0], [-1, -4, -1]]]], dtype=torch.float, device=device
        )

    weights = weights.repeat(2, 1, 1, 1)

    dfdx = nnf.conv2d(f, weights, padding=1, groups=2)
    dfdy = nnf.conv2d(f, weights.permute(0, 1, 3, 2), padding=1, groups=2)

    F = torch.empty_like(f)

    F[:, 0, :, :] = (dfdx[:, 0, :, :] + dfdy[:, 1, :, :]) / 12
    F[:, 1, :, :] = (-dfdx[:, 1, :, :] + dfdy[:, 0, :, :]) / 12

    v = poisson_solver_2d_fft(F)

    return v[..., 0]


def gridgen(f, j_lb=0.4, j_ub=4.0, n_euler=20, inv=False):
    """Grid generation"""
    with torch.no_grad():
        f[:, 0:1, :, :] = f[:, 0:1, :, :] / torch.mean(
            f[:, 0:1, :, :], dim=(-1, -2), keepdims=True
        )

        k = torch.tensor(
            [[[[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]]],
            dtype=torch.float,
            device=device,
        )

        # while torch.max(f[:, 0, :, :]) > j_ub and torch.min(f[:, 0, :, :]) < j_lb:
        while torch.max(f[:, 0, :, :]) > j_ub or torch.min(f[:, 0, :, :]) < j_lb:
            f[:, 0:1, :, :] = nnf.conv2d(
                nnf.pad(f[:, 0:1, :, :], (1, 1, 1, 1), "replicate"), k, padding=0
            )

        d = torch.zeros_like(f)
        d[:, 0, :, :] = 1

        v = div_curl_solver_2d(f - d, inv)
        pos = euler_2d(v, f[:, 0:1, :, :], n_euler)
        return pos


def gridgen_no_clamp(f, n_euler=20, inv=False):
    """Grid generation with no normalization of radial deformation component.
    It could be useful with PyTorch automatic gradient computation"""

    d = torch.zeros_like(f)
    d[:, 0, :, :] = 1

    v = div_curl_solver_2d(f - d, inv)
    pos = euler_2d(v, f[:, 0:1, :, :], n_euler)
    return pos
