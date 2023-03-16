import torch

# from src.torch import gridgen
from src.pytorch import gridgen
import torch.nn.functional as nnf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RegParam:
    """Define registration parameters"""

    def __init__(
        self,
        mx_iter=20.0,
        t=0.5,
        t_up=1.0,
        t_dn=2.0 / 3.0,
        mn_t=0.01,
        j_lb=0.25,
        j_ub=2.0,
    ):
        self.mx_iter = mx_iter
        self.t = t
        self.t_up = t_up
        self.t_dn = t_dn
        self.mn_t = mn_t
        self.j_lb = j_lb
        self.j_ub = j_ub


def find_cost(pos, im_s, im_t):
    """Compute similarity cost"""
    im_w = gridgen.mygriddata(pos, im_t)
    metric = torch.mean(torch.square(im_w - im_s), dim=(-1, -2))
    return im_w, metric


def grad_2d(im_t, im_ri):
    """compute gradient"""
    im_d = im_ri - im_t

    im_ri = nnf.pad(im_ri, (1, 1, 1, 1), "replicate")

    k = torch.tensor(
        [[[[-1, -4, -1], [0, 0, 0], [1, 4, 1]]]], dtype=torch.float, device=device
    )

    rx1 = -nnf.conv2d(im_ri, k, padding=0)
    rx2 = -nnf.conv2d(im_ri, k.permute(0, 1, 3, 2), padding=0)

    g_u = torch.empty_like(im_t).repeat(1, 2, 1, 1)

    g_u[:, 0:1, :, :] = im_d * rx1 / 12
    g_u[:, 1:2, :, :] = im_d * rx2 / 12

    g_f = gridgen.poisson_solver_2d_fft(g_u)[..., 0]

    g_f11 = nnf.conv2d(g_f[:, 0:1, :, :], k, padding=1)
    g_f12 = nnf.conv2d(g_f[:, 1:2, :, :], k.permute(0, 1, 3, 2), padding=1)
    g_f21 = nnf.conv2d(g_f[:, 0:1, :, :], k.permute(0, 1, 3, 2), padding=1)
    g_f22 = -nnf.conv2d(g_f[:, 1:2, :, :], k, padding=1)

    g_f1 = g_f11 + g_f12
    g_f2 = g_f21 + g_f22

    max_f1 = torch.amax(torch.abs(g_f1), dim=(-1, -2), keepdims=True)
    max_f2 = torch.amax(torch.abs(g_f2), dim=(-1, -2), keepdims=True)

    max_f1[max_f1 == 0] = 1
    max_f2[max_f2 == 0] = 1

    return g_f1 / max_f1, g_f2 / max_f2


def torch_registration(im_s, im_t, prm):
    """torch version of diffeomorphic registration"""
    with torch.no_grad():
        g_f1 = torch.zeros_like(im_s, device=device)
        g_f2 = torch.zeros_like(im_s, device=device)

        tstep = torch.tensor(prm.t).to(device).repeat(im_s.shape[0])
        better = torch.tensor(True).to(device).repeat(im_s.shape[0])
        iter_ = torch.tensor(0).to(device).repeat(im_s.shape[0])

        f = torch.ones(
            (im_s.shape[0], 2, im_s.shape[-2], im_s.shape[-1]),
            device=device,
            requires_grad=True,
        )

        f[:, 1, :, :] = 0

        pos = gridgen.gridgen(f, j_lb=prm.j_lb, j_ub=prm.j_ub)

        im_w, smeasure = find_cost(pos, im_s, im_t)

        while (max(tstep) > prm.mn_t) and (min(iter_) < prm.mx_iter):
            iter_[better] += 1

            if len(better.shape) == 0:
                better = torch.tensor([better,])

            if torch.any(better):
                g_f1[better, ...], g_f2[better, ...] = grad_2d(
                    im_s[better, ...], im_w[better, ...]
                )

            f_n = torch.empty_like(f)
            f_n[:, 0:1, ...] = f[:, 0:1, ...] - g_f1 * tstep[:, None, None, None]
            f_n[:, 1:2, ...] = f[:, 1:2, ...] - g_f2 * tstep[:, None, None, None]

            pos = gridgen.gridgen(f_n, j_lb=prm.j_lb, j_ub=prm.j_ub)

            im_wt, smeasure_new = find_cost(pos, im_s, im_t)

            better = torch.squeeze(smeasure_new < smeasure)

            tstep[~better] *= prm.t_dn
            tstep[better] = torch.clamp(tstep[better] * prm.t_up, 0.0, 0.9)
            im_w[better, ...] = im_wt[better, ...]
            smeasure[better] = smeasure_new[better]
            f[better, ...] = f_n[better, ...]

        pos = gridgen.gridgen(f, j_lb=prm.j_lb, j_ub=prm.j_ub)

    return pos


def register_sequence(ims, prm=RegParam):
    """Register a sequence of images. The input is xsize x ysize x nframes"""
    with torch.no_grad():
        ims = (
            torch.from_numpy(ims)
            .float()
            .to(device)
            .repeat(1, 1, 1, 1)
            .permute(3, 0, 1, 2)
        )
        print(ims.shape)

        imt = torch.roll(ims, shifts=-1, dims=0)
        pos_f = torch_registration(ims, imt, prm)

        nx, ny = torch.meshgrid(
            torch.arange(0, pos_f.shape[-2], device=device),
            torch.arange(0, pos_f.shape[-1], device=device),
        )

        pos_f[:, 0:1, :, :] = (pos_f.shape[-2] - 1) * (pos_f[:, 0:1, :, :] + 1) / 2
        pos_f[:, 1:2, :, :] = (pos_f.shape[-1] - 1) * (pos_f[:, 1:2, :, :] + 1) / 2

        pos_f[:, 0:1, :, :] = pos_f[:, 0:1, :, :] - nx
        pos_f[:, 1:2, :, :] = pos_f[:, 1:2, :, :] - ny

        pos_f = pos_f.permute(2, 3, 0, 1)

        pos_b = torch_registration(imt, ims, prm)

        pos_b = torch.flip(pos_b, dims=[0])

        pos_b[:, 0:1, :, :] = (pos_b.shape[-2] - 1) * (pos_b[:, 0:1, :, :] + 1) / 2
        pos_b[:, 1:2, :, :] = (pos_b.shape[-1] - 1) * (pos_b[:, 1:2, :, :] + 1) / 2

        pos_b[:, 0:1, :, :] = pos_b[:, 0:1, :, :] - nx
        pos_b[:, 1:2, :, :] = pos_b[:, 1:2, :, :] - ny

        pos_b = pos_b.permute(2, 3, 0, 1)

        pos_f, pos_b = torch.flip(pos_f, dims=[-1]), torch.flip(pos_b, dims=[-1])

        return pos_f.detach().cpu().numpy(), pos_b.detach().cpu().numpy()


def register_pair(im_s, im_t, prm):
    """Register set of template images im_t with set of study images im_s"""
    with torch.no_grad():
        im_s = torch.from_numpy(im_s).float().to(device).repeat(1, 1, 1, 1)
        im_t = torch.from_numpy(im_t).float().to(device).repeat(1, 1, 1, 1)

        pos = torch_registration(im_s, im_t, prm)
        return pos.detach().cpu().numpy()
