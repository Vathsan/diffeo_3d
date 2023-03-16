# reg_3d_pycardiac.py
#
# This implements the registration using pytorch.
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################
import torch

import nibabel as nib
import numpy as np
from src.pytorch import gridgen_3d_pycardiac as gridgen_3d
import torch.nn.functional as nnf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class RegParam:
    """Define registration parameters"""

    def __init__(
        self,
        mx_iter=20.0,
        n_euler=20.0,
        t=0.5,
        t_up=1.0,
        t_dn=2.0 / 3.0,
        mn_t=0.01,
        j_lb=0.25,
        j_ub=6.0,
    ):
        self.mx_iter = mx_iter
        self.n_euler = n_euler
        self.t = t
        self.t_up = t_up
        self.t_dn = t_dn
        self.mn_t = mn_t
        self.j_lb = j_lb
        self.j_ub = j_ub

def find_cost(pos, im_s, im_t):
    """Compute similarity cost"""
    im_w = gridgen_3d.mygriddata_3d(pos, im_t)
    metric = torch.mean(torch.square(im_w - im_s), dim=(-1, -2, -3))

    # im_s_np = torch.squeeze(im_s).cpu().numpy()
    # im_w_np = torch.squeeze(im_w).cpu().numpy()
    #
    # im_s_sitk = sitk.GetImageFromArray(im_s_np, isVector = False)
    # im_w_sitk = sitk.GetImageFromArray(im_w_np, isVector = False)
    #
    # # img1 = sitk.ReadImage(im_s, sitk.sitkFloat32)
    # # img2 = sitk.ReadImage(im_t, sitk.sitkFloat32)
    # registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation()
    # metric = registration_method.MetricEvaluate(im_s_sitk, im_w_sitk)
    #
    # metric_py = torch.tensor([metric], device=device)
    # metric_py = metric_py.unsqueeze(0)

    return im_w, metric

def grad_3d(im_t, im_ri):
    """compute gradient"""

    ### Updated with 3D Sobel kernels ###
    x1_kernel = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], \
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                              [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float, device=device)

    x2_kernel = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], \
                              [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], \
                              [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float, device=device)

    x3_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], \
                              [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], \
                              [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float, device=device)


    x1_kernel = torch.unsqueeze(torch.unsqueeze(x1_kernel,0),0)
    x2_kernel = torch.unsqueeze(torch.unsqueeze(x2_kernel,0),0)
    x3_kernel = torch.unsqueeze(torch.unsqueeze(x3_kernel,0),0)

    im_d = im_ri - im_t

    im_ri = nnf.pad(im_ri, (1, 1, 1, 1, 1, 1), "replicate")

    # rx1 = -nnf.conv3d(im_ri, x1_kernel, padding=0) # but I don't have to keep center like in numpy... why?
    # rx2 = -nnf.conv3d(im_ri, x2_kernel, padding=0)
    # rx3 = -nnf.conv3d(im_ri, x3_kernel, padding=0)
    rx1 = nnf.conv3d(im_ri, x1_kernel, padding=0) # but I don't have to keep center like in numpy... why?
    rx2 = nnf.conv3d(im_ri, x2_kernel, padding=0)
    rx3 = nnf.conv3d(im_ri, x3_kernel, padding=0)

    g_u = torch.empty_like(im_t).repeat(1, 3, 1, 1, 1)

    g_u[:, 0:1, :, :, :] = im_d * rx1 / 32 # This is for the new Sobel kernels.
    g_u[:, 1:2, :, :, :] = im_d * rx2 / 32
    g_u[:, 2:3, :, :, :] = im_d * rx3 / 32

    g_f = gridgen_3d.poisson_solver_3d_fft(g_u)[..., 0]

    dF1_df1_filter = x1_kernel
    dF2_df1_filter = x2_kernel
    dF3_df1_filter = x3_kernel

    dF2_df2_filter = -dF3_df1_filter
    dF3_df2_filter = dF2_df1_filter
    dF1_df3_filter = dF3_df1_filter

    dF3_df3_filter = -dF1_df1_filter
    dF1_df4_filter = -dF2_df1_filter
    dF2_df4_filter = dF1_df1_filter

    dF1_df2_filter = torch.zeros_like(dF1_df1_filter, device=device)
    dF2_df3_filter = torch.zeros_like(dF2_df1_filter, device=device)
    dF3_df4_filter = torch.zeros_like(dF3_df1_filter, device=device)

    # g_f11 = -nnf.conv3d(g_f[:, 0:1, :, :, :], dF1_df1_filter, padding=1)
    # g_f12 = -nnf.conv3d(g_f[:, 1:2, :, :, :], dF2_df1_filter, padding=1)
    # g_f13 = -nnf.conv3d(g_f[:, 2:3, :, :, :], dF3_df1_filter, padding=1)
    #
    # g_f21 = -nnf.conv3d(g_f[:, 0:1, :, :, :], dF1_df2_filter, padding=1)
    # g_f22 = -nnf.conv3d(g_f[:, 1:2, :, :, :], dF2_df2_filter, padding=1)
    # g_f23 = -nnf.conv3d(g_f[:, 2:3, :, :, :], dF3_df2_filter, padding=1)
    #
    # g_f31 = -nnf.conv3d(g_f[:, 0:1, :, :, :], dF1_df3_filter, padding=1)
    # g_f32 = -nnf.conv3d(g_f[:, 1:2, :, :, :], dF2_df3_filter, padding=1)
    # g_f33 = -nnf.conv3d(g_f[:, 2:3, :, :, :], dF3_df3_filter, padding=1)
    #
    # g_f41 = -nnf.conv3d(g_f[:, 0:1, :, :, :], dF1_df4_filter, padding=1)
    # g_f42 = -nnf.conv3d(g_f[:, 1:2, :, :, :], dF2_df4_filter, padding=1)
    # g_f43 = -nnf.conv3d(g_f[:, 2:3, :, :, :], dF3_df4_filter, padding=1)

    # g_f11 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF1_df1_filter, padding=0)
    # g_f12 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF2_df1_filter, padding=0)
    # g_f13 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF3_df1_filter, padding=0)
    #
    # g_f21 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF1_df2_filter, padding=0)
    # g_f22 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF2_df2_filter, padding=0)
    # g_f23 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF3_df2_filter, padding=0)
    #
    # g_f31 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF1_df3_filter, padding=0)
    # g_f32 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF2_df3_filter, padding=0)
    # g_f33 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF3_df3_filter, padding=0)
    #
    # g_f41 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF1_df4_filter, padding=0)
    # g_f42 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF2_df4_filter, padding=0)
    # g_f43 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), dF3_df4_filter, padding=0)

    g_f11 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF1_df1_filter, padding=0)
    g_f12 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF2_df1_filter, padding=0)
    g_f13 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF3_df1_filter, padding=0)

    g_f21 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF1_df2_filter, padding=0)
    g_f22 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF2_df2_filter, padding=0)
    g_f23 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF3_df2_filter, padding=0)

    g_f31 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF1_df3_filter, padding=0)
    g_f32 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF2_df3_filter, padding=0)
    g_f33 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF3_df3_filter, padding=0)

    g_f41 = -nnf.conv3d(nnf.pad(g_f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF1_df4_filter, padding=0)
    g_f42 = -nnf.conv3d(nnf.pad(g_f[:, 1:2, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF2_df4_filter, padding=0)
    g_f43 = -nnf.conv3d(nnf.pad(g_f[:, 2:3, :, :, :], (1, 1, 1, 1, 1, 1), "constant", value=1), dF3_df4_filter, padding=0)

    g_f1 = g_f11 + g_f12 + g_f13
    g_f2 = g_f21 + g_f22 + g_f23
    g_f3 = g_f31 + g_f32 + g_f33
    g_f4 = g_f41 + g_f42 + g_f43

    max_f1 = torch.amax(torch.abs(g_f1), dim=(-1, -2, -3), keepdims=True)
    max_f2 = torch.amax(torch.abs(g_f2), dim=(-1, -2, -3), keepdims=True)
    max_f3 = torch.amax(torch.abs(g_f3), dim=(-1, -2, -3), keepdims=True)
    max_f4 = torch.amax(torch.abs(g_f4), dim=(-1, -2, -3), keepdims=True)

    max_f1[max_f1 == 0] = 1
    max_f2[max_f2 == 0] = 1
    max_f3[max_f3 == 0] = 1
    max_f4[max_f4 == 0] = 1

    return g_f1 / max_f1, g_f2 / max_f2, g_f3 / max_f3, g_f4 / max_f4


def torch_registration_3d(im_s, im_t, prm):
    """torch version of diffeomorphic registration"""
    with torch.no_grad():

        smeasure_new_list = []
        smeasure_list = []
        tstep_list = []

        tstep_list_tensor = torch.tensor((), device=device)
        smeasure_new_list_tensor = torch.tensor((), device=device)
        smeasure_list_tensor = torch.tensor((), device=device)

        g_f1 = torch.zeros_like(im_s, device=device)
        g_f2 = torch.zeros_like(im_s, device=device)
        g_f3 = torch.zeros_like(im_s, device=device)
        g_f4 = torch.zeros_like(im_s, device=device)

        tstep = torch.tensor(prm.t).to(device).repeat(im_s.shape[0]) # repeat by batch size
        better = torch.tensor(True).to(device).repeat(im_s.shape[0])
        iter_ = torch.tensor(0).to(device).repeat(im_s.shape[0])

        # 1 div component, 3 curl components
        # ones for div. zeros for curl.
        f = torch.ones(
            (im_s.shape[0], 4, im_s.shape[-3], im_s.shape[-2], im_s.shape[-1]),
            device=device,
            requires_grad=True,
        )

        # initialize curl to zeros, keep div equal to ones
        f[:, 1:2, :, :, :] = 0
        f[:, 2:3, :, :, :] = 0
        f[:, 3:4, :, :, :] = 0

        f_n = torch.empty_like(f)

        pos, f = gridgen_3d.gridgen_3d(f, j_lb=prm.j_lb, j_ub=prm.j_ub, n_euler=prm.n_euler)

        im_w, smeasure = find_cost(pos, im_s, im_t)
        print ('first smeasure: ' + str(smeasure))

        while (max(tstep) > prm.mn_t) and (min(iter_) < prm.mx_iter):
            # iter_[better] += 1

            print ('iter_: ' + str(iter_))

            if len(better.shape) == 0:
                better = torch.tensor([better,])

            if torch.any(better):
                print ('calculate grad')
                iter_[better] += 1
                g_f1[better, ...], g_f2[better, ...], g_f3[better, ...], g_f4[better, ...] = grad_3d(
                    im_s[better, ...], im_w[better, ...]
                )

            # f_n = torch.empty_like(f)
            f_n[:, 0:1, ...] = f[:, 0:1, ...] - g_f1 * tstep[:, None, None, None, None] # check size of t step
            f_n[:, 1:2, ...] = f[:, 1:2, ...] - g_f2 * tstep[:, None, None, None, None]
            f_n[:, 2:3, ...] = f[:, 2:3, ...] - g_f3 * tstep[:, None, None, None, None]
            f_n[:, 3:4, ...] = f[:, 3:4, ...] - g_f4 * tstep[:, None, None, None, None]

            # pos = gridgen_3d.gridgen_3d(f_n, j_lb=prm.j_lb, j_ub=prm.j_ub)
            pos, f_n = gridgen_3d.gridgen_3d(f_n, j_lb=prm.j_lb, j_ub=prm.j_ub, n_euler=prm.n_euler)

            im_wt, smeasure_new = find_cost(pos, im_s, im_t)
            print('smeasure_new before if statement: ' + str(smeasure_new))

            better = torch.squeeze(smeasure_new < smeasure)

            # If better=False, reduce size of tstep by 2/3rds of current value
            tstep[~better] *= prm.t_dn

            # If better=True, keep value of t step the same.
            tstep[better] = torch.clamp(tstep[better] * prm.t_up, 0.0, 0.9)
            im_w[better, ...] = im_wt[better, ...]
            smeasure[better] = smeasure_new[better]
            f[better, ...] = f_n[better, ...]
            #####################################

            print('better: ' + str(better))
            print('smeasure_new: ' + str(smeasure_new))
            print('smeasure: ' + str(smeasure))
            print ("smeasure_new - smeasure: " + str(smeasure_new-smeasure))
            print('tstep: ' + str(tstep))

            smeasure_new_list_tensor = torch.cat((smeasure_new_list_tensor, smeasure_new), 0)
            smeasure_list_tensor = torch.cat((smeasure_list_tensor, smeasure), 0)
            tstep_list_tensor = torch.cat((tstep_list_tensor, tstep), 0)

        pos, f = gridgen_3d.gridgen_3d(f, j_lb=prm.j_lb, j_ub=prm.j_ub, n_euler=prm.n_euler)


    return pos, f, smeasure_new_list_tensor, smeasure_list_tensor, tstep_list_tensor


def register_pair_3d(im_s, im_t, prm):
    """Register set of template images im_t with set of study images im_s"""
    with torch.no_grad():
        im_s = torch.from_numpy(im_s).float().to(device).repeat(1, 1, 1, 1, 1)
        im_t = torch.from_numpy(im_t).float().to(device).repeat(1, 1, 1, 1, 1)

        pos, f, smeasure, smeasure_new, tstep = torch_registration_3d(im_s, im_t, prm)

        # return pos.detach().cpu().numpy(), f.detach().cpu().numpy(), smeasure, smeasure_new, tstep
        return pos.detach().cpu().numpy(), f.detach().cpu().numpy(), smeasure.detach().cpu().numpy(), smeasure_new.detach().cpu().numpy(), tstep.detach().cpu().numpy()




def register_sequence_3d(ims, prm=RegParam):
    """Register a sequence of images. The input is xsize x ysize x zsize x nframes"""
    with torch.no_grad():
        # ims = (
        #     torch.from_numpy(ims)
        #     .float()
        #     .to(device)
        #     .repeat(1, 1, 1, 1)
        #     .permute(3, 0, 1, 2)
        # )
        ims = (
            torch.from_numpy(ims)
            .float()
            .to(device)
            .repeat(1, 1, 1, 1, 1)
            .permute(4, 0, 1, 2, 3) # check this
        )

        imt = torch.roll(ims, shifts=-1, dims=0) # make sure this is correct

        ### Registration in forward direction ###
        # pos_f = torch_registration_3d(ims, imt, prm)
        # pos_f, _, _, _ = torch_registration_3d_new(ims, imt, prm)
        pos_f, _, _, _, _ = torch_registration_3d(ims, imt, prm)

        nx, ny, nz = torch.meshgrid(
            torch.arange(0, pos_f.shape[-3], device=device),
            torch.arange(0, pos_f.shape[-2], device=device),
            torch.arange(0, pos_f.shape[-1], device=device),
        )

        pos_f[:, 0:1, :, :, :] = (pos_f.shape[-3] - 1) * (pos_f[:, 0:1, :, :, :] + 1) / 2
        pos_f[:, 1:2, :, :, :] = (pos_f.shape[-2] - 1) * (pos_f[:, 1:2, :, :, :] + 1) / 2
        pos_f[:, 2:3, :, :, :] = (pos_f.shape[-1] - 1) * (pos_f[:, 2:3, :, :, :] + 1) / 2

        pos_f[:, 0:1, :, :, :] = pos_f[:, 0:1, :, :, :] - nx
        pos_f[:, 1:2, :, :, :] = pos_f[:, 1:2, :, :, :] - ny
        pos_f[:, 2:3, :, :, :] = pos_f[:, 2:3, :, :, :] - nz

        # pos_f = pos_f.permute(2, 3, 0, 1)
        pos_f = pos_f.permute(2, 3, 4, 0, 1) # check this

        ### Registration in reverse direction ###
        # pos_b = torch_registration_3d(imt, ims, prm)
        # pos_b, _, _, _ = torch_registration_3d_new(imt, ims, prm)
        pos_b, _, _, _, _ = torch_registration_3d(imt, ims, prm)

        pos_b = torch.flip(pos_b, dims=[0]) # check this

        pos_b[:, 0:1, :, :, :] = (pos_b.shape[-3] - 1) * (pos_b[:, 0:1, :, :, :] + 1) / 2
        pos_b[:, 1:2, :, :, :] = (pos_b.shape[-2] - 1) * (pos_b[:, 1:2, :, :, :] + 1) / 2
        pos_b[:, 2:3, :, :, :] = (pos_b.shape[-1] - 1) * (pos_b[:, 2:3, :, :, :] + 1) / 2

        pos_b[:, 0:1, :, :, :] = pos_b[:, 0:1, :, :, :] - nx
        pos_b[:, 1:2, :, :, :] = pos_b[:, 1:2, :, :, :] - ny
        pos_b[:, 2:3, :, :, :] = pos_b[:, 2:3, :, :, :] - nz

        # pos_b = pos_b.permute(2, 3, 0, 1)
        pos_b = pos_b.permute(2, 3, 4, 0, 1) # check this

        pos_f, pos_b = torch.flip(pos_f, dims=[-1]), torch.flip(pos_b, dims=[-1]) # check this, didn't change

        return pos_f.detach().cpu().numpy(), pos_b.detach().cpu().numpy()