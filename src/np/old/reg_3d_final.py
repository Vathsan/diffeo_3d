# reg_3d.py
#
# This the 3d version of the registration code. Should match reg_3d.py from torch directory. The code performs a step-
# then-correct optimization procedure.
#
# IMPORTANT:
# Use numpy_registration_3d_new and register_sequence_3d_new
#
# Notes:
# 1. Optional - use jit from numba to reduce the run time. This was used in Kumar's original 2D numba version.
# 2. 5-18-21 - Removed the negative sign in rx1, rx2, rx3 in the grad_3d function
# 3. The original kernels were incorrect. They have been changed to 3D Sobel filters.
# 4. Divide by 16 and not by 20 in the grad function. This is because one side of filter adds to 16.
#
# Deepa Krishnaswamy
# University of Alberta
# March 31 2021
#
# 06-15-21 - Modified the RegParam to include n_euler=20.0, and changed the default values of j_lb to 0.1 and j_ub to
#            6.0, as these values seemed to work better for the ACDC MRI datasets and US datasets from the Mazankowski
#          - Included the prm.n_euler in calls to gridgen_3d.gridgen_3d()
#          - In the grad function, changed r_x1 etc to division by 32
# 06-18-21 - Removed all uses of jit. Was causing problems in the euler function
# 07-14-21 - updated conv functions
########################################################################################################################

import nibabel as nib
from src.np import gridgen_3d_final as gridgen_3d
import numpy as np
import scipy.ndimage


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

def find_cost(posx, posy, posz, im_s, im_t):
    """Compute similarity cost"""

    im_w = gridgen_3d.mygriddata_3d(posx, posy, posz, im_t)
    npts = im_s.shape[0] * im_s.shape[1] * im_s.shape[2]
    metric = np.sum(np.square(im_s-im_w)) / npts

    return im_w, metric

def padarray_3d(x):
    '''Pad array in 3 dimensions, repeated rows, cols, slices'''

    x = np.vstack([x[0:1, :, :], x, x[-1:, :, :]])
    x = np.hstack([x[:, 0:1, :], x, x[:, -1:, :]])
    x = np.dstack([x[:, :, 0:1], x, x[:, :, -1]])

    return x

def padarray_zeros_3d(x):

    x = np.vstack([np.zeros((1, x.shape[1], x.shape[2])), x, np.zeros((1, x.shape[1], x.shape[2]))])
    x = np.hstack([np.zeros((x.shape[0], 1, x.shape[2])), x, np.zeros((x.shape[0], 1, x.shape[2]))])
    x = np.dstack([np.zeros((x.shape[0], x.shape[1], 1)), x, np.zeros((x.shape[0], x.shape[1], 1))])

    return x

def grad_3d(im_t, im_ri):

    """compute gradient"""

    ### 3D Sobel kernels ###
    x1_kernel = np.zeros((3, 3, 3))
    x1_kernel[0, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    x1_kernel[2, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    x2_kernel = np.zeros((3, 3, 3))
    x2_kernel[:, 0, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    x2_kernel[:, 2, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    x3_kernel = np.zeros((3, 3, 3))
    x3_kernel[:, :, 0] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    x3_kernel[:, :, 2] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    im_d = im_ri - im_t

    ### Try this ###
    # im_ri = padarray_3d(im_ri)

    # rx1 = -scipy.ndimage.convolve(im_ri, x1_kernel, mode='constant', cval=0.0) # had to add in negative
    # rx2 = -scipy.ndimage.convolve(im_ri, x2_kernel, mode='constant', cval=0.0) # had to add in negative
    # rx3 = -scipy.ndimage.convolve(im_ri, x3_kernel, mode='constant', cval=0.0) # had to add in negative
    #
    # rx1 = rx1[1:-1, 1:-1, 1:-1] / 32.0 # This is for the new Sobel kernels.
    # rx2 = rx2[1:-1, 1:-1, 1:-1] / 32.0
    # rx3 = rx3[1:-1, 1:-1, 1:-1] / 32.0

    ### Try this ###
    # rx1 = -scipy.ndimage.convolve(im_ri, x1_kernel, mode='constant', cval=0.0) # had to add in negative
    # rx2 = -scipy.ndimage.convolve(im_ri, x2_kernel, mode='constant', cval=0.0) # had to add in negative
    # rx3 = -scipy.ndimage.convolve(im_ri, x3_kernel, mode='constant', cval=0.0) # had to add in negative
    #
    # rx1 = rx1 / 32.0 # This is for the new Sobel kernels.
    # rx2 = rx2 / 32.0
    # rx3 = rx3 / 32.0

    ### Try this ###

    im_ri = padarray_3d(im_ri)

    rx1 = -scipy.ndimage.convolve(im_ri, x1_kernel, mode='wrap')
    rx2 = -scipy.ndimage.convolve(im_ri, x2_kernel, mode='wrap')
    rx3 = -scipy.ndimage.convolve(im_ri, x3_kernel, mode='wrap')

    rx1 = rx1 / 32.0 # This is for the new Sobel kernels.
    rx2 = rx2 / 32.0
    rx3 = rx3 / 32.0

    rx1 = rx1[1:-1,1:-1,1:-1]
    rx2 = rx2[1:-1,1:-1,1:-1]
    rx3 = rx3[1:-1,1:-1,1:-1]



    g_ux = im_d * rx1
    g_uy = im_d * rx2
    g_uz = im_d * rx3

    g_f1 = gridgen_3d.poisson_solver_3d_fft(g_ux)
    g_f2 = gridgen_3d.poisson_solver_3d_fft(g_uy)
    g_f3 = gridgen_3d.poisson_solver_3d_fft(g_uz)

    ### 3D Sobel filters ###
    dF1_df1_filter = np.zeros((3, 3, 3))
    dF1_df1_filter[0, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    dF1_df1_filter[2, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    dF2_df1_filter = np.zeros((3, 3, 3))
    dF2_df1_filter[:, 0, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    dF2_df1_filter[:, 2, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    dF3_df1_filter = np.zeros((3, 3, 3))
    dF3_df1_filter[:, :, 0] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    dF3_df1_filter[:, :, 2] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    dF3_df3_filter = -dF1_df1_filter
    dF2_df4_filter = dF1_df1_filter

    dF1_df4_filter = -dF2_df1_filter
    dF3_df2_filter = dF2_df1_filter

    dF2_df2_filter = -dF3_df1_filter
    dF1_df3_filter = dF3_df1_filter

    dF1_df2_filter = np.zeros((3, 3, 3))
    dF2_df3_filter = np.zeros((3, 3, 3))
    dF3_df4_filter = np.zeros((3, 3, 3))

    # g_f1 = padarray_zeros_3d(g_f1)
    # g_f2 = padarray_zeros_3d(g_f2)
    # g_f3 = padarray_zeros_3d(g_f3)
    #
    # g_f11 = scipy.ndimage.convolve(g_f1[1:-1, 1:-1, 1:-1], dF1_df1_filter, mode='constant', cval=0.0)
    # g_f12 = scipy.ndimage.convolve(g_f2[1:-1, 1:-1, 1:-1], dF2_df1_filter, mode='constant', cval=0.0)
    # g_f13 = scipy.ndimage.convolve(g_f3[1:-1, 1:-1, 1:-1], dF3_df1_filter, mode='constant', cval=0.0)
    #
    # g_f21 = scipy.ndimage.convolve(g_f1[1:-1, 1:-1, 1:-1], dF1_df2_filter, mode='constant', cval=0.0)
    # g_f22 = scipy.ndimage.convolve(g_f2[1:-1, 1:-1, 1:-1], dF2_df2_filter, mode='constant', cval=0.0)
    # g_f23 = scipy.ndimage.convolve(g_f3[1:-1, 1:-1, 1:-1], dF3_df2_filter, mode='constant', cval=0.0)
    #
    # g_f31 = scipy.ndimage.convolve(g_f1[1:-1, 1:-1, 1:-1], dF1_df3_filter, mode='constant', cval=0.0)
    # g_f32 = scipy.ndimage.convolve(g_f2[1:-1, 1:-1, 1:-1], dF2_df3_filter, mode='constant', cval=0.0)
    # g_f33 = scipy.ndimage.convolve(g_f3[1:-1, 1:-1, 1:-1], dF3_df3_filter, mode='constant', cval=0.0)
    #
    # g_f41 = scipy.ndimage.convolve(g_f1[1:-1, 1:-1, 1:-1], dF1_df4_filter, mode='constant', cval=0.0)
    # g_f42 = scipy.ndimage.convolve(g_f2[1:-1, 1:-1, 1:-1], dF2_df4_filter, mode='constant', cval=0.0)
    # g_f43 = scipy.ndimage.convolve(g_f3[1:-1, 1:-1, 1:-1], dF3_df4_filter, mode='constant', cval=0.0)

    g_f11 = scipy.ndimage.convolve(g_f1, dF1_df1_filter, mode='constant', cval=0.0)
    g_f12 = scipy.ndimage.convolve(g_f2, dF2_df1_filter, mode='constant', cval=0.0)
    g_f13 = scipy.ndimage.convolve(g_f3, dF3_df1_filter, mode='constant', cval=0.0)

    g_f21 = scipy.ndimage.convolve(g_f1, dF1_df2_filter, mode='constant', cval=0.0)
    g_f22 = scipy.ndimage.convolve(g_f2, dF2_df2_filter, mode='constant', cval=0.0)
    g_f23 = scipy.ndimage.convolve(g_f3, dF3_df2_filter, mode='constant', cval=0.0)

    g_f31 = scipy.ndimage.convolve(g_f1, dF1_df3_filter, mode='constant', cval=0.0)
    g_f32 = scipy.ndimage.convolve(g_f2, dF2_df3_filter, mode='constant', cval=0.0)
    g_f33 = scipy.ndimage.convolve(g_f3, dF3_df3_filter, mode='constant', cval=0.0)

    g_f41 = scipy.ndimage.convolve(g_f1, dF1_df4_filter, mode='constant', cval=0.0)
    g_f42 = scipy.ndimage.convolve(g_f2, dF2_df4_filter, mode='constant', cval=0.0)
    g_f43 = scipy.ndimage.convolve(g_f3, dF3_df4_filter, mode='constant', cval=0.0)

    g_f1 = g_f11 + g_f12 + g_f13
    g_f2 = g_f21 + g_f22 + g_f23
    g_f3 = g_f31 + g_f32 + g_f33
    g_f4 = g_f41 + g_f42 + g_f43

    g_f1_max = np.max(np.absolute(g_f1))
    g_f2_max = np.max(np.absolute(g_f2))
    g_f3_max = np.max(np.absolute(g_f3))
    g_f4_max = np.max(np.absolute(g_f4))

    if g_f1_max > 0:
        g_f1 = g_f1 / g_f1_max

    if g_f2_max > 0:
        g_f2 = g_f2 / g_f2_max

    if g_f3_max > 0:
        g_f3 = g_f3 / g_f3_max

    if g_f4_max > 0:
        g_f4 = g_f4 / g_f4_max

    return g_f1, g_f2, g_f3, g_f4

### This is the version based off of Kumar's 2D numpy version ###
### But it is missing parts that were in the original 2D numba version ###
### We need gridgen to return f1 to f4, so it can be reassigned if smeasure_new < smeasure)?? don't do this in function.
def numpy_registration_3d(im_s, im_t, prm):
    """numpy version of diffeomorphic registration"""

    smeasure_new_list = []
    smeasure_list = []
    tstep_list = []

    tstep, j_lb, j_ub, n_euler = prm.t, prm.j_lb, prm.j_ub, prm.n_euler
    better = True
    iter_ = 0

    # 1 div component, 3 curl components
    f1, f2, f3, f4 = np.ones_like(im_s), np.zeros_like(im_s), np.zeros_like(im_s), np.zeros_like(im_s)

    posx, posy, posz, _, _, _, _ = gridgen_3d.gridgen_3d(f1, f2, f3, f4, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    im_w, smeasure = find_cost(posx, posy, posz, im_s, im_t)
    # print ('smeasure: ' + str(smeasure))

    while (tstep > prm.mn_t) and (iter_ < prm.mx_iter):

        if (better):
            iter_ += 1
            g_f1, g_f2, g_f3, g_f4 = grad_3d(im_s, im_w)

        # f1 = f1 - g_f1 * tstep
        # f2 = f2 - g_f2 * tstep
        # f3 = f3 - g_f3 * tstep
        # f4 = f4 - g_f4 * tstep
        f1_n = f1 - g_f1 * tstep
        f2_n = f2 - g_f2 * tstep
        f3_n = f3 - g_f3 * tstep
        f4_n = f4 - g_f4 * tstep

        # posx, posy, posz, _, _, _, _ = gridgen_3d.gridgen_3d(f1, f2, f3, f4, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)
        posx, posy, posz, _, _, _, _ = gridgen_3d.gridgen_3d(f1_n, f2_n, f3_n, f4_n, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)


        im_wt, smeasure_new = find_cost(posx, posy, posz, im_s, im_t)

        if (smeasure_new > smeasure):
            tstep *= prm.t_dn
            better = False
        else:
            tstep = np.minimum(tstep * prm.t_up, 0.9)
            better = True
            im_w = im_wt
            smeasure = smeasure_new
            f1, f2, f3, f4 = f1_n, f2_n, f3_n, f4_n


        print('smeasure_new: ' + str(smeasure_new))
        print('smeasure: ' + str(smeasure))
        print('tstep: ' + str(tstep))

        smeasure_new_list.append(smeasure_new)
        smeasure_list.append(smeasure)
        tstep_list.append(tstep)


    posx, posy, posz, _, _, _, _ = gridgen_3d.gridgen_3d(f1, f2, f3, f4, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    # return posx, posy, posz
    return posx, posy, posz, smeasure_new_list, smeasure_list, tstep_list

### This version follows the original 2D numba version ###
### The major difference is that gridgen returns f1 to f4 ###
### which are assigned to f1c_new etc if smeasure _new < smeasure ###
def numpy_registration_3d_new(im_s, im_t, prm):
    """numpy version of diffeomorphic registration"""

    smeasure_new_list = []
    smeasure_list = []
    tstep_list = []

    tstep, j_lb, j_ub, n_euler = prm.t, prm.j_lb, prm.j_ub, prm.n_euler
    better = True
    iter_ = 0

    # 1 div component, 3 curl components
    f1c, f2c, f3c, f4c = np.ones_like(im_s), np.zeros_like(im_s), np.zeros_like(im_s), np.zeros_like(im_s)

    posx, posy, posz, f1c, f2c, f3c, f4c = gridgen_3d.gridgen_3d(f1c, f2c, f3c, f4c, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    im_w, smeasure = find_cost(posx, posy, posz, im_s, im_t)

    while (tstep > prm.mn_t) and (iter_ < prm.mx_iter):

        print ('iter_: ' + str(iter_))

        if (better):
            print ('better')
            iter_ += 1
            g_f1, g_f2, g_f3, g_f4 = grad_3d(im_s, im_w)

        f1c_new = f1c - g_f1 * tstep
        f2c_new = f2c - g_f2 * tstep
        f3c_new = f3c - g_f3 * tstep
        f4c_new = f4c - g_f4 * tstep

        posx, posy, posz, f1c_new, f2c_new, f3c_new, f4c_new = gridgen_3d.gridgen_3d(f1c_new, f2c_new, f3c_new, f4c_new, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

        im_wt, smeasure_new = find_cost(posx, posy, posz, im_s, im_t)
        print ('smeasure_new before if statement: ' + str(smeasure_new))

        # temporarily save out at iter = 10
        if (iter_ == 10):
            print ("in iter = 10.")
            print ('iter_: ' + str(iter_))
            print ('smeasure_new: ' + str(smeasure_new))
            pos_all = np.zeros((posx.shape[0], posx.shape[1], posx.shape[2],3))
            pos_all[:,:,:,0] = posx
            pos_all[:,:,:,1] = posy
            pos_all[:,:,:,2] = posz
            img = nib.Nifti1Image(pos_all, np.eye(4))
            output_filename = r"D:\Deepa\projects\reg_3D_pytorch\test_reg_pycardiac_vs_numpy\pos_all_np.nii"
            nib.save(img, output_filename)

            img = nib.Nifti1Image(im_wt, np.eye(4))
            output_filename = r"D:\Deepa\projects\reg_3D_pytorch\test_reg_pycardiac_vs_numpy\im_wt_np.nii"
            nib.save(img, output_filename)

        if (smeasure_new > smeasure):
            tstep *= prm.t_dn
            better = False
        else:
            tstep = np.minimum(tstep * prm.t_up, 0.9)
            better = True
            im_w = im_wt
            smeasure = smeasure_new
            f1c, f2c, f3c, f4c = f1c_new, f2c_new, f3c_new, f4c_new
        print ('better: ' + str(better))

        print('smeasure_new: ' + str(smeasure_new))
        print('smeasure: ' + str(smeasure))
        print('tstep: ' + str(tstep))

        smeasure_new_list.append(smeasure_new)
        smeasure_list.append(smeasure)
        tstep_list.append(tstep)

    posx, posy, posz, _, _, _, _ = gridgen_3d.gridgen_3d(f1c, f2c, f3c, f4c, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    # return posx, posy, posz
    return posx, posy, posz, smeasure_new_list, smeasure_list, tstep_list


# def register_sequence_3d(ims, prm=RegParam):
#     """Register a sequence of images. The input is xsize x ysize x zsize x nframes"""
#
#     # frames is first dimension??
#     imt = np.roll(ims, shifts=-1, dims=0)
#
#     # registration in forward direction
#     pos_fx, pos_fy, pos_fz = numpy_registration_3d(ims, imt, prm)
#
#     # registration in reverse direction
#     pos_bx, pos_by, pos_bz = numpy_registration_3d(imt, ims, prm)
#
#     # reorder the reverse deformation fields
#     pos_bx = np.flip(pos_bx, dim=0)
#     pos_by = np.flip(pos_by, dim=0)
#     pos_bz = np.flip(pos_bz, dim=0)
#
#     return pos_fx, pos_fy, pos_fz, pos_bx, pos_by, pos_bz

def register_sequence_3d_new(ims, prm=RegParam):
    """Register a sequence of images. The input is xsize x ysize x zsize x nframes"""

    # frames is first dimension??
    imt = np.roll(ims, shifts=-1, dims=0)

    # registration in forward direction
    pos_fx, pos_fy, pos_fz ,_, _, _, _ = numpy_registration_3d_new(ims, imt, prm)

    # registration in reverse direction
    pos_bx, pos_by, pos_bz, _, _, _, _ = numpy_registration_3d_new(imt, ims, prm)

    # reorder the reverse deformation fields
    pos_bx = np.flip(pos_bx, dim=0)
    pos_by = np.flip(pos_by, dim=0)
    pos_bz = np.flip(pos_bz, dim=0)

    return pos_fx, pos_fy, pos_fz, pos_bx, pos_by, pos_bz
