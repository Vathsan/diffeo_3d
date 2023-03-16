# reg_3d_pycardiac.py
#
# This implements the registration using numpy and jit.
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################


import nibabel as nib
from src.np import gridgen_3d_pycardiac as gridgen_3d
import numpy as np
import scipy.ndimage
from numba import jit

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

@jit
def find_cost(posx, posy, posz, im_s, im_t):
    """Compute similarity cost"""

    im_w = gridgen_3d.mygriddata_3d(posx, posy, posz, im_t)
    npts = im_s.shape[0] * im_s.shape[1] * im_s.shape[2]
    metric = np.sum(np.square(im_s-im_w)) / npts

    return im_w, metric

@jit
def padarray_3d(x):
    '''Pad array in 3 dimensions, repeated rows, cols, slices'''

    x = np.vstack([x[0:1, :, :], x, x[-1:, :, :]])
    x = np.hstack([x[:, 0:1, :], x, x[:, -1:, :]])
    x = np.dstack([x[:, :, 0:1], x, x[:, :, -1]])

    return x

@jit
def padarray_zeros_3d(x):

    x = np.vstack([np.zeros((1, x.shape[1], x.shape[2])), x, np.zeros((1, x.shape[1], x.shape[2]))])
    x = np.hstack([np.zeros((x.shape[0], 1, x.shape[2])), x, np.zeros((x.shape[0], 1, x.shape[2]))])
    x = np.dstack([np.zeros((x.shape[0], x.shape[1], 1)), x, np.zeros((x.shape[0], x.shape[1], 1))])

    return x

@jit
def padarray_ones_3d(x):

    x = np.vstack([np.ones((1, x.shape[1], x.shape[2])), x, np.ones((1, x.shape[1], x.shape[2]))])
    x = np.hstack([np.ones((x.shape[0], 1, x.shape[2])), x, np.ones((x.shape[0], 1, x.shape[2]))])
    x = np.dstack([np.ones((x.shape[0], x.shape[1], 1)), x, np.ones((x.shape[0], x.shape[1], 1))])

    return x

@jit
# def Gradient_3D_GGB(T, Ri, disp_x, disp_y, disp_z, mu, mylambda):
def grad_3d(T, Ri):

    filter_x1 = np.zeros((3, 3, 3))
    filter_x1[0, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    filter_x1[2, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    filter_x2 = np.zeros((3, 3, 3))
    filter_x2[:, 0, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    filter_x2[:, 2, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    filter_x3 = np.zeros((3, 3, 3))
    filter_x3[:, :, 0] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    filter_x3[:, :, 2] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    RT_diff = Ri - T

    Ri = padarray_3d(Ri) # This should be the same as the replicate in torch - nnf.pad(im_ri, (1, 1, 1, 1), "replicate")
    # Ri = padarray_ones_3d(Ri) # ?

    # R_x1 = -ndimage.convolve(Ri, filter_x1, mode='reflect') # should be same as -nnf.conv2d(im_ri, k, padding=0) -- mode=constant
    # R_x2 = -ndimage.convolve(Ri, filter_x2, mode='reflect')
    # R_x3 = -ndimage.convolve(Ri, filter_x3, mode='reflect')

    R_x1 = -scipy.ndimage.convolve(Ri, filter_x1, mode='wrap') # should be same as -nnf.conv2d(im_ri, k, padding=0) -- mode=constant
    R_x2 = -scipy.ndimage.convolve(Ri, filter_x2, mode='wrap')
    R_x3 = -scipy.ndimage.convolve(Ri, filter_x3, mode='wrap')

    R_x1 = R_x1 / 32.0  # adding up all values of kernel = 32.
    R_x2 = R_x2 / 32.0
    R_x3 = R_x3 / 32.0

    # added
    R_x1 = R_x1[1:-1,1:-1,1:-1]
    R_x2 = R_x2[1:-1,1:-1,1:-1]
    R_x3 = R_x3[1:-1,1:-1,1:-1]

    G_ux = RT_diff * R_x1
    G_uy = RT_diff * R_x2
    G_uz = RT_diff * R_x3

    G_F1 = gridgen_3d.poisson_solver_3d_fft(G_ux)
    G_F2 = gridgen_3d.poisson_solver_3d_fft(G_uy)
    G_F3 = gridgen_3d.poisson_solver_3d_fft(G_uz)

    # G_F1 = padarray_zeros_3d(G_F1)
    # G_F2 = padarray_zeros_3d(G_F2)
    # G_F3 = padarray_zeros_3d(G_F3)
    G_F1 = padarray_ones_3d(G_F1) # g_f11 = nnf.conv2d(g_f[:, 0:1, :, :], k, padding=1)
    G_F2 = padarray_ones_3d(G_F2)
    G_F3 = padarray_ones_3d(G_F3)

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

    G_f1_1 = scipy.ndimage.convolve(G_F1, dF1_df1_filter, mode='wrap')[1:-1,1:-1,1:-1] # g_f11 = nnf.conv2d(g_f[:, 0:1, :, :], k, padding=1)
    G_f1_2 = scipy.ndimage.convolve(G_F2, dF2_df1_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f1_3 = scipy.ndimage.convolve(G_F3, dF3_df1_filter, mode='wrap')[1:-1,1:-1,1:-1]

    G_f2_1 = scipy.ndimage.convolve(G_F1, dF1_df2_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f2_2 = scipy.ndimage.convolve(G_F2, dF2_df2_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f2_3 = scipy.ndimage.convolve(G_F3, dF3_df2_filter, mode='wrap')[1:-1,1:-1,1:-1]

    G_f3_1 = scipy.ndimage.convolve(G_F1, dF1_df3_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f3_2 = scipy.ndimage.convolve(G_F2, dF2_df3_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f3_3 = scipy.ndimage.convolve(G_F3, dF3_df3_filter, mode='wrap')[1:-1,1:-1,1:-1]

    G_f4_1 = scipy.ndimage.convolve(G_F1, dF1_df4_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f4_2 = scipy.ndimage.convolve(G_F2, dF2_df4_filter, mode='wrap')[1:-1,1:-1,1:-1]
    G_f4_3 = scipy.ndimage.convolve(G_F3, dF3_df4_filter, mode='wrap')[1:-1,1:-1,1:-1]

    ##################################

    G_f1 = G_f1_1 + G_f1_2 + G_f1_3
    G_f2 = G_f2_1 + G_f2_2 + G_f2_3
    G_f3 = G_f3_1 + G_f3_2 + G_f3_3
    G_f4 = G_f4_1 + G_f4_2 + G_f4_3

    # G_f1_temp, G_f2_temp, G_f3_temp, G_f4_temp = G_f1, G_f2, G_f3, G_f4

    # # added?
    # G_f1 = G_f1[1:-1,1:-1,1:-1]
    # G_f2 = G_f2[1:-1,1:-1,1:-1]
    # G_f3 = G_f3[1:-1,1:-1,1:-1]
    # G_f4 = G_f4[1:-1,1:-1,1:-1]

    G_f1_max = np.max(np.absolute(G_f1))
    G_f2_max = np.max(np.absolute(G_f2))
    G_f3_max = np.max(np.absolute(G_f3))
    G_f4_max = np.max(np.absolute(G_f4))

    if G_f1_max > 0:
        G_f1 = G_f1 / G_f1_max

    if G_f2_max > 0:
        G_f2 = G_f2 / G_f2_max

    if G_f3_max > 0:
        G_f3 = G_f3 / G_f3_max

    if G_f4_max > 0:
        G_f4 = G_f4 / G_f4_max

    return G_f1, G_f2, G_f3, G_f4


# def diffeomorphic_nonrigid_registration_3d(S, T, prm):
def numpy_registration_3d(S, T, prm):

    tstep, j_lb, j_ub, n_euler = prm.t, prm.j_lb, prm.j_ub, prm.n_euler
    better = True
    iter = 0

    f1c, f2c, f3c, f4c = np.ones_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)

    smeasure_new_list = []
    smeasure_list = []
    tstep_list = []
    f1c_new_list = []

    pos_x, pos_y, pos_z, f1c, f2c, f3c, f4c = gridgen_3d.gridgen_3d(f1c, f2c, f3c, f4c, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    [Ti, smeasure] = find_cost(pos_x, pos_y, pos_z, S, T)
    print ('first smeasure: ' + str(smeasure))

    while (tstep > prm.mn_t) and (iter < prm.mx_iter):

        print('iter: ' + str(iter))

        if better:
            # print('better')
            print('calculate grad')
            iter = iter + 1
            [G_f1c, G_f2c, G_f3c, G_f4c] = grad_3d(S, Ti)

        f1c_new = f1c - G_f1c * tstep
        f2c_new = f2c - G_f2c * tstep
        f3c_new = f3c - G_f3c * tstep
        f4c_new = f4c - G_f4c * tstep

        pos_x, pos_y, pos_z, f1c_new, f2c_new, f3c_new, f4c_new = gridgen_3d.gridgen_3d(f1c_new, f2c_new, f3c_new, f4c_new,
                                                                                                j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)


        # save out the f1 values to see if any actually fall below 0 -- this would be mesh folding.
        f1c_new_list.append(f1c_new)

        # T_temp, smeasure_new = find_cost_3d(pos_x, pos_y, pos_z, S, T, mu, mylambda, npts)
        T_temp, smeasure_new = find_cost(pos_x, pos_y, pos_z, S, T)
        print('smeasure_new before if statement: ' + str(smeasure_new))


        if (smeasure_new > smeasure):
            tstep *= prm.t_dn
            better = False
        else:
            # tstep = np.min(tstep * prm.t_up, 0.9)
            tstep = np.minimum(tstep * prm.t_up, 0.9)
            better = True
            Ti = T_temp
            smeasure = smeasure_new
            f1c, f2c, f3c, f4c = f1c_new, f2c_new, f3c_new, f4c_new

        print('smeasure_new: ' + str(smeasure_new))
        print('smeasure: ' + str(smeasure))
        print('tstep: ' + str(tstep))

        smeasure_new_list.append(smeasure_new)
        smeasure_list.append(smeasure)
        tstep_list.append(tstep)

    pos_x, pos_y, pos_z, f1c, f2c, f3c, f4c, = gridgen_3d.gridgen_3d(f1c, f2c, f3c, f4c, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler)

    # return pos_x, pos_y, pos_z, smeasure_new_list, smeasure_list, tstep_list
    return pos_x, pos_y, pos_z, f1c, f2c, f3c, f4c, smeasure_new_list, smeasure_list, tstep_list