# test_div_curl_det_Jacobian_2D.py
#
# Understand the relationship between f1 (radial) and f2 (rotational) and the divergence, curl and determinant of the
# Jacobian. I thought that the minimum and maximum of f1 would be equal to the minimum and maximum of the divergence.
# I also thought that the curl values would not affect the computation of the determinant of the Jacobian.
#
# Note:
# 1. Also see why Ameneh's version of the determinant does not exactly match the determinant from jacob_divcurl. Looks
#    like there are probably issues at the edges.
#
# Deepa Krishnaswamy
# University of Alberta
# June 28 2021
########################################################################################################################

# copied from nonrigidreg_numba.py from Kumar's code
def jacob_divcurl(posx, posy):
    ux = 0.5 * posx[2:, 1:-1] - 0.5 * posx[:-2, 1:-1]
    uy = 0.5 * posx[1:-1, 2:] - 0.5 * posx[1:-1, :-2]
    vx = 0.5 * posy[2:, 1:-1] - 0.5 * posy[:-2, 1:-1]
    vy = 0.5 * posy[1:-1, 2:] - 0.5 * posy[1:-1, :-2]

    return ux * vy - uy * vx, vx - uy, ux + vy
    # Det, C, D

# ### 2D function was from Ameneh ###
# def calculate_determinant_Ameneh_2D(dispx, dispy):
#
#     [gx_y, gx_x] = np.gradient(dispx)
#     [gy_y, gy_x] = np.gradient(dispy)
#
#     # gx_x = gx_x + 1
#     # gy_y = gy_y + 1
#
#     # det
#     det_J = np.multiply(gx_x, gy_y) - np.multiply(gy_x, gx_y)
#
#     return det_J

### 2D function was from Ameneh ###
def calculate_determinant_Ameneh_2D(dispx, dispy):

    [gx_x, gx_y] = np.gradient(dispx)
    [gy_x, gy_y] = np.gradient(dispy)

    # gx_x = gx_x + 1
    # gy_y = gy_y + 1

    # det
    det_J = np.multiply(gx_x, gy_y) - np.multiply(gy_x, gx_y)

    return det_J

# copied from Kumar's original 2D numba code
def padarray(x):
    x = np.vstack([x[0, :], x, x[x.shape[0] - 1, :]])
    return np.hstack([x[:, 0:1], x, x[:, x.shape[1] - 1:x.shape[1]]])


def calculate_det_curl_div(posx, posy):

    [du_dx, du_dy] = np.gradient(posx, edge_order=1) # check between edge_order=1 and edge_order=2?
    [dv_dx, dv_dy] = np.gradient(posy, edge_order=1)

    det = (du_dx * dv_dy) - (du_dy * dv_dx)
    curl = dv_dx - du_dy
    div = du_dx + dv_dy

    return det, curl, div

# copied from Kumar's code.
def display_im_grid(xgrid, ygrid, im, ngrid):
    """Display both grids and image"""
    clear_output(wait=True)
    plt.imshow(im, cmap="gray")
    plt.plot(ygrid[::ngrid, ::ngrid], xgrid[::ngrid, ::ngrid], "b", lw=1.0)
    plt.plot(ygrid[::ngrid, ::ngrid].T, xgrid[::ngrid, ::ngrid].T, "b", lw=1.0)

    plt.axis("off")
    plt.axis("equal")
    # plt.pause(0.1)
    plt.pause(0.5)

########################################################################################################################

import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.np.gridgen import gridgen
from src.np.gridgen import div_curl_solver_2d
from src.np.gridgen import euler_2d
from src.np.gridgen import mygriddata

from IPython.display import clear_output

import nibabel as nib

########################################################################################################################

main_directory = r"D:\Deepa\projects\reg_3D_pytorch"
output_directory = os.path.join(main_directory, "test_div_curl_det_Jacobian_2D")
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# test_type = 'f1_constant_f2_change'
test_type = 'f1_change_f2_constant'

########################################################################################################################

if (test_type=='f1_constant_f2_change'):
    j_lb = 0.25 # previously 0.1
    j_ub = 6.0
### Test without DC ###
elif (test_type=='f1_change_f2_constant'):
    j_lb = -10.0
    j_ub = 10.0

n_euler = 20
# n_euler = 1000 # just to see if a higher value decreases the difference between the original method and one with numpy gradient

szx, szy = 121, 91
# keep radial the same and only change the curl
im = np.zeros((szx, szy))
# Create a grid image
im[::5, :], im[:, ::5] = 1, 1
im[::5, ::5] = 1

nframes = 10

# Generate radial and rotational fields
# f11 = np.linspace(1.0, 1.0, nframes) # div
# set f11 to a value near to j_lb
if (test_type=='f1_constant_f2_change'):
    f11 = (j_lb + 0.05) * np.linspace(1.0, 1.0, nframes)
    f21 = np.linspace(0.0, 1.0, nframes) # curl
elif (test_type=='f1_change_f2_constant'):
    # f11 = (j_lb + 9.95) * np.linspace(1.0, 1.0, nframes)
    # f21 = np.linspace(0.0, 1.0, nframes) # curl
    f11 = np.linspace(5.0,-5.0,nframes) # div
    f21 = np.linspace(1.0, 1.0, nframes) # curl

cz = int(0.1 * min(szx, szy))

# radial field
f1 = np.ones((szx,szy))
# set f1 to a value near to j_lb
# f1 = (j_lb + 0.05) * np.ones((szx,szy))

# rotational field
f2 = np.zeros((szx,szy))

imw_all = np.zeros((szx,szy,1,nframes))


for i in range(0,nframes):

    print ('************* frame: ' + str(i) + ' *****************')
    print ('input f1 div value: ' + str(f11[i]))
    print ('input f2 curl value: ' + str(f21[i]))

    f1[
        szx // 3 - 1 - cz : szx // 3 + cz, szy // 3 - 1 - cz : szy // 3 + cz
    ] = f11[i]
    f2[
        szx // 3 - 1 - cz : szx // 3 + cz, szy // 3 - 1 - cz : szy // 3 + cz
    ] = f21[i]

    # calculate disp.
    pos_x, pos_y = gridgen(f1, f2, j_lb=j_lb, j_ub=j_ub, n_euler=n_euler, inv=False)
    print ('pos_x min: ' + str(np.min(pos_x)) + ' max: ' + str(np.max(pos_x)))
    print ('pos_y min: ' + str(np.min(pos_y)) + ' max: ' + str(np.max(pos_y)))

    # Should be able to get the inverse without recomputing
    # for applying to image
    pos_x_inv, pos_y_inv = gridgen(f1, f2, j_lb, j_ub, inv=True)
    imw = mygriddata(pos_x_inv, pos_y_inv, im)

    m, n = szx, szy
    # should be 0:m instead of 1:m as in Kumar's original 2D code -- because his griddata doesn't subtract 1 from the
    # original rx and ry
    xI, yI = scipy.mgrid[0:m, 0:n]
    # xI, yI = scipy.mgrid[1:m+1, 1:n+1]
    disp_x, disp_y = pos_x - xI, pos_y - yI
    pos_x_inv2, pos_y_inv2 = -disp_x + xI, -disp_y + yI
    imw2 = mygriddata(pos_x_inv2, pos_y_inv2, im)

    # There will be a difference because the "inverse" calculated from using gridgen with inv=True is not the true
    # inverse! Because of the runge_kutta most likely - in reality we'd probably need a very small step size with
    # many iterations in order to result in the true inverse
    # But by subtracting the xI, yI grid, we are getting the true inverse
    diff_imw = np.sum(imw-imw2)
    print ('diff_imw: ' + str(diff_imw))
    diff_imw_min = np.min(imw-imw2)
    diff_imw_max = np.max(imw-imw2)
    print ('min: ' + str(diff_imw_min) + ' max: ' + str(diff_imw_max))

    # must pad the pos_x and pos_y before computing the det_J, curl and div
    # get disp from pos
    m, n = f1.shape
    # xI, yI = scipy.mgrid[1:m + 1, 1:n + 1] # is this supposed to be 1:m+1 or 0:m???? 0 for new gridgen version
    xI, yI = scipy.mgrid[0:m, 0:n]
    disp_x, disp_y = pos_x - xI, pos_y - yI
    # pad disp
    disp_x_pad = padarray(disp_x)
    disp_y_pad = padarray(disp_y)
    # calculate new pos
    xI_pad, yI_pad = scipy.mgrid[0:m + 2, 0:n + 2] # is this correct with Kumar's code?? should go from -1?
    pos_x_pad = disp_x_pad + xI_pad
    pos_y_pad = disp_y_pad + yI_pad

    print ('disp_x min: ' + str(np.min(disp_x)) + ' max: ' + str(np.max(disp_x)))
    print ('disp_y min: ' + str(np.min(disp_y)) + ' max: ' + str(np.max(disp_y)))

    det_J, curl, div = jacob_divcurl(pos_x_pad, pos_y_pad)
    print ('values after gridgen')
    print ('det_J min: ' + str(np.min(det_J)))
    print ('det J max: ' + str(np.max(det_J)))
    print ('divergence min: ' + str(np.min(div)))
    print ('divergence max: ' + str(np.max(div)))
    print ('curl min: ' + str(np.min(curl)))
    print ('curl max: ' + str(np.max(curl)))

    # calculate the difference between f1 (monitor function) and det J
    diff_f1_det_J = f1 - det_J
    min_diff = np.min(diff_f1_det_J)
    max_diff = np.max(diff_f1_det_J)
    sum_diff = np.sum(diff_f1_det_J)
    print ('f1 minus det_J min diff: ' + str(min_diff) + ' max_diff: ' + str(max_diff) + ' sum_diff: ' + str(sum_diff))

    # # calculate using method similar to Ameneh's
    # det_J, curl, div = calculate_det_curl_div(pos_x, pos_y)
    # print ('\nalternate values after gridgen \n')
    # print ('det_J min: ' + str(np.min(det_J)))
    # print ('det J max: ' + str(np.max(det_J)))
    # print ('divergence min: ' + str(np.min(div)))
    # print ('divergence max: ' + str(np.max(div)))
    # print ('curl min: ' + str(np.min(curl)))
    # print ('curl max: ' + str(np.max(curl)))

    display_im_grid(pos_x, pos_y, imw, 5)

    # plt.subplot(1,3,1)
    # plt.imshow(imw2, cmap='gray')
    # plt.title('original Kumar')
    # plt.subplot(1,3,2)
    # plt.imshow(imw, cmap='gray')
    # plt.title('new computed Kumar')
    # plt.subplot(1,3,3)
    # plt.imshow(imw2-imw, cmap='gray')
    # plt.title('diff')
    # plt.pause(1)

    # plt.show()
    plt.clf()

    # plt.figure()
    # plt.subplot(3,3,1)
    # plt.imshow(f1, cmap='gray')
    # plt.title('f1')
    # plt.subplot(3,3,2)
    # plt.imshow(f2, cmap='gray')
    # plt.title('f2')
    #
    # plt.subplot(3,3,4)
    # plt.imshow(pos_x, cmap='gray')
    # plt.title('pos x')
    # plt.subplot(3,3,5)
    # plt.imshow(pos_y, cmap='gray')
    # plt.title('pos y')
    #
    # plt.subplot(3,3,7)
    # plt.imshow(det_J, cmap='gray')
    # plt.title('det J')
    # plt.subplot(3,3,8)
    # plt.imshow(div, cmap='gray')
    # plt.title('div')
    # plt.subplot(3,3,9)
    # plt.imshow(curl, cmap='gray')
    # plt.title('curl')
    #
    # plt.show()
    #

    imw_all[:,:,0,i] = imw2


# save out nifti file
print ('imw_all: ' + str(imw_all.shape))
output_filename = os.path.join(output_directory, 'imw_all.nii')
img = nib.Nifti1Image(imw_all, np.eye(4))
nib.save(img, output_filename)
