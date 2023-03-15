# test_div_curl_det_Jacobian_3D.py
#
# Understand the relationship between f1 (radial) and f2,f3,f4 (rotational) and the divergence, curl and determinant of the
# Jacobian. I thought that the minimum and maximum of f1 would be equal to the minimum and maximum of the divergence.
# I also thought that the curl values would not affect the computation of the determinant of the Jacobian.
#
# According to Kumar -
# The determinant of the Jacobian and the monitor function (f1) - should more or less have the same values.
#
# Note:
# 1. Also see why Ameneh's version of the determinant does not exactly match the determinant from jacob_divcurl. Looks
#    like there are probably issues at the edges.
#
# Deepa Krishnaswamy
# University of Alberta
# July 8 2021
########################################################################################################################

# copied from pycardiac_3D_functions_final.py
def jacob_divcurl_3d(posx, posy, posz):
    ux = 0.5 * posx[2:, 1:-1, 1:-1] - 0.5 * posx[:-2, 1:-1, 1:-1]
    uy = 0.5 * posx[1:-1, 2:, 1:-1] - 0.5 * posx[1:-1, :-2, 1:-1]
    uz = 0.5 * posx[1:-1, 1:-1, 2:] - 0.5 * posx[1:-1, 1:-1, :-2]
    vx = 0.5 * posy[2:, 1:-1, 1:-1] - 0.5 * posy[:-2, 1:-1, 1:-1]
    vy = 0.5 * posy[1:-1, 2:, 1:-1] - 0.5 * posy[1:-1, :-2, 1:-1]
    vz = 0.5 * posy[1:-1, 1:-1, 2:] - 0.5 * posy[1:-1, 1:-1, :-2]
    wx = 0.5 * posz[2:, 1:-1, 1:-1] - 0.5 * posz[:-2, 1:-1, 1:-1]
    wy = 0.5 * posz[1:-1, 2:, 1:-1] - 0.5 * posz[1:-1, :-2, 1:-1]
    wz = 0.5 * posz[1:-1, 1:-1, 2:] - 0.5 * posz[1:-1, 1:-1, :-2]

    det = ux * (vy * wz - wy * vz) - uy * (vx * wz - wx * vz) + uz * (vx * wy - wx * vy)
    curl = (wy - vz) + (uz - wx) + (vx - uy)
    div = ux + vy + wz

    return det, curl, div

# copied from pycardiac_3D_functions_final.py
def padarray_3d(x):
    '''Pad array in 3 dimensions, repeated rows, cols, slices'''

    x = np.vstack([x[0:1, :, :], x, x[-1:, :, :]])
    x = np.hstack([x[:, 0:1, :], x, x[:, -1:, :]])
    x = np.dstack([x[:, :, 0:1], x, x[:, :, -1]])

    return x

########################################################################################################################

import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

from pycardiac_3D_functions_modified_griddata import mygriddata3
from pycardiac_3D_functions_modified_griddata import fc2pos_3d

# from src.np.gridgen import gridgen
# from src.np.gridgen import div_curl_solver_2d
# from src.np.gridgen import euler_2d
# from src.np.gridgen import mygriddata
#
# from IPython.display import clear_output

import nibabel as nib

########################################################################################################################

main_directory = r"D:\Deepa\projects\reg_3D_pytorch"
output_directory = os.path.join(main_directory, "test_div_curl_det_Jacobian_3D")
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

########################################################################################################################

# j_lb = 0.1
j_lb = 0.25
j_ub = 6.0
n_euler = 20
# n_euler = 1000 # just to see if a higher value decreases the difference between the original method and one with numpy gradient

szx, szy, szz = 121, 91, 91
# keep radial the same and only change the curl
im = np.zeros((szx, szy, szz))
# Create a grid image
im[::5, :, :], im[:, ::5, :], im[:, :, ::5] = 1, 1, 1
im[::5, ::5, ::5] = 1

nframes = 10

# Generate radial and rotational fields
# f11 = np.linspace(1.0, 1.0, nframes) # div
# set f11 to a value near to j_lb
f11 = (j_lb + 0.05) * np.linspace(1.0, 1.0, nframes)

f21 = np.linspace(0.0, 1.0, nframes) # curl

cz = int(0.1 * min(szx, szy, szz))
print ('cz: ' + str(cz))

# radial field
f1 = np.ones((szx,szy,szz))

# rotational field
f2 = np.zeros((szx,szy,szz))
f3 = np.zeros((szx,szy,szz))
f4 = np.zeros((szx,szy,szz))

imw_all = np.zeros((szx,szy,szz,nframes))

for i in range(0,nframes):

    print ('************* frame: ' + str(i) + ' *****************')
    print ('input f1 div value: ' + str(f11[i]))
    print ('input f2 curl value: ' + str(f21[i]))

    f1[
        szx // 3 - 1 - cz : szx // 3 + cz, szy // 3 - 1 - cz : szy // 3 + cz, szz // 3 - 1 - cz : szz // 3 + cz
    ] = f11[i]
    f2[
        szx // 3 - 1 - cz : szx // 3 + cz, szy // 3 - 1 - cz : szy // 3 + cz, szz // 3 - 1 - cz : szz // 3 + cz
    ] = f21[i]

    # Calculate the pos fields
    pos_x, pos_y, pos_z,_, _, _, _, _, _, _, _ = fc2pos_3d(f1, f2, f3, f4, n_euler, j_lb, j_ub, 0)

    # Get the warped image
    m, n, o = szx, szy, szz
    xI, yI, zI = scipy.mgrid[1:m+1, 1:n+1, 1:o+1]
    disp_x, disp_y, disp_z = pos_x - xI, pos_y - yI, pos_z - zI
    pos_x_inv, pos_y_inv, pos_z_inv = -disp_x + xI, -disp_y + yI, -disp_z + zI
    imw = mygriddata3(pos_x_inv, pos_y_inv, pos_z_inv, im)

    # must pad the pos_x and pos_y before computing the det_J, curl and div
    # get disp from pos
    m, n, o = f1.shape
    xI, yI, zI = scipy.mgrid[1:m + 1, 1:n + 1, 1:o + 1]
    disp_x, disp_y, disp_z = pos_x - xI, pos_y - yI, pos_z - zI
    # pad disp
    disp_x_pad = padarray_3d(disp_x)
    disp_y_pad = padarray_3d(disp_y)
    disp_z_pad = padarray_3d(disp_z)
    # calculate new pos
    xI_pad, yI_pad, zI_pad = scipy.mgrid[0:m + 2, 0:n + 2, 0:o + 2]
    pos_x_pad = disp_x_pad + xI_pad
    pos_y_pad = disp_y_pad + yI_pad
    pos_z_pad = disp_z_pad + zI_pad

    det_J, curl, div = jacob_divcurl_3d(pos_x_pad, pos_y_pad, pos_z_pad)
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

    # display_im_grid(pos_x, pos_y, imw, 5)

    imw_all[:,:,:,i] = imw

# save out nifti file
print ('imw_all: ' + str(imw_all.shape))
output_filename = os.path.join(output_directory, 'imw_all.nii')
img = nib.Nifti1Image(imw_all, np.eye(4))
nib.save(img, output_filename)


