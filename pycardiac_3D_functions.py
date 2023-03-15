# pycardiac_3D_functions.py 
# 
# These functions are used only for computing auto contours, transforming meshes, 
# getting the point correspondence, etc. Please use the registration functions in 
# src/np and src/pytorch for the actual registration code. 
#
# Deepa Krishnaswamy
# University of Alberta
# June 2021
###########################################################################################

import os

if ('CONDA_DEFAULT_ENV' in os.environ.keys()):
    print(os.environ['CONDA_DEFAULT_ENV'])
    if (os.environ['CONDA_DEFAULT_ENV'] == "dipy2"):
        import dipy
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import SSDMetric, CCMetric
    elif (os.environ['CONDA_DEFAULT_ENV'] == "reg_3d_paper"):
        import dipy
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import SSDMetric, CCMetric

import sys
import re

import numpy as np
from numpy import pi
import scipy
from scipy import ndimage
import scipy.io as sio

from numba import jit
import vtk
import nibabel as nib

import time

from reg_functions_validation import get_points_from_mesh, get_faces_from_mesh
from reg_functions_validation import form_mesh_from_vertices_and_faces

###########################################################################################

class registration_parameters:
    def __init__(self, nRK=20.0, mx_iter=20.0, t=0.5, t_up=1.0, t_dn=2.0 / 3.0, mn_t=0.01, J_lb=0.25, J_ub=4.0):
        self.nRK = nRK
        self.mx_iter = mx_iter
        self.t = t
        self.t_up = t_up
        self.t_dn = t_dn
        self.mn_t = mn_t
        self.J_lb = J_lb
        self.J_ub = J_ub

def compute_auto_contours_vtk(dispm_file_3D_filename, num_frames, frame_list, contour_list, NGRIDPTS, spacing, img_list,
                              perform_forward_and_reverse=1):
    """Compute or recompute auto contours based on the displacement matrix."""

    cardiacnumofimages = num_frames

    try:
        DM_i = np.load(dispm_file_3D_filename, 'rb')
    except:
        DM_i = sio.loadmat(dispm_file_3D_filename)

    disp_x = DM_i["DM_fx"]
    print('disp_x: ' + str(disp_x.shape))

    i_mc = frame_list
    pts_list = contour_list

    endo_contour_list = [None] * num_frames
    # grid_contour_list = [None] * num_frames
    nii_warped_forward_list_vtk = [None] * num_frames
    nii_warped_reverse_list_vtk = [None] * num_frames

    nii_warped_forward_vtk = [None] * num_frames
    nii_warped_reverse_vtk = [None] * num_frames

    ind_shift_all = {}
    for i in range(len(i_mc)):
        if (perform_forward_and_reverse):
            contour, nii_warped_forward_vtk, nii_warped_reverse_vtk, index1, index2, ind_shift_mean, ind_direction = \
                get_pointcorrespondence_3d_forward_and_reverse_vtk(DM_i, i_mc[i], i_mc[(i + 1) % len(i_mc)],
                                                                   pts_list[i], pts_list[(i + 1) % len(i_mc)],
                                                                   cardiacnumofimages, spacing, img_list[i],
                                                                   img_list[(i + 1) % len(i_mc)])
        else:
            contour, nii_warped_forward_vtk, index1, index2, ind_shift_mean, ind_direction = \
                get_pointcorrespondence_3d_forward_vtk(DM_i, i_mc[i], i_mc[(i + 1) % len(i_mc)], pts_list[i],
                                                       pts_list[(i + 1) % len(i_mc)], cardiacnumofimages, spacing,
                                                       img_list[i], img_list[(i + 1) % len(i_mc)])

        ind_shift_all[index2] = (ind_shift_mean, ind_direction)

        print('contour: ' + str(len(contour)))
        print('index1: ' + str(index1))
        print('index2: ' + str(index2))

        for k in range(index1 + 1, index2 + 1):
            ### new --> This makes ED and ES match up finally!!
            endo_contour_list[k % cardiacnumofimages] = contour[k - index1 - 1]
            # grid_contour_list[k % cardiacnumofimages] = mesh[k % cardiacnumofimages]
            nii_warped_forward_list_vtk[k % cardiacnumofimages] = nii_warped_forward_vtk[k - index1 - 1]
            nii_warped_reverse_list_vtk[k % cardiacnumofimages] = nii_warped_reverse_vtk[k - index1 - 1]

    # return endo_contour_list, grid_contour_list, mesh
    return endo_contour_list, nii_warped_forward_list_vtk, nii_warped_reverse_list_vtk


def transform_mesh(mesh, img, DM_fx, DM_fy, DM_fz, dim, spacing):
    '''Transform mesh and img by the displacement fields'''

    # Create vtk image
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(dim[0:3])
    img_vtk.SetSpacing(spacing)
    info = vtk.vtkInformation()
    img_vtk.SetNumberOfScalarComponents(3, info)
    # img_vtk.AllocateScalars(vtk.VTK_FLOAT, 3)
    img_vtk.AllocateScalars(vtk.VTK_DOUBLE, 3)

    # Fill every entry of the image data
    for z in range(dim[2]):
        for y in range(dim[1]):
            for x in range(dim[0]):
                img_vtk.SetScalarComponentFromDouble(x, y, z, 0, DM_fx[x, y, z])
                img_vtk.SetScalarComponentFromDouble(x, y, z, 1, DM_fy[x, y, z])
                img_vtk.SetScalarComponentFromDouble(x, y, z, 2, DM_fz[x, y, z])

    # Get transform
    transform = vtk.vtkGridTransform()
    transform.SetDisplacementGridData(img_vtk)
    transform.SetInterpolationModeToLinear()
    transform.Update()

    ### Divide the mesh points by spacing before transforming ###
    mesh_points = get_points_from_mesh(mesh)
    mesh_faces = get_faces_from_mesh(mesh)
    #     mesh_points[:,0] = mesh_points[:,0] / spacing[0]
    #     mesh_points[:,1] = mesh_points[:,1] / spacing[1]
    #     mesh_points[:,2] = mesh_points[:,2] / spacing[2]
    mesh = form_mesh_from_vertices_and_faces(mesh_points, mesh_faces)

    # print ('mesh from transform_mesh form_mesh_from_vertices_and_faces: ' + str(mesh))

    # Transform the polydata by the vtk grid transform

    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetInputData(mesh)
    filter.SetTransform(transform)
    filter.Update()

    # Get the transformed polydata
    mesh_transformed = vtk.vtkPolyData()
    mesh_transformed = filter.GetOutput()

    ### Multiply by the spacing to get the mesh points back in mm space again ###
    mesh_transformed_points = get_points_from_mesh(mesh_transformed)
    #     mesh_transformed_points[:,0] = mesh_transformed_points[:,0] * spacing[0]
    #     mesh_transformed_points[:,1] = mesh_transformed_points[:,1] * spacing[1]
    #     mesh_transformed_points[:,2] = mesh_transformed_points[:,2] * spacing[2]
    mesh_transformed = form_mesh_from_vertices_and_faces(mesh_transformed_points, mesh_faces)

    # print ('mesh_transformed: ' + str(mesh_transformed))

    ### Warp the image by the deformation fields ###
    m, n, s = dim[0], dim[1], dim[2]
    xI, yI, zI = scipy.mgrid[1:m + 1, 1:n + 1, 1:s + 1] ### CHECK THIS. Might need to start from 0? Because I now subtract 1 from griddata
    # xI, yI, zI = scipy.mgrid[0:m, 0:n, 0:s]
    # had to change the line below - since we want the opposite deformation
    pos_x, pos_y, pos_z = -DM_fx + xI, -DM_fy + yI, -DM_fz + zI

    image_array = mygriddata3(pos_x * 1.0, pos_y * 1.0, pos_z * 1.0, img * 1.0)

    return mesh_transformed, image_array


def get_pointcorrespondence_3d_forward_vtk(DM_i, index1, index2, pts1, pts2, nframes, spacing, img1, img2):
    closed = 1

    """Get point correspondence using a displacement matrix for images lies between index 1 and 2."""
    """Does not use find_contour_index_association"""

    ind_shift, ind_direction = None, None
    DM_fx, DM_fy, DM_fz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"]

    index2 += nframes if index2 <= index1 else 0  # add nframes if index2 is smaller than index1

    contour_f, contour = [], {}
    nii_warped_forward_list_f, nii_warped_forward_list = [], {}

    dim = DM_fx[0].shape

    pts1_temp = pts1
    img1_temp = img1

    for k in np.arange(index1, index2 - 1):
        # mesh_transformed_forward, nii_warped_forward = transform_mesh(pts1_temp, img1_temp, DM_fx[k-1], DM_fy[k-1], DM_fz[k-1], dim, spacing)
        mesh_transformed_forward, nii_warped_forward = transform_mesh(pts1_temp, img1_temp, DM_fx[k % nframes],
                                                                      DM_fy[k % nframes], DM_fz[k % nframes], dim,
                                                                      spacing)
        contour_f.append(mesh_transformed_forward)
        nii_warped_forward_list_f.append(nii_warped_forward)
        ### Instead of using the same pts1, need to keep resetting it!!! ###
        pts1_temp = mesh_transformed_forward
        img1_temp = nii_warped_forward

    print('contour_f: ' + str(len(contour_f)))
    print('nii_warped_forward_list_f: ' + str(len(nii_warped_forward_list_f)))

    for k in range(index1, index2 - 1):
        contour[k - index1] = contour_f[k - index1]
        nii_warped_forward_list[k - index1] = nii_warped_forward_list_f[k - index1]

    print('contour: ' + str(len(contour)))
    print('nii_warped_forward_list: ' + str(len(nii_warped_forward_list)))

    contour[index2 - index1 - 1] = pts2
    # nii_warped_forward_list[index2 - index1 - 1] = img
    nii_warped_forward_list[index2 - index1 - 1] = img2

    print('contour: ' + str(len(contour)))
    print('nii_warped_forward_list: ' + str(len(nii_warped_forward_list)))

    return contour, nii_warped_forward_list, index1, index2, ind_shift, ind_direction


def get_pointcorrespondence_3d_forward_and_reverse_vtk(DM_i, index1, index2, pts1, pts2, nframes, spacing, img1, img2):
    closed = 1

    """Get point correspondence using a displacement matrix for images lies between index 1 and 2."""
    """Does not use find_contour_index_association"""

    ind_shift, ind_direction = None, None
    DM_fx, DM_fy, DM_fz, DM_bx, DM_by, DM_bz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"], DM_i["DM_bx"], DM_i[
        "DM_by"], DM_i["DM_bz"]

    index2 += nframes if index2 <= index1 else 0  # add nframes if index2 is smaller than index1

    contour_f, contour_b, contour = [], [], {}
    nii_warped_forward_list_f, nii_warped_forward_list = [], {}
    nii_warped_reverse_list_b, nii_warped_reverse_list = [], {}

    dim = DM_fx[0].shape

    pts1_temp = pts1
    img1_temp = img1

    for k in np.arange(index1, index2 - 1):
        # mesh_transformed_forward, nii_warped_forward = transform_mesh(pts1_temp, img1_temp, DM_fx[k-1], DM_fy[k-1], DM_fz[k-1], dim, spacing)
        mesh_transformed_forward, nii_warped_forward = transform_mesh(pts1_temp, img1_temp, DM_fx[k % nframes],
                                                                      DM_fy[k % nframes], DM_fz[k % nframes], dim,
                                                                      spacing)
        mesh_transformed_forward_points = get_points_from_mesh(mesh_transformed_forward)
        contour_f.append(mesh_transformed_forward_points)
        nii_warped_forward_list_f.append(nii_warped_forward)
        ### Instead of using the same pts1, need to keep resetting it!!! ###
        pts1_temp = mesh_transformed_forward
        img1_temp = nii_warped_forward

    print('contour_f: ' + str(len(contour_f)))
    print('nii_warped_forward_list_f: ' + str(len(nii_warped_forward_list_f)))

    # pts1_temp = pts1
    # img1_temp = img1

    pts2_temp = pts2
    img2_temp = img2

    for k in np.arange(index2, index1 + 1, -1):
        # mesh_transformed_reverse, nii_warped_reverse = transform_mesh(pts1_temp, img_temp, DM_bx[(nframes - k) % nframes], DM_by[(nframes - k) % nframes], DM_bz[(nframes - k) % nframes], dim, spacing)
        # mesh_transformed_reverse, nii_warped_reverse = transform_mesh(pts1_temp, img1_temp, DM_bx[k-1], DM_by[k-1], DM_bz[k-1], dim, spacing)
        # mesh_transformed_reverse, nii_warped_reverse = transform_mesh(pts2_temp, img2_temp, DM_bx[k-1], DM_by[k-1], DM_bz[k-1], dim, spacing)
        mesh_transformed_reverse, nii_warped_reverse = transform_mesh(pts2_temp, img2_temp, DM_bx[k % nframes],
                                                                      DM_by[k % nframes], DM_bz[k % nframes], dim,
                                                                      spacing)

        mesh_transformed_reverse_points = get_points_from_mesh(mesh_transformed_reverse)
        contour_b.insert(0, mesh_transformed_reverse_points)
        nii_warped_reverse_list_b.insert(0, nii_warped_reverse)
        ### Instead of using the same pts1, need to keep resetting it!!! ###
        # pts1_temp = mesh_transformed_reverse
        # img1_temp = nii_warped_reverse
        pts2_temp = mesh_transformed_reverse
        img2_temp = nii_warped_reverse

    print('contour_b: ' + str(len(contour_b)))
    print('nii_warped_reverse_list_b: ' + str(len(nii_warped_reverse_list_b)))

    for k in range(index1, index2 - 1):
        w = 1.0 * (k - index1 + 1) / (1.0 * (index2 - index1))
        # contour[k - index1], ind_shift, ind_direction = find_contour_index_association(closed, contour_f[k - index1], contour_b[k - index1], w)
        contour[k - index1] = ((1.0 - w) * contour_f[k - index1]) + (w * contour_b[k - index1])
        nii_warped_forward_list[k - index1] = nii_warped_forward_list_f[k - index1]
        nii_warped_reverse_list[k - index1] = nii_warped_reverse_list_b[k - index1]

    print('contour: ' + str(len(contour)))
    print('nii_warped_forward_list: ' + str(len(nii_warped_forward_list)))
    print('nii_warped_reverse_list: ' + str(len(nii_warped_reverse_list)))

    faces = get_faces_from_mesh(pts2)
    pts2_points = get_points_from_mesh(pts2)
    contour[index2 - index1 - 1] = pts2_points
    # nii_warped_forward_list[index2 - index1 - 1] = img
    # nii_warped_reverse_list[index2 - index1 - 1] = img
    nii_warped_forward_list[index2 - index1 - 1] = img2
    nii_warped_reverse_list[index2 - index1 - 1] = img2

    print('contour in get_pointcorrespondence_3d_forward_and_reverse_vtk: ' + str(len(contour)))
    print('nii_warped_forward_list: ' + str(len(nii_warped_forward_list)))
    print('nii_warped_reverse_list: ' + str(len(nii_warped_reverse_list)))

    # Convert back to meshes
    contour_mesh = {}
    faces = get_faces_from_mesh(pts1)

    for k in range(0, len(contour)):
        contour_mesh[k] = form_mesh_from_vertices_and_faces(contour[k], faces)

    return contour_mesh, nii_warped_forward_list, nii_warped_reverse_list, index1, index2, ind_shift, ind_direction


###########################################################################


# Similar to code from Kumar's pycardiac analysis code
def compute_auto_contours(dispm_file_3D_filename, num_frames, frame_list, contour_list, NGRIDPTS,
                          perform_forward_and_reverse=1):
    """Compute or recompute auto contours based on the displacement matrix."""

    # NGRIDPTS = 30
    # cardiacnumofimages = num_frames - 1
    cardiacnumofimages = num_frames

    # DM_i = np.load(file(dispm_file_3D_filename, 'rb'))
    DM_i = np.load(dispm_file_3D_filename, 'rb')

    disp_x = DM_i["DM_fx"]
    print('disp_x: ' + str(disp_x.shape))

    # meshpts = np.meshgrid(np.linspace(DM_i["bb"][0], DM_i["bb"][1], NGRIDPTS), np.linspace(DM_i["bb"][2], DM_i["bb"][3], NGRIDPTS))
    x = DM_i["DM_fx"].shape[0]
    y = DM_i["DM_fx"].shape[1]
    z = DM_i["DM_fx"].shape[2]
    meshpts = np.meshgrid(np.linspace(0, x, NGRIDPTS), np.linspace(0, y, NGRIDPTS), np.linspace(0, z, NGRIDPTS))

    if (perform_forward_and_reverse):
        mesh = get_pointcorrespondence_mesh_3d_forward_and_reverse(DM_i=DM_i, pts=np.vstack(
            [meshpts[0].flatten(), meshpts[1].flatten(), meshpts[2].flatten()]).T, nframes=cardiacnumofimages)
    else:
        mesh = get_pointcorrespondence_mesh_3d_forward(DM_i=DM_i, pts=np.vstack(
            [meshpts[0].flatten(), meshpts[1].flatten(), meshpts[2].flatten()]).T, nframes=cardiacnumofimages)

    i_mc = frame_list
    pts_list = contour_list

    endo_contour_list = [None] * num_frames
    grid_contour_list = [None] * num_frames

    ind_shift_all = {}
    for i in range(len(i_mc)):
        # contour, index1, index2, ind_shift_mean, ind_direction = get_pointcorrespondence_3d(DM_i, i_mc[i], i_mc[(i + 1) % len(i_mc)], pts_list[i], pts_list[(i + 1) % len(i_mc)], cardiacnumofimages)
        if (perform_forward_and_reverse):
            contour, index1, index2, ind_shift_mean, ind_direction = get_pointcorrespondence_3d_forward_and_reverse(
                DM_i, i_mc[i], i_mc[(i + 1) % len(i_mc)], pts_list[i], pts_list[(i + 1) % len(i_mc)],
                cardiacnumofimages)
        else:
            contour, index1, index2, ind_shift_mean, ind_direction = get_pointcorrespondence_3d_forward(DM_i, i_mc[i],
                                                                                                        i_mc[(
                                                                                                                         i + 1) % len(
                                                                                                            i_mc)],
                                                                                                        pts_list[i],
                                                                                                        pts_list[(
                                                                                                                             i + 1) % len(
                                                                                                            i_mc)],
                                                                                                        cardiacnumofimages)

        ind_shift_all[index2] = (ind_shift_mean, ind_direction)

        print('contour: ' + str(len(contour)))
        print('index1: ' + str(index1))
        print('index2: ' + str(index2))

        for k in range(index1 + 1, index2 + 1):
            # foutname = os.path.join(foutdir, "%04d.bin" % (self.cardiacnumofimages * i_ser + k % self.cardiacnumofimages) )
            # print ('k: ' + str(k) + ' ' + str(k - index1 - 1) + ' ' + str(k % cardiacnumofimages))
            # endo_contour_list[k] = contour[k - index1 - 1]
            # grid_contour_list[k] = mesh[k % cardiacnumofimages]
            ### before ###
            # endo_contour_list[k-1] = contour[k - index1 - 1]
            # grid_contour_list[k-1] = mesh[k % cardiacnumofimages]
            ##############
            # grid_contour_list[k-1] = mesh[(k-1) % cardiacnumofimages]

            ### new --> This makes ED and ES match up finally!!
            endo_contour_list[k % cardiacnumofimages] = contour[k - index1 - 1]
            grid_contour_list[k % cardiacnumofimages] = mesh[k % cardiacnumofimages]

    return endo_contour_list, grid_contour_list, mesh


# # Similar to code from Kumar's pycardiac analysis code
# def compute_auto_contours(dispm_file_3D_filename, num_frames, ED_frame, ED_contour, ES_frame, ES_contour):
#     """Compute or recompute auto contours based on the displacement matrix."""
#
#     NGRIDPTS = 30
#     # cardiacnumofimages = num_frames - 1
#     cardiacnumofimages = num_frames
#
#     DM_i = np.load(file(dispm_file_3D_filename, 'rb'))
#     disp_x = DM_i["DM_fx"]
#     print ('disp_x: ' + str(disp_x.shape))
#
#     # meshpts = np.meshgrid(np.linspace(DM_i["bb"][0], DM_i["bb"][1], NGRIDPTS), np.linspace(DM_i["bb"][2], DM_i["bb"][3], NGRIDPTS))
#     x = DM_i["DM_fx"].shape[0]
#     y = DM_i["DM_fx"].shape[1]
#     z = DM_i["DM_fx"].shape[2]
#     meshpts = np.meshgrid(np.linspace(0,x,NGRIDPTS), np.linspace(0,y,NGRIDPTS), np.linspace(0,z,NGRIDPTS))
#     mesh = get_pointcorrespondence_mesh_3d_forward_and_reverse(DM_i=DM_i, pts=np.vstack([meshpts[0].flatten(), meshpts[1].flatten(), meshpts[2].flatten()]).T, nframes=cardiacnumofimages)
#
#     i_mc = []
#     pts_list = []
#
#     i_mc.append(ED_frame)
#     pts_list.append(ED_contour)
#
#     i_mc.append(ES_frame)
#     pts_list.append(ES_contour)
#
#     endo_contour_list = [None] * num_frames
#     grid_contour_list = [None] * num_frames
#
#     ind_shift_all = {}
#     for i in range(len(i_mc)):
#         contour, index1, index2, ind_shift_mean, ind_direction = get_pointcorrespondence_3d(DM_i, i_mc[i], i_mc[(i + 1) % len(i_mc)], pts_list[i], pts_list[(i + 1) % len(i_mc)], cardiacnumofimages)
#         ind_shift_all[index2] = (ind_shift_mean, ind_direction)
#
#         print ('contour: ' + str(len(contour)))
#         print ('index1: ' + str(index1))
#         print ('index2: ' + str(index2))
#
#         for k in range(index1 + 1, index2 + 1):
#             # foutname = os.path.join(foutdir, "%04d.bin" % (self.cardiacnumofimages * i_ser + k % self.cardiacnumofimages) )
#             # print ('k: ' + str(k) + ' ' + str(k - index1 - 1) + ' ' + str(k % cardiacnumofimages))
#             # endo_contour_list[k] = contour[k - index1 - 1]
#             # grid_contour_list[k] = mesh[k % cardiacnumofimages]
#             endo_contour_list[k-1] = contour[k - index1 - 1]
#             grid_contour_list[k-1] = mesh[k % cardiacnumofimages]
#
#
#     return endo_contour_list, grid_contour_list


@jit
def griddata_1d_3d(nx, ny, nz, V):
    m, n, s = V.shape
    res = np.zeros((nx.shape[0]))

    for i in range(nx.shape[0]):
        Rx = nx[i] - 1.0
        Ry = ny[i] - 1.0
        Rz = nz[i] - 1.0
        # Rx = nx[i]
        # Ry = ny[i]
        # Rz = nz[i]
        if (Rx >= (m - 1.0)):
            Rx = m - 1.0
            cRx = int(Rx)
            fRx = cRx - 1
        else:
            if (Rx < 0):
                Rx = 0

            fRx = int(Rx)
            cRx = fRx + 1

        if (Ry >= (n - 1.0)):
            Ry = n - 1.0
            cRy = int(Ry)
            fRy = cRy - 1
        else:
            if (Ry < 0):
                Ry = 0

            fRy = int(Ry)
            cRy = fRy + 1

        if (Rz >= (s - 1.0)):
            Rz = s - 1.0
            cRz = int(Rz)
            fRz = cRz - 1
        else:
            if (Rz < 0):
                Rz = 0

            fRz = int(Rz)
            cRz = fRz + 1

        res[i] = V[fRx, fRy, fRz] * (cRx - Rx) * (cRy - Ry) * (cRz - Rz) + \
                 V[fRx, fRy, cRz] * (cRx - Rx) * (cRy - Ry) * (Rz - fRz) + \
                 V[fRx, cRy, fRz] * (cRx - Rx) * (Ry - fRy) * (cRz - Rz) + \
                 V[fRx, cRy, cRz] * (cRx - Rx) * (Ry - fRy) * (Rz - fRz) + \
                 V[cRx, fRy, fRz] * (Rx - fRx) * (cRy - Ry) * (cRz - Rz) + \
                 V[cRx, fRy, cRz] * (Rx - fRx) * (cRy - Ry) * (Rz - fRz) + \
                 V[cRx, cRy, fRz] * (Rx - fRx) * (Ry - fRy) * (cRz - Rz) + \
                 V[cRx, cRy, cRz] * (Rx - fRx) * (Ry - fRy) * (Rz - fRz)

    return res

def get_pointcorrespondence_mesh_3d_forward_and_reverse(DM_i, pts, nframes):
    """Get point correspondence using the displacement matrix for all images in a series."""
    DM_fx, DM_fy, DM_fz, DM_bx, DM_by, DM_bz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"], DM_i["DM_bx"], DM_i[
        "DM_by"], DM_i["DM_bz"]

    iuf, ivf, iwf = pts[:, 0], pts[:, 1], pts[:, 2]
    iub, ivb, iwb = iuf, ivf, iwf
    contour_f, contour_b, contour = [], [], {}

    for k in range(nframes):
        if k > 0:
            ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fx[k - 1]).astype(np.float32))
            iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fy[k - 1]).astype(np.float32))
            iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fz[k - 1]).astype(np.float32))
            iuf, ivf, iwf = iuf + ix, ivf + iy, iwf + iz

            ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_bx[k - 1]).astype(np.float32))
            iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_by[k - 1]).astype(np.float32))
            iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_bz[k - 1]).astype(np.float32))
            # ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_bx[nframes-k]).astype(np.float32))
            # iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_by[nframes-k]).astype(np.float32))
            # iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_bz[nframes-k]).astype(np.float32))
            iub, ivb, iwb = iub + ix, ivb + iy, iwb + iz

            contour_b.insert(0, np.vstack([iub, ivb, iwb]))

        contour_f.append(np.vstack([iuf, ivf, iwf]))

    contour_b.insert(0, np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]]))

    for k in range(nframes):
        w = 1.0 * (k + 1) / (1.0 * nframes)
        contour[k] = ((1.0 - w) * contour_f[k]) + (w * contour_b[k])

    return contour


def get_pointcorrespondence_mesh_3d_forward(DM_i, pts, nframes):
    """Get point correspondence using the displacement matrix for all images in a series."""
    DM_fx, DM_fy, DM_fz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"]

    iuf, ivf, iwf = pts[:, 0], pts[:, 1], pts[:, 2]
    contour_f, contour = [], {}

    for k in range(nframes):
        if k > 0:
            ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fx[k - 1]).astype(np.float32))
            iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fy[k - 1]).astype(np.float32))
            iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                                (DM_fz[k - 1]).astype(np.float32))
            iuf, ivf, iwf = iuf + ix, ivf + iy, iwf + iz

        contour_f.append(np.vstack([iuf, ivf, iwf]))

    contour = contour_f

    return contour


def get_pointcorrespondence_3d_forward(DM_i, index1, index2, pts1, pts2, nframes):
    closed = 1

    """Get point correspondence using a displacement matrix for images lies between index 1 and 2."""
    """Does not use find_contour_index_association"""

    ind_shift, ind_direction = None, None
    DM_fx, DM_fy, DM_fz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"]

    index2 += nframes if index2 <= index1 else 0  # add nframes if index2 is smaller than index1

    iuf, ivf, iwf = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    contour_f, contour = [], {}

    for k in np.arange(index1, index2 - 1):
        ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fx[k % nframes]).astype(np.float32))
        iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fy[k % nframes]).astype(np.float32))
        iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fz[k % nframes]).astype(np.float32))
        iuf, ivf, iwf = iuf + ix, ivf + iy, iwf + iz

        contour_f.append(np.vstack([iuf, ivf, iwf]))

    for k in range(index1, index2 - 1):
        # w = 1.0 * (k - index1 + 1) / (1.0 * (index2 - index1))
        # contour[k - index1], ind_shift, ind_direction = find_contour_index_association(closed, contour_f[k - index1], contour_b[k - index1], w)
        # contour[k - index1] = (1 - w) * contour_f[k - index1] + w * contour_b[k - index1]
        contour[k - index1] = contour_f[k - index1]

    contour[index2 - index1 - 1] = pts2

    return contour, index1, index2, ind_shift, ind_direction


# def get_pointcorrespondence(closed, DM_i, index1, index2, pts1, pts2, nframes):
# def get_pointcorrespondence_3d(DM_i, index1, index2, pts1, pts2, nframes):
def get_pointcorrespondence_3d_forward_and_reverse(DM_i, index1, index2, pts1, pts2, nframes):
    closed = 1

    """Get point correspondence using a displacement matrix for images lies between index 1 and 2."""
    """Does not use find_contour_index_association"""

    ind_shift, ind_direction = None, None
    DM_fx, DM_fy, DM_fz, DM_bx, DM_by, DM_bz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"], DM_i["DM_bx"], DM_i[
        "DM_by"], DM_i["DM_bz"]

    index2 += nframes if index2 <= index1 else 0  # add nframes if index2 is smaller than index1

    iuf, ivf, iwf = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    iub, ivb, iwb = pts2[:, 0], pts2[:, 1], pts2[:, 2]
    contour_f, contour_b, contour = [], [], {}

    for k in np.arange(index1, index2 - 1):
        ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fx[k % nframes]).astype(np.float32))
        iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fy[k % nframes]).astype(np.float32))
        iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32),
                            (DM_fz[k % nframes]).astype(np.float32))
        iuf, ivf, iwf = iuf + ix, ivf + iy, iwf + iz

        contour_f.append(np.vstack([iuf, ivf, iwf]))

    for k in np.arange(index2, index1 + 1, -1):
        ix = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32),
                            (DM_bx[(nframes - k) % nframes]).astype(np.float32))
        iy = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32),
                            (DM_by[(nframes - k) % nframes]).astype(np.float32))
        iz = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32),
                            (DM_bz[(nframes - k) % nframes]).astype(np.float32))
        iub, ivb, iwb = iub + ix, ivb + iy, iwb + iz

        contour_b.insert(0, np.vstack([iub, ivb, iwb]))

    for k in range(index1, index2 - 1):
        w = 1.0 * (k - index1 + 1) / (1.0 * (index2 - index1))
        # contour[k - index1], ind_shift, ind_direction = find_contour_index_association(closed, contour_f[k - index1], contour_b[k - index1], w)
        contour[k - index1] = ((1.0 - w) * contour_f[k - index1]) + (w * contour_b[k - index1])

    contour[index2 - index1 - 1] = pts2

    return contour, index1, index2, ind_shift, ind_direction


# # def get_pointcorrespondence(closed, DM_i, index1, index2, pts1, pts2, nframes):
# def get_pointcorrespondence_3d(DM_i, index1, index2, pts1, pts2, nframes):
#
#     closed = 1
#
#     """Get point correspondence using a displacement matrix for images lies between index 1 and 2."""
#
#     ind_shift, ind_direction = None, None
#     DM_fx, DM_fy, DM_fz, DM_bx, DM_by, DM_bz = DM_i["DM_fx"], DM_i["DM_fy"], DM_i["DM_fz"], DM_i["DM_bx"], DM_i["DM_by"], DM_i["DM_bz"]
#
#     index2 += nframes if index2 <= index1 else 0  # add nframes if index2 is smaller than index1
#
#     iuf, ivf, iwf = pts1[:, 0], pts1[:, 1], pts1[:, 2]
#     iub, ivb, iwb = pts2[:, 0], pts2[:, 1], pts2[:, 2]
#     contour_f, contour_b, contour = [], [], {}
#
#     for k in np.arange(index1, index2 - 1):
#         ix = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_fx[k % nframes]).astype(np.float32))
#         iy = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_fy[k % nframes]).astype(np.float32))
#         iz = griddata_1d_3d((iuf).astype(np.float32), (ivf).astype(np.float32), (iwf).astype(np.float32), (DM_fz[k % nframes]).astype(np.float32))
#         iuf, ivf, iwf = iuf + ix, ivf + iy, iwf + iz
#
#         contour_f.append(np.vstack([iuf, ivf, iwf]))
#
#     for k in np.arange(index2, index1 + 1, -1):
#         ix = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32), (DM_bx[(nframes - k) % nframes]).astype(np.float32))
#         iy = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32), (DM_by[(nframes - k) % nframes]).astype(np.float32))
#         iz = griddata_1d_3d((iub).astype(np.float32), (ivb).astype(np.float32), (iwb).astype(np.float32), (DM_bz[(nframes - k) % nframes]).astype(np.float32))
#         iub, ivb, iwb = iub + ix, ivb + iy, iwb + iz
#
#         contour_b.insert(0, np.vstack([iub, ivb, iwb]))
#
#     for k in range(index1, index2 - 1):
#         w = 1.0 * (k - index1 + 1) / (1.0 * (index2 - index1))
#         contour[k - index1], ind_shift, ind_direction = find_contour_index_association(closed, contour_f[k - index1], contour_b[k - index1], w)
#
#     contour[index2 - index1 - 1] = pts2
#
#     return contour, index1, index2, ind_shift, ind_direction

def find_contour_index_association(closed, ctr1, ctr2, w):
    """Find point correspondence between two contours"""
    ctr2_reverse = np.fliplr(ctr2)

    # if self.closedOnAction.isChecked():
    if (closed):
        dist_vector = np.empty((ctr2.shape[1]))
        for i in range(ctr2.shape[1]):
            dist_vector[i] = np.sum(np.linalg.norm(ctr1 - np.roll(ctr2, shift=i, axis=1), axis=0))

        dist_vector_reverse = np.empty((ctr2.shape[1]))
        for i in range(ctr2.shape[1]):
            dist_vector_reverse[i] = np.sum(np.linalg.norm(ctr1 - np.roll(ctr2_reverse, shift=i, axis=1), axis=0))

    else:
        dist_vector = np.sum(np.linalg.norm(ctr1 - np.roll(ctr2, shift=0, axis=1), axis=0))
        dist_vector_reverse = np.sum(np.linalg.norm(ctr1 - np.roll(ctr2_reverse, shift=0, axis=1), axis=0))

    if np.min(dist_vector) < np.min(dist_vector_reverse):
        return (1.0 - w) * ctr1 + w * np.roll(ctr2, shift=np.argmin(dist_vector), axis=1), np.argmin(dist_vector), 1
    else:
        return (1.0 - w) * ctr1 + w * np.roll(ctr2_reverse, shift=np.argmin(dist_vector_reverse), axis=1), np.argmin(
            dist_vector), -1

########################################################################################################################

@jit
def mygriddata3(nx, ny, nz, V):
    m, n, s = nx.shape
    res = np.zeros((m, n, s))

    for i in range(m):

        for j in range(n):

            for k in range(s):

                Rx = nx[i, j, k] - 1.0
                Ry = ny[i, j, k] - 1.0
                Rz = nz[i, j, k] - 1.0
                # Rx = nx[i, j, k]
                # Ry = ny[i, j, k]
                # Rz = nz[i, j, k]

                if (Rx >= (m - 1.0)):
                    Rx = m - 1.0
                    cRx = int(Rx)
                    fRx = cRx - 1
                else:
                    if (Rx < 0):
                        Rx = 0

                    fRx = int(Rx)
                    cRx = fRx + 1

                if (Ry >= (n - 1.0)):
                    Ry = n - 1.0
                    cRy = int(Ry)
                    fRy = cRy - 1
                else:
                    if (Ry < 0):
                        Ry = 0

                    fRy = int(Ry)
                    cRy = fRy + 1

                if (Rz >= (s - 1.0)):
                    Rz = s - 1.0
                    cRz = int(Rz)
                    fRz = cRz - 1
                else:
                    if (Rz < 0):
                        Rz = 0

                    fRz = int(Rz)
                    cRz = fRz + 1

                res[i, j, k] = V[fRx, fRy, fRz] * (cRx - Rx) * (cRy - Ry) * (cRz - Rz) + \
                               V[fRx, fRy, cRz] * (cRx - Rx) * (cRy - Ry) * (Rz - fRz) + \
                               V[fRx, cRy, fRz] * (cRx - Rx) * (Ry - fRy) * (cRz - Rz) + \
                               V[fRx, cRy, cRz] * (cRx - Rx) * (Ry - fRy) * (Rz - fRz) + \
                               V[cRx, fRy, fRz] * (Rx - fRx) * (cRy - Ry) * (cRz - Rz) + \
                               V[cRx, fRy, cRz] * (Rx - fRx) * (cRy - Ry) * (Rz - fRz) + \
                               V[cRx, cRy, fRz] * (Rx - fRx) * (Ry - fRy) * (cRz - Rz) + \
                               V[cRx, cRy, cRz] * (Rx - fRx) * (Ry - fRy) * (Rz - fRz)

    return res



@jit
def m_norm(x):
    return x * np.size(x) / np.sum(x)


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

