# reg_functions_validation.py 
# 
# Holds functions for validation, calculating metrics, forming meshes, etc
# 
# 
# Deepa Krishnaswamy 
# University of Alberta
# June 2020
# 
# 06-15-20 - added function perform_validation_SSD - difference between 
#               gt and warped images 
#            added function calculate_min_jacobian_det - calculates min 
#               Jacobian det per frame of cycle 
#            added to perform_validation_function_average the SSD and jacobian
# 12-11-20 - calculate_min_jacobian_det - changed np.expand_dims from 5 to 4
# 06-21-20 - added the curl and divergence along with the jacobian det.
###############################################################################

import os 
if ('CONDA_DEFAULT_ENV' in os.environ.keys()): 
    print (os.environ['CONDA_DEFAULT_ENV'])
    if (os.environ['CONDA_DEFAULT_ENV']=="dipy2"):
        import dipy 
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import SSDMetric, CCMetric
    elif (os.environ['CONDA_DEFAULT_ENV'] == "reg_3d_paper"):
        import dipy
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import SSDMetric, CCMetric

        
from numba import jit
import numba

# import pycardiac_3D_functions
# import pycardiac_3D_functions_final as pycardiac_3D_functions
# import pycardiac_3D_functions_modified_griddata as pycardiac_3D_functions
import pycardiac_3D_functions

import vtk, numpy as np, sys, os, glob, vtk.util.numpy_support as vn
import time, matplotlib.animation as animation, matplotlib.pyplot as plt
from vtk.util.colors import brown_ochre, tomato, banana, azure, black
import scipy 
from numpy import pi
import nibabel as nib 

from skimage import draw
from skimage import measure

from PIL import Image
import math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D

import re 
import json 
import cv2

from vtk.util.numpy_support import vtk_to_numpy
import SimpleITK as sitk 
import scipy 
import scipy.io as sio 


from scipy.stats.stats import pearsonr


#########################################################################
# https://simpleitk.readthedocs.io/en/master/link_DemonsRegistration1_docs.html    
def command_iteration(filter) :
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

def perform_sitk_demons_registration(frame1, frame2):
        
    fixed = sitk.GetImageFromArray(frame1)
    moving = sitk.GetImageFromArray(frame2)  
    
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving,fixed)
    
    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in SimpleITK
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations( 50 )
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations( 1.0 )
    
    demons.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(demons) )
    
    displacementField = demons.Execute( fixed, moving )
    
    # print("-------")
    # print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
    # print(" RMS: {0}".format(demons.GetRMSChange()))
        
    outTx = sitk.DisplacementFieldTransform( displacementField )
    disp_field = outTx.GetDisplacementField()
    disp_field_array = sitk.GetArrayFromImage(disp_field)
    
    disp_x_sitk = disp_field_array[:,:,:,0]
    disp_y_sitk = disp_field_array[:,:,:,1]
    disp_z_sitk = disp_field_array[:,:,:,2]
    
    return disp_x_sitk, disp_y_sitk, disp_z_sitk
    

# https://simpleitk.readthedocs.io/en/master/link_DemonsRegistration2_docs.html#lbl-demons-registration2

def perform_sitk_fsf_demons_registration(frame1, frame2):

    fixed = sitk.GetImageFromArray(frame1)
    moving = sitk.GetImageFromArray(frame2)  
    
    matcher = sitk.HistogramMatchingImageFilter()
    if (fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8)):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)
    
    # The fast symmetric forces Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(200)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)
    
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
    
    displacementField = demons.Execute(fixed, moving)
    
    # print("-------")
    # print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
    # print(" RMS: {0}".format(demons.GetRMSChange()))
    
    outTx = sitk.DisplacementFieldTransform(displacementField)
    disp_field = outTx.GetDisplacementField()
    disp_field_array = sitk.GetArrayFromImage(disp_field)
    
    disp_x_sitk = disp_field_array[:,:,:,0]
    disp_y_sitk = disp_field_array[:,:,:,1]
    disp_z_sitk = disp_field_array[:,:,:,2]
    
    return disp_x_sitk, disp_y_sitk, disp_z_sitk



# def ndreg_LDDMM(frame1, frame2):
 
# https://dipy.org/documentation/1.1.1./examples_built/syn_registration_3d/
# https://dipy.org/documentation/1.1.1./reference/dipy.align/#symmetricdiffeomorphicregistration   
def perform_dipy_registration(frame1, frame2):
    
    # metric = CCMetric(3)
    metric = SSDMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    # (metric, level_iters=None, step_length=0.25, ss_sigma_factor=0.2, opt_tol=1e-05, inv_iter=20, inv_tol=0.001, callback=None)
    mapping = sdr.optimize(frame2, frame1)
    
    # disp = mapping.get_backward_field()
    disp = mapping.get_forward_field()
    
    dispx = disp[:,:,:,0]
    dispy = disp[:,:,:,1]
    dispz = disp[:,:,:,2]
    
    return dispx, dispy, dispz 


# def perform_dipy(S,T):
#     
#     ### Original ###
#     metric = SSDMetric(2) 
#     level_iters = [200, 100, 50, 25]
#     sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
#     
# #     sigma_diff = 3.0
# #     radius = 4
# #     metric = CCMetric(2, sigma_diff, radius)
# #     level_iters = [200, 100, 50, 25]
# #     sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
#     
#     # mapping = sdr.optimize(static, moving)
#     mapping = sdr.optimize(T, S)
#     
#     # disp = mapping.get_forward_field() 
#     disp = mapping.get_backward_field()
# 
#     dispx = disp[:,:,0]
#     dispy = disp[:,:,1]
# 
#     return dispx, dispy
#     

# copied from common_functions.py from Maz_LV_4D_seg 
# https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up 
def calculate_LV_volume(vertices, faces):
    """Calculates the volume in mL of the LV mesh segmentation by taking the volume of each face connected to the origin""" 

    num_tri = faces.shape[0] 
    
    orig = np.mean(vertices,axis=0)
    vol_dt = np.zeros((num_tri,1))
    for n in range(0,num_tri):
        pt = np.int64(faces[n,:])
        v1 = vertices[pt[0],:]-orig
        v2 = vertices[pt[1],:]-orig
        v3 = vertices[pt[2],:]-orig
        d = np.zeros((3,3))
        d[0,0] = v1[0]; d[1,0] = v1[1]; d[2,0] = v1[2];
        d[0,1] = v2[0]; d[1,1] = v2[1]; d[2,1] = v2[2]; 
        d[0,2] = v3[0]; d[1,2] = v3[1]; d[2,2] = v3[2]; 
        vol_dt[n] = (1.0/6.0) * np.abs(np.linalg.det(d))
    
    # returns result in mL
    vol_dt_sum = np.sum(vol_dt) / 1000.0
    
    return vol_dt_sum

def perform_validation_function_volume(mesh_gt_filelist, mesh_filelist, ED_frame=-1,ES_frame=-1):
    
    num_files = len(mesh_gt_filelist)
    volume_gt_for_graph = [] 
    volume_for_graph = [] 
    volume_gt_for_correlation = [] 
    volume_for_correlation = [] 
        
    for n in range(0,num_files):
        
        
        # read the mesh_gt_filelist
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_gt_filelist[n])
        reader.Update()
        mesh_gt = reader.GetOutput()
        # get points
        mesh_gt_points = get_points_from_mesh(mesh_gt) 
        # get faces
        mesh_gt_faces = get_faces_from_mesh(mesh_gt)
        
        # read the mesh_filelist 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_filelist[n])
        reader.Update()
        mesh = reader.GetOutput()
        # get points
        mesh_points = get_points_from_mesh(mesh)
        # get faces 
        mesh_faces = get_faces_from_mesh(mesh)
        
        # Calculate volume for gt  
        volume_gt = calculate_LV_volume(mesh_gt_points, mesh_gt_faces)
        volume_gt_for_graph.append(volume_gt)
        volume_gt_for_correlation.append(volume_gt)
        
        # Calculate volume 
        volume = calculate_LV_volume(mesh_points, mesh_faces)
        volume_for_graph.append(volume)
        volume_for_correlation.append(volume)
        
    volume_gt_for_graph = np.asarray(volume_gt_for_graph)
    volume_for_graph = np.asarray(volume_for_graph)
    
    # volume_gt_for_correlation = volume_gt_for_graph 
    # volume_for_correlation = volume_for_graph 
    
    ### Volume for correlation - without ED and ES frame ### 
    if (ED_frame!=-1):
        volume_gt_for_correlation[ED_frame] = 0.0
        volume_for_correlation[ED_frame] = 0.0
    if (ES_frame!=-1):
        volume_gt_for_correlation[ES_frame] = 0.0
        volume_for_correlation[ES_frame] = 0.0
        
    ### Now find the -1's and remove ###
        
    ind = np.squeeze(np.where(np.asarray(volume_gt_for_correlation) <= 0.5))
    print ('ind: ' + str(ind))
    volume_gt_for_correlation = np.delete(volume_gt_for_correlation, ind)
    volume_for_correlation = np.delete(volume_for_correlation, ind)
    
    # calculate correlation value 
    volume_corr = pearsonr(volume_gt_for_correlation, volume_for_correlation)
        
    return volume_gt_for_graph, volume_for_graph, volume_gt_for_correlation, volume_for_correlation, volume_corr 
    

def perform_validation_function_average(output_validation_directory, output_validation_average_filename):
    
    # read all files from output validation directory 
    files = [os.path.join(output_validation_directory,s) for s in os.listdir(output_validation_directory)] 
    num_files = len(files)
    MAD_all = []
    Dice_all = []
    HD_all = []  
    SSD_all = [] 
    SSD_forward_all = [] 
    SSD_reverse_all = [] 
    min_jacobian_det_forward_all = [] 
    max_jacobian_det_forward_all = []
    min_jacobian_det_reverse_all = []
    max_jacobian_det_reverse_all = []
    min_curl_forward_all = []
    max_curl_forward_all = []
    min_curl_reverse_all = []
    max_curl_reverse_all = []
    min_div_forward_all = []
    max_div_forward_all = []
    min_div_reverse_all = []
    max_div_reverse_all = []
    elapsed_time_all = [] 
    
    for n in range(0,num_files):
        # distances = np.load(files[n])
        distances = np.load(files[n], allow_pickle=True)
        MAD_all.append(distances['MAD'])
        Dice_all.append(distances['Dice'])
        HD_all.append(distances['HD'])
        SSD_all.append(distances['SSD'])
        SSD_forward_all.append(distances['SSD_forward'])
        SSD_reverse_all.append(distances['SSD_reverse'])
        min_jacobian_det_forward_all.append(distances['min_jacobian_det_forward'])
        max_jacobian_det_forward_all.append(distances['max_jacobian_det_forward'])
        min_jacobian_det_reverse_all.append(distances['min_jacobian_det_reverse'])
        max_jacobian_det_reverse_all.append(distances['max_jacobian_det_reverse'])
        min_curl_forward_all.append(distances['min_curl_forward'])
        max_curl_forward_all.append(distances['max_curl_forward'])
        min_curl_reverse_all.append(distances['min_curl_reverse'])
        max_curl_reverse_all.append(distances['max_curl_reverse'])
        min_div_forward_all.append(distances['min_div_forward'])
        max_div_forward_all.append(distances['max_div_forward'])
        min_div_reverse_all.append(distances['min_div_reverse'])
        max_div_reverse_all.append(distances['max_div_reverse'])

        elapsed_time_all.append(distances['elapsed_time'])
    
    MAD_all = np.asarray(np.concatenate(MAD_all)) 
    Dice_all = np.asarray(np.concatenate(Dice_all))
    HD_all = np.asarray(np.concatenate(HD_all))
    SSD_all = np.asarray(np.concatenate(SSD_all))
    SSD_forward_all = np.asarray(np.concatenate(SSD_forward_all))
    SSD_reverse_all = np.asarray(np.concatenate(SSD_reverse_all))
    min_jacobian_det_forward_all = np.asarray(np.concatenate(min_jacobian_det_forward_all))
    max_jacobian_det_forward_all = np.asarray(np.concatenate(max_jacobian_det_forward_all))
    min_jacobian_det_reverse_all = np.asarray(np.concatenate(min_jacobian_det_reverse_all))
    max_jacobian_det_reverse_all = np.asarray(np.concatenate(max_jacobian_det_reverse_all))
    min_curl_forward_all = np.asarray(np.concatenate(min_curl_forward_all))
    max_curl_forward_all = np.asarray(np.concatenate(max_curl_forward_all))
    min_curl_reverse_all = np.asarray(np.concatenate(min_curl_reverse_all))
    max_curl_reverse_all = np.asarray(np.concatenate(max_curl_reverse_all))
    min_div_forward_all = np.asarray(np.concatenate(min_div_forward_all))
    max_div_forward_all = np.asarray(np.concatenate(max_div_forward_all))
    min_div_reverse_all = np.asarray(np.concatenate(min_div_reverse_all))
    max_div_reverse_all = np.asarray(np.concatenate(max_div_reverse_all))

    elapsed_time_all = np.asarray(np.concatenate(elapsed_time_all))
    
    # Get mean and standard deviation 
    MAD_mean = np.mean(MAD_all)
    MAD_std = np.std(MAD_all)
    Dice_mean = np.mean(Dice_all)
    Dice_std = np.std(Dice_all)
    HD_mean = np.mean(HD_all)
    HD_std = np.std(HD_all)
    SSD_mean = np.mean(SSD_all)
    SSD_std = np.std(SSD_all)
    SSD_forward_mean = np.mean(SSD_forward_all)
    SSD_forward_std = np.std(SSD_forward_all)
    SSD_reverse_mean = np.mean(SSD_reverse_all)
    SSD_reverse_std = np.std(SSD_reverse_all)
    # min_jacobian_det_mean = np.mean(min_jacobian_det_all) 
    # min_jacobian_det_std = np.std(min_jacobian_det_all)
    elapsed_time_mean = np.mean(elapsed_time_all)
    elapsed_time_std = np.std(elapsed_time_all)
    
    # only do something different for min and max jacobian det, and curl and div
    min_jacobian_det_forward_overall = np.min(min_jacobian_det_forward_all)
    max_jacobian_det_forward_overall = np.max(max_jacobian_det_forward_all)
    min_jacobian_det_reverse_overall = np.min(min_jacobian_det_reverse_all)
    max_jacobian_det_reverse_overall = np.max(max_jacobian_det_reverse_all)
    min_jacobian_det_overall = np.min([min_jacobian_det_forward_overall, min_jacobian_det_reverse_overall])
    max_jacobian_det_overall = np.max([max_jacobian_det_forward_overall, max_jacobian_det_reverse_overall])

    min_curl_forward_overall = np.min(min_curl_forward_all)
    max_curl_forward_overall = np.max(max_curl_forward_all)
    min_curl_reverse_overall = np.min(min_curl_reverse_all)
    max_curl_reverse_overall = np.max(max_curl_reverse_all)
    min_curl_overall = np.min([min_curl_forward_overall, min_curl_reverse_overall])
    max_curl_overall = np.max([max_curl_forward_overall, max_curl_reverse_overall])

    min_div_forward_overall = np.min(min_div_forward_all)
    max_div_forward_overall = np.max(max_div_forward_all)
    min_div_reverse_overall = np.min(min_div_reverse_all)
    max_div_reverse_overall = np.max(max_div_reverse_all)
    min_div_overall = np.min([min_div_forward_overall, min_div_reverse_overall])
    max_div_overall = np.max([max_div_forward_overall, max_div_reverse_overall])
    
    # Write out to text file
    fid = open(output_validation_average_filename, 'w')
    fid.write("MAD mean: %s\n" % str(MAD_mean))
    fid.write("MAD std: %s\n" % str(MAD_std))
    fid.write("Dice mean: %s\n" % str(Dice_mean))
    fid.write("Dice std: %s\n" % str(Dice_std))
    fid.write("HD mean: %s\n" % str(HD_mean))
    fid.write("HD std: %s\n" % str(HD_std))
    fid.write("SSD mean: %s\n" % str(SSD_mean))
    fid.write("SSD std: %s\n" % str(SSD_std))
    fid.write("SSD forward mean: %s\n" % str(SSD_forward_mean))
    fid.write("SSD forward std: %s\n" % str(SSD_forward_std))
    fid.write("SSD reverse mean: %s\n" % str(SSD_reverse_mean))
    fid.write("SSD reverse std: %s\n" % str(SSD_reverse_std))
    # fid.write("min Jacobian det mean: %s\n" % str(min_jacobian_det_mean))
    # fid.write("min Jacobian det std: %s\n" % str(min_jacobian_det_std))
    fid.write("min Jacobian det: %s\n" % str(min_jacobian_det_overall))
    fid.write("max Jacobian det: %s\n" % str(max_jacobian_det_overall))
    fid.write("min curl: %s\n" % str(min_curl_overall))
    fid.write("max curl: %s\n" % str(max_curl_overall))
    fid.write("min div: %s\n" % str(min_div_overall))
    fid.write("max div: %s\n" % str(max_div_overall))
    
    fid.write("elapsed time mean: %s\n" % str(elapsed_time_mean))
    fid.write("elapsed time std: %s\n" % str(elapsed_time_std))
    fid.close()
    
    
    
def perform_timing_information_function_average(output_validation_directory, output_timing_average_filename):
    
    # read all files from output validation directory 
    files = [os.path.join(output_validation_directory,s) for s in os.listdir(output_validation_directory)] 
    num_files = len(files)
    
    timing_forward_all = []
    timing_reverse_all = []  
    
    for n in range(0,num_files):
        
        _, ext = os.path.splitext(files[n])
        if (ext=='.mat'):
            timing_information = sio.loadmat(files[n])
        else: 
            timing_information = np.load(files[n], allow_pickle=True)
            
                
        timing_forward_all.append(timing_information['elapsed_time_forward'])
        timing_reverse_all.append(timing_information['elapsed_time_reverse']) 
        
    
    timing_forward_all = np.asarray(np.concatenate(timing_forward_all))
    timing_reverse_all = np.asarray(np.concatenate(timing_reverse_all))
    timing_all = np.asarray(np.concatenate(timing_forward_all, timing_reverse_all))
    
    # Get mean and standard deviation
    timing_mean = np.mean(timing_all)
    timing_std = np.std(timing_all)
    
    # Write out to text file
    fid = open(output_timing_average_filename, 'w')
    fid.write("timing mean: %s\n" % str(timing_mean))
    fid.write("timing std: %s\n" % str(timing_std))
    fid.close()
    
        
            
        


def perform_validation_function(mesh_gt_filelist, mesh_filelist, volume_gt_filelist, volume_filelist, ED_frame=-1, ES_frame=-1):
    '''Performs validation for each patient given a list of files'''
    
    num_files = len(mesh_gt_filelist)
    MAD_all = []
    Dice_all = []
    HD_all = [] 
        
    for n in range(0,num_files):
        
        if (num_files>1) and ((ED_frame!=-1 and n==ED_frame) or (ES_frame!=-1 and n==ES_frame)): 
            continue
                
        # read the mesh_gt_filelist
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_gt_filelist[n])
        reader.Update()
        mesh_gt = reader.GetOutput()
        # read the mesh_filelist 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_filelist[n])
        reader.Update()
        mesh = reader.GetOutput()
        # read the volume_gt_filelist
        nii = nib.load(volume_gt_filelist[n])
        volume_gt = nii.get_data()
        # read the volume_filelist
        nii = nib.load(volume_filelist[n])
        volume = nii.get_data()
        
        ### Calculate MAD ### 
        _,_,MAD = calculate_MAD(mesh_gt, mesh)
        ### Calculate Dice ### 
        Dice = calculate_Dice_score(volume_gt, volume)
        ### Calculate HD ###
        mesh_gt_points = get_points_from_mesh(mesh_gt)
        mesh_points = get_points_from_mesh(mesh)
        _,_,_,_,HD = calculate_HD(mesh_gt_points, mesh_points)        
        
        ### Save to list ###
        MAD_all.append(MAD)
        Dice_all.append(Dice)
        HD_all.append(HD)
        
        
    MAD_all = np.asarray(MAD_all)
    Dice_all = np.asarray(Dice_all)
    HD_all = np.asarray(HD_all)
    
    
    
    return MAD_all, Dice_all, HD_all
     
     
def perform_validation_function_SSD(data_cropped, data_cropped_warped_forward, data_cropped_warped_reverse, ED_frame=-1, ES_frame=-1):
    
    num_files = data_cropped.shape[3] # number of frames  
    SSD_all_forward = []
    SSD_all_reverse = [] 
        
    for n in range(0,num_files):
        
        if (num_files>1) and ((ED_frame!=-1 and n==ED_frame) or (ES_frame!=-1 and n==ES_frame)): 
            continue    
        
        gt_frame = data_cropped[:,:,:,n]
        my_frame_forward = data_cropped_warped_forward[:,:,:,n]
        my_frame_reverse = data_cropped_warped_reverse[:,:,:,n]
        
        # sum of squared differences
        forward_diff = np.sum((gt_frame-my_frame_forward)**2)
        reverse_diff = np.sum((gt_frame-my_frame_reverse)**2) 
        
        SSD_all_forward.append(forward_diff)
        SSD_all_reverse.append(reverse_diff)
        
    SSD_all_forward = np.asarray(SSD_all_forward)
    SSD_all_reverse = np.asarray(SSD_all_reverse)
    SSD = np.concatenate((SSD_all_forward,SSD_all_reverse))
    
    return SSD, SSD_all_forward, SSD_all_reverse


### NEW VERSION ###
# Taken from jacob_divcurl_3d function 
# No mask input - need to calculate on whole image 
# 2-5-21 - changed n in for loop to n 
def calculate_min_jacobian_det(dx, dy, dz):
    
    num_frames, m, n, o = dx.shape[0], dx.shape[1], dx.shape[2], dx.shape[3]
    dim = [m,n,o]

    # jacobian_det_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
    # min_max_per_frame = np.zeros((num_frames,2))
    # min_max = np.zeros((2,1))
    det_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
    det_min_max_per_frame = np.zeros((num_frames,2))
    det_min_max = np.zeros((2,1))

    curl_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
    curl_min_max_per_frame = np.zeros((num_frames,2))
    curl_min_max = np.zeros((2,1))

    div_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
    div_min_max_per_frame = np.zeros((num_frames,2))
    div_min_max = np.zeros((2,1))

    # xI, yI, zI = scipy.mgrid[1:m + 1, 1:n + 1, 1:o + 1]
    xI_pad, yI_pad, zI_pad = scipy.mgrid[0:m + 2, 0:n + 2, 0:o + 2]
    
    for nn in range(0,num_frames):
        
        dispx = dx[nn]
        dispy = dy[nn]
        dispz = dz[nn]
    
        D1 = pycardiac_3D_functions.padarray_3d(dispx)
        D2 = pycardiac_3D_functions.padarray_3d(dispy)
        D3 = pycardiac_3D_functions.padarray_3d(dispz)
        posx_pad = D1 + xI_pad
        posy_pad = D2 + yI_pad
        posz_pad = D3 + zI_pad 
        
        # det, _, _, = pycardiac_3D_functions.jacob_divcurl_3d(posx_pad + 0.0, posy_pad + 0.0, posz_pad + 0.0)
        det, curl, div = pycardiac_3D_functions.jacob_divcurl_3d(posx_pad + 0.0, posy_pad + 0.0, posz_pad + 0.0)

        ### det ###
        min_jacobian_det = np.min(det)
        max_jacobian_det = np.max(det)
        # jacobian_det_vol_all_frames[:,:,:,nn] = det
        det_vol_all_frames[:,:,:,nn] = det
        
        # min_max_per_frame[nn,0] = min_jacobian_det
        # min_max_per_frame[nn,1] = max_jacobian_det
        det_min_max_per_frame[nn,0] = min_jacobian_det
        det_min_max_per_frame[nn,1] = max_jacobian_det

        ### curl ###
        min_curl = np.min(curl)
        max_curl = np.max(curl)
        curl_vol_all_frames[:,:,:,nn] = curl
        curl_min_max_per_frame[nn,0] = min_curl
        curl_min_max_per_frame[nn,1] = max_curl

        ### div ###
        min_div = np.min(div)
        max_div = np.max(div)
        div_vol_all_frames[:,:,:,nn] = div
        div_min_max_per_frame[nn,0] = min_div
        div_min_max_per_frame[nn,1] = max_div
        
    # min_max[0] = np.min(min_max_per_frame[:,0])
    # min_max[1] = np.max(min_max_per_frame[:,1])
    det_min_max[0] = np.min(det_min_max_per_frame[:,0])
    det_min_max[1] = np.max(det_min_max_per_frame[:,1])

    curl_min_max[0] = np.min(curl_min_max_per_frame[:,0])
    curl_min_max[1] = np.max(curl_min_max_per_frame[:,1])

    div_min_max[0] = np.min(div_min_max_per_frame[:,0])
    div_min_max[1] = np.max(div_min_max_per_frame[:,1])

    # return jacobian_det_vol_all_frames, min_max_per_frame, min_max
    return det_vol_all_frames, det_min_max_per_frame, det_min_max, \
           curl_vol_all_frames, curl_min_max_per_frame, curl_min_max, \
           div_vol_all_frames, div_min_max_per_frame, div_min_max
    

# def jacob_divcurl_3d(posx, posy, posz):
#     
#     ux = 0.5 * posx[2:, 1:-1, 1:-1] - 0.5 * posx[:-2, 1:-1, 1:-1] 
#     uy = 0.5 * posx[1:-1, 2:, 1:-1] - 0.5 * posx[1:-1, :-2, 1:-1]
#     uz = 0.5 * posx[1:-1, 1:-1, 2:] - 0.5 * posx[1:-1, 1:-1, :-2] 
#     vx = 0.5 * posy[2:, 1:-1, 1:-1] - 0.5 * posy[:-2, 1:-1, 1:-1] 
#     vy = 0.5 * posy[1:-1, 2:, 1:-1] - 0.5 * posy[1:-1, :-2, 1:-1]
#     vz = 0.5 * posy[1:-1, 1:-1, 2:] - 0.5 * posy[1:-1, 1:-1, :-2] 
#     wx = 0.5 * posz[2:, 1:-1, 1:-1] - 0.5 * posz[:-2, 1:-1, 1:-1] 
#     wy = 0.5 * posz[1:-1, 2:, 1:-1] - 0.5 * posz[1:-1, :-2, 1:-1]
#     wz = 0.5 * posz[1:-1, 1:-1, 2:] - 0.5 * posz[1:-1, 1:-1, :-2] 
#     
#     det = ux*(vy*wz - wy*vz) - uy*(vx*wz-wx*vz) + uz*(vx*wy-wx*vy)
#     curl = (wy-vz) + (uz-wx) + (vx-uy) 
#     div = ux + vy + wz
#      
#     return det, curl, div 


# ### OLD VERSION ### 
# # https://github.com/SimpleITK/SimpleITK/issues/932
# # mask for each frame of disp. 
# # def calculate_min_jacobian_det(disp, mask=None):
# # def calculate_min_jacobian_det(disp, mask):
# def calculate_min_jacobian_det(dx, dy, dz, mask):
# 
#     num_frames = np.shape(dx)[0]
#     min_jacobian_det = [] 
# 
#     dx = np.expand_dims(dx,4)
#     dy = np.expand_dims(dy,4)
#     dz = np.expand_dims(dz,4)
# 
#     disp2 = np.concatenate((dx,dy,dz),axis=4)
#     
#     dim = dx.shape[1:4]
#     jacobian_det_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
#     print ('jacobian_det_vol_all_frames: ' + str(jacobian_det_vol_all_frames.shape))
#     min_max_per_frame = np.zeros((num_frames,2))
#     min_max = np.zeros((2,1))
#     
#     for n in range(0,num_frames):
#         
#         sitk_displacement_field = sitk.GetImageFromArray(disp2[n,:,:,:,:], isVector=True)
#         jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
#         jacobian_det_volume_np = sitk.GetArrayFromImage(jacobian_det_volume)
#         
#         # if a mask is included, only calculate the jacobian inside this area  
#         # if (mask != None):
#         # if mask is not empty 
#         if (mask != []):
#             mask_frame = mask[:,:,:,n]
#             # Make mask 0/1 if not. 
#             mask_frame_max = np.max(mask)
#             mask_frame = mask_frame / mask_frame_max 
#             # Just include mask area        
#             jacobian_det_volume_np = jacobian_det_volume_np * mask_frame
#             ind = np.where(mask_frame>0)
#             jacobian_ind = jacobian_det_volume_np[ind]
#             min_jacobian_det = np.min(jacobian_ind)
#             max_jacobian_det = np.max(jacobian_ind)
#             jacobian_det_vol_all_frames[:,:,:,n] = jacobian_det_volume_np
#         else:
#             min_jacobian_det = np.min(jacobian_det_volume_np)
#             max_jacobian_det = np.max(jacobian_det_volume_np)
#             jacobian_det_vol_all_frames[:,:,:,n] = jacobian_det_volume_np 
#             # min_jacobian_det.append(np.min(jacobian_det_volume_np))
#             
#         # Assign
#         min_max_per_frame[n,0] = min_jacobian_det 
#         min_max_per_frame[n,1] = max_jacobian_det
#     
#     min_max[0] = np.min(min_max_per_frame[:,0])
#     min_max[1] = np.max(min_max_per_frame[:,1])
#         
#     # Returns the 4D jacobian det volume, the minimum and max per frame and the overall min and max
#     return jacobian_det_vol_all_frames, min_max_per_frame, min_max

# # https://github.com/SimpleITK/SimpleITK/issues/932
# # mask for each frame of disp. 
# # def calculate_min_jacobian_det(disp, mask=None):
# def calculate_min_jacobian_det(disp, mask):
# 
#     temp = disp["DM_fx"]
#     num_frames = np.shape(temp)[0]
#     min_jacobian_det = [] 
#     
# #     dx = disp["DM_fx"]
# #     dx = np.expand_dims(dx,5)
# #     dy = disp["DM_fy"]
# #     dy = np.expand_dims(dy,5)
# #     dz = disp["DM_fz"]
# #     dz = np.expand_dims(dz,5)
# 
#     ### changed 12-11-20 ### 
#     dx = disp["DM_fx"]
#     dx = np.expand_dims(dx,4)
#     dy = disp["DM_fy"]
#     dy = np.expand_dims(dy,4)
#     dz = disp["DM_fz"]
#     dz = np.expand_dims(dz,4)
# 
#     disp2 = np.concatenate((dx,dy,dz),axis=4)
#     
#     dim = dx.shape[1:4]
#     jacobian_det_vol_all_frames = np.zeros((dim[0], dim[1], dim[2], num_frames))
#     print ('jacobian_det_vol_all_frames: ' + str(jacobian_det_vol_all_frames.shape))
#     min_max_per_frame = np.zeros((num_frames,2))
#     min_max = np.zeros((2,1))
#     
#     for n in range(0,num_frames):
#         
#         sitk_displacement_field = sitk.GetImageFromArray(disp2[n,:,:,:,:], isVector=True)
#         jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
#         jacobian_det_volume_np = sitk.GetArrayFromImage(jacobian_det_volume)
#         
#         # if a mask is included, only calculate the jacobian inside this area  
#         # if (mask != None):
#         # if mask is not empty 
#         if (mask != []):
#             mask_frame = mask[:,:,:,n]
#             # Make mask 0/1 if not. 
#             mask_frame_max = np.max(mask)
#             mask_frame = mask_frame / mask_frame_max 
#             # Just include mask area        
#             jacobian_det_volume_np = jacobian_det_volume_np * mask_frame
#             ind = np.where(mask_frame>0)
#             jacobian_ind = jacobian_det_volume_np[ind]
#             min_jacobian_det = np.min(jacobian_ind)
#             max_jacobian_det = np.max(jacobian_ind)
#             jacobian_det_vol_all_frames[:,:,:,n] = jacobian_det_volume_np
#         else:
#             min_jacobian_det = np.min(jacobian_det_volume_np)
#             max_jacobian_det = np.max(jacobian_det_volume_np)
#             jacobian_det_vol_all_frames[:,:,:,n] = jacobian_det_volume_np 
#             # min_jacobian_det.append(np.min(jacobian_det_volume_np))
#             
#         # Assign
#         min_max_per_frame[n,0] = min_jacobian_det 
#         min_max_per_frame[n,1] = max_jacobian_det
#     
#     min_max[0] = np.min(min_max_per_frame[:,0])
#     min_max[1] = np.max(min_max_per_frame[:,1])
#         
#     # Returns the 4D jacobian det volume, the minimum and max per frame and the overall min and max
#     return jacobian_det_vol_all_frames, min_max_per_frame, min_max
    

def calculate_MAD(mesh_A, mesh_B):
    '''Calculates the mean absolute distance (MAD) between two meshes'''
     
    ### calculate MAD metric ###
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetSignedDistance(0)
    # distance between ref and my mesh 
    if vtk.VTK_MAJOR_VERSION <= 5:
        distance_filter.SetInput(0,mesh_A)
        distance_filter.SetInput(1,mesh_B)
    else:
        # distance_filter.SetInputConnection(0,mesh_A)
        # distance_filter.SetInputConnection(1,mesh_B)
        distance_filter.SetInputData(0,mesh_A)
        distance_filter.SetInputData(1,mesh_B)
    distance_filter.Update()
    # get distance     
    distances = vtk_to_numpy(distance_filter.GetOutput().GetPointData().GetScalars())
    mean_distance = np.mean(distances)
    # get distance meshes 
    distance_mesh = vtk.vtkPolyData()
    distance_mesh = distance_filter.GetOutput() 
     
    return distance_mesh, distances, mean_distance 
 
# https://stackoverflow.com/questions/13692801/distance-matrix-of-curves-in-python
def calculate_HD(A,B):
    '''Calculates the Hausdorff distance between two contours'''
     
    D = scipy.spatial.distance.cdist(A, B, 'euclidean')
    # need this for the calculation of distance maps
    D1 = np.min(D, axis=1)
    D2 = np.min(D, axis=0)
    # HD in both directions 
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))
    H = np.max([H1,H2])
     
    return D1, D2, H1, H2, H 
 
def calculate_Dice_score(maskA, maskB):
    '''Calculate the Dice score between two masks'''
     
    maskA_flat = maskA.flatten()
    maskA_flat_ind = np.asarray(np.where(maskA_flat>0))
    maskA_flat_ind_len = np.float64(np.shape(maskA_flat_ind)[1])
     
    maskB_flat = maskB.flatten()
    maskB_flat_ind = np.asarray(np.where(maskB_flat>0))
    maskB_flat_ind_len = np.float64(np.shape(maskB_flat_ind)[1])
     
    intersection_ind = np.intersect1d(maskA_flat_ind, maskB_flat_ind)
    intersection_len = np.float64(len(intersection_ind))
     
    Dice_score = 2.0 * (intersection_len / (maskA_flat_ind_len + maskB_flat_ind_len))
 
    return Dice_score 

def create_mesh_and_faces(S_contour_points_temp, spacing, bb, mesh_for_volume=0):
    '''Creates a set of faces, and a mesh, with or without the extra base added'''
    '''Orders the contours'''
    
    ### Get the min and max points ###  
    S_contour_points_new = [] 
    
    min_z = np.int32(np.min(S_contour_points_temp[:,2]))
    max_z = np.int32(np.max(S_contour_points_temp[:,2]))
    # print ('min_z: ' + str(min_z))
    # print ('max_z: ' + str(max_z))

    ### Get the ordered contours ### 
    for n in range(min_z,max_z):
        # print ('n: ' + str(n))
        ctr1 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n),:])
        ctr2 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n+1),:])
        ctr2_new = find_contour_index_association2(ctr1, ctr2)
        if (n==min_z):
            S_contour_points_new.append(ctr1)
        S_contour_points_new.append(ctr2_new)
    S_contour_points_new = np.asarray(S_contour_points_new)
    S_contour_points_new = np.concatenate(S_contour_points_new, 0)
    
    ### Create faces ### 
    num_contour_pairs = max_z-min_z
    num_points = 241 
    # Get faces 
    faces = create_faces(num_contour_pairs, num_points)
    # Get min faces 
    faces_min = create_faces_min(S_contour_points_new, min_z, num_points)    
    # Get max faces     
    faces_max = create_faces_max(S_contour_points_new, min_z, max_z, num_points)


    points_min = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==min_z),:])
    mean_min_z = np.mean(points_min,0)
    points_max = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==max_z),:])
    mean_max_z = np.mean(points_max,0)

    # Add the faces and the points to the previous mesh, and save
    mean_min_z = np.transpose(np.expand_dims(mean_min_z,1))
    mean_max_z = np.transpose(np.expand_dims(mean_max_z,1))
    
    ### changed ####
    # Add points to make sure the binary mask created extends for top slice
    if (mesh_for_volume==1):
        mean_max_z = mean_max_z + 1.0

    
    
    S_contour_points_new = np.asarray(np.concatenate((S_contour_points_new, mean_min_z, mean_max_z)))
    
    
    
    # print ('S_contour_points_new: ' + str(S_contour_points_new.shape))
     
    faces_temp = np.asarray(np.concatenate((faces, faces_min, faces_max)))
    # delete last face - the extra triangle
    # faces = faces_temp[0:-2,:]
    # faces = faces_temp[0:-2,:] 
    # faces = faces_temp
    # faces = np.vstack((faces_temp[0:-2,:], faces_temp[-1,:]))
    
    faces = faces_temp[0:-2,:] 
    a = faces[-1,0] + 1
    b = faces[-1,1] - num_points + 1 
    c = faces[-1,2] 
    faces = np.vstack((faces,[a,b,c]))
    
    # print ('faces: ' + str(faces.shape))
    # print ('faces: ' + str(faces))
    # print ('S_contour_points_new: ' + str(S_contour_points_new.shape))
    
    
    ### Crop the S_contour_points_new by the 2D bb ###
    S_contour_points_new[:,0] = S_contour_points_new[:,0] - bb[0]
    S_contour_points_new[:,1] = S_contour_points_new[:,1] - bb[2]
    
        
    # multiply contour points by spacing
    S_contour_points_new[:,0] = S_contour_points_new[:,0] * spacing[0]
    S_contour_points_new[:,1] = S_contour_points_new[:,1] * spacing[1]
    S_contour_points_new[:,2] = S_contour_points_new[:,2] * spacing[2]
    # switch x and y points?!
    S_contour_points_new = np.transpose(np.vstack((S_contour_points_new[:,1], S_contour_points_new[:,0], S_contour_points_new[:,2])))
     
    S_mesh_new = form_mesh_from_vertices_and_faces(S_contour_points_new, faces)
    
    # print ('faces: ' + str(faces))
    
    
    # print ('S_contour_points_new: ' + str(S_contour_points_new.shape))
    # print ('faces: ' + str(faces.shape))
    # print ('S_mesh_new: ' + str(S_mesh_new))
    # print ('spacing: ' + str(spacing))
    
    
    
    
    return S_mesh_new


def create_mesh_for_frame(im_files, contour_files, min_z, max_z, spacing, tol, bb_max):
    '''Creates a mesh, mesh for volume and binary volume given a list of image files and contour files for a particular frame'''
    
    # Get the dimensions from the first file 
    data = np.load(im_files[0])
    dim = data.shape 
    
    # Read the set of im_files 
    num_files = len(im_files)
    im_data = [] 
    contour_data = []
    contour_data_spline = [] 
    bb_across_slices = [] 
    
    for n in range(0,num_files):
        data = np.load(im_files[n])
        # original points 
        S2_pts = np.load(contour_files[n])
        # added - since Kumar saved out the contour points 
        S2_pts = np.roll(S2_pts, 1, 1)
        # spline points
        S2_pts_spline = np.transpose(get_pts_from_spline(np.transpose(S2_pts), spacing[0:2]))  
        S2_pts_spline = np.asarray(np.hstack((S2_pts,n*np.ones((S2_pts.shape[0],1)))))
        # append
        im_data.append(data)
        contour_data.append(S2_pts)
        contour_data_spline.append(S2_pts_spline)
#         plt.figure()
#         plt.imshow(data, cmap='gray')
#         plt.plot(S2_pts[:,0], S2_pts[:,1], '.r')
#         plt.show()
        # get the bounding box for the slice
        
        # if (bb_max==[]):
        if not list(bb_max):
        
            bb = [] 
            bb = find_boundingbox(S2_pts[:,0], S2_pts[:,1], dim[1], dim[0], bb, tol)
            bb_across_slices.append(bb)

    # concatenate
    contour_points = np.concatenate(contour_data_spline)

    ### get the bounding box across slices ###
    
    # if (bb_max==[]): 
    if not list(bb_max): 
    
        bb_across_slices = np.asarray(bb_across_slices) # Nx4 
        bb_max = np.zeros((1,6))
        bb_max[0,0] = np.min(bb_across_slices[:,0]) 
        bb_max[0,1] = np.max(bb_across_slices[:,1])
        bb_max[0,2] = np.min(bb_across_slices[:,2])
        bb_max[0,3] = np.max(bb_across_slices[:,3])
        # bb_max = np.array([np.min(bb_across_slices[:,0]), np.max(bb_across_slices[:,1]), np.min(bb_across_slices[:,2]), np.max(bb_across_slices[:,3])])
        # Get the z dimensions for the bounding box 
        # slices_ = np.asarray(slices_)
        # bb_max[0,4] = slices_[0]
        # bb_max[0,5] = slices_[-1]
        bb_max[0,4] = min_z
        bb_max[0,5] = max_z
        bb_max = np.asarray(bb_max, np.int32)
        bb_max = np.squeeze(bb_max)
    
         
    ### Create the 3D set of contour points ###
    # with the extra base 

    ### contour_points_with_extra_base - Adding a second row of the bottom contour points ###
    # min_points = np.squeeze(contour_points[np.where(contour_points[:,2]==min_z),:])
    min_points = np.squeeze(contour_points[np.where(contour_points[:,2]==0),:])

    min_points[:,2] = -1 
    temp_points = np.concatenate((min_points, contour_points))
    contour_points_with_extra_base = temp_points

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(contour_points[:,0], contour_points[:,1], contour_points[:,2], '.r')
#     ax.plot(contour_points_with_extra_base[:,0], contour_points_with_extra_base[:,1], contour_points_with_extra_base[:,2], '.b')
#     plt.show()
    
    
    # contour_points, contour_points_with_extra_base
    
    ########### Create the meshes ######################
    ### This function also crops all of the data by the bounding box given ###
    # Data is already cropped in the z direction! 
    mesh = create_mesh_and_faces(contour_points, spacing, bb_max[0:4], mesh_for_volume=0) # bb in 2D 
    mesh_for_volume = create_mesh_and_faces(contour_points_with_extra_base, spacing, bb_max[0:4], mesh_for_volume=1)
    
    # Create a binary volume of the mesh
    
    # dim = [dim[0], dim[1], bb_max[5]]
    # dim = [(bb[1]-bb[0])+1, (bb[3]-bb[2])+1, bb_max[5]]
    
    # dim = [(bb_max[3]-bb_max[2])+1, (bb_max[1]-bb_max[0])+1, bb_max[5]]
    #### CHANGED ####
    dim = [(bb_max[3]-bb_max[2])+1, (bb_max[1]-bb_max[0])+1, (bb_max[5]-bb_max[4])+1]


    # print ('dim: ' + str(dim))
    
    volume = convert_mesh_to_mask(mesh_for_volume, dim, spacing)
    # print ('mesh_for_volume: ' + str(volume.shape))
    
    
    return mesh, mesh_for_volume, volume, bb_max  
        


def convert_ACDC_sequence(ACDC_directory, ACDC_sequence_directory, ACDC_sequence_output_directory, paraview_time, tol, num_patients):
    '''Converts the data from ACDC_sequence_directory into a format for the validation'''

    # Get a list of the folders - patient numbers
    subdirs = glob.glob(ACDC_sequence_directory+'/*')
    # subdirs = os.path.dirname(ACDC_sequence_directory) 
    # num_patients = len(subdirs)
    
    # print ('num_patients: ' + str(num_patients))
    # patient_name = os.path.basename(patient_dir)

    # For each patient 
    
    bb_all = [] 
    
    for patient in range(0,num_patients):
    # for patient in range(0,1):
    # for patient in range(0,2):
        
        patient_subdir = os.path.basename(subdirs[patient])
        
        # Create directories for output 
        output_mesh_warped_directory = os.path.join(ACDC_sequence_output_directory, patient_subdir, 'mesh_warped')
        output_mesh_warped_for_volume_directory = os.path.join(ACDC_sequence_output_directory, patient_subdir, 'mesh_warped_for_volume')
        output_mesh_mask_warped_directory = os.path.join(ACDC_sequence_output_directory, patient_subdir, 'mesh_mask_warped')
        
        if not os.path.isdir(ACDC_sequence_output_directory):
            os.mkdir(ACDC_sequence_output_directory)
        if not os.path.isdir(os.path.join(ACDC_sequence_output_directory, patient_subdir)): 
            os.mkdir(os.path.join(ACDC_sequence_output_directory, patient_subdir))
        if not os.path.isdir(output_mesh_warped_directory):
            os.mkdir(output_mesh_warped_directory)
        if not os.path.isdir(output_mesh_warped_for_volume_directory):
            os.mkdir(output_mesh_warped_for_volume_directory)
        if not os.path.isdir(output_mesh_mask_warped_directory):
            os.mkdir(output_mesh_mask_warped_directory)
            
        # Get the spacing from the original ACDC_directory
        ACDC_filename = os.path.join(ACDC_directory, patient_subdir, patient_subdir+'_4d.nii.gz')
        nii = nib.load(ACDC_filename)
        spacing = [nii.header['pixdim'][1], nii.header['pixdim'][2], nii.header['pixdim'][3]]
        
        # get all image files 
        image_files = [f for f in os.listdir(os.path.join(ACDC_sequence_directory, patient_subdir)) if 'image' in f]
        num_image_files = len(image_files)
        # print ('num_image_files: ' + str(num_image_files))
        # print (image_files)
        
        # get all files
        files = [f for f in os.listdir(os.path.join(ACDC_sequence_directory, patient_subdir)) if 'lv_endo_contour' in f]
        num_files = len(files)
        # print ('num_files: ' + str(num_files))
        
        # split string to get ser01, ser02... ser09 etc. 
        ser_strings = [s.split('_')[1] for s in files]
        ser_strings_unique = sort_nicely(list(set(ser_strings)))
        num_ser_strings_unique = len(ser_strings_unique)
        # print (ser_strings_unique)
        
        # need to get the min and max z slices numbers
        # ser_strings_int = [int(s) for s in str.split() if s.isdigit()]
        ser_strings_int = list(map(lambda sub:int(''.join([ele for ele in sub if ele.isnumeric()])), ser_strings_unique))
        min_z = np.min(ser_strings_int)
        max_z = np.max(ser_strings_int)
        
        # get the number of frames 
        im_strings = [s.split('_')[2] for s in files]
        im_strings_unique = sort_nicely(list(set(im_strings)))
        num_im_strings_unique = len(im_strings_unique)
        # print (im_strings_unique) 
        num_frames = num_im_strings_unique

        
        ### For each frame ### 
        
        mesh_mask_warped_all = [] 
        
        for frame in range(0,num_frames):
            
            if (frame==0):
                bb_max = [] 
            
            # print ('frame: ' + str(frame))
            # Get a list of the image files for the frame 
            im_files = [s for s in image_files if im_strings_unique[frame] in s]
            im_files = [os.path.join(ACDC_sequence_directory, patient_subdir, f) for f in im_files]
            contour_files = [s for s in files if im_strings_unique[frame] in s]
            contour_files = [os.path.join(ACDC_sequence_directory, patient_subdir, f) for f in contour_files]
            # Get the mesh, mesh_for_volume, binary volume
            mesh_warped, mesh_warped_for_volume, mesh_mask_warped, bb_max = create_mesh_for_frame(im_files, contour_files, min_z, max_z, spacing, tol, bb_max)
            # Save out the mesh_warped
            writer = vtk.vtkPolyDataWriter()
            output_filename_mesh = os.path.join(output_mesh_warped_directory, 'mesh.time.' + str(paraview_time*frame) + '.vtk')
            # writer.SetInput(mesh_warped)
            writer.SetInputData(mesh_warped)
            writer.SetFileName(output_filename_mesh)
            writer.Write()
            # Save out the mesh_warped_for_volume
            writer = vtk.vtkPolyDataWriter()
            output_filename_mesh = os.path.join(output_mesh_warped_for_volume_directory, 'mesh.time.' + str(paraview_time*frame) + '.vtk')
            # writer.SetInput(mesh_warped_for_volume)
            writer.SetInputData(mesh_warped_for_volume)
            writer.SetFileName(output_filename_mesh)
            writer.Write()
            # Save out the volume
            img = nib.Nifti1Image(mesh_mask_warped, np.eye(4))
            output_filename = os.path.join(output_mesh_mask_warped_directory, str(frame)+'.nii')
            img.header['pixdim'][1] = spacing[0]
            img.header['pixdim'][2] = spacing[1]
            img.header['pixdim'][3] = spacing[2]
            nib.save(img, output_filename)
            
            mesh_mask_warped = np.expand_dims(mesh_mask_warped,4)
            mesh_mask_warped_all.append(mesh_mask_warped)
            
        
        ### Save out a 4D volume of the mesh_mask_warped ### 
        mesh_mask_warped_all = np.asarray(mesh_mask_warped_all)
        mesh_mask_warped_all = np.rollaxis(mesh_mask_warped_all,0,4)
        mesh_mask_warped_all = np.squeeze(mesh_mask_warped_all)
        print ('mesh_mask_warped_all: ' + str(mesh_mask_warped_all.shape))
        
        img = nib.Nifti1Image(mesh_mask_warped_all, np.eye(4))
        output_filename = os.path.join(ACDC_sequence_output_directory, patient_subdir, "mesh_mask_warped.nii")
        img.header['pixdim'][1] = spacing[0]
        img.header['pixdim'][2] = spacing[1]
        img.header['pixdim'][3] = spacing[2]
        nib.save(img, output_filename)
        
        
        ### bounding box ### 
        bb_all.append(bb_max)
        
        ### save out paraview mesh series file ### 
        # filenames
        output_filename_paraview_mesh_series = os.path.join(output_mesh_warped_directory, 'mesh.vtk.series')
        output_filename_paraview_mesh_warped_series = os.path.join(output_mesh_warped_for_volume_directory, 'mesh.vtk.series')
        paraview_mesh_series_data = {}
        paraview_mesh_series_data['file-series-version'] = "1.0"
        paraview_mesh_series_data['files'] = [] 
        for nn in range(0,num_frames):
            paraview_mesh_series_data['files'].append({
                'name': 'mesh.time.'+str(paraview_time*nn)+'.vtk',
                # 'time': paraview_time*nn
                'time': np.int(paraview_time*nn)
            })
        
        with open(output_filename_paraview_mesh_series, 'w') as outfile:
            json.dump(paraview_mesh_series_data, outfile)
        with open(output_filename_paraview_mesh_warped_series, 'w') as outfile: 
            json.dump(paraview_mesh_series_data, outfile)
        
    
    bb_all = np.asarray(bb_all)    
    bb_all = np.squeeze(bb_all)
    
    return bb_all 






# copied from cardiacanalysis.py 
def get_pts_from_spline(contour, spacing):        
    """Create a vtk spline widget and retrieve points"""
    """2xN array"""
    
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(contour.shape[1])
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(contour.shape[1])
                           
    # for i in xrange(contour.shape[1]):
    for i in range(0,contour.shape[1]):  
        points.SetPoint(i, spacing[0] * contour[0, i], spacing[1] * (contour[1, i]), 0)
        lines.InsertNextCell(i)
    
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)
    
    mySpline = vtk.vtkCardinalSpline()
    mySpline.SetLeftConstraint(2)
    mySpline.SetLeftValue(0.0)
    mySpline.SetRightConstraint(2)
    mySpline.SetRightValue(0.0)
    
    spline = vtk.vtkSplineFilter()
    spline.SetSpline(mySpline)
    # spline.SetInput(pd)
    spline.SetInputData(pd)
    spline.SetSubdivideToSpecified()
    # spline.SetNumberOfSubdivisions(settings_.NSRESOLUTION)
    spline.SetNumberOfSubdivisions(240)
    spline.Update()
    
    endo_points = spline.GetOutput().GetPoints()
    endo_contour = np.zeros((2, endo_points.GetNumberOfPoints()))
    
    for i in range(endo_points.GetNumberOfPoints()):
        endo_contour[0, i] = endo_points.GetPoint(i)[0] / spacing[0]
        endo_contour[1, i] = endo_points.GetPoint(i)[1] / spacing[1]
         
    return endo_contour

def find_contour_index_association2(ctr1, ctr2):
    """Find point correspondence between two contours""" 
    """Make contour 2 align with contour 1"""
    ctr2_reverse = np.fliplr(ctr2)
    
    closed = 1 
    
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
        # return (1.0 - w) * ctr1 + w * np.roll(ctr2, shift=np.argmin(dist_vector), axis=1), np.argmin(dist_vector), 1
        return np.roll(ctr2, shift=np.argmin(dist_vector), axis=1)
    else:
        # return (1.0 - w) * ctr1 + w * np.roll(ctr2_reverse, shift=np.argmin(dist_vector_reverse), axis=1), np.argmin(dist_vector), -1
        return np.roll(ctr2_reverse, shift=np.argmin(dist_vector_reverse), axis=1)
    
            
def find_boundingbox(msx, msy, imxmax, imymax, bb, tol):
    """Find the bounding box based on a contour. This will allow for computing registration only within the bounding box to save time."""
    if bb == []:
        b1, b2 = np.floor(max(msx.min() - tol, 0)), np.ceil(min(msx.max() + tol, imxmax))
        b3, b4 = np.floor(max(msy.min() - tol, 0)), np.ceil(min(msy.max() + tol, imymax))    
    else:
        b1, b2 = np.floor(max(min(msx.min() - tol, bb[0]), 0)), np.ceil(min(max(msx.max() + tol, bb[1]), imxmax))
        b3, b4 = np.floor(max(min(msy.min() - tol, bb[2]), 0)), np.ceil(min(max(msy.max() + tol, bb[3]), imymax))    
    
    return np.array([b1, b2, b3, b4])


def get_bounding_box(ED_contourfile, contour_type, tol):
    
    nii = nib.load(ED_contourfile)
    contourdata = nii.get_data()
    
    num_slices = contourdata.shape[2] 
    dim = contourdata.shape[0:2]
    
    slices_ = [] 
    bb_across_slices = [] 
    
    ### Get where contourdata equals 3 - LV endocardium ###
    for m in range(0,num_slices):
    # for m in range(0,num_slices+1):
        
        S2_endo = np.zeros((dim))
        S2_truth = contourdata[:,:,m]        
        
        # get index 
        endo = np.asarray(np.where(S2_truth==contour_type))
        
        if (endo.any()):
            slices_.append(m)
            ind = np.unravel_index(endo, S2_endo.shape)
            ind = np.asarray(ind)
            S2_endo[ind[:,0], ind[:,1]] = 1 
        
            # get contour of endocardium 
            S2_pts = measure.find_contours(S2_endo, 0)
            # S2_pts = S2_pts[1]
            S2_pts = S2_pts[-1] # the last one is the longest, may need to change this? 
            S2_pts = np.roll(S2_pts,1,axis=1)
            
            # calculate the bounding box per image slice 
            bb = [] 
            bb = find_boundingbox(S2_pts[:,0], S2_pts[:,1], dim[1], dim[0], bb, tol)
            bb_across_slices.append(bb)
            
    ### now save the bounding boxes ###
    bb_across_slices = np.asarray(bb_across_slices) # Nx4 
    
    bb_max = np.zeros((1,6))
    bb_max[0,0] = np.min(bb_across_slices[:,0]) 
    bb_max[0,1] = np.max(bb_across_slices[:,1])
    bb_max[0,2] = np.min(bb_across_slices[:,2])
    bb_max[0,3] = np.max(bb_across_slices[:,3])
    # bb_max = np.array([np.min(bb_across_slices[:,0]), np.max(bb_across_slices[:,1]), np.min(bb_across_slices[:,2]), np.max(bb_across_slices[:,3])])
    
    # Get the z dimensions for the bounding box 
    slices_ = np.asarray(slices_)
    bb_max[0,4] = slices_[0]
    # bb_max[0,5] = slices_[-1]+1
    bb_max[0,5] = slices_[-1]
    
    bb_max = np.asarray(bb_max, np.int32)
    
    # print ('slices_: ' + str(slices_))
    # print ('bb_max: ' + str(bb_max))
    
    return bb_max 
    

def get_ACDC_data(patient_dir, contour_type, tol): 
    '''Processes the ACDC data''' 
              
    ### Get the info file ###
    info_cfg_filename = os.path.join(patient_dir, 'Info.cfg')
    file = open(info_cfg_filename, 'rb')
    data = file.readlines()
    
    ### Split the patient_dir to form the subdir and patient_name ###
    # subdir = os.path.dirname(patient_dir)
    patient_name = os.path.basename(patient_dir)

    ### Get the ED frame ###
    ED = str(data[0])
    # ED_frame = np.int16(ED[4]) - 1 # make sure ACDC uses MATLAB indexing??
    # ED_frame = np.int16(ED[6]) - 1 # make sure ACDC uses MATLAB indexing?? 
    ED_frame = int(re.search(r'\d+', ED).group()) - 1
    print ('ED_frame: ' + str(ED_frame))
    
    ### Get the ES frame ###
    ES = str(data[1]) 
    # ES_frame = np.int16(ES[4:]) - 1 
    # ES_frame = np.int16(ES[6:]) - 1 
    ES_frame = int(re.search(r'\d+', ES).group()) - 1
    print ('ES_frame: ' + str(ES_frame))

    
    ### Get the ED filename ###
    if (ED_frame < 9): 
        ED_contourfile = os.path.join(patient_dir, patient_name + '_frame0' + str(ED_frame+1) + '_gt.nii.gz')
    else: 
        ED_contourfile = os.path.join(patient_dir, patient_name + '_frame' + str(ED_frame+1) + '_gt.nii.gz')
    # Load the ED file 
    nii = nib.load(ED_contourfile)
    ED_gt_temp = nii.get_data()
    
    # keep where ED_gt = contour_type 
    ED_gt = np.zeros((ED_gt_temp.shape))
    ED_gt[np.where(ED_gt_temp==contour_type)] = 1
    

    ### Get the ES filename ###
    if (ES_frame < 9): 
        ES_contourfile = os.path.join(patient_dir, patient_name + '_frame0' + str(ES_frame+1) + '_gt.nii.gz')
    else: 
        ES_contourfile = os.path.join(patient_dir, patient_name + '_frame' + str(ES_frame+1) + '_gt.nii.gz')
    # Load the ES file 
    nii = nib.load(ES_contourfile)
    ES_gt_temp = nii.get_data()
    
    # keep where ES_gt = contour_type
    ES_gt = np.zeros((ES_gt_temp.shape))
    ES_gt[np.where(ES_gt_temp==contour_type)] = 1
    
    ### Get the datafile name and load ### 
    datafile = os.path.join(patient_dir, patient_name + '_4d.nii.gz')
    nii = nib.load(datafile)
    data = nii.get_data()
    dim = [data.shape[0], data.shape[1]]
    num_slices = data.shape[2]
    num_frames = data.shape[3]
    header = nii.header
    sx = header['pixdim'][1]
    sy = header['pixdim'][2]
    sz = header['pixdim'][3]
    spacing = [sx, sy, sz]

    ### Get the bounding box from the ED frame ###
    # If no tolerance given, assume that we use the full volume for registration 
    if (tol==0):
        bb = [0,dim[0],0,dim[1],0,dim[2]]
    # Gets the maximum bounding box plus tolerance across the slices
    # Only x and y   
    else:
        bb = get_bounding_box(ED_contourfile, contour_type, tol)
        # print ('bb in get_ACDC_data: ' + str(bb))
    bb = np.squeeze(bb)        
        
    
        
    return data, spacing, bb, ED_frame, ES_frame, ED_gt, ES_gt 


def get_ACDC_contour_with_endpoints(contourdata, dim, spacing):
    '''Add points to the bottom and top in the z direction''' 
    
    num_slices = contourdata.shape[2]
    slices_ = [] 
    S2_pts_with_z_list = [] 
    
    for m in range(0,num_slices):
        
        S2_endo = np.zeros((dim[0], dim[1]))
        S2_truth = contourdata[:,:,m]
        
        # get index 
        # endo = np.asarray(np.where(S2_truth==contour_type))
        endo = np.asarray(np.where(S2_truth==1))
                
        if (endo.any()):
            
            slices_.append(m)
            ind = np.unravel_index(endo, S2_endo.shape)
            ind = np.asarray(ind)
            S2_endo[ind[:,0], ind[:,1]] = 1 
        
            # get contour of endocardium 
            S2_pts = measure.find_contours(S2_endo, 0)
            # S2_pts = S2_pts[1]
            S2_pts = S2_pts[-1] # the last one is the longest, may need to change this? 
            S2_pts = np.roll(S2_pts,1,axis=1)
            
            # S2_pts[:,0] = S2_pts[:,0] - bb[2]
            # S2_pts[:,1] = S2_pts[:,1] - bb[0]
            
            # create a spline
            S2_pts = np.transpose(get_pts_from_spline(np.transpose(S2_pts), spacing[0:2]))  
            S2_pts_with_z = np.asarray(np.hstack((S2_pts,m*np.ones((S2_pts.shape[0],1)))))
            S2_pts_with_z_list.append(S2_pts_with_z) 
            
            
            
    S2_pts_with_z_list = np.concatenate(S2_pts_with_z_list)
    contour_points = S2_pts_with_z_list 
    
    
    ### Add points to the top and bottom slices ### 
    # Get the contour points for where the z value is the smallest 
    min_z = np.min(contour_points[:,2])
    contour_points_min_z = np.squeeze(contour_points[np.where(contour_points[:,2]==min_z),:])
    # contour_points_min_z = contour_points_min_z[:,0:2]
    # Fill this contour with open cv
    filled_contour_min_z = np.zeros((contourdata.shape[0], contourdata.shape[1]), dtype=np.uint8)    
    cv2.fillPoly(filled_contour_min_z, np.int32([contour_points_min_z[:,0:2]]), 255)
    filled_contour_min_z = np.swapaxes(filled_contour_min_z,0,1)
    
#     plt.figure()
#     plt.imshow(filled_contour_min_z)
#     plt.plot(contour_points_min_z[:,0], contour_points_min_z[:,1], 'g')
#     plt.show()
    
    # Now get these points 
    contour_pixel_min_z = np.where(filled_contour_min_z>0) 
    contour_pixel_min_z = np.transpose(np.asarray(contour_pixel_min_z))
    contour_pixel_min_z = np.asarray(np.hstack((contour_pixel_min_z,min_z*np.ones((contour_pixel_min_z.shape[0],1)))))
    
    
    # Get the contour points for where the z value is the largest 
    max_z = np.max(contour_points[:,2])
    contour_points_max_z = np.squeeze(contour_points[np.where(contour_points[:,2]==max_z),:])
    # contour_points_max_z = contour_points_max_z[:,0:2]
    # Fill this contour with open cv 
    filled_contour_max_z = np.zeros((contourdata.shape[0], contourdata.shape[1]), dtype=np.uint8)    
    cv2.fillPoly(filled_contour_max_z, np.int32([contour_points_max_z[:,0:2]]), 255)
    filled_contour_max_z = np.swapaxes(filled_contour_max_z,0,1)
    
#     plt.figure()
#     plt.imshow(filled_contour_max_z)
#     plt.plot(contour_points_max_z[:,0], contour_points_max_z[:,1], 'g')
#     plt.show()
    
    # Now get these points
    contour_pixel_max_z = np.where(filled_contour_max_z>0)
    contour_pixel_max_z = np.transpose(np.asarray(contour_pixel_max_z))    
    contour_pixel_max_z = np.asarray(np.hstack((contour_pixel_max_z,max_z*np.ones((contour_pixel_max_z.shape[0],1)))))

    # print ('min_z: ' + str(min_z))
    # print ('max_z: ' + str(max_z))
    # print ('contour_points_min_z: ' + str(contour_points_min_z.shape))
    # print ('contour_points_max_z: ' + str(contour_points_max_z.shape))
    # print ('contour_pixel_min_z: ' + str(contour_pixel_min_z.shape))
    # print ('contour_pixel_max_z: ' + str(contour_pixel_max_z.shape))
    
    # Add these points 
    contour_points_final = [] 
    
#     # if min_z == 1 
#     if (np.int32(min_z)==1):
#         print ('if min_z == 1: ' + str(min_z))
#         min_points = np.squeeze(contour_points[np.where(contour_points[:,2]==min_z),:])
#         min_points[:,2] = 0 
#         temp_points = np.concatenate((min_points, contour_points))
#         contour_points = temp_points
#     else: 
#         contour_points_final.append(contour_points)
    
    contour_points_final.append(contour_points)
    contour_points_final.append(contour_pixel_min_z)
    contour_points_final.append(contour_pixel_max_z)
    contour_points_final = np.concatenate(contour_points_final)

    ### contour_points_with_extra_base - Adding a second row of the bottom contour points ###
    min_points = np.squeeze(contour_points[np.where(contour_points[:,2]==min_z),:])
    min_points[:,2] = -1 
    temp_points = np.concatenate((min_points, contour_points))
    contour_points_with_extra_base = temp_points

     
    
    
    # return contour_points, contour_pixel_min_z, contour_pixel_max_z, contour_points_final
    return contour_points, contour_pixel_min_z, contour_pixel_max_z, contour_points_with_extra_base



def create_faces(num_contour_pairs, num_points):
    
    faces = [] 
    
    for n in range(0,num_contour_pairs):
        for m in range(0,num_points): # number of points in spline
            if (m<(num_points-1)): 
                face = [(n)*num_points+m, (n)*num_points+m+1, (n)*num_points+m+1+(num_points-1)]
                faces.append(face)
                face = [(n)*num_points+m+1, (n)*num_points+m+1+(num_points-1), (n)*num_points+m+1+num_points]
                faces.append(face)
            # connect last point back to first
            else:
                face = [(n)*num_points+(num_points-1), (n)*num_points, (n)*num_points+(num_points-1)+num_points]
                faces.append(face)
                face = [(n)*num_points, (n)*num_points+(num_points-1)+num_points, (n)*num_points+num_points]
                faces.append(face)
    faces = np.asarray(faces)
    
    return faces 

#     ### 3. Create the faces for each set of slices ###
#     # num_contour_pairs = max_z-min_z-1
#     num_contour_pairs = max_z-min_z 
#     faces = []  
# 
#     for n in range(0,num_contour_pairs):
#         for m in range(0,241): # number of points in spline
#             if (m<240): 
#                 face = [(n)*241+m, (n)*241+m+1, (n)*241+m+1+240]
#                 faces.append(face)
#                 face = [(n)*241+m+1, (n)*241+m+1+240, (n)*241+m+1+241]
#                 faces.append(face)
#             # connect last point back to first
#             else:
#                 face = [(n)*241+240, (n)*241, (n)*241+240+241]
#                 faces.append(face)
#                 face = [(n)*241, (n)*241+240+241, (n)*241+241]
#                 faces.append(face)
#     faces = np.asarray(faces)

# def create_faces_min(S_contour_points_temp, S_contour_points_new, min_z, num_points):
def create_faces_min(S_contour_points_new, min_z, num_points):

    
    ### Add one point at base and apex and create appropriate triangles to fill for now ###
    ### min ### 
    # points_min = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==min_z),:])
    points_min = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==min_z),:])
    mean_min_z = np.mean(points_min,0)
    mean_min_z_index = S_contour_points_new.shape[0]
    # make faces
    faces_min = []  
    for n in range(0,num_points):
        if (n<(num_points-1)):
            faces_min.append([n,n+1,mean_min_z_index])
        else:
            faces_min.append([(num_points-1),0,mean_min_z_index])
    faces_min = np.asarray(faces_min)
    
    return faces_min 
    
#     ### Add one point at base and apex and create appropriate triangles to fill for now ###
#     ### min ### 
#     points_min = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==min_z),:])
#     mean_min_z = np.mean(points_min,0)
#     mean_min_z_index = S_contour_points_new.shape[0]
#     # make faces
#     faces_min = []  
#     for n in range(0,241):
#         if (n<240):
#             faces_min.append([n,n+1,mean_min_z_index])
#         else:
#             faces_min.append([240,0,mean_min_z_index])
#     faces_min = np.asarray(faces_min)
    
# def create_faces_max(S_contour_points_temp, S_contour_points_new, min_z, max_z, num_points):
def create_faces_max(S_contour_points_new, min_z, max_z, num_points):

    
    ### max ### 
    # points_max = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==max_z),:])
    points_max = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==max_z),:])
    mean_max_z = np.mean(points_max,0)
    mean_max_z_index = S_contour_points_new.shape[0]+1
    # make faces 
    faces_max = []
    
    for n in range(num_points*(max_z-min_z),num_points*(max_z-min_z+1)+1): 
        if n<(num_points*(max_z-min_z+1)):
            faces_max.append([n,n+1,mean_max_z_index])
        else:
            faces_max.append([num_points*(max_z-min_z+1),num_points*(max_z-min_z),mean_max_z_index])
    faces_max = np.asarray(faces_max)    
    
    return faces_max 
    
#     ### max ### 
#     points_max = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==max_z),:])
#     mean_max_z = np.mean(points_max,0)
#     mean_max_z_index = S_contour_points_new.shape[0]+1
#     # make faces 
#     faces_max = []
#     
#     for n in range(241*(max_z-min_z),241*(max_z-min_z+1)+1): 
#         if n<(241*(max_z-min_z+1)):
#             faces_max.append([n,n+1,mean_max_z_index])
#         else:
#             faces_max.append([241*(max_z-min_z+1),241*(max_z-min_z),mean_max_z_index])
#     faces_max = np.asarray(faces_max)    
    
def get_ACDC_gt_mesh_and_volume(ED_gt, dim, spacing):
    
    # output_directory_temp = r"D:\Deepa\projects\reg_3D\output_ACDC\patient001"
        
    ###############################
    ### Create the updated mesh ###
    ###############################
    
    # S_contour_points_temp, S_contour_points_min_z, S_contour_points_max_z, S_contour_points = get_ACDC_contour_with_endpoints(ED_gt, dim, spacing) 
    # S_contour_points_temp, S_contour_points_min_z, S_contour_points_max_z, S_contour_points_extra_base = get_ACDC_contour_with_endpoints(ED_gt, dim, spacing) 

    # S_contour_points_for_mesh, S_contour_points_min_z, S_contour_points_max_z, S_contour_points_temp = get_ACDC_contour_with_endpoints(ED_gt, dim, spacing) 

    S_contour_points_temp, S_contour_points_min_z, S_contour_points_max_z, S_contour_points_extra_base = get_ACDC_contour_with_endpoints(ED_gt, dim, spacing) 

    
    
    ####################### For ED_gt_mesh ########################
    
    # print ('ED gt mesh')
    
    # print ('S_contour_points_temp: ' + str(S_contour_points_temp.shape))

    
    ### Get the min and max points ###  
    S_contour_points_new = [] 
    
    min_z = np.int32(np.min(S_contour_points_temp[:,2]))
    max_z = np.int32(np.max(S_contour_points_temp[:,2]))
    # print ('min_z: ' + str(min_z))
    # print ('max_z: ' + str(max_z))

    ### Get the ordered contours ### 
    for n in range(min_z,max_z):
        # print ('n: ' + str(n))
        ctr1 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n),:])
        ctr2 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n+1),:])
        ctr2_new = find_contour_index_association2(ctr1, ctr2)
        if (n==min_z):
            S_contour_points_new.append(ctr1)
        S_contour_points_new.append(ctr2_new)
    S_contour_points_new = np.asarray(S_contour_points_new)
    S_contour_points_new = np.concatenate(S_contour_points_new, 0)
    
    ### Create faces ### 
    num_contour_pairs = max_z-min_z
    num_points = 241 
    # Get faces 
    faces = create_faces(num_contour_pairs, num_points)
    # Get min faces 
    # faces_min = create_faces_min(S_contour_points_temp, S_contour_points_new, min_z, num_points)
    faces_min = create_faces_min(S_contour_points_new, min_z, num_points)

    
    # Get max faces     
    # faces_max = create_faces_max(S_contour_points_temp, S_contour_points_new, min_z, max_z, num_points)
    faces_max = create_faces_max(S_contour_points_new, min_z, max_z, num_points)


#     points_min = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==min_z),:])
#     mean_min_z = np.mean(points_min,0)
#     points_max = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==max_z),:])
#     mean_max_z = np.mean(points_max,0)
    points_min = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==min_z),:])
    mean_min_z = np.mean(points_min,0)
    points_max = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==max_z),:])
    mean_max_z = np.mean(points_max,0)

    # Add the faces and the points to the previous mesh, and save
    mean_min_z = np.transpose(np.expand_dims(mean_min_z,1))
    mean_max_z = np.transpose(np.expand_dims(mean_max_z,1))
    S_contour_points_new = np.asarray(np.concatenate((S_contour_points_new, mean_min_z, mean_max_z)))
     
    faces_temp = np.asarray(np.concatenate((faces, faces_min, faces_max)))
    # delete last face
    faces = faces_temp[0:-2,:]
        
    # multiply contour points by spacing
    S_contour_points_new[:,0] = S_contour_points_new[:,0] * spacing[0]
    S_contour_points_new[:,1] = S_contour_points_new[:,1] * spacing[1]
    S_contour_points_new[:,2] = S_contour_points_new[:,2] * spacing[2]
    # switch x and y points?!
    S_contour_points_new = np.transpose(np.vstack((S_contour_points_new[:,1], S_contour_points_new[:,0], S_contour_points_new[:,2])))
     
    S_mesh_new = form_mesh_from_vertices_and_faces(S_contour_points_new, faces)
    
    # print ('faces: ' + str(faces))
    
    
    # print ('S_contour_points_new: ' + str(S_contour_points_new.shape))
    # print ('faces: ' + str(faces.shape))
    # print ('S_mesh_new: ' + str(S_mesh_new))
    # print ('dim: ' + str(dim))
    # print ('spacing: ' + str(spacing))
    
    
    ####################### For ED_gt_mesh_for_volume ########################
    
    # print ('ED gt mesh for volume')
    del S_contour_points_temp
    # Add points to S_contour_points_temp for the ED_gt_mesh_for_volume
    S_contour_points_temp = S_contour_points_extra_base
    # print ('S_contour_points_temp: ' + str(S_contour_points_temp.shape))
    
    ### Get the min and max points ###  
    S_contour_points_new = [] 
        
    min_z = np.int32(np.min(S_contour_points_temp[:,2]))
    max_z = np.int32(np.max(S_contour_points_temp[:,2]))
    # print ('min_z: ' + str(min_z))
    # print ('max_z: ' + str(max_z))

    ### Get the ordered contours ### 
    for n in range(min_z,max_z):
        # print ('n: ' + str(n))
        ctr1 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n),:])
        ctr2 = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==n+1),:])
        ctr2_new = find_contour_index_association2(ctr1, ctr2)
        if (n==min_z):
            S_contour_points_new.append(ctr1)
        S_contour_points_new.append(ctr2_new)
    S_contour_points_new = np.asarray(S_contour_points_new)
    S_contour_points_new = np.concatenate(S_contour_points_new, 0)
    
    ### Create faces ### 
    num_contour_pairs = max_z-min_z
    num_points = 241 
    # Get faces 
    faces = create_faces(num_contour_pairs, num_points)
    # Get min faces 
    # faces_min = create_faces_min(S_contour_points_temp, S_contour_points_new, min_z, num_points)
    faces_min = create_faces_min(S_contour_points_new, min_z, num_points)

    
    # Get max faces     
    # faces_max = create_faces_max(S_contour_points_temp, S_contour_points_new, min_z, max_z, num_points)
    faces_max = create_faces_max(S_contour_points_new, min_z, max_z, num_points)


#     points_min = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==min_z),:])
#     mean_min_z = np.mean(points_min,0)
#     points_max = np.squeeze(S_contour_points_temp[np.where(S_contour_points_temp[:,2]==max_z),:])
#     mean_max_z = np.mean(points_max,0)
    points_min = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==min_z),:])
    mean_min_z = np.mean(points_min,0)
    points_max = np.squeeze(S_contour_points_new[np.where(S_contour_points_new[:,2]==max_z),:])
    mean_max_z = np.mean(points_max,0)

    # Add the faces and the points to the previous mesh, and save
    mean_min_z = np.transpose(np.expand_dims(mean_min_z,1))
    
    ####### CHANGED ###### 
    # mean_max_z = np.transpose(np.expand_dims(mean_max_z,1))
    mean_max_z = np.transpose(np.expand_dims(mean_max_z,1)) + 0.1
    # mean_max_z = np.transpose(np.expand_dims(mean_max_z,1)) + 10.0

    
    
    S_contour_points_new = np.asarray(np.concatenate((S_contour_points_new, mean_min_z, mean_max_z)))
    # print ('S_contour_points_new: ' + str(S_contour_points_new))
    
    
    
    
     
    faces_temp = np.asarray(np.concatenate((faces, faces_min, faces_max)))
    # faces = np.asarray(np.concatenate((faces, faces_min)))
    # faces_temp = np.asarray(faces)
    # print ('faces_temp: ' + str(faces_temp))
    # delete last face
    faces = faces_temp[0:-2,:]
    # print ('faces: ' + str(faces))
        
    # multiply contour points by spacing
    S_contour_points_new[:,0] = S_contour_points_new[:,0] * spacing[0]
    S_contour_points_new[:,1] = S_contour_points_new[:,1] * spacing[1]
    S_contour_points_new[:,2] = S_contour_points_new[:,2] * spacing[2]
    # switch x and y points?!
    S_contour_points_new = np.transpose(np.vstack((S_contour_points_new[:,1], S_contour_points_new[:,0], S_contour_points_new[:,2])))
    
    
#     ### Plot the endo contours ### 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(S_contour_points_new[:,0], S_contour_points_new[:,1], S_contour_points_new[:,2], c='r', marker='o')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
    
    
     
    S_mesh_new2 = form_mesh_from_vertices_and_faces(S_contour_points_new, faces)
    
    
    # print ('S_contour_points_new: ' + str(S_contour_points_new.shape))
    # print ('faces: ' + str(faces.shape))
    # print ('S_mesh_new: ' + str(S_mesh_new))
    # print ('dim: ' + str(dim))
    # print ('spacing: ' + str(spacing))
    
    
    ########################################################
    
    
    
    
    ### Convert the mesh to a mask binary volume ### 
    S_mask_new2 = convert_mesh_to_mask(S_mesh_new2, dim, spacing)
    # print ('S_mask_new2: ' + str(S_mask_new2.shape))
        
    ### Final output ###    
    ED_gt_mesh = S_mesh_new
    ED_gt_mesh_for_volume = S_mesh_new2 
    ED_gt_volume = S_mask_new2     
    
    
    
    ### Create the updated volume ### 

    
    return ED_gt_mesh, ED_gt_mesh_for_volume, ED_gt_volume 




# https://stackoverflow.com/questions/5491913/sorting-list-in-python
def sort_nicely( l ): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    l.sort( key=alphanum_key )

    return l 


def mkVtkIdList(it):
    '''Make vtk id list''' 
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil

def form_mesh_from_vertices_and_faces(vertices, faces):
    """Form the mesh polydata structure from arrays of vertices and faces"""
    
    # vertices = self.vertices
    # faces = self.faces 
    
    # set vertices
    mesh_vertices_vtk = vtk.vtkPoints()
    num_pts = vertices.shape[0]
    for n in range(0,num_pts):
        p = vertices[n,:]
        mesh_vertices_vtk.InsertPoint(n,tuple(p))
    mesh_vertices_vtk.SetNumberOfPoints(num_pts)
    
    # set faces
    mesh_faces_vtk = vtk.vtkCellArray()
    num_tri = faces.shape[0]
    for n in range(0,num_tri):
        tri_p = faces[n,:]
        mesh_faces_vtk.InsertNextCell(mkVtkIdList(tuple(tri_p)))
    mesh_faces_vtk.SetNumberOfCells(num_tri)
    
    # create the polydata mesh 
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(mesh_vertices_vtk)
    mesh.SetPolys(mesh_faces_vtk)
    # mesh.SetStrips(mesh_faces_vtk)
            
    # self.LV_mesh = mesh 
    
    return mesh 


def get_points_from_mesh(mesh):
    '''Get an array of points from mesh polydata'''
    
    points_vtk = mesh.GetPoints()
    num_pts = points_vtk.GetNumberOfPoints()
    points = [] 
    for n in range(0,num_pts):
        p = points_vtk.GetPoint(n)
        points.append(p)
    points = np.asarray(points)    
    
    return points 

def get_faces_from_mesh(mesh):
    '''Get an array of faces from mesh polydata'''
    
    num_faces = mesh.GetNumberOfCells()
    faces = [] 
    for nn in range(0,num_faces):
        tri = mesh.GetCell(nn)
        faces.append([tri.GetPointIds().GetId(0), tri.GetPointIds().GetId(1), tri.GetPointIds().GetId(2)])
    faces = np.asarray(faces)
    
    return faces 

# copied from common_functions.py from Maz_LV_4D_seg 
def convert_mesh_to_mask(mesh, mesh_dim, mesh_spacing):
    '''Converts a mesh to a filled binary volume mask'''
    
    # get stencil of mesh polydata 
    imageStencil = vtk.vtkPolyDataToImageStencil()
    # imageStencil.SetInput(mesh)
    imageStencil.SetInputData(mesh)
    
    imageStencil.SetOutputOrigin(0.0, 0.0, 0.0)
    imageStencil.SetOutputSpacing(mesh_spacing[0], mesh_spacing[1], mesh_spacing[2])
    imageStencil.SetOutputWholeExtent(0, mesh_dim[0]-1, 0, mesh_dim[1]-1, 0, mesh_dim[2]-1)
    imageStencil.Update()
    
    # convert stencil to image 
    imageStencilToImage = vtk.vtkImageStencilToImage()
    # imageStencilToImage.SetInput(imageStencil.GetOutput())
    imageStencilToImage.SetInputData(imageStencil.GetOutput())
    imageStencilToImage.SetInsideValue(255)
    imageStencilToImage.Update() 
    
    # set to image data 
    imageData = vtk.vtkImageData()
    imageData = imageStencilToImage.GetOutput()
    # imageData.Update()
    
#     # get number of voxels
#     imageAccum = vtk.vtkImageAccumulate() 
#     imageAccum.SetStencil(imageStencil.GetOutput())
#     imageAccum.SetInput(imageData)
#     imageAccum.Update()
    
    # convert imageData vtk to mask numpy array 
    vtk_data = imageData.GetPointData().GetScalars() 
    mask = vn.vtk_to_numpy(vtk_data)
    
    mask2 = np.zeros((mesh_dim[0], mesh_dim[1], mesh_dim[2]))
    mask2 = np.reshape(mask, mesh_dim, order="F")
    
    return mask2  