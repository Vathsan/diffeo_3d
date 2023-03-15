# run_ACDC_Slicer_Elastix_3D.py 
# 
# So far we have been running Elastix in MATLAB. Simple ITK Elastix has not given the 
# correct results for Ameneh and me. Therefore, this script (to be run on command line)
# is to run Slicer's Elastix. 
# 
# To run on command line in windows: 
# 
# C:\Users\deepa\AppData\Local\NA-MIC\Slicer 5.1.0-2022-05-24\Slicer.exe --no-main-window --python-script G:\My Drive\git\diffeo_3d\run_ACDC_Slicer_Elastix_3D.py
#
# "C:\Users\deepa\AppData\Local\NA-MIC\Slicer 5.1.0-2022-05-24\Slicer.exe" --no-main-window --python-script "G:\My Drive\git\diffeo_3d\run_ACDC_Slicer_Elastix_3D.py"
# 
# Deepa Krishnaswamy
# deepa@ualberta.ca
# July 10 2022
#####################################################################################################

import os 
import sys
import numpy as np
import time 
import scipy 
import scipy.io as sio
import re 
import json 
import SimpleITK as sitk 
import glob 

import slicer
from slicer.ScriptedLoadableModule import *
from Elastix import ElastixLogic

os.path.dirname(sys.executable)

# slicer.util.pip_install("jupyter ipywidgets pandas ipyevents ipycanvas --no-warn-script-location")
slicer.util.pip_install("nibabel")
slicer.util.pip_install("pynrrd")
slicer.util.pip_install("scikit-image")

import nibabel as nib
import nrrd
nrrd.reader.ALLOW_DUPLICATE_FIELD = True
import skimage
from skimage import measure

print ('imported packages for Slicer')

#####################################################
### Functions ###
#####################################################

def get_ACDC_frame_numbers(patient_dir):
  '''Gets the frame numbers from the Info cfg file''' 
              
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

  ### Get the ES frame ###
  ES = str(data[1]) 
  # ES_frame = np.int16(ES[4:]) - 1 
  # ES_frame = np.int16(ES[6:]) - 1 
  ES_frame = int(re.search(r'\d+', ES).group()) - 1

  return ED_frame, ES_frame


def get_ACDC_ED_and_ES_filenames(patient_dir, ED_frame, ES_frame): 
  ''' Gets the name of the ED and ES image files and the gt files'''

  patient_name = os.path.basename(patient_dir)

  ### Get the ED filename ###
  if (ED_frame < 9): 
    ED_filename = os.path.join(patient_dir, patient_name + '_frame0' + str(ED_frame+1) + '.nii.gz')
    ED_contourfile = os.path.join(patient_dir, patient_name + '_frame0' + str(ED_frame+1) + '_gt.nii.gz')
  else: 
    ED_filename = os.path.join(patient_dir, patient_name + '_frame' + str(ED_frame+1) + '.nii.gz')
    ED_contourfile = os.path.join(patient_dir, patient_name + '_frame' + str(ED_frame+1) + '_gt.nii.gz')

  ### Get the ES filename ###
  if (ES_frame < 9): 
    ES_filename = os.path.join(patient_dir, patient_name + '_frame0' + str(ES_frame+1) + '.nii.gz')
    ES_contourfile = os.path.join(patient_dir, patient_name + '_frame0' + str(ES_frame+1) + '_gt.nii.gz')
  else: 
    ES_filename = os.path.join(patient_dir, patient_name + '_frame' + str(ES_frame+1) + '.nii.gz')
    ES_contourfile = os.path.join(patient_dir, patient_name + '_frame' + str(ES_frame+1) + '_gt.nii.gz')

  return ED_filename, ES_filename, ED_contourfile, ES_contourfile


def get_ACDC_data(ED_contourfile, contour_type):
  """Returns the binary mask of the segment we want""" 

  # Load the ED file 
  nii = nib.load(ED_contourfile)
  ED_gt_temp = nii.get_data()

  # keep where ED_gt = contour_type 
  ED_gt = np.zeros((ED_gt_temp.shape))
  ED_gt[np.where(ED_gt_temp==contour_type)] = 1

  return ED_gt 

# include code for 2D and 3D warping linear interp and nearest neighbor 

def get_slice_indices(ED_gt):

  """Get the list of slices that contain a label"""
  slice_indices = [] 
  num_slices = ED_gt.shape[2]
  for n in range(0,num_slices):
    img_slice = ED_gt[:,:,n]
    if (img_slice.any()):
      slice_indices.append(n)

  return slice_indices 

def perform_elastix_registration_3D(S,T):
  """Registers volumes moving S to fixed T and returns the warped volume along with 
    the displacement field and the corresponding transform node"""
    
  S = np.swapaxes(S,0,2)
  T = np.swapaxes(T,0,2)
  
  movingVolumeNode = slicer.util.addVolumeFromArray(S, ijkToRAS=np.diag([1.0, 1.0, 1.0, 1.0]), name=None, nodeClassName=None)
  fixedVolumeNode = slicer.util.addVolumeFromArray(T, ijkToRAS=np.diag([1.0, 1.0, 1.0, 1.0]), name=None, nodeClassName=None)
  
  logic = ElastixLogic()
  
  # Just B spline
  RegistrationPresets_ParameterFilenames = 5
  parameterFilenames = ['Parameters_BSpline.txt']
  
  print ('parameterFilenames: ' + str(parameterFilenames))
  
  VolumeNodename = 'Volume_{:02d}'.format(0)
  TransformNodename = 'Transform_{:02d}'.format(0)
  
  outputVolume = slicer.vtkMRMLScalarVolumeNode()
  slicer.mrmlScene.AddNode(outputVolume)
  outputVolume.CreateDefaultDisplayNodes()
  outputVolume.SetName(VolumeNodename)
  
  outputTransform = slicer.vtkMRMLTransformNode()
  slicer.mrmlScene.AddNode(outputTransform)
  outputTransform.CreateDefaultDisplayNodes()
  outputTransform.SetName(TransformNodename)
  
  
  logic.registerVolumes(fixedVolumeNode, \
                        movingVolumeNode, \
                        parameterFilenames = parameterFilenames, \
                        outputVolumeNode = outputVolume, \
                        outputTransformNode = outputTransform, \
                        forceDisplacementFieldOutputTransform = True)
  
  # Get the S_3D_warped from the OutputVolume 
  S_3D_warped = slicer.util.arrayFromVolume(outputVolume)
  
  # Get the dispx, dispy, dispz from the outputTransform 
  disp = slicer.util.arrayFromGridTransform(outputTransform)
  dispx, dispy, dispz = disp[:,:,:,0], disp[:,:,:,1], disp[:,:,:,2]
  
  # swap back 
  S_3D_warped = np.swapaxes(S_3D_warped,0,2)
  dispx = np.swapaxes(dispx,0,2)
  dispy = np.swapaxes(dispy,0,2)
  dispz = np.swapaxes(dispz,0,2)
  
  return S_3D_warped, dispx, dispy, dispz, outputTransform 


def perform_elastix_registration_2D(S,T):
  """Registers volumes moving S to fixed T and returns the warped volume along with 
    the displacement field and the corresponding transform node"""
    
  S = np.swapaxes(S,0,1)
  T = np.swapaxes(T,0,1)
  S = np.expand_dims(S,2)
  T = np.expand_dims(T,2)
  
  movingVolumeNode = slicer.util.addVolumeFromArray(S, ijkToRAS=np.diag([1.0, 1.0, 1.0, 1.0]), name=None, nodeClassName=None)
  fixedVolumeNode = slicer.util.addVolumeFromArray(T, ijkToRAS=np.diag([1.0, 1.0, 1.0, 1.0]), name=None, nodeClassName=None)
  
  logic = ElastixLogic()
  
  # Just B spline
  RegistrationPresets_ParameterFilenames = 5
  parameterFilenames = ['Parameters_BSpline.txt']
  # parameterFilenames = ['Bsplines_parameter-file.txt']
  
  print ('parameterFilenames: ' + str(parameterFilenames))
  
  VolumeNodename = 'Volume_{:02d}'.format(0)
  TransformNodename = 'Transform_{:02d}'.format(0)
  
  outputVolume = slicer.vtkMRMLScalarVolumeNode()
  slicer.mrmlScene.AddNode(outputVolume)
  outputVolume.CreateDefaultDisplayNodes()
  outputVolume.SetName(VolumeNodename)
  
  outputTransform = slicer.vtkMRMLTransformNode()
  slicer.mrmlScene.AddNode(outputTransform)
  outputTransform.CreateDefaultDisplayNodes()
  outputTransform.SetName(TransformNodename)
  
  
  logic.registerVolumes(fixedVolumeNode, \
                        movingVolumeNode, \
                        parameterFilenames = parameterFilenames, \
                        outputVolumeNode = outputVolume, \
                        outputTransformNode = outputTransform, \
                        forceDisplacementFieldOutputTransform = True)
  
  # Get the S_3D_warped from the OutputVolume 
  S_2D_warped = slicer.util.arrayFromVolume(outputVolume)
  
  # Get the dispx, dispy, dispz from the outputTransform 
  disp = slicer.util.arrayFromGridTransform(outputTransform)
  print('disp: ' + str(disp.shape))
  dispx, dispy = disp[:,:,0,0], disp[:,:,0,1]
  
  # swap back 
  S_2D_warped = np.swapaxes(S_2D_warped,0,1)
  dispx = np.swapaxes(dispx,0,1)
  dispy = np.swapaxes(dispy,0,1)
  
  return S_2D_warped, dispx, dispy, outputTransform 
  
def warp_volume(outputTransform, input_volume):
  """ Applies the transform to the input volume and returns a warped volume"""
  
  inputVolumeNode = slicer.util.addVolumeFromArray(input_volume, ijkToRAS=np.diag([1.0, 1.0, 1.0, 1.0]), name=None, nodeClassName=None)
  # inputVolumeNode.SetAndObserveTransformNodeID(outputTransform.GetID())
  
  # inputVolumeNode.ApplyTransform(outputTransform)

  # resampled_volume = slicer.util.arrayFromVolume(inputVolumeNode)
  
  transformedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
  resampled_volume = slicer.util.arrayFromVolume(transformedVolumeNode)

  return resampled_volume 

##################################################
### Inputs ###
##################################################

# ACDC_directory = '/content/gdrive/My Drive/Colab Notebooks/Ameneh/ACDC'
ACDC_directory = r"G:\My Drive\Colab Notebooks\Ameneh\ACDC"
output_directory_3D = r"C:\Users\deepa\deepa\UofA\Ameneh\output_ACDC_Slicer_Elastix_3D"
output_directory_2D = r"C:\Users\deepa\deepa\UofA\Ameneh\output_ACDC_Slicer_Elastix_2D"

contour_type = 3 # LV endocardium 
perform_registration_3D = 1
perform_registration_2D = 1
save_nifti = 1 # set to 1 if you want to save the images/masks as nifti for a quick check.
save_npz = 1

##################################################
### Registration ### 
##################################################

if not os.path.isdir(output_directory_3D):
  os.mkdir(output_directory_3D)
if not os.path.isdir(output_directory_2D):
  os.mkdir(output_directory_2D)

patient_names = os.listdir(ACDC_directory)
num_patients = len(patient_names)


# for patient in range(0,num_patients):
for patient in range(0,1):

  #--- Set patient name ---#
  patient_name = patient_names[patient]

  #--- Set the folder name ---#
  patient_directory = os.path.join(ACDC_directory, patient_name)

  #--- Load the Info.cfg file and get the ED and ES frame numbers ---#
  ED_frame, ES_frame = get_ACDC_frame_numbers(patient_directory)

  #--- Get the ED and ES filenames for the images and ground truth ---#
  ED_filename, ES_filename, ED_contourfile, ES_contourfile = get_ACDC_ED_and_ES_filenames(patient_directory, ED_frame, ES_frame)

  #--- Load the image files ---#
  ED_nii = nib.load(ED_filename)
  ED_img = ED_nii.get_fdata()
  ES_nii = nib.load(ES_filename)
  ES_img = ES_nii.get_fdata()

  #--- Get the ground truth masks ---#
  ED_gt = get_ACDC_data(ED_contourfile, contour_type)
  ES_gt = get_ACDC_data(ES_contourfile, contour_type)

  ###############################
  ### Perform 3D registration ### 
  ###############################
  if (perform_registration_3D):
    
    output_patient_directory = os.path.join(output_directory_3D, patient_name)
    if not os.path.isdir(output_patient_directory):
      os.mkdir(output_patient_directory)
    
    #--- Perform registration in both directions ---#
    # The deformation field produced is the reverse of the input. 
    # (source, target)
    # (forward = ED to ES, reverse is ES to ED)
    
    ED_img_warped, fx, fy, fz, outputTransformNode_forward = perform_elastix_registration_3D(ED_img, ES_img)
    ES_img_warped, bx, by, bz, outputTransformNode_reverse = perform_elastix_registration_3D(ES_img, ED_img)
    
    disp_field_forward = np.zeros((1,3,fx.shape[0],fx.shape[1],fx.shape[2]))
    disp_field_forward[0,0,:,:,:] = fx 
    disp_field_forward[0,1,:,:,:] = fy 
    disp_field_forward[0,2,:,:,:] = fz 
    
    disp_field_reverse = np.zeros((1,3,fx.shape[0],fx.shape[1],fx.shape[2]))
    disp_field_reverse[0,0,:,:,:] = bx 
    disp_field_reverse[0,1,:,:,:] = by 
    disp_field_reverse[0,2,:,:,:] = bz 
    
    #--- Warp the ED_img and ES_img by the appropriate displacement fields ---# 
    ED_gt_warped = warp_volume(outputTransformNode_forward, ED_gt)
    ES_gt_warped = warp_volume(outputTransformNode_reverse, ES_gt)
  
    #--- Save all as nifti ---# 
    if (save_nifti):
      # ED img warped to ES 
      ED_img_warped_nii = nib.Nifti1Image(ED_img_warped, ED_nii.affine, ED_nii.header)
      output_filename = os.path.join(output_patient_directory, 'ED_img_warped_to_ES.nii')
      nib.save(ED_img_warped_nii, output_filename)
      # ES img warped to ED 
      ES_img_warped_nii = nib.Nifti1Image(ES_img_warped, ED_nii.affine, ED_nii.header)
      output_filename = os.path.join(output_patient_directory, 'ES_img_warped_to_ED.nii')
      nib.save(ES_img_warped_nii, output_filename)
      # ED gt warped to ES 
      ED_gt_warped_nii = nib.Nifti1Image(ED_gt_warped, ED_nii.affine, ED_nii.header)
      output_filename = os.path.join(output_patient_directory, 'ED_gt_warped_to_ES.nii')
      nib.save(ED_gt_warped_nii, output_filename)
      # ES gt warped to ED 
      ES_gt_warped_nii = nib.Nifti1Image(ES_gt_warped, ED_nii.affine, ED_nii.header)
      output_filename = os.path.join(output_patient_directory, 'ES_gt_warped_to_ED.nii')
      nib.save(ES_gt_warped_nii, output_filename)
  
    #--- Save all as npz file ---#
    if (save_npz):
      output_npz_filename = os.path.join(output_patient_directory, patient_name+'_output.npz')
      np.savez(output_npz_filename, 
              ED_img_warped=ED_img_warped,
              ES_img_warped=ES_img_warped,
              ED_gt_warped=ED_gt_warped,
              ES_gt_warped=ES_gt_warped,
              pos_f=disp_field_forward,
              pos_b=disp_field_reverse)
      
  ###############################
  ### Perform 2D registration ###
  ###############################    
  if (perform_registration_2D):
    
    output_patient_directory = os.path.join(output_directory_2D,patient_name)
    if not os.path.isdir(output_patient_directory):
      os.mkdir(output_patient_directory)
    
    #--- Get the slices that contain a label ---#
    slice_indices = get_slice_indices(ED_gt)

    ####################################################
    #--- Pairwise registration for each axial slice ---#
    ####################################################
    # for n in range(0,len(slice_indices)):
    for n in range(0,1):
      
      ED_slice = ED_img[:,:,slice_indices[n]]
      ES_slice = ES_img[:,:,slice_indices[n]]

      ED_gt_slice = ED_gt[:,:,slice_indices[n]]
      ES_gt_slice = ES_gt[:,:,slice_indices[n]]

      #---Perform registration in both direction ---#
      # The deformation field produced is the reverse of the input. 
      # (source, target)
      # (forward = ED to ES, reverse is ES to ED)
  
      ED_slice_warped, fx, fy, outputTransformNode_forward = perform_elastix_registration_2D(ED_slice, ES_slice)
      ES_slice_warped, bx, by, outputTransformNode_reverse = perform_elastix_registration_2D(ES_slice, ED_slice)
      
      disp_field_forward = np.zeros((1,2,fx.shape[0],fx.shape[1]))
      disp_field_forward[0,0,:,:] = fx 
      disp_field_forward[0,1,:,:] = fy 
      
      disp_field_reverse = np.zeros((1,2,fx.shape[0],fx.shape[1]))
      disp_field_reverse[0,0,:,:] = bx 
      disp_field_reverse[0,1,:,:] = by 
      
      #--- Warp the ED_img and ES_img by the appropriate displacement fields ---# 
      ED_gt_slice_temp = np.expand_dims(ED_gt_slice,2)
      ES_gt_slice_temp = np.expand_dims(ES_gt_slice,2)
      ED_gt_slice_warped_temp = warp_volume(outputTransformNode_forward, ED_gt_slice_temp)
      ES_gt_slice_warped_temp = warp_volume(outputTransformNode_reverse, ES_gt_slice_temp)
      ED_gt_slice_warped = np.squeeze(ED_gt_slice_warped_temp)
      ES_gt_slice_warped = np.squeeze(ES_gt_slice_warped_temp)



      #--- Save all as nifti ---# 
      if (save_nifti):
        # ED img warped to ES 
        header_2D = ED_nii.header 
        header_2D['pixdim'][3] = 1 
        ED_slice_warped_nii = nib.Nifti1Image(ED_slice_warped, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ED_img_warped_to_ES.nii')
        nib.save(ED_slice_warped_nii, output_filename)
        # ES img warped to ED 
        ES_slice_warped_nii = nib.Nifti1Image(ES_slice_warped, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ES_img_warped_to_ED.nii')
        nib.save(ES_slice_warped_nii, output_filename)
        # ED gt warped to ES 
        ED_gt_slice_warped_nii = nib.Nifti1Image(ED_gt_slice_warped, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ED_gt_warped_to_ES.nii')
        nib.save(ED_gt_slice_warped_nii, output_filename)
        # ES gt warped to ED 
        ES_gt_slice_warped_nii = nib.Nifti1Image(ES_gt_slice_warped, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ES_gt_warped_to_ED.nii')
        nib.save(ES_gt_slice_warped_nii, output_filename)
        ##### save the gt as well since the original are in 3D #####
        # ED slice 
        ED_slice_nii = nib.Nifti1Image(ED_slice, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ED_img.nii')
        nib.save(ED_slice_nii, output_filename)
        # ES slice 
        ES_slice_nii = nib.Nifti1Image(ES_slice, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ES_img.nii')
        nib.save(ES_slice_nii, output_filename)
        # ED gt 
        ED_gt_slice_nii = nib.Nifti1Image(ED_gt_slice, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ED_gt.nii')
        nib.save(ED_gt_slice_nii, output_filename)
        # ES gt 
        ES_gt_slice_nii = nib.Nifti1Image(ES_gt_slice, ED_nii.affine, header_2D)
        output_filename = os.path.join(output_patient_directory, str(n) + '_' + 'ES_gt.nii')
        nib.save(ES_gt_slice_nii, output_filename)


      #--- Save all as npz file ---#
      if (save_npz):
        output_npz_filename = os.path.join(output_patient_directory, patient_name + '_' + str(n) + '_output.npz')
        np.savez(output_npz_filename, 
                ED_slice_warped=ED_slice_warped,
                ES_slice_warped=ES_slice_warped,
                ED_gt_slice_warped=ED_gt_slice_warped,
                ES_gt_slice_warped=ES_gt_slice_warped,
                pos_f=disp_field_forward,
                pos_b=disp_field_reverse)
    
    
    
    
    
    
    
    
    
    
    
    
    
    


