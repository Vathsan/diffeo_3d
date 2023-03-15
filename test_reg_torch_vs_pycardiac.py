# test_reg_torch_vs_pycardiac.py
#
# This uses the ACDC dataset to perform pairwise registration and compare numpy and pytorch. The displacement fields,
# and the warped image are saved and compared. I would say that if the  min and max differences for each voxel
# are between  -10^-5 to +10^-5, then the difference is acceptable. The summed difference might be on the order of
# 10^-1 or 10^-2, but I believe that this is fine.
#
# Note to self:
#   copied pycardiac code into numpy
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################

import torch
import torch.nn.functional as nnf

print ('imported torch: ' + str(torch.__version__))

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy

from src.np.reg_3d_pycardiac import numpy_registration_3d as register_pair_3d_np
from src.pytorch.reg_3d_pycardiac import register_pair_3d as register_pair_3d_py

from src.np.gridgen_3d_pycardiac import mygriddata_3d as mygriddata_3d_np
from src.pytorch.gridgen_3d_pycardiac import mygriddata_3d as mygriddata_3d_py

########################################################################################################################
# diff_posx: 0.01783373737311833 min: -7.576082364835202e-05 max: 7.77626650858565e-05
# diff_posy: -0.02279517589939084 min: -5.4335576280095665e-05 max: 5.5725859866684324e-05
# diff_posz: -0.0021013141896911725 min: -7.43926294344277e-06 max: 8.375315477593404e-06
# diff_imw: 0.10375906065821372 min: -0.011600110350570958 max: 0.010314755619901916

########################################################################################################################

def crop_center(im, crx=64, cry=64):
    """Center crop the image"""
    y, x = im.shape[-2:]
    sx = x // 2 - (crx // 2)
    sy = y // 2 - (cry // 2)
    return im[..., sy : sy + cry, sx : sx + crx]

def crop_center_3d(im, crx=64, cry=64, crz=7):
    """Center crop the image"""
    x, y, z = im.shape[0:3]
    sx = x // 2 - (crx // 2)
    sy = y // 2 - (cry // 2)
    sz = z // 2 - (crz // 2)
    return im[sx:sx+crx, sy:sy+cry, sz:sz+crz, :]

########################################################################################################################

main_directory = r"/home/srivathsan/PycharmProjects/diffeo_3d/output"

output_directory = os.path.join(main_directory, 'echoFusion')

# if not os.path.isdir(main_directory):
#     os.mkdir(main_directory)
if not os.path.isdir(output_directory):
	os.mkdir(output_directory)

########################################################################################################################

frameA = 0 # ED
frameB = 1 # approx ES

# Download patient001 data from ACDC and place it in /data/external/ACDC/"""
# f_im = r"data/external/ACDC/patient001_4d.nii.gz"
f_im = r"/home/srivathsan/Documents/EchoFusion/01/nifti/1.nii.gz"

im4d = nib.load(f_im).get_fdata().astype(np.uint16)
print ('im4d: ' + str(im4d.shape)) # (216, 256, 10, 30)

szy, szx, szz, nfr = im4d.shape
ims = im4d

ims = crop_center_3d(ims, 128, 96, 7)
print('ims: ' + str(ims.shape))
szx, szy, szz, nfr = ims.shape
print ('szx: ' + str(szx) + ' szy: ' + str(szy) + ' szz: ' + str(szz))

img = nib.Nifti1Image(im4d, np.eye(4))
output_filename = os.path.join(output_directory, 'im4d.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(ims, np.eye(4))
output_filename = os.path.join(output_directory, 'ims.nii')
nib.save(img, output_filename)

########################################################################################################################

# define registration parameters
class RegParam:
    """Define registration parameters"""

    def __init__(
            self,
            # mx_iter=20.0,
            mx_iter = 10.0,
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

prm = RegParam()

#####################
### Numpy version ###
#####################

im_s = ims[:,:,:,frameA]
im_t = ims[:,:,:,frameB]

im_s = np.float32(im_s)
im_t = np.float32(im_t)
print ('im_s: ' + str(im_s.dtype))
print ('im_y: ' + str(im_t.dtype))

# Register from im_s to im_t
posx_np, posy_np, posz_np, f1c, f2c, f3c, f4c, smeasure_new_list_np, smeasure_list_np, tstep_list_np = \
	register_pair_3d_np(im_s, im_t, prm)

pos_np = np.zeros((szx, szy, szz, 3))
pos_np[:,:,:,0] = posx_np
pos_np[:,:,:,1] = posy_np
pos_np[:,:,:,2] = posz_np

# Get the inverse pos and apply to get a warped image
m, n, o = szx, szy, szz
xI, yI, zI = scipy.mgrid[1:m + 1, 1:n + 1, 1:o + 1]
dispx_np, dispy_np, dispz_np = posx_np - xI, posy_np - yI, posz_np - zI
posx_inv_np, posy_inv_np, posz_inv_np = -dispx_np + xI, -dispy_np + yI, -dispz_np + zI
imw_np = mygriddata_3d_np(posx_inv_np, posy_inv_np, posz_inv_np, im_s)

########################
### Pytorch version ####
########################

pos_py, fc, smeasure_new_list_py, smeasure_list_py, tstep_list_py = register_pair_3d_py(im_s, im_t, prm)

# smeasure_new_list_py = np.squeeze(np.concatenate(smeasure_new_list_py, axis=0))
# smeasure_list_py = np.squeeze(np.concatenate(smeasure_list_py, axis=0))
# tstep_list_py = np.squeeze(np.concatenate(tstep_list_py, axis=0))

smeasure_new_list_py = np.asarray(smeasure_new_list_py)  # .numpy())
smeasure_new_list_py = smeasure_new_list_py.tolist()
smeasure_list_py = np.asarray(smeasure_list_py)  # .numpy())
smeasure_list_py = smeasure_list_py.tolist()
tstep_list_py = np.asarray(tstep_list_py)  # .numpy())
tstep_list_py = tstep_list_py.tolist()

print ('pos_py: ' + str(pos_py.shape))
posx_py = np.squeeze(pos_py[0,0,:,:,:])
posy_py = np.squeeze(pos_py[0,1,:,:,:])
posz_py = np.squeeze(pos_py[0,2,:,:,:])
# Rescale the deformation fields from pytorch
pos_py_rescaled = np.zeros((szx, szy, szz, 3))
pos_py_rescaled[:, :, :, 0] = ((szx - 1) * (posx_py + 1) / 2) + 1
pos_py_rescaled[:, :, :, 1] = ((szy - 1) * (posy_py + 1) / 2) + 1
pos_py_rescaled[:, :, :, 2] = ((szz - 1) * (posz_py + 1) / 2) + 1

posx_rescaled_py = pos_py_rescaled[:,:,:,0]
posy_rescaled_py = pos_py_rescaled[:,:,:,1]
posz_rescaled_py = pos_py_rescaled[:,:,:,2]

print ('pos_np: ' + str(pos_np.shape))
print ('pos_py_rescaled: ' + str(pos_py_rescaled.shape))

# check the correlation between the two, even if the values aren't the same.
pos_np_flatten = pos_np.flatten()
pos_py_rescaled_flatten = pos_py_rescaled.flatten()
corr_coef = np.corrcoef(pos_np_flatten, pos_py_rescaled_flatten)
print('corr_coef: ' + str(corr_coef[0, 1]))

# Get the inverse pos and apply to get a warped image
m, n, o = szx, szy, szz
xI, yI, zI = scipy.mgrid[1:m + 1, 1:n + 1, 1:o + 1]
dispx_py, dispy_py, dispz_py = posx_rescaled_py - xI, posy_rescaled_py - yI, posz_rescaled_py - zI
posx_rescaled_inv_py, posy_rescaled_inv_py, posz_rescaled_inv_py = -dispx_py + xI, -dispy_py + yI, -dispz_py + zI
# use the numpy function to warp.
imw_py = mygriddata_3d_np(posx_rescaled_inv_py, posy_rescaled_inv_py, posz_rescaled_inv_py, im_s)

##############
### Plot #####
##############

print ('smeasure_new_list_np: ' + str(smeasure_new_list_np))
print ('smeasure_new_list_py: ' + str(smeasure_new_list_py))

print ('smeasure_list_np: ' + str(smeasure_list_np))
print ('smeasure_list_py: ' + str(smeasure_list_py))

print ('tstep_list_np: ' + str(tstep_list_np))
print ('tstep_list_py: ' + str(tstep_list_py))

### Plot ###
plt.figure()
plt.subplot(2,2,1)
plt.plot(smeasure_list_np, 'r')
plt.plot(smeasure_new_list_np, 'g')
plt.legend(['smeasure', 'smeasure_new'])
plt.subplot(2,2,3)
plt.plot(tstep_list_np, 'b')
plt.legend(['tstep'])
plt.title('np')
plt.subplot(2,2,2)
plt.plot(smeasure_list_py, 'r')
plt.plot(smeasure_new_list_py, 'g')
plt.legend(['smeasure', 'smeasure_new'])
plt.subplot(2,2,4)
plt.plot(tstep_list_py, 'b')
plt.legend(['tstep'])
plt.title('py')
plt.savefig(os.path.join(output_directory,'smeasure.png'))

#############################
### Calculate differences ###
#############################

diff_posx = np.sum(posx_np-posx_rescaled_py)
diff_posy = np.sum(posy_np-posy_rescaled_py)
diff_posz = np.sum(posz_np-posz_rescaled_py)
diff_imw = np.sum(imw_np-imw_py)
print ('diff_posx: ' + str(diff_posx) + ' min: ' + str(np.min(posx_np-posx_rescaled_py)) + ' max: ' + str(np.max(posx_np-posx_rescaled_py)))
print ('diff_posy: ' + str(diff_posy) + ' min: ' + str(np.min(posy_np-posy_rescaled_py)) + ' max: ' + str(np.max(posy_np-posy_rescaled_py)))
print ('diff_posz: ' + str(diff_posz) + ' min: ' + str(np.min(posz_np-posz_rescaled_py)) + ' max: ' + str(np.max(posz_np-posz_rescaled_py)))
print ('diff_imw: ' + str(diff_imw) + ' min: ' + str(np.min(imw_np-imw_py)) + ' max: ' + str(np.max(imw_np-imw_py)))


### Get the correlation
corr_posx = np.corrcoef(posx_np.flatten(), posx_py.flatten())
corr_posy = np.corrcoef(posy_np.flatten(), posy_py.flatten())
corr_posz = np.corrcoef(posz_np.flatten(), posz_py.flatten())
corr_imw = np.corrcoef(imw_np.flatten(), imw_py.flatten())
print ('corr_posx: ' + str(corr_posx))
print ('corr_posy: ' + str(corr_posy))
print ('corr_posz: ' + str(corr_posz))
print ('corr_imw: ' + str(corr_imw))



#######################
### Save out niftis ###
#######################

img = nib.Nifti1Image(im_s, np.eye(4))
output_filename = os.path.join(output_directory, 'im_s.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(im_t, np.eye(4))
output_filename = os.path.join(output_directory, 'im_t.nii')
nib.save(img, output_filename)

### Numpy ###

img = nib.Nifti1Image(posx_np, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_np, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_np, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_np, np.eye(4))
output_filename = os.path.join(output_directory, 'im_warped_np.nii')
nib.save(img, output_filename)

### Pytorch ###

img = nib.Nifti1Image(posx_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_py, np.eye(4))
output_filename = os.path.join(output_directory, 'im_warped_py.nii')
nib.save(img, output_filename)

### Save out differences ###

img = nib.Nifti1Image(posx_np-posx_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_np-posy_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_np-posz_rescaled_py, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_np-imw_py, np.eye(4))
output_filename = os.path.join(output_directory, 'im_warped_np_minus_py.nii')
nib.save(img, output_filename)
