# test_gridgen_torch_vs_pycardiac.py
#
# This function performs a series of tests to compare the gridgen pytorch vs numpy versions. A grid is created for four
# different cases:
#   1. positive divergence
#   2. negative divergence
#   3. negative divergence and positive curl = radial inwards and rotate to right
#   4. positive divergence and negative curl = radial outwards and rotate to left
#
# The displacement fields are saved out for the original, and the inverse, as well as the warped image. The differences
# are computed between the numpy and pytorch versions. I would say that if the  min and max differences for each voxel
# are between  -10^-5 to +10^-5, then the difference is acceptable. The summed difference might be on the order of
# 10^-1 or 10^-2, but I believe that this is fine.
#
# Note to self:
#   Still using functions from the numpy folder
#   They were mostly copied from pycardiac functions
#
# Deepa Krishnaswamy
# University of Alberta
# July 15 2021
########################################################################################################################

### Output from test with test type = "pos div" ###
# diff_posx: -0.22677157521459945 min: -2.543595210369176e-05 max: 2.5519907410398446e-05
# diff_posy: -0.27728580592621743 min: -3.74063713266537e-05 max: 3.686253086243596e-05
# diff_posz: -0.14843080290606037 min: -5.088088362015242e-05 max: 5.117732948178855e-05
# diff_posx_inv: 0.3397125466560298 min: -2.564259279580483e-05 max: 2.5675183358941922e-05
# diff_posy_inv: 0.06102487080558516 min: -2.140596090072222e-05 max: 2.2601094300966906e-05
# diff_posz_inv: 0.07484414409330897 min: -3.1230681344140976e-05 max: 2.8925633955623198e-05
# diff_imw: 0.04725160048948028 min: -4.4410793674090794e-05 max: 4.8485924555482975e-05

### Output from test with test_type = "neg_div" ###
# diff_posx: -0.10758943710416302 min: -2.383472063571812e-05 max: 2.3811938747120287e-05
# diff_posy: -0.2681397516095403 min: -2.1304678334388427e-05 max: 2.2361717967100958e-05
# diff_posz: -0.16022513437963237 min: -3.1273825214839235e-05 max: 2.9459635399575745e-05
# diff_posx_inv: 0.19316840805877378 min: -2.5669945216577617e-05 max: 2.5731541555273907e-05
# diff_posy_inv: 0.03919129443345737 min: -3.7529605506847474e-05 max: 3.721822532298802e-05
# diff_posz_inv: 0.0664328472596073 min: -5.120685051451801e-05 max: 5.1354006416204356e-05
# diff_imw: -0.06938254677619198 min: -5.003035855323084e-05 max: 4.4477373441287674e-05

### Output from test with test_type = "neg_div_pos_curl" ###
# diff_posx: -0.1076027660731278 min: -2.3823656799981663e-05 max: 2.381200732137767e-05
# diff_posy: -0.2731014720808942 min: -3.5710998533389215e-05 max: 3.573076092633354e-05
# diff_posz: 0.01895551038059584 min: -3.133858622561547e-05 max: 2.941327100813851e-05
# diff_posx_inv: 0.19232261616299373 min: -2.5669076237022637e-05 max: 2.5731554501362552e-05
# diff_posy_inv: 0.03857341242174961 min: -3.7599866175241914e-05 max: 3.7092475302813455e-05
# diff_posz_inv: -0.034363414868389675 min: -4.755838260861456e-05 max: 5.1314823807047105e-05
# diff_imw: -0.0458005674037633 min: -5.127175289409893e-05 max: 5.21905048413312e-05

### Output from test with test_type = "pos_div_neg_curl" ###
# diff_posx: -0.22818272316846544 min: -2.5434671641733075e-05 max: 2.5520672323864346e-05
# diff_posy: -0.35419172440623214 min: -3.718995348123144e-05 max: 3.6829404471916405e-05
# diff_posz: -0.0503940019107445 min: -5.1295509294391195e-05 max: 5.113690032487739e-05
# diff_posx_inv: 0.3400602508595689 min: -2.5642163933525808e-05 max: 2.567501218209145e-05
# diff_posy_inv: 0.1206425155687445 min: -3.7502707534997626e-05 max: 3.641392568454194e-05
# diff_posz_inv: -0.018564744337792538 min: -3.120461671812791e-05 max: 2.867498641023758e-05
# diff_imw: 0.031178891427415215 min: -4.7337528424401835e-05 max: 4.877942160671708e-05

########################################################################################################################

import torch

print ('imported torch: ' + str(torch.__version__))

import os
import nibabel as nib
import numpy as np

from src.pytorch.gridgen_3d_pycardiac import gridgen_3d as gridgen_3d_py
from src.pytorch.gridgen_3d_pycardiac import mygriddata_3d as mygriddata_3d_py
from src.np.gridgen_3d_pycardiac import gridgen_3d as gridgen_3d_np
from src.np.gridgen_3d_pycardiac import mygriddata_3d as mygriddata_3d_np

########################################################################################################################

### curl is in the yz plane ###
test_type = "pos_div"
# test_type = "neg_div"
# test_type = "neg_div_pos_curl" # radial inwards + rotate to the right
# test_type = "pos_div_neg_curl" # radial outwards + rotate to the left

main_directory = r"D:\Deepa\projects\reg_3D_pytorch\test_gridgen_torch_vs_pycardiac"

output_directory = os.path.join(main_directory, test_type)

if not os.path.isdir(main_directory):
    os.mkdir(main_directory)
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

########################################################################################################################

#####################
### Numpy version ###
#####################

print ('*********performing numpy gridgen*********')

im = np.zeros((41, 61, 81))
im[::5, :, :], im[:, ::5, :], im[:, :, ::5] = 1, 1, 1
im[::5, ::5, ::5] = 1

print ('im: ' + str(im.shape))

szx, szy, szz = im.shape

nframes = 5
j_lb = 0.1
j_ub = 4.0

# Generate radial and rotational fields
f11 = np.linspace(1.0, 0.3, nframes)
f12 = np.linspace(1.0, 3.0, nframes)
f21 = np.linspace(0.0, 1.0, nframes)
f22 = np.linspace(0.0, -1.0, nframes)

cz = int(0.1 * min(szx, szy, szz))
# cz = 3 # simple case
print ('cz: ' + str(cz))

# radial field
f1 = np.ones((szx, szy, szz))

# rotational field
f2 = np.zeros((szx, szy, szz))
f3 = np.zeros((szx, szy, szz))
f4 = np.zeros((szx, szy, szz))

posx_np_all = np.zeros((szx, szy, szz, nframes))
posy_np_all = np.zeros((szx, szy, szz, nframes))
posz_np_all = np.zeros((szx, szy, szz, nframes))
posx_inv_np_all = np.zeros((szx, szy, szz, nframes))
posy_inv_np_all = np.zeros((szx, szy, szz, nframes))
posz_inv_np_all = np.zeros((szx, szy, szz, nframes))
imw_np_all = np.zeros((szx, szy, szz, nframes))

for i in range(0,nframes):

    print ('frame: ' + str(i))

    if (test_type=="pos_div"):
        # f12 = positive divergence outwards
        f1[
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f12[i]

    elif (test_type=="neg_div"):
        # f11 = negative divergence inwards
        f1[
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f11[i]

    elif (test_type=="neg_div_pos_curl"):
        # f11 = negative divergence inwards
        f1[
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f11[i]
        # f21 = positive curl to the right
        f2[
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f21[i]

    elif (test_type == "pos_div_neg_curl"):
        # f12 = positive divergence outwards
        f1[
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f12[i]
        # f22 = negative curl to the left
        f2[
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f22[i]


    posx, posy, posz, _, _, _, _ = gridgen_3d_np(f1, f2, f3, f4, j_lb=j_lb, j_ub=j_ub, n_euler=20, inv=False)
    posx_inv, posy_inv, posz_inv, _, _, _, _ = gridgen_3d_np(f1, f2, f3, f4, j_lb=j_lb, j_ub=j_ub, n_euler=20, inv=True)

    imw = mygriddata_3d_np(posx_inv, posy_inv, posz_inv, im)

    posx_np_all[:,:,:,i] = posx
    posy_np_all[:,:,:,i] = posy
    posz_np_all[:,:,:,i] = posz
    posx_inv_np_all[:,:,:,i] = posx_inv
    posy_inv_np_all[:,:,:,i] = posy_inv
    posz_inv_np_all[:,:,:,i] = posz_inv
    imw_np_all[:,:,:,i] = imw



img = nib.Nifti1Image(posx_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posx_inv_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_inv_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_inv_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_np_all, np.eye(4))
output_filename = os.path.join(output_directory, 'imw_np.nii')
nib.save(img, output_filename)

##############################
### Pytorch implementation ###
##############################

print ('*********performing pytorch gridgen*********')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# im_py = torch.zeros((10, 2, 25, 27,29), device=device)
# Create a grid image
# im_py[:, :, ::3, :, :], im_py[:, :, :, ::3, :], im_py[:, :, :, :, ::3] = 1, 1, 1
# im_py[:, :, ::3, ::3, ::3] = 1

im_py = torch.zeros((nframes, 2, 41, 61, 81), device=device)
im_py[:, :, ::5, :, :], im_py[:, :, :, ::5, :], im_py[:, :, :, :, ::5] = 1, 1, 1
im_py[:, :, ::5, ::5, ::5] = 1

bsz, c, szx, szy, szz = im_py.shape

print ('im_py: ' + str(im_py.shape))

### save out original im_py to check ###
im_py_np = im_py.detach().numpy()
im_py_np = np.squeeze(im_py_np[:,0,:,:,:])
im_py_np = np.moveaxis(im_py_np,0,3)
print ('im_py_np: ' + str(im_py_np.shape))
img = nib.Nifti1Image(im_py_np, np.eye(4))
output_filename = os.path.join(output_directory, 'im_py_np.nii')
nib.save(img, output_filename)


# f1 - radial, f2, f3, f4 - rotational
f = torch.ones((bsz, 4, szx, szy, szz), device=device, requires_grad=True)
f[:, 1, :, :, :] = 0
f[:, 2, :, :, :] = 0
f[:, 3, :, :, :] = 0

cz = int(0.1 * min(szx, szy, szz))
# cz = 3 # from the original simple case
print ('cz: ' + str(cz))

for i in range(nframes):

    print('frame: ' + str(i))

    if (test_type=="pos_div"):
        # f12 = positive divergence outwards
        f[i,0,
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f12[i]

    elif (test_type=="neg_div"):
        # f11 = negative divergence inwards
        f[i,0,
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f11[i]

    elif (test_type=="neg_div_pos_curl"):
        # f11 = negative divergence inwards
        f[i,0,
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f11[i]
        # f21 = positive curl to the right
        f[i,1,
        szx // 3 - 1 - cz: szx // 3 + cz,
        szy // 3 - 1 - cz: szy // 3 + cz,
        szz // 3 - 1 - cz: szz // 3 + cz
        ] = f21[i]

    elif (test_type=="pos_div_neg_curl"):
        # f12 = positive divergence outwards
        f[i,0,
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f12[i]
        # f22 = negative curl to the left
        f[i,1,
        2 * szx // 3 - 1 - cz: 2 * szx // 3 + cz,
        2 * szy // 3 - 1 - cz: 2 * szy // 3 + cz,
        2 * szz // 3 - 1 - cz: 2 * szz // 3 + cz
        ] = f22[i]


pos_py, _ = gridgen_3d_py(f, j_lb=j_lb, j_ub=j_ub, n_euler=20, inv=False)
pos_inv_py, _ = gridgen_3d_py(f, j_lb=j_lb, j_ub=j_ub, n_euler=20, inv=True)

print ('pos_py: ' + str(pos_py.shape))
print ('pos_inv_py: ' + str(pos_inv_py.shape))

# imw_py_all = mygriddata_3d_py(pos_inv_py, im_py[:, 0:1, :, :, :])
imw_py_all = im_py
# print ('imw_py_all before griddata: ' + str(imw_py_all.shape))

imw_py_all[:,0:1,:,:,:] = mygriddata_3d_py(pos_inv_py, im_py[:, 0:1, :, :, :])
print ('imw_py_all after griddata: ' + str(imw_py_all.shape))
imw_py_all = im_py.detach().numpy()
imw_py_all = np.squeeze(imw_py_all[:,0,:,:,:]) # (10, 25, 27, 29)
imw_py_all = np.moveaxis(imw_py_all,0,3)
print ('imw_py_all: ' + str(imw_py_all.shape))

pos_py = pos_py.detach().numpy()
print ('pos_py: ' + str(pos_py.shape)) # (10, 3, 25, 27, 29)
posx_py_all = pos_py[:,0,:,:,:]
posy_py_all = pos_py[:,1,:,:,:]
posz_py_all = pos_py[:,2,:,:,:]
posx_py_all = np.moveaxis(posx_py_all,0,3)
posy_py_all = np.moveaxis(posy_py_all,0,3)
posz_py_all = np.moveaxis(posz_py_all,0,3)
# rescale!
# posx_py_all = (szx - 1) * (posx_py_all + 1) / 2
# posy_py_all = (szy - 1) * (posy_py_all + 1) / 2
# posz_py_all = (szz - 1) * (posz_py_all + 1) / 2
posx_py_all = ((szx - 1) * (posx_py_all + 1) / 2) + 1
posy_py_all = ((szy - 1) * (posy_py_all + 1) / 2) + 1
posz_py_all = ((szz - 1) * (posz_py_all + 1) / 2) + 1

pos_inv_py = pos_inv_py.detach().numpy()
posx_inv_py_all = pos_inv_py[:,0,:,:,:]
posy_inv_py_all = pos_inv_py[:,1,:,:,:]
posz_inv_py_all = pos_inv_py[:,2,:,:,:]
posx_inv_py_all = np.moveaxis(posx_inv_py_all,0,3)
posy_inv_py_all = np.moveaxis(posy_inv_py_all,0,3)
posz_inv_py_all = np.moveaxis(posz_inv_py_all,0,3)
# rescale
# posx_inv_py_all = (szx - 1) * (posx_inv_py_all + 1) / 2
# posy_inv_py_all = (szy - 1) * (posy_inv_py_all + 1) / 2
# posz_inv_py_all = (szz - 1) * (posz_inv_py_all + 1) / 2
posx_inv_py_all = ((szx - 1) * (posx_inv_py_all + 1) / 2) + 1
posy_inv_py_all = ((szy - 1) * (posy_inv_py_all + 1) / 2) + 1
posz_inv_py_all = ((szz - 1) * (posz_inv_py_all + 1) / 2) + 1

img = nib.Nifti1Image(posx_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posx_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'imw_py.nii')
nib.save(img, output_filename)

### save out differences nii ###

img = nib.Nifti1Image(posx_np_all-posx_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_np_all-posy_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_np_all-posz_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np_minus_py.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posx_inv_np_all-posx_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posx_np_minus_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posy_inv_np_all-posy_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posy_np_minus_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(posz_inv_np_all-posz_inv_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'posz_np_minus_py_inv.nii')
nib.save(img, output_filename)

img = nib.Nifti1Image(imw_np_all-imw_py_all, np.eye(4))
output_filename = os.path.join(output_directory, 'imw_np_minus_py.nii')
nib.save(img, output_filename)

### find differences ###

diff_posx = np.sum(posx_np_all-posx_py_all)
diff_posy = np.sum(posy_np_all-posy_py_all)
diff_posz = np.sum(posz_np_all-posz_py_all)
diff_posx_inv = np.sum(posx_inv_np_all-posx_inv_py_all)
diff_posy_inv = np.sum(posy_inv_np_all-posy_inv_py_all)
diff_posz_inv = np.sum(posz_inv_np_all-posz_inv_py_all)
diff_imw = np.sum(imw_np_all-imw_py_all)

print ('diff_posx: ' + str(diff_posx) + ' min: ' + str(np.min(posx_np_all-posx_py_all)) + ' max: ' + str(np.max(posx_np_all-posx_py_all)))
print ('diff_posy: ' + str(diff_posy) + ' min: ' + str(np.min(posy_np_all-posy_py_all)) + ' max: ' + str(np.max(posy_np_all-posy_py_all)))
print ('diff_posz: ' + str(diff_posz) + ' min: ' + str(np.min(posz_np_all-posz_py_all)) + ' max: ' + str(np.max(posz_np_all-posz_py_all)))
print ('diff_posx_inv: ' + str(diff_posx_inv) + ' min: ' + str(np.min(posx_inv_np_all-posx_inv_py_all)) + ' max: ' + str(np.max(posx_inv_np_all-posx_inv_py_all)))
print ('diff_posy_inv: ' + str(diff_posy_inv) + ' min: ' + str(np.min(posy_inv_np_all-posy_inv_py_all)) + ' max: ' + str(np.max(posy_inv_np_all-posy_inv_py_all)))
print ('diff_posz_inv: ' + str(diff_posz_inv) + ' min: ' + str(np.min(posz_inv_np_all-posz_inv_py_all)) + ' max: ' + str(np.max(posz_inv_np_all-posz_inv_py_all)))
print ('diff_imw: ' + str(diff_imw) + ' min: ' + str(np.min(imw_np_all-imw_py_all)) + ' max: ' + str(np.max(imw_np_all-imw_py_all)))