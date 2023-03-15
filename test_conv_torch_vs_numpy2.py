# test_conv_torch_vs_numpy2.py
#
# I have the pytorch conv functions that seem to work and don't cause fc2pos to be stuck in the while loop, and also
# don't cause oscillations in the smeasure/smeasure_new values
#
# The three functions to get equivalent numpy versions for:
# Div_curl - nnf.conv3d(f, dx_kernel, padding=1, groups=4)
# Gridgen - f[:, 0:1, :, :, :] = nnf.conv3d(nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), k, padding=0)
# Gradient - nnf.conv3d(g_f[:, 0:1, :, :, :], dF1_df1_filter, padding=1)
#
# Deepa Krishnaswamy
# University of Alberta
# July 14 2021
########################################################################################################################

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as nnf

from scipy import ndimage


########################################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################
### div curl kernels1 ###
#######################

print('*************** testing kernels in div curl function 1 *************')
print ('This is what I actually use')

### Numpy ###
szx, szy, szz = 15, 17, 19

f1 = np.random.randn(szx, szy, szz)
# set one value below zero
f1[10,10,10] = -0.1
print('f1: ' + str(f1.shape))


dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
dx_kernel[0, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
dx_kernel[2, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

# This is what I actually use in the pycardiac code.
f1c_out_numpy = scipy.ndimage.convolve(f1, dx_kernel, mode='wrap') # this is what I had in my numpy code. didn't oscillate.

### Pytorch ###

f2 = np.zeros((szx, szy, szz))
f3 = np.zeros((szx, szy, szz))
f4 = np.zeros((szx, szy, szz))
f1, f2, f3, f4 = np.expand_dims(f1,0), np.expand_dims(f2,0), np.expand_dims(f3,0), np.expand_dims(f4,0)
f = np.concatenate([f1, f2, f3, f4], axis=0)
f = np.expand_dims(f,0) # batch size
f = torch.from_numpy(f)
print ('f: ' + str(f.shape))

# This dx kernel is switched from the one above
dx_kernel = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], \
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                          [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.double, device=device)

print('dx_kernel: ' + str(dx_kernel.shape))

dx_kernel = torch.unsqueeze(torch.unsqueeze(dx_kernel,0),0)
dx_kernel = dx_kernel.repeat(4, 1, 1, 1, 1)

# f1c_out_torch = nnf.conv3d(f, dx_kernel, padding=1, groups=4) # doesn't match
# f1c_out_torch = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "replicate"), dx_kernel, padding=0,groups=4)
f1c_out_torch = nnf.conv3d(nnf.pad(f, (1, 1, 1, 1, 1, 1), "circular"), dx_kernel, padding=0, groups=4) # this one matches!

f1c_out_torch = f1c_out_torch.detach().cpu().numpy()
f1c_out_torch = np.squeeze(f1c_out_torch)
f1c_out_torch = np.squeeze(f1c_out_torch[0,:,:,:])
print ('f1c_out_torch: ' + str(f1c_out_torch.shape))

f1c_diff = np.sum(f1c_out_numpy-f1c_out_torch)
print ('f1c_diff: ' + str(f1c_diff)) # 0.0




#######################
### div curl kernels2 ###
#######################

print('*************** testing kernels in div curl function 2 *************')
print ('This is not what Im using in my code')

### Numpy ###
szx, szy, szz = 15, 17, 19

f1 = np.random.randn(szx, szy, szz)
print('f1: ' + str(f1.shape))

dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
dx_kernel[0, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
dx_kernel[2, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

# this one doesn't match
# f1c_out_numpy = scipy.ndimage.convolve(f1, dx_kernel, mode='wrap') # this is what I had in my numpy code. didn't oscillate.

# Either of the ones below match
# f1c_out_numpy = scipy.signal.convolve(f1, dx_kernel, mode='same') # 1.8 e-6 difference
f1c_out_numpy = scipy.ndimage.convolve(f1, dx_kernel, mode='constant', cval=0.0) # 0 difference

### Pytorch ###

f2 = np.zeros((szx, szy, szz))
f3 = np.zeros((szx, szy, szz))
f4 = np.zeros((szx, szy, szz))
f1, f2, f3, f4 = np.expand_dims(f1,0), np.expand_dims(f2,0), np.expand_dims(f3,0), np.expand_dims(f4,0)
f = np.concatenate([f1, f2, f3, f4], axis=0)
f = np.expand_dims(f,0) # batch size
f = torch.from_numpy(f)
print ('f: ' + str(f.shape))

# This dx kernel is switched from the one above
dx_kernel = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], \
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                          [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.double, device=device)

print('dx_kernel: ' + str(dx_kernel.shape))

dx_kernel = torch.unsqueeze(torch.unsqueeze(dx_kernel,0),0)
dx_kernel = dx_kernel.repeat(4, 1, 1, 1, 1)
f1c_out_torch = nnf.conv3d(f, dx_kernel, padding=1, groups=4)
f1c_out_torch = f1c_out_torch.detach().cpu().numpy()
f1c_out_torch = np.squeeze(f1c_out_torch)
f1c_out_torch = np.squeeze(f1c_out_torch[0,:,:,:])
print ('f1c_out_torch: ' + str(f1c_out_torch.shape))

f1c_diff = np.sum(f1c_out_numpy-f1c_out_torch)
print ('f1c_diff: ' + str(f1c_diff))

#######################
### gridgen kernels ###
#######################

print('*************** testing kernels in gridgen while loop function *************')

### Numpy ###
j_lb = 0.25
szx, szy, szz = 15, 17, 19

f1 = np.random.randn(szx, szy, szz)
# set one value below zero
f1[10,10,10] = -0.1
print('f1: ' + str(f1.shape))

ker = np.ones((3, 3, 3)) / 27.0

# I had this one
f1c_out_numpy = ndimage.convolve(f1, ker, mode='wrap') # diff = f1c_diff: -3.9968028886505635e-15
# f1c_out_numpy = scipy.signal.convolve(f1, ker, mode="same") # boundary="wrap" doesn't exist? # try this one.

### Pytorch ###

k = torch.ones((3, 3, 3), dtype=torch.double, device=device) / 27.0
k = torch.unsqueeze(k, 0)
k = torch.unsqueeze(k, 0)
print('k: ' + str(k.shape))

f2 = np.zeros((szx, szy, szz))
f3 = np.zeros((szx, szy, szz))
f4 = np.zeros((szx, szy, szz))
f1, f2, f3, f4 = np.expand_dims(f1, 0), np.expand_dims(f2, 0), np.expand_dims(f3, 0), np.expand_dims(f4, 0)
f = np.concatenate([f1, f2, f3, f4], axis=0)
f = np.expand_dims(f, 0)  # batch size
f = torch.from_numpy(f)
print('f: ' + str(f.shape))

# f1c_out_torch = nnf.conv3d(nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "replicate"), k, padding=0)
f1c_out_torch = nnf.conv3d(nnf.pad(f[:, 0:1, :, :, :], (1, 1, 1, 1, 1, 1), "circular"), k, padding=0)
f1c_out_torch = f1c_out_torch.detach().cpu().numpy()
f1c_out_torch = np.squeeze(f1c_out_torch)
print ('f1c_out_torch: ' + str(f1c_out_torch.shape))

f1c_diff = np.sum(f1c_out_numpy-f1c_out_torch)
print ('f1c_diff: ' + str(f1c_diff)) # f1c_diff: 7.198018223131264e-15



#######################
### gradient kernels ###
#######################

print('*************** testing kernels in gradient function *************')

### Numpy ###
szx, szy, szz = 15, 17, 19

f1 = np.random.randn(szx, szy, szz)
print('f1: ' + str(f1.shape))

dx_kernel = np.zeros((3, 3, 3), dtype=np.float32)
dx_kernel[0, :, :] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
dx_kernel[2, :, :] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]

# this one doesn't match
# f1c_out_numpy = scipy.ndimage.convolve(f1, dx_kernel, mode='wrap') # this is what I had in my numpy code. didn't oscillate.

# Either of the ones below match
# f1c_out_numpy = scipy.signal.convolve(f1, dx_kernel, mode='same') # 1.8 e-6 difference
f1c_out_numpy = scipy.ndimage.convolve(f1, dx_kernel, mode='constant', cval=0.0) # 0 difference

### Pytorch ###

f2 = np.zeros((szx, szy, szz))
f3 = np.zeros((szx, szy, szz))
f4 = np.zeros((szx, szy, szz))
f1, f2, f3, f4 = np.expand_dims(f1,0), np.expand_dims(f2,0), np.expand_dims(f3,0), np.expand_dims(f4,0)
f = np.concatenate([f1, f2, f3, f4], axis=0)
f = np.expand_dims(f,0) # batch size
f = torch.from_numpy(f)
print ('f: ' + str(f.shape))

# This dx kernel is switched from the one above
dx_kernel = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], \
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                          [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.double, device=device)

print('dx_kernel: ' + str(dx_kernel.shape))

dx_kernel = torch.unsqueeze(torch.unsqueeze(dx_kernel,0),0)
# dx_kernel = dx_kernel.repeat(4, 1, 1, 1, 1)
dx_kernel = dx_kernel.repeat(1,1,1,1,1)
# f1c_out_torch = nnf.conv3d(f, dx_kernel, padding=1, groups=4)
f1c_out_torch = nnf.conv3d(f[:, 0:1, :, :, :], dx_kernel, padding=1)

f1c_out_torch = f1c_out_torch.detach().cpu().numpy()
f1c_out_torch = np.squeeze(f1c_out_torch)
print ('f1c_out_torch: ' + str(f1c_out_torch.shape))

f1c_diff = np.sum(f1c_out_numpy-f1c_out_torch)
print ('f1c_diff: ' + str(f1c_diff))