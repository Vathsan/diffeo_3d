import torch
import os
import nrrd
import nibabel as nib
import numpy as np
from src.pytorch.gridgen_3d_pycardiac import mygriddata_3d as mygriddata_3d_py
from src.pytorch.reg_3d_pycardiac import torch_registration_3d as torch_registration_3d
from src.pytorch.reg_3d_pycardiac import register_sequence_3d as register_sequence_3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class RegParam:
    """Define registration parameters"""

    def __init__(
        self,
        mx_iter=20.0,   #
        n_euler=20.0,   # increase
        t=0.5,
        t_up=1.0,
        t_dn=2.0 / 3.0,
        mn_t=0.01,
        j_lb=0.9,  # 0.8/0.9
        j_ub=1.5,   # 1.5/2.0
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

def transform(image, pos):
    im_t_py = torch.from_numpy(image).float().to(device).repeat(1, 1, 1, 1, 1)
    im_w_py = mygriddata_3d_py(pos, im_t_py)

    im_w = im_w_py.detach().cpu().numpy()
    im_w = np.squeeze(im_w)

    return im_w

def register_sequence(input_dir, rigidMaskDir, fixed_img, frame_no, output_dir):
    im_s, hdr_s = nrrd.read(os.path.join(input_dir, fixed_img))

    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if filename != fixed_img and filename.endswith(".nrrd"):
            im_t, hdr_t = nrrd.read(os.path.join(input_dir, filename))
            dim = hdr_t["sizes"][1:4]
            frames = hdr_t["sizes"][0]
            output_nrrd = np.ndarray(shape=(frames, dim[0], dim[1], dim[2]))
            print("Registering ", filename)

            im_s_py = im_s[frame_no, :, :, :]
            im_t_py = im_t[frame_no, :, :, :]

            im_s_py = torch.from_numpy(im_s_py).float().to(device).repeat(1, 1, 1, 1, 1)
            im_t_py = torch.from_numpy(im_t_py).float().to(device).repeat(1, 1, 1, 1, 1)

            pos, f, smeasure, smeasure_new, tstep = torch_registration_3d(im_s_py, im_t_py, prm)

            mask = filename.rsplit("_", 1)[0] + "_Mask.nrrd"
            maskData, maskHeader = nrrd.read(os.path.join(rigidMaskDir, mask))
            mask_frames, x, y, z = maskData.shape
            output_mask = np.ndarray(shape=(mask_frames, x, y, z))

            for i in range(frames):
                print("Starting transformation for frame ", str(i))
                output_nrrd[i, :, :, :] = transform(im_t[i, :, :, :], pos)
                reg_mask = transform(maskData[i, :, :, :], pos)
                reg_mask[reg_mask > 0] = 1
                output_mask[i, :, :, :] = reg_mask

            output_nrrd = ((output_nrrd - np.min(output_nrrd)) / (np.max(output_nrrd) - np.min(output_nrrd))) * 255
            output_nrrd = output_nrrd.astype('uint8')

            output_name = filename.rsplit("_", 1)[0] + "_diffeo.nrrd"
            print("Writing transformed image")
            nrrd.write(os.path.join(output_dir, output_name), output_nrrd, hdr_t)
            print("Writing transformed mask")
            nrrd.write(os.path.join(output_dir, mask), output_mask, header=maskHeader)
            print("*" * 20)

input_dir = "/media/srivathsan/New Volume/EchoFusion/3DNew/SS/registered/rigid/"
fixed_img = "E9PCCZHF_ApSt.nrrd"
frame_no = 0
output_dir = "/media/srivathsan/New Volume/EchoFusion/3DNew/SS/registered/diffeo/"
rigidMaskDir = "/media/srivathsan/New Volume/EchoFusion/3DNew/SS/binary_mask_rigid_reduced/"

register_sequence(input_dir, rigidMaskDir, fixed_img, frame_no, output_dir)
