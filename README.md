diffeo_3d
==============================

The code in this repository performs 3D-to-3D diffeomorphic image registration, where the 2D version 
has been published in:
Punithakumar K, Boulanger P, Noga M. A GPU-accelerated deformable image registration algorithm with 
applications to right ventricular segmentation. IEEE Access. 2017 Sep 26;5:20374-82. 

The src/np/ directory contains the registration functions implemented in numpy+jit, while the src/pytorch
directory contains the registration functions implemented using pytorch. The grid generation functions
are available in src/np/gridgen_3d_pycardiac.py and src/pytorch/gridgen_3d_pycardiac.py, and the code for the 
pairwise registration are available in src/np/reg_3d_pycardiac.py and src/pytorch/reg_3d_pycardiac.py. 
Old versions of the gridgen and reg code are in the src/np/old and src/pytorch/old directories. 

The following are the descriptions of the scripts included: 

The script test_gridgen_torch_vs_pycardiac.py contains code for testing the numpy vs pytorch versions of the
gridgen_3d_pycardiac.py scripts. Four tests are performed using 3D image grids, positive divergence, negative 
divergence, negative divergence and positive curl, and positive divergence and negative curl. The nifti
files of the warped images are saved and can be viewed for correctness by the user. I have also computed the 
summed, min and max difference of each of these files (pos, warped images) to make sure that the two methods
are giving the same values. Min and max differences between -10^-5 and 10^-5 are acceptable in my opinion, 
but may result in summed differences on the order of 10^-2. 

The script test_reg_torch_vs_pycardiac.py contains code for testing the numpy vs pytorch versions of the 
reg_3d.py scripts. The ACDC dataset is used, where pairwise registration is performed between the first
two frames. The difference in the pos for x, y, and z are saved as nifti files and can be viewed by 
the user. Again, I have computed the summed, min and max differences for each of these files to make sure 
the two methods are giving the same values. Min and max differences between -10^-5 and 10^-5 are acceptable in 
my opinion, but may result in summed differences on the order of 10^-2. For this test, it is important to 
look at the final cost plot in order to easily compare the numpy and pycardiac versions. 

The pycardiac_3D_functions.py holds the other necessary functions for evaluation, including the 3D versions 
of get_pointcorrespondence, computing auto contours, transforming meshes, etc. The reg_functions_validation.py 
script holds functions necessary for computing the Dice score, HD, MAD, etc. The older versions of the pycardiac
code are located in the pycardiac_old directory. 

Additional scripts have been included for testing. These include the test_div_curl_det_Jacobian for 2D and 3D.
These tests were done to make sure the Jacobian determinant values computed were correct. The script 
test_conv_torch_vs_numpy2.py holds tests done to compare strictly the convlutional functions between 
torch and numpy. Old/personal testing scripts are located in the tests_old directory. 

Python 3.6 was used, and pytorch version 1.17. I have provided two requirements files, one for use on your
personal computer, and another for setting up on compute canada. 


--------

