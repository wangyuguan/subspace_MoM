import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))


from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.basis import FBBasis3D
from utils import * 
from viewing_direction import * 
from moments import *
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mrcfile 
from scipy.io import savemat 
from utils_BO import align_BO

np.random.seed(42)

with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())
    
L = vol.shape[0]
ds_res = 64
vol_ds = vol_downsample(vol, ds_res)
with mrcfile.new('vol_ds.mrc', overwrite=True) as mrc:
    mrc.set_data(vol_ds)  # Set the volume data
    mrc.voxel_size = 1.0  
    
    

'''
    
ell_max_vol = 8
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
vol_ds = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol = vol_ds.reshape(ds_res,ds_res,ds_res,order='F')




vol = vol.reshape([ds_res,ds_res,ds_res])

with mrcfile.new('vol_L8.mrc', overwrite=True) as mrc:
    mrc.set_data(vol)  # Set the volume data
    mrc.voxel_size = 1.0  




rots = np.zeros((10,3,3))
for i in range(10):
    alpha = np.random.uniform(0,2*np.pi)
    beta = np.random.uniform(0,np.pi)
    gamma = np.random.uniform(0,2*np.pi)
    rots[i] = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    
I_projs = vol_proj(vol, rots)


plt.imshow(I_projs[2], cmap='gray')
plt.axis('off')  # Hide axes
plt.show()


plt.imshow(I_projs[9], cmap='gray')
plt.axis('off')  # Hide axes
plt.show()

'''