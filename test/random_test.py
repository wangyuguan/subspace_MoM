import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))

from aspire.volume import Volume
from aspire.utils.rotation import Rotation
import mrcfile
import numpy as np 
import numpy.linalg as LA 
from viewing_direction import *
from utils import *
from aspire.basis.basis_utils import lgwt
from volume import *
from moments import * 
import matplotlib.pyplot as plt
from scipy.io import savemat
from utils_BO import align_BO

import jax
import jax.numpy as jnp
from jax import grad, jit 
from jax.numpy.linalg import norm



with mrcfile.open('../data/emd_34948.map') as mrc:
    data = mrc.data


data = data/LA.norm(data.flatten())
Vol = Volume(data)
ds_res = 64 
Vol = Vol.downsample(ds_res)
vol = Vol.asnumpy()
vol = vol[0]

ell_max_vol = 3
# spherical bessel transform 
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol, ell_max_vol)
vol_coef_ref = reflect_sphFB(vol_coef, ell_max_vol, k_max, indices_vol)


alpha = np.random.uniform(0,2*np.pi)
beta = np.random.uniform(0,np.pi)
gamma = np.random.uniform(0,2*np.pi)
Rot0 = Rz(alpha)@Ry(beta)@Rz(gamma)



vol_coef_rot = rotate_sphFB(vol_coef_ref, ell_max_vol, k_max, indices_vol, (alpha, beta, gamma))
vol_coef_rot_est,cost,Rot,reflect = align_vol_coef(vol_coef, vol_coef_rot, ell_max_vol, k_max, indices_vol)

print(LA.norm(vol_coef_rot_est-vol_coef)/LA.norm(vol_coef))
'''

vol = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol = vol.reshape([ds_res,ds_res,ds_res])
vol_ref = coef_t_vol(vol_coef_ref, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol_ref = vol_ref.reshape([ds_res,ds_res,ds_res])




with mrcfile.new('vol.mrc', overwrite=True) as mrc:
    mrc.set_data(vol)  # Set the volume data
    mrc.voxel_size = 1.0 
    
    
    
with mrcfile.new('vol_ref.mrc', overwrite=True) as mrc:
    mrc.set_data(vol_ref)  # Set the volume data
    mrc.voxel_size = 1.0 
'''