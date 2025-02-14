import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))


from aspire.volume import Volume
import mrcfile
import numpy as np 
import numpy.linalg as LA 
import finufft 
from scipy.special import spherical_jn
from scipy.io import savemat
from utils import *
from volume import * 
from viewing_direction import * 

import random

random.seed(1)


np.set_printoptions(threshold=sys.maxsize)

with mrcfile.open('../data/emd_34948.map') as mrc:
    data = mrc.data

data = data/LA.norm(data.flatten())
Vol = Volume(data)
ds_res = 64 
Vol = Vol.downsample(ds_res)
vol = Vol.asnumpy()
vol = vol[0]


ell_max_vol = 3

vol_coef, k_max, r0, indices_vol = sphFB_transform(vol, ell_max_vol)
# savemat('vol_coef1.mat',{'vol_coef1':vol_coef})


c = 2
kappa = 4
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)
#print(centers,w_vmf)


def my_fun(th,ph):
    grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
    return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]

print(my_fun(1,2))

ell_max_half_view = 2
sph_coef, indices = sph_harm_transform(my_fun, ell_max_half_view)
rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
rot_coef[0] = 1
sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
b = np.real(sph_c_t_r @ rot_coef)
rot_coef = sph_r_t_c @ b
b = b[1:]

savemat('vmf_params.mat',{'centers':centers,'w_vmf':w_vmf,'kappa':kappa,'sph_coef':sph_coef,'rot_coef':rot_coef,'b':b})