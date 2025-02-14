import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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


np.random.seed(1)

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
sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
a = np.real(sphFB_c_t_r @ vol_coef)
savemat('vol_coef.mat',{'vol_coef':vol_coef, 'a':a})


c = 2
kappa = 4
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)
print(centers,w_vmf)


ell_max_half_view = 2
rot_coef, indices_view = vmf_t_rot_coef(centers,w_vmf,kappa,ell_max_half_view)

sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
b = np.real(sph_c_t_r @ rot_coef)
b = b[1:]
savemat('vmf_params.mat',{'centers':centers,'w_vmf':w_vmf,'kappa':kappa,'b':b})
savemat('rot_coef.mat',{'rot_coef':rot_coef,'b':b})


grid = get_2d_unif_grid(ds_res,1/ds_res)
grid = Grid_3d(type='euclid', xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))
grid.ths[2080] = np.pi/2
precomp_vol_basis,radial_part,angular_part = precompute_sphFB_basis(ell_max_vol, k_max, r0, indices_vol, grid)

savemat('precomp_vol_basis.mat',{'precomp_vol_basis':precomp_vol_basis,'radial_part':radial_part,'angular_part':angular_part})
savemat('grid_info.mat',{'rs':grid.rs,'ths':grid.ths,'phs':grid.phs})








