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
import time 
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

# get ground truth a and b 

c = 10
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)


ngrid = 50 
_ths = np.pi*np.arange(ngrid)/ngrid
_phs = 2*np.pi*np.arange(ngrid)/ngrid

ths, phs = np.meshgrid(_ths,_phs,indexing='ij')
ths, phs = ths.flatten(), phs.flatten()

grid = Grid_3d(type='spherical', ths=ths, phs=phs)


# 
# f_vmf = vMF_density(centers,w_vmf,kappa,grid)
# f_vmf = f_vmf*np.sin(ths)
# f_vmf = f_vmf.reshape((ngrid,ngrid))


# generate rotation distribution 
kappa = 5
def my_fun(th,ph):
    grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
    return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]

ell_max_half_view = 2
sph_coef, indices = sph_harm_transform(my_fun, ell_max_half_view)
rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
rot_coef[0] = 1
sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
b = np.real(sph_c_t_r @ rot_coef)
rot_coef = sph_r_t_c @ b
b = b[1:]


# get the spherical FB coefficient of the volume
with mrcfile.open('../data/emd_34948.map') as mrc:
    data = mrc.data


# load volume data 
data = data/LA.norm(data.flatten())
Vol = Volume(data)
ds_res = 64 
Vol = Vol.downsample(ds_res)
vol = Vol.asnumpy()
vol = vol[0]
vol = vol/LA.norm(vol.flatten())


ell_max_vol = 3
# spherical bessel transform 
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol, ell_max_vol)
sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
a = np.real(sphFB_c_t_r @ vol_coef)
vol_coef = sphFB_r_t_c @ a 
vol = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol = vol.reshape([ds_res,ds_res,ds_res])
with mrcfile.new('vol.mrc', overwrite=True) as mrc:
    mrc.set_data(vol)  # Set the volume data
    mrc.voxel_size = 1.0 

# form the moments 
r2_max = 200 
r3_max = 60 
tol2 = 1e-12
tol3 = 1e-8
grid = get_2d_unif_grid(ds_res,1/ds_res)
grid = Grid_3d(xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))

opts = {}
opts['r2_max'] = r2_max
opts['r3_max'] = r3_max
opts['tol2'] = tol2 
opts['tol3'] = tol3 
opts['grid'] = grid



subMoMs = coef_t_subspace_moments(vol_coef, ell_max_vol, k_max, r0, indices_vol, rot_coef, ell_max_half_view, opts)
m1_emp = subMoMs['m1']
m2_emp = subMoMs['m2']
m3_emp = subMoMs['m3']
U2 = subMoMs['U2']
U3 = subMoMs['U3']


print(m1_emp.shape)
print(m2_emp.shape)
print(m3_emp.shape)

subspaces = {}
subspaces['m2'] = U2 
subspaces['m3'] = U3 



quadrature_rules = {} 
quadrature_rules['m2'] = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
quadrature_rules['m3'] = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)



# precomputation 

Phi_precomps, Psi_precomps = precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid)


if False:
    
    # constraints 
    xtrue =  np.concatenate([a,b])
    na = len(a)
    nb = len(b)
    view_constr, rhs, Psi0 = get_linear_ineqn_constraint(ell_max_half_view)
    A_constr = np.zeros([len(rhs), len(xtrue)])
    A_constr[:,na:] = view_constr 
    
    
    # stage 1 
    
    
    
    '''
    
    
    
    def ineq_constrains(x):
        return rhs - A_constr @ x 
    
    l1 = LA.norm(m1_emp.flatten())**2
    l2 = LA.norm(m2_emp.flatten())**2
    def objective(x):
        return find_cost(x,quadrature_rules,Phi_precomps,Psi_precomps,m1_emp,m2_emp,m3_emp,l1=l1,l2=l2,l3=0)
        
    def gradient(x):
        return find_grad(x,quadrature_rules,Phi_precomps,Psi_precomps,m1_emp,m2_emp,m3_emp,l1=l1,l2=l2,l3=0)
    
    
    _f, _g = find_cost_grad(x0,quadrature_rules,Phi_precomps,Psi_precomps,m1_emp,m2_emp,m3_emp,l1=l1,l2=l2,l3=0)
    f = objective(x0)
    g = gradient(x0)
    '''
    
    
    loss2 = 100000
    x20 = None 
    
    for i in range(5):
        a0 = 1e-4*np.random.normal(0,1,a.shape)
        centers = np.random.normal(0,1,size=(c,3))
        centers /= LA.norm(centers, axis=1, keepdims=True)
        w_vmf = np.random.uniform(0,1,c)
        w_vmf = w_vmf/np.sum(w_vmf)
        def my_fun(th,ph):
            grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
            return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
        sph_coef, indices = sph_harm_transform(my_fun, ell_max_half_view)
        rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
        rot_coef[0] = 1
        b0 = sph_c_t_r @ rot_coef
        b0 = jnp.real(b0[1:])
        x0 = jnp.concatenate([a0,b0])
        res2 = moment_LS(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l3=0)
        if res2.fun<loss2:
          x2 = res2.x
          loss2 = res2.fun
      
    savemat('res2.mat',res2)
    a_est = x2[:na]
    vol_coef_est_m2 = sphFB_r_t_c @ a_est
    
    
    
    res3 = moment_LS(x2, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs)
    x3 = res3.x 
    savemat('x3.mat',{'x3':x3})
    savemat('res3.mat',res3)
    a_est = x3[:na]
    vol_coef_est_m3 = sphFB_r_t_c @ a_est
    
    
    vol_coef_est_m2, _ , _ , _ = align_vol_coef(vol_coef,vol_coef_est_m2,ell_max_vol,k_max,indices_vol)
    vol_coef_est_m3, _ , _ , _ = align_vol_coef(vol_coef,vol_coef_est_m3,ell_max_vol,k_max,indices_vol)
    
    vol_est_m2_aligned = coef_t_vol(vol_coef_est_m2, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m3_aligned = coef_t_vol(vol_coef_est_m3, ell_max_vol, ds_res, k_max, r0, indices_vol)
    
    vol_est_m2_aligned = np.reshape(vol_est_m2_aligned, [ds_res,ds_res,ds_res])
    vol_est_m3_aligned = np.reshape(vol_est_m3_aligned, [ds_res,ds_res,ds_res])
    
    with mrcfile.new('vol_est_m2_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_est_m2_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    
    with mrcfile.new('vol_est_m3_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_est_m3_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    
