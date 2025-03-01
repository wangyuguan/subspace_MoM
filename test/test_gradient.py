import sys
from pathlib import Path

# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))

from aspire.volume import Volume
import numpy as np 
import numpy.linalg as LA 
import mrcfile 
from utils import * 
from viewing_direction import * 
from volume import * 
from moments import * 
import time 

import jax
import jax.numpy as jnp
from jax import grad, jit 
from jax.numpy.linalg import norm


# generate view distribution
c = 10
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)
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

data = data/LA.norm(data.flatten())
Vol = Volume(data)
ds_res = 64 
Vol = Vol.downsample(ds_res)
vol = Vol.asnumpy()
vol = vol[0]


ell_max_vol = 4
# spherical bessel transform 
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol, ell_max_vol)
sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
a = np.real(sphFB_c_t_r @ vol_coef)
vol_coef = sphFB_r_t_c @ a 


# form the moments 
r2_max = 10 
r3_max = 60 
tol2 = 1e-8
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


# precomputation 
euler_nodes, w_so3 = load_so3_quadrature(ell_max_vol, 2*ell_max_half_view)
Phi =  precomp_sphFB_all(U2, ell_max_vol, k_max, r0, indices_vol, euler_nodes, grid)
Psi = precomp_wignerD_all(ell_max_half_view, euler_nodes)



if False:

    # test cost, gradient over the first moment at the ground truth 
    x0 = jnp.concatenate([a,b])
    l1 = LA.norm(m1_emp.flatten())**2
    cost0, grad0 = find_cost_grad_m1(x0, w_so3, Phi, Psi, m1_emp, l1)
    print(cost0, LA.norm(grad0))
    
    
    
    # check gradient over the first moment via fdm  
    
    x = np.random.normal(0,1,x0.shape)
    l1 = 1 
    cost, grad = find_cost_grad_m1(x, w_so3, Phi, Psi, m1_emp, l1)
    
    
    def my_grad(x):
        h = 1e-6 
        I = jnp.eye(len(x))
        grad = np.zeros(len(x))
        for i in range(len(x)):
            xph = x+h*I[i,:]
            xmh = x-h*I[i,:]
            costph = find_cost_m1(xph, w_so3, Phi, Psi, m1_emp, l1)
            costmh = find_cost_m1(xmh, w_so3, Phi, Psi, m1_emp, l1)
            grad[i] = (costph-costmh)/2/h 
        return grad 
    
    grad_fdm = my_grad(x)
    
    LA.norm(grad-grad_fdm)/LA.norm(grad_fdm)
    
    
    # precomputation 
    
    euler_nodes, w_so3 = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
    Phi =  precomp_sphFB_all(U2, ell_max_vol, k_max, r0, indices_vol, euler_nodes, grid)
    Psi = precomp_wignerD_all(ell_max_half_view, euler_nodes)
    
    
    l2 = LA.norm(m2_emp.flatten())**2
    cost0, grad0 = find_cost_grad_m2(x0, w_so3, Phi, Psi, m2_emp, l2)
    print(cost0, LA.norm(grad0))
    
    
    # check gradient over the second moment via fdm  
    
    x = np.random.normal(0,1,x0.shape)
    l2 = 1 
    cost, grad = find_cost_grad_m2(x, w_so3, Phi, Psi, m2_emp, l2)
    
    def my_grad(x):
        h = 1e-6 
        I = jnp.eye(len(x))
        grad = np.zeros(len(x))
        for i in range(len(x)):
            xph = x+h*I[i,:]
            xmh = x-h*I[i,:]
            costph = find_cost_m2(xph, w_so3, Phi, Psi, m2_emp, l2)
            costmh = find_cost_m2(xmh, w_so3, Phi, Psi, m2_emp, l2)
            grad[i] = (costph-costmh)/2/h 
        return grad 
    
    grad_fdm = my_grad(x)
    
    LA.norm(grad-grad_fdm)/LA.norm(grad_fdm)
    
    
    # precomputation 
    
    euler_nodes, w_so3 = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)
    Phi =  precomp_sphFB_all(U3, ell_max_vol, k_max, r0, indices_vol, euler_nodes, grid)
    Psi = precomp_wignerD_all(ell_max_half_view, euler_nodes)
    
    
    
    # test cost, gradient over the third moment at the ground truth 
    
    l3 = np.max(np.abs(m3_emp.flatten()))**2
    cost0, grad0 = find_cost_grad_m3(x0, w_so3, Phi, Psi, m3_emp, l3)
    print(cost0, LA.norm(grad0))
    
    
    # check gradient over the third moment via fdm  
    
    x = np.random.normal(0,1,x0.shape)
    l3 = 1
    cost, grad = find_cost_grad_m3(x, w_so3, Phi, Psi, m3_emp, l3)
    
    def my_grad(x):
        h = 1e-6 
        I = jnp.eye(len(x))
        grad = np.zeros(len(x))
        for i in range(len(x)):
            xph = x+h*I[i,:]
            xmh = x-h*I[i,:]
            costph = find_cost_m3(xph, w_so3, Phi, Psi, m3_emp, l3)
            costmh = find_cost_m3(xmh, w_so3, Phi, Psi, m3_emp, l3)
            grad[i] = (costph-costmh)/2/h
        return grad 
    
    grad_fdm = my_grad(x)
    
    LA.norm(grad-grad_fdm)/LA.norm(grad_fdm)
    
    
