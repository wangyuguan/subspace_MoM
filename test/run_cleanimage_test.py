import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
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

with mrcfile.open('vol_test.mrc') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())
    
# preprocess the volume 
ell_max_vol = 4
Vol = Volume(vol)
expand = False 

if expand:
    fb_aspire = FBBasis3D(size=vol.shape[0], ell_max=ell_max_vol)
    fb_aspire_coef = fb_aspire.evaluate_t(Vol)
    Vol = fb_aspire.evaluate(fb_aspire_coef)
    vol = Vol.asnumpy()
    vol = vol[0]
    Vol = Volume(vol)


if False :
    
    with mrcfile.new('vol_test.mrc', overwrite=True) as mrc:
        mrc.set_data(vol)  # Set the volume data
        mrc.voxel_size = 1.0  

    Ntot = 20 
    rots = np.zeros((Ntot,3,3),dtype=np.float32)
    alpha = np.random.uniform(0,2*np.pi,Ntot)
    beta = np.random.uniform(0,np.pi,Ntot)
    gamma = np.random.uniform(0,2*np.pi,Ntot)
    for i in range(Ntot):
        rot= Rz(alpha[0]) @ Ry(beta[0]) @ Rz(gamma[i])
        rots[i,:,:] = rot
        
        

    Rots = Rotation(rots)
    imags = Vol.project(Rots).downsample(ds_res=64, zero_nyquist=False)
    imags.show()
        
#        imags = imags.asnumpy()
#        for imag in imags:
#            print(imag.shape)



# set up viewing direction distribution 
c = 10
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)
kappa = 5
def my_fun(th,ph):
    grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
    return np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
ell_max_half_view = 2
sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)



# form moments 
Ntot = 10000
rots = np.zeros((Ntot,3,3),dtype=np.float32)
print('sampling viewing directions')
samples = sample_sph_coef(Ntot, sph_coef, ell_max_half_view)
print('done')
gamma = np.random.uniform(0,2*np.pi,Ntot)
for i in range(Ntot):
    _, beta, alpha = cart2sph(samples[i,0], samples[i,1], samples[i,2])
    rots[i,:,:] = Rz(alpha) @ Ry(beta) @ Rz(gamma[i])

ds_res = 64
params = {'r2_max':50, 'r3_max':30, 'tol2':1e-12, 'tol3':1e-8, 'ds_res':ds_res}
U2, U3, U2_fft, U3_fft, t_sketch = momentPCA_rNLA(vol, rots, params)
m1_emp, m2_emp, m3_emp, t_form = form_subspace_moments(vol, rots, U2, U3)



# precomputation 
subspaces = {}
subspaces['m2'] = U2_fft 
subspaces['m3'] = U3_fft 
quadrature_rules = {} 
quadrature_rules['m2'] = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
quadrature_rules['m3'] = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)
k_max, r0, indices_vol = get_sphFB_indices(ds_res, ell_max_vol)
sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
grid = get_2d_unif_grid(ds_res,1/ds_res)
grid = Grid_3d(xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))
Phi_precomps, Psi_precomps = precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid)
na = len(indices_vol)
nb = len(indices_view)-1
view_constr, rhs, _ = get_linear_ineqn_constraint(ell_max_half_view)
A_constr = np.zeros([len(rhs), na+nb])
A_constr[:,na:] = view_constr 





# reconstruction 
loss2 = 100000
x20 = None 
l1 = LA.norm(m1_emp.flatten())**2
l2 = LA.norm(m2_emp.flatten())**2
l3 = LA.norm(m3_emp.flatten())**2
for i in range(5):
    a0 = 1e-4*np.random.normal(0,1,na)
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    sph_coef, _ = sph_harm_transform(my_fun, ell_max_half_view)
    rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
    rot_coef[0] = 1
    b0 = sph_c_t_r @ rot_coef
    b0 = jnp.real(b0[1:])
    x0 = jnp.concatenate([a0,b0])
    res2 = moment_LS(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l3=0)

    if res2.fun<loss2:
      x2 = res2.x
      loss2 = res2.fun
  
a_est = x2[:na]
vol_coef_est_m2 = sphFB_r_t_c @ a_est
res3 = moment_LS(x2, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs)
x3 = res3.x 
a_est = x3[:na]
vol_coef_est_m3 = sphFB_r_t_c @ a_est


# save results
vol_est_m2 = coef_t_vol(vol_coef_est_m2, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol_est_m3 = coef_t_vol(vol_coef_est_m3, ell_max_vol, ds_res, k_max, r0, indices_vol)

with mrcfile.new('vol_est_m2.mrc', overwrite=True) as mrc:
    mrc.set_data(vol_est_m2_aligned)  # Set the volume data
    mrc.voxel_size = 1.0  

with mrcfile.new('vol_est_m3.mrc', overwrite=True) as mrc:
    mrc.set_data(vol_est_m3_aligned)  # Set the volume data
    mrc.voxel_size = 1.0  







