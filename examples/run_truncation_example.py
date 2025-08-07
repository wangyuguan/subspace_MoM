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


def get_subsmom_errors(vol_path, ell_max_vol, out_file_name):

    np.random.seed(2025)
    out = {}

    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    
    # preprocess the volume 
    vol = vol/tns_norm(vol)
    img_size = vol.shape[0]
    ds_res = 64
    vol_ds = vol_downsample(vol, ds_res)
    vol_ds = vol_ds/tns_norm(vol_ds)
    vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
    sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
    a = np.real(sphFB_c_t_r @ vol_coef)
    vol_coef = sphFB_r_t_c @ a 
    vol = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol = vol.reshape([ds_res,ds_res,ds_res])
    

    

    with mrcfile.new('vol_gt.mrc', overwrite=True) as mrc:
        mrc.set_data(vol)  # Set the volume data
        mrc.voxel_size = 1.0  

    # set up viewing direction distribution 
    c = 12
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    kappa = 4
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    ell_max_half_view = 2
    sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)
    # f_sph, ths, phs = plot_sph_harm(sph_coef, ell_max_half_view, fname='spherical_ground_truth.pdf')
    rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
    rot_coef[0] = 1
    sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
    b = np.real(sph_c_t_r @ rot_coef)
    rot_coef = sph_r_t_c @ b
    b = b[1:]
    



    # form the moments 
    opts = {}
    opts['r2_max'] = 50
    opts['r3_max'] = 50
    opts['tol2'] = 1e-12
    opts['tol3'] = 1e-12
    opts['img_size'] = img_size
    opts['ds_res'] = ds_res
    grid = get_2d_unif_grid(ds_res,1/ds_res)
    grid = Grid_3d(xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))
    opts['grid'] = grid
    moms_out = analytical_subspace_moments_gaussian_m1(vol_coef, ell_max_vol, k_max, r0, indices_vol, rot_coef, ell_max_half_view, opts)
    m1_emp = moms_out['m1']
    m2_emp = moms_out['m2']
    m3_emp = moms_out['m3'] 
    U2 = moms_out['U2']
    U3 = moms_out['U3']
    # np.savez(out_file_name, **out)
    
    quadrature_rules = {} 
    quadrature_rules['m2'] = load_so3_quadrature(16, 2*ell_max_half_view)
    quadrature_rules['m3'] = load_so3_quadrature(16, 2*ell_max_half_view)
    subspaces = {}
    subspaces['m2'] = U2 
    subspaces['m3'] = U3 
    Phi_precomps, Psi_precomps = precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid)

    x_true = np.concatenate([a,b])

    l1 = LA.norm(m1_emp.flatten())**2
    l2 = LA.norm(m2_emp.flatten())**2
    l3 = LA.norm(m3_emp.flatten())**2
    
    errs = {}
    objective = lambda x: find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1=l1, l2=0, l3=0)
    cost, grad = objective(x_true)
    errs['m1'] = np.sqrt(cost)

    objective = lambda x: find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1=0, l2=l2, l3=0)
    cost, grad = objective(x_true)
    errs['m2'] = np.sqrt(cost)

    objective = lambda x: find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1=0, l2=0, l3=l3)
    cost, grad = objective(x_true)
    errs['m3'] = np.sqrt(cost)


    np.savez(out_file_name, **errs)



def run_reconstruction(vol_path, ell_max_vol, out_file_name ):
    
    np.random.seed(42)
    out = {}

    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    pixel_size = 1.04 
    # preprocess the volume 
    vol = vol/tns_norm(vol)
    img_size = vol.shape[0]
    ds_res = 64
    vol_ds = vol_downsample(vol, ds_res)
    vol_ds = vol_ds/tns_norm(vol_ds)
    vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
    sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
    a = np.real(sphFB_c_t_r @ vol_coef)
    na = len(a)
    vol_coef = sphFB_r_t_c @ a 
    vol = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol = vol.reshape([ds_res,ds_res,ds_res])
    
    out['vol_coef'] = vol_coef
    out['vol'] = vol
    np.savez(out_file_name, **out)

    fname = 'vol_'+str(ell_max_vol)+'.mrc'
    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(vol)  # Set the volume data
        mrc.voxel_size = 1.0  

    # set up viewing direction distribution 
    c = 12
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    kappa = 4
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    ell_max_half_view = 2
    sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)
    # f_sph, ths, phs = plot_sph_harm(sph_coef, ell_max_half_view, fname='spherical_ground_truth.pdf')
    rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
    rot_coef[0] = 1
    sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)
    b = np.real(sph_c_t_r @ rot_coef)
    rot_coef = sph_r_t_c @ b
    b = b[1:]
    
    out['sph_coef'] = sph_coef
    np.savez(out_file_name, **out)


    momsout = np.load('subsmom.npz')
    m1_emp = momsout['m1_emp'] 
    m2_emp = momsout['m2_emp'] 
    m3_emp = momsout['m3_emp'] 
    U2 = momsout['U2'] 
    U3 = momsout['U3'] 


    res = sequential_moment_matching(m1_emp,m2_emp,m3_emp,U2,U3,ds_res,ell_max_vol,ell_max_half_view,L2=16,L3=16)
    for key, value in res.items():
        out[key] = value
    np.savez(out_file_name, **out)

    # align the volume 
    x1 = out['x1']
    a_est1 = x1[:na]
    b_est1 = x1[na:]
    vol_coef_est_m1 = sphFB_r_t_c @ a_est1
    out['a_est1'] = a_est1
    out['b_est1'] = b_est1
    out['vol_coef_est_m1'] = vol_coef_est_m1
    vol_est_m1 = coef_t_vol(vol_coef_est_m1, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m1 = vol_est_m1.reshape(ds_res,ds_res,ds_res)


    x2 = out['x2']
    a_est2 = x2[:na]
    b_est2 = x2[na:]
    vol_coef_est_m2 = sphFB_r_t_c @ a_est2
    out['a_est2'] = a_est2
    out['b_est2'] = b_est2
    out['vol_coef_est_m2'] = vol_coef_est_m2
    vol_est_m2 = coef_t_vol(vol_coef_est_m2, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m2 = vol_est_m2.reshape(ds_res,ds_res,ds_res)


    x3 = out['x3']
    a_est3 = x3[:na]
    b_est3 = x3[na:]
    vol_coef_est_m3 = sphFB_r_t_c @ a_est3
    out['a_est3'] = a_est3
    out['b_est3'] = b_est3
    out['vol_coef_est_m3'] = vol_coef_est_m3
    vol_est_m3 = coef_t_vol(vol_coef_est_m3, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m3 = vol_est_m3.reshape(ds_res,ds_res,ds_res)
    np.savez(out_file_name, **out)


    # align volume
    Vol_gt = Volume.load("vol_gt_L=12.mrc", dtype=np.float32)
    Vol1 = Volume(vol_est_m1)
    Vol2 = Volume(vol_est_m2)
    Vol3 = Volume(vol_est_m3)

    para =['wemd',64,300,True] 
    [R01,R1]=align_BO(Vol_gt,Vol1,para,reflect=True) 
    [R02,R2]=align_BO(Vol_gt,Vol2,para,reflect=True) 
    [R03,R3]=align_BO(Vol_gt,Vol3,para,reflect=True) 


    vol_gt = Vol_gt.asnumpy()
    vol_gt = vol_gt[0]
    vol1_aligned = Vol1.rotate(Rotation(R1)).asnumpy()
    vol1_aligned = np.array(vol1_aligned[0],dtype=np.float32)
    vol2_aligned = Vol2.rotate(Rotation(R2)).asnumpy()
    vol2_aligned = np.array(vol2_aligned[0],dtype=np.float32)
    vol3_aligned = Vol3.rotate(Rotation(R3)).asnumpy()
    vol3_aligned = np.array(vol3_aligned[0],dtype=np.float32)

    fname1 = 'vol_est_m1_aligned_'+str(ell_max_vol)+'.mrc'
    fname2 = 'vol_est_m2_aligned_'+str(ell_max_vol)+'.mrc'
    fname3 = 'vol_est_m3_aligned_'+str(ell_max_vol)+'.mrc'
    with mrcfile.new(fname1, overwrite=True) as mrc:
        mrc.set_data(vol1_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    with mrcfile.new(fname2, overwrite=True) as mrc:
        mrc.set_data(vol2_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    with mrcfile.new(fname3, overwrite=True) as mrc:
        mrc.set_data(vol3_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  

    resolution1 = get_fsc(vol_ds, vol1_aligned, pixel_size*img_size/ds_res)
    resolution2 = get_fsc(vol_ds, vol2_aligned, pixel_size*img_size/ds_res)
    resolution3 = get_fsc(vol_ds, vol3_aligned, pixel_size*img_size/ds_res)
    # alignment
    out['R1'] = R1
    out['R2'] = R2
    out['R3'] = R3
    out['vol1_aligned'] = vol1_aligned
    out['vol2_aligned'] = vol2_aligned
    out['vol3_aligned'] = vol3_aligned
    
    out['resolution1'] = resolution1
    out['resolution2'] = resolution2
    out['resolution3'] = resolution3
    np.savez(out_file_name, **out)
    


if __name__ == "__main__":
    vol_path = '../data/emd_34948.map'

    ell_max_vol = 4 
    out_file_name = 'out_4.npz'
    run_reconstruction(vol_path, ell_max_vol, out_file_name)


    ell_max_vol = 6 
    out_file_name = 'out_6.npz'
    run_reconstruction(vol_path, ell_max_vol, out_file_name)


    ell_max_vol = 8
    out_file_name = 'out_8.npz'
    run_reconstruction(vol_path, ell_max_vol, out_file_name)


    ell_max_vol = 10 
    out_file_name = 'out_10.npz'
    run_reconstruction(vol_path, ell_max_vol, out_file_name)



    ell_max_vol = 12
    out_file_name = 'out_12.npz'
    run_reconstruction(vol_path, ell_max_vol, out_file_name)

    