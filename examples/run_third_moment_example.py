import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../../fast-cryoEM-PCA').resolve()
sys.path.append(str(src_path))
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))


import mrcfile
import time 
import numpy as np 
import numpy.linalg as LA 
from viewing_direction import *
from utils import *
from optimization import *
from volume import *
from moments import * 



def run_third_moment_analytical_test(vol_path, tol3, out_file_name):

    np.random.seed(2025)
    out = {}

    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    
    # preprocess the volume 
    vol = vol/tns_norm(vol)
    img_size = vol.shape[0]
    ds_res = 64
    vol_ds = vol_downsample(vol, ds_res)
    ell_max_vol = 5
    vol_ds = vol_ds/tns_norm(vol_ds)
    vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
    sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
    a = np.real(sphFB_c_t_r @ vol_coef)
    vol_coef = sphFB_r_t_c @ a 
    vol = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol = vol.reshape([ds_res,ds_res,ds_res])
    
    out['vol_coef'] = vol_coef
    out['vol'] = vol
    np.savez(out_file_name, **out)
    

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
    
    out['sph_coef'] = sph_coef
    np.savez(out_file_name, **out)



    # form the moments 
    opts = {}
    opts['r2_max'] = 250
    opts['r3_max'] = 120
    opts['tol2'] = 1e-12
    opts['tol3'] = tol3
    opts['img_size'] = img_size
    opts['ds_res'] = ds_res
    grid = get_2d_unif_grid(ds_res,1/ds_res)
    grid = Grid_3d(xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))
    opts['grid'] = grid
    moms_out = analytical_subspace_moments_gaussian(vol_coef, ell_max_vol, k_max, r0, indices_vol, rot_coef, ell_max_half_view, opts)
    m1_emp = moms_out['m1']
    m2_emp = moms_out['m2']
    m3_emp = moms_out['m3'] 
    U1 = moms_out['U1']
    U2 = moms_out['U2']
    U3 = moms_out['U3']
    out['relerr2'] = moms_out['relerr2']
    out['relerr3'] = moms_out['relerr3']
    out['m1_emp'] = m1_emp
    out['m2_emp'] = m2_emp
    out['m3_emp'] = m3_emp
    out['U1'] = U1
    out['U2'] = U2
    out['U3'] = U3
    np.savez(out_file_name, **out)


    # precomputation
    subspaces = {}
    subspaces['m1'] = U1
    subspaces['m2'] = U2 
    subspaces['m3'] = U3 
    
    
    quadrature_rules = {} 
    quadrature_rules['m1'] = load_so3_quadrature(ell_max_vol, 2*ell_max_half_view)
    quadrature_rules['m2'] = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
    quadrature_rules['m3'] = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)
    

    print('doing precomputation')
    Phi_precomps, Psi_precomps = precomputation_analytical_test(ell_max_vol, k_max, 
                                                                r0, indices_vol, ell_max_half_view, 
                                                                subspaces, quadrature_rules, grid)


    # constraint 
    na = len(indices_vol)
    nb = len(indices_view)-1
    view_constr, rhs, _ = get_linear_ineqn_constraint(ell_max_half_view)
    A_constr = np.zeros([len(rhs), na+nb])
    A_constr[:,na:] = view_constr 



    # initialization 
    a0 = 1e-6*np.random.normal(0,1,a.shape)
    c = 5 
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    kappa = 2
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    sph_coef, indices = sph_harm_transform(my_fun, ell_max_half_view)
    rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
    rot_coef[0] = 1
    b0 = sph_c_t_r @ rot_coef
    b0 = np.real(b0[1:])

    x0 = jnp.concatenate([a0,b0])
    
    l1 = tns_norm(m1_emp)**2
    l2 = tns_norm(m2_emp)**2
    l3 = tns_norm(m3_emp)**2
    
    
    # sequential moment matching 
    t_m1 = time.time()
    res1 = moment_LS_analytical_test(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l1=l1, l2=0, l3=0)
    x1 = res1.x
    t_m1 = time.time()-t_m1
    
    
    t_m2 = time.time()
    res2 = moment_LS_analytical_test(x1, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l1=l1, l2=l2, l3=0)
    x2 = res2.x
    t_m2 = time.time()-t_m2


    t_m3 = time.time()
    res3 = moment_LS_analytical_test(x2, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l1=l1, l2=l2, l3=l3)
    x3 = res3.x
    t_m3 = time.time()-t_m3

    
    out['x1'] = x1
    out['x2'] = x2    
    out['x3'] = x3    
    out['t_m1'] = t_m1
    out['t_m2'] = t_m2
    out['t_m3'] = t_m3
    np.savez(out_file_name, **out)


    # align the volume 
    a_est1 = x1[:na]
    b_est1 = x1[na:]
    vol_coef_est_m1 = sphFB_r_t_c @ a_est1
    out['a_est1'] = a_est1
    out['b_est1'] = b_est1
    out['vol_coef_est_m1'] = vol_coef_est_m1
    
    a_est2 = x2[:na]
    b_est2 = x2[na:]
    vol_coef_est_m2 = sphFB_r_t_c @ a_est2
    out['a_est2'] = a_est2
    out['b_est2'] = b_est2
    out['vol_coef_est_m2'] = vol_coef_est_m2


    a_est3 = x3[:na]
    b_est3 = x3[na:]
    vol_coef_est_m3 = sphFB_r_t_c @ a_est3
    out['a_est3'] = a_est3
    out['b_est3'] = b_est3
    out['vol_coef_est_m3'] = vol_coef_est_m3
    np.savez(out_file_name, **out)
    
    
    vol_coef_est_m1, R1 , _ , ref1 = align_vol_coef(vol_coef,vol_coef_est_m1,ell_max_vol,k_max,indices_vol)
    vol_coef_est_m2, R2 , _ , ref2 = align_vol_coef(vol_coef,vol_coef_est_m2,ell_max_vol,k_max,indices_vol)
    vol_coef_est_m3, R3 , _ , ref3 = align_vol_coef(vol_coef,vol_coef_est_m3,ell_max_vol,k_max,indices_vol)
    
    vol_est_m1_aligned = coef_t_vol(vol_coef_est_m1, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m2_aligned = coef_t_vol(vol_coef_est_m2, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m3_aligned = coef_t_vol(vol_coef_est_m3, ell_max_vol, ds_res, k_max, r0, indices_vol)
    
    vol_est_m1_aligned = np.reshape(vol_est_m1_aligned, [ds_res,ds_res,ds_res])
    vol_est_m2_aligned = np.reshape(vol_est_m2_aligned, [ds_res,ds_res,ds_res])
    vol_est_m3_aligned = np.reshape(vol_est_m3_aligned, [ds_res,ds_res,ds_res])
    
    out['vol_est_m1_aligned'] = vol_est_m1_aligned
    out['vol_est_m2_aligned'] = vol_est_m2_aligned
    out['vol_est_m3_aligned'] = vol_est_m3_aligned
    np.savez(out_file_name, **out)
    
    
    
    with mrcfile.new('vol_est_m1_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_est_m1_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    
    with mrcfile.new('vol_est_m2_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_est_m2_aligned)  # Set the volume data
        mrc.voxel_size = 1.0 

    with mrcfile.new('vol_est_m3_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_est_m3_aligned)  # Set the volume data
        mrc.voxel_size = 1.0 

    # align the viewing direction distribution
    sph_coef_est_m1 = np.insert(b_est1,0,1)
    sph_coef_est_m1_aligned = rot_t_sph_coef(sph_r_t_c @ sph_coef_est_m1, ell_max_half_view)
    if ref1:
        sph_coef_est_m1_aligned = reflect_sph_coef(sph_coef_est_m1_aligned, ell_max_half_view)
        sph_coef_est_m1_aligned = rotate_sph_coef(sph_coef_est_m1_aligned, R1, ell_max_half_view)
    else:
        sph_coef_est_m1_aligned = rotate_sph_coef(sph_coef_est_m1, R1, ell_max_half_view)
    _ = plot_sph_harm(sph_coef_est_m1_aligned, ell_max_half_view, fname="m1_viewing_recons.pdf")

    sph_coef_est_m2 = np.insert(b_est2,0,1)
    sph_coef_est_m2_aligned = rot_t_sph_coef(sph_r_t_c @ sph_coef_est_m2, ell_max_half_view)
    if ref2:
        sph_coef_est_m2_aligned = reflect_sph_coef(sph_coef_est_m2_aligned, ell_max_half_view)
        sph_coef_est_m2_aligned = rotate_sph_coef(sph_coef_est_m2_aligned, R2, ell_max_half_view)
    else:
        sph_coef_est_m2_aligned = rotate_sph_coef(sph_coef_est_m2, R2, ell_max_half_view)
    _ = plot_sph_harm(sph_coef_est_m2_aligned, ell_max_half_view, fname="m2_viewing_recons.pdf")


    sph_coef_est_m3 = np.insert(b_est3,0,1)
    sph_coef_est_m3_aligned = rot_t_sph_coef(sph_r_t_c @ sph_coef_est_m3, ell_max_half_view)
    if ref3:
        sph_coef_est_m3_aligned = reflect_sph_coef(sph_coef_est_m3_aligned, ell_max_half_view)
        sph_coef_est_m3_aligned = rotate_sph_coef(sph_coef_est_m3_aligned, R3, ell_max_half_view)
    else:
        sph_coef_est_m3_aligned = rotate_sph_coef(sph_coef_est_m3, R3, ell_max_half_view)
    _ = plot_sph_harm(sph_coef_est_m3_aligned, ell_max_half_view, fname="m3_viewing_recons.pdf")



    out['sph_coef_est_m1_aligned'] = sph_coef_est_m1_aligned
    out['sph_coef_est_m2_aligned'] = sph_coef_est_m2_aligned
    out['sph_coef_est_m3_aligned'] = sph_coef_est_m3_aligned
    np.savez(out_file_name, **out)

    # f_sph_m1, _, _ = plot_sph_harm(sph_coef_est_m1_aligned, ell_max_half_view, fname='spherical_recons_m1.pdf')
    # f_sph_m2, _, _ = plot_sph_harm(sph_coef_est_m2_aligned, ell_max_half_view, fname='spherical_recons_m2.pdf')
    



if __name__ == "__main__":
    vol_path = '../data/emd_34948.map'
    
    tol3 = 1e-4 
    out_file_name = 'third_moment_analytical_test_4.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)

    tol3 = 1e-5
    out_file_name = 'third_moment_analytical_test_5.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)

    tol3 = 1e-6 
    out_file_name = 'third_moment_analytical_test_6.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)

    tol3 = 1e-7 
    out_file_name = 'third_moment_analytical_test_7.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)

    tol3 = 1e-8 
    out_file_name = 'third_moment_analytical_test_8.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)


    tol3 = 1e-9
    out_file_name = 'third_moment_analytical_test_9.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)


    tol3 = 1e-10
    out_file_name = 'third_moment_analytical_test_10.npz'
    run_third_moment_analytical_test(vol_path, tol3, out_file_name)


    