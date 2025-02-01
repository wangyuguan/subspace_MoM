import numpy as np 
import numpy.linalg as LA 
from utils import * 
from viewing_direction import * 
from volume import * 
import time 


def coef_t_subspace_moments(vol_coef, ell_max_vol, k_max, r0, indices_vol, rot_coef, ell_max_half_view, opts):
    
    c = 0.5 
    r2_max = opts['r2_max']
    r3_max = opts['r3_max']
    sketch_size = opts['sketch_size']
    tol2 = opts['tol2']
    tol3 = opts['tol3']
    grid = opts['grid']

    n_grid = len(grid.xs)
    n_basis = len(indices_vol)
    
    precomp_vol_basis = precompute_sphFB_basis(ell_max_vol, k_max, r0, indices_vol, grid)

    # form the uncompressed first moment  
    euler_nodes, weights = load_so3_quadrature(ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M1 = 0 
    print('forming the first moment')
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        M1 += weights[i]*rot_density[i]*fft_Img


    # form the projected second moment 
    euler_nodes, weights = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M2 = 0 
    G = np.random.normal(0,1,[n_grid, sketch_size])
    print('sampling the second moment')
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M2 += (weights[i]*rot_density[i]*fft_Img) @ (np.conj(fft_Img).T @ G)

    subMoMs = {}
    subMoMs['G'] = G 
    subMoMs['M1'] = M1 
    subMoMs['M2'] = M2 
    return subMoMs