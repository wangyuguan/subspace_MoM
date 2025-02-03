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
    print('getting the first moment')
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M1 += weights[i]*rot_density[i]*fft_Img


    # form the projected second moment 
    euler_nodes, weights = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M2 = 0 
    G = np.random.normal(0,1,[n_grid, r2_max])
    print('getting the second moment')
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M2 += (weights[i]*rot_density[i]*fft_Img) @ (np.conj(fft_Img).T @ G)

    U2, S2, Vh2 = LA.svd(M2, full_matrices=False)

    m2 = 0 
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        fft_Img = np.conj(U2).T @ fft_Img
        m2 += weights[i]*rot_density[i]*(fft_Img @ np.conj(fft_Img).T)


    # form the projected third moment 
    print('getting the third moment')
    euler_nodes, weights = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M3 = 0 
    G1 = np.random.normal(0,1,[n_grid, r3_max])
    G2 = np.random.normal(0,1,[n_grid, r3_max])
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M3 += (weights[i]*rot_density[i]*fft_Img) @ ((np.conj(fft_Img).T @ G1) * (np.conj(fft_Img).T @ G2))

    U3, S3, Vh3 = LA.svd(M3, full_matrices=False)
    m3 = 0
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = np.conj(U3).T @ fft_Img
        m3 += weights[i]*rot_density[i]*np.einsum('i,j,k->ijk', fft_Img, fft_Img, fft_Img)
    

    subMoMs = {}
    subMoMs['G'] = G 
    subMoMs['G1'] = G1 
    subMoMs['G2'] = G2 
    subMoMs['M1'] = M1 
    subMoMs['m1'] = np.conj(U2).T @ M1 
    subMoMs['M2'] = M2 
    subMoMs['m2'] = m2 
    subMoMs['M3'] = M3 
    subMoMs['m3'] = m3 
    subMoMs['U2'] = U2 
    subMoMs['U3'] = U3 
    return subMoMs


def precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid):

    U1 = subspaces['m1']
    U2 = subspaces['m2']
    U3 = subspaces['m3']

    nodes1, weights1 = quadrature_rules['m1']
    nodes2, weights2 = quadrature_rules['m2']
    nodes3, weights3 = quadrature_rules['m3']

    Phi_precomps, Psi_precomps = {}, {}


    return Phi_precomps, Psi_precomps

def precomp_sphFB_all(U, ell_max, k_max, r0, indices, euler_nodes, grid):
    
    c = 0.5 
    ndim = U.shape[1]
    n_so3 = euler_nodes.shape[0]
    n_basis = len(indices)
    n_grid = len(grid.rs)
    r_idx =  (grid.rs>c)
    Phi_precomp = np.zeros((n_so3, ndim, n_basis), dtype=np.complex128)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(grid.ths))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,n_grid), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*grid.phs)

    sphFB_r_t_c, _ = get_sphFB_r_t_c_mat(ell_max, k_max, indices)

    jlk = {} 
    for ell in range(0,ell_max+1):
        for k in range(0,k_max[ell]):
            z0k = r0[ell][k]
            js = spherical_jn(ell, grid.rs*z0k/c)
            djs = spherical_jn(ell, z0k, True)
            js = js*np.sqrt(2/c**3)/abs(djs)
            js[r_idx] = 0
            jlk[(ell,k)] = js 

    Yl = {} 
    for ell in range(0,ell_max+1):
        yl = np.zeros((n_grid, 2*ell+1), dtype=np.complex128)
        for m in range(-ell,ell+1):
            lpmn = lpall[ell,abs(m),:]
            if m<0:
                lpmn = (-1)**m * lpmn
            yl[:,m+ell] = lpmn*exp_all[m+ell_max,:]
        Yl[ell] = yl  


    for i in range(n_so3):
        alpha, beta, gamma = euler_nodes[i,:]
        for ell in range(0,ell_max+1):
            D_l = wignerD(ell, alpha, beta, gamma)
            Yl_rot = Yl[ell] @ np.conj(D_l).T 
            for k in range(0,k_max[ell]):
                Flk = np.einsum('i,ij->ij', jlk[(ell,k)], Yl_rot)
                Phi_precomp[i,:,indices[(ell,k,-ell)]:indices[(ell,k,ell)]+1] = np.conj(U).T @ Flk

        Phi_precomp[i,:,:] = Phi_precomp[i,:,:] @ sphFB_r_t_c


    return Phi_precomp


def precomp_wignerD_all(ell_max_half, euler_nodes):
    ell_max = 2*ell_max_half
    n_grid = euler_nodes.shape[0]
    indices = {}
    n_coef = 0 
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1

    sph_r_t_c , _ =  get_sph_r_t_c_mat(ell_max_half)
    Psi_precomp = np.zeros((n_grid, n_coef), dtype=np.complex128)
    for i in range(n_grid):
        alpha,beta,gamma = euler_nodes[i,:]
        for ell in range(ell_max+1):
            if ell%2 == 0:
                Dl = wignerD(ell,alpha,beta,gamma)
                Psi_precomp[i,indices[(ell,-ell)]:indices[(ell,ell)]+1] = Dl[:,ell]
    
    Psi_precomp = np.real(Psi_precomp @ sph_r_t_c)
    
    return Psi_precomp