import numpy as np 
import numpy.linalg as LA 
import jax
from jax import jit
import jax.numpy as jnp
from jax.numpy.linalg import norm
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


def find_cost_grad(a, b, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):
    
    _, w_so3 = quadrature_rules['m2']
    Phi = Phi_precomps['m2']
    Psi = Psi_precomps['m3']

    # covert to jax array 
    a, b, w_so3 = jnp.array(a), jnp.array(b), jnp.array(w_so3)

    # compute the cost and gradient from the three moments
    cost1, grad1 = find_cost_grad_m1(a, b, w_so3, Phi, Psi, m1_emp, l1)
    cost2, grad2 = find_cost_grad_m2(a, b, w_so3, Phi, Psi, m2_emp, l2)
    cost3, grad3 = find_cost_grad_m3(a, b, w_so3, Phi, Psi, m3_emp, l3)
    cost = cost1+cost2+cost3 
    grad = grad1+grad2+grad3 
    return np.array(cost), np.array(grad)

# def find_cost_grad_m1(x, w_so3, Phi, Psi, m1_emp, l1):
    
#     na = Phi.shape[2]
#     a, b = x[:na],x[na:]
#     b1 = np.concatenate([np.array([1]), b])
#     n = len(w_so3)
#     PCs = np.einsum('ijk,k->ij', Phi, a)
#     w = w_so3*np.real(Psi @ b1) 

#     m1_emp = m1_emp.flatten()
#     m1 = np.zeros(PCs.shape[1], dtype=np.complex128)
#     for i in range(n):
#         m1 = m1+w[i]*PCs[i,:]
#     C1 = m1-m1_emp
#     C1_conj = np.conj(C1)

#     cost = LA.norm(C1.flatten())**2 


#     grad_a = np.zeros(Phi.shape[2])
#     grad_rho = np.zeros(n)
#     for i in range(n):
#         grad_a = grad_a + 2*w[i]*np.real(np.conj(Phi[i,:,:]).T @ C1)
#         # print(PCs[i,:].shape, C1_conj.shape)
#         grad_rho[i] = 2*w_so3[i]*np.sum(np.real(PCs[i,:]*C1_conj))

#     grad_b = np.conj(Psi).T @ grad_rho
#     grad = np.concatenate([grad_a, grad_b[1:]])

#     return cost / l1,  np.real(grad) / l1


@jit 
def find_cost_m1(x, w_so3, Phi, Psi, m1_emp, l1):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    m1_emp = m1_emp.flatten()
    m1 = jnp.zeros(PCs.shape[1], dtype=jnp.complex128)
    
    def body_fun(i, m1):
        return m1 + w[i] * PCs[i, :]
    
    m1 = jax.lax.fori_loop(0, n, body_fun, m1)
    
    C1 = m1 - m1_emp
    cost = norm(C1.flatten())**2 
    
    return cost / l1
    


@jit 
def find_cost_grad_m1(x, w_so3, Phi, Psi, m1_emp, l1):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    m1_emp = m1_emp.flatten()
    m1 = jnp.zeros(PCs.shape[1], dtype=jnp.complex128)
    
    def body_fun(i, m1):
        return m1 + w[i] * PCs[i, :]
    
    m1 = jax.lax.fori_loop(0, n, body_fun, m1)
    
    C1 = m1 - m1_emp
    C1_conj = jnp.conj(C1)
    
    cost = norm(C1.flatten())**2 
    
    grad_a = jnp.zeros(Phi.shape[2])
    grad_rho = jnp.zeros(n)
    
    def grad_body_fun(i, val):
        grad_a, grad_rho = val
        grad_a += 2 * w[i] * jnp.real(jnp.conj(Phi[i, :, :]).T @ C1)
        grad_rho = grad_rho.at[i].set(2 * w_so3[i] * jnp.sum(jnp.real(PCs[i, :] * C1_conj)))
        return grad_a, grad_rho
    
    grad_a, grad_rho = jax.lax.fori_loop(0, n, grad_body_fun, (grad_a, grad_rho))
    
    grad_b = jnp.conj(Psi).T @ grad_rho
    grad = jnp.concatenate([grad_a, grad_b[1:]])
    
    return cost / l1, jnp.real(grad) / l1
    

# def find_cost_grad_m2(x, w_so3, Phi, Psi, m2_emp, l2):
    
#     na = Phi.shape[2]
#     a, b = x[:na],x[na:]
#     b1 = np.concatenate([np.array([1]), b])
#     n = len(w_so3)
#     PCs = np.einsum('ijk,k->ij', Phi, a)
#     w = w_so3*np.real(Psi @ b1) 

#     d = PCs.shape[1]
#     m2 = np.zeros((d,d), dtype=np.complex128)
#     PC_dots = np.zeros((n,d,d), dtype=np.complex128)
#     for i in range(n):
#         Img =  PCs[i,:].reshape((-1,1))
#         PC_dots[i,:,:] = Img @ np.conj(Img).T
#         m2 = m2+w[i]*PC_dots[i,:,:]
#     C2 = m2-m2_emp
#     C2_conj = np.conj(C2)

#     cost = LA.norm(C2.flatten())**2 


#     grad_a = np.zeros(Phi.shape[2])
#     grad_rho = np.zeros(n)
#     for i in range(n):
#         grad_a = grad_a + 4*w[i]*np.real(Phi[i,:,:].T @ (C2_conj @ np.conj(PCs[i,:])))
#         grad_rho[i] = 2*w_so3[i]*np.real(np.sum(PC_dots[i,:,:]*C2_conj))

#     grad_b = np.conj(Psi).T @ grad_rho
#     grad = np.concatenate([grad_a, grad_b[1:]])

#     return cost / l2,  np.real(grad) / l2


@jit 
def find_cost_m2(x, w_so3, Phi, Psi, m2_emp, l2):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m2 = jnp.zeros((d, d), dtype=jnp.complex128)
    
    def body_fun(i, m2):
        Img = PCs[i, :].reshape((-1, 1))
        PC_dot = Img @ jnp.conj(Img).T
        return m2 + w[i] * PC_dot
    
    m2 = jax.lax.fori_loop(0, n, body_fun, m2)
    
    C2 = m2 - m2_emp
    cost = norm(C2.flatten())**2 
    return cost / l2


@jit 
def find_cost_grad_m2(x, w_so3, Phi, Psi, m2_emp, l2):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m2 = jnp.zeros((d, d), dtype=jnp.complex128)
    
    def body_fun(i, m2):
        Img = PCs[i, :].reshape((-1, 1))
        PC_dot = Img @ jnp.conj(Img).T
        return m2 + w[i] * PC_dot
    
    m2 = jax.lax.fori_loop(0, n, body_fun, m2)
    
    C2 = m2 - m2_emp
    C2_conj = jnp.conj(C2)

    cost = norm(C2.flatten())**2 
    
    grad_a = jnp.zeros(Phi.shape[2])
    grad_rho = jnp.zeros(n)
    
    def grad_body_fun(i, val):
        grad_a, grad_rho = val
        grad_a += 4 * w[i] * jnp.real(Phi[i, :, :].T @ (C2_conj @ jnp.conj(PCs[i, :])))
        Img = PCs[i, :].reshape((-1, 1))
        PC_dot = Img @ jnp.conj(Img).T  # Compute on-the-fly
        grad_rho = grad_rho.at[i].set(2 * w_so3[i] * jnp.real(jnp.sum(PC_dot * C2_conj)))
        return grad_a, grad_rho
    
    grad_a, grad_rho = jax.lax.fori_loop(0, n, grad_body_fun, (grad_a, grad_rho))
    
    grad_b = jnp.conj(Psi).T @ grad_rho
    grad = jnp.concatenate([grad_a, grad_b[1:]])
    
    return cost / l2, jnp.real(grad) / l2


# def find_cost_grad_m3(x, w_so3, Phi, Psi, m3_emp, l3):
    
#     na = Phi.shape[2]
#     a, b = x[:na],x[na:]
#     b1 = np.concatenate([np.array([1]), b])
#     n = len(w_so3)
#     PCs = np.einsum('ijk,k->ij', Phi, a)
#     w = w_so3*np.real(Psi @ b1) 

#     d = PCs.shape[1]
#     m3 = np.zeros((d,d,d), dtype=np.complex128)
#     PC_dots = np.zeros((n,d,d,d), dtype=np.complex128)
#     for i in range(n):
#         Img =  PCs[i,:]
#         PC_dots[i,:,:,:] = np.einsum('i,j,k->ijk',Img,Img,Img)
#         m3 = m3+w[i]*PC_dots[i,:,:,:]
#     C3 = m3-m3_emp
#     C3_conj = np.conj(C3)
#     C3_conj_mat = np.reshape(C3_conj,[d,d**2])

#     cost = LA.norm(C3.flatten())**2 


#     grad_a = np.zeros(Phi.shape[2])
#     grad_rho = np.zeros(n)
#     for i in range(n):
#         Img = PCs[i,:]
#         Img2 = np.einsum('i,j->ij',Img,Img)
#         tmp = C3_conj_mat @ Img2.flatten()
#         grad_a = grad_a + 6*w[i]*np.real(Phi[i,:,:].T @ tmp)
#         grad_rho[i] = 2*w_so3[i]*np.real(np.sum(PC_dots[i,:,:,:]*C3_conj))

#     grad_b = np.conj(Psi).T @ grad_rho
#     grad = np.concatenate([grad_a, grad_b[1:]])

#     return cost / l3,  np.real(grad) / l3


@jit 
def find_cost_m3(x, w_so3, Phi, Psi, m3_emp, l3):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m3 = jnp.zeros((d, d, d), dtype=jnp.complex128)
    
    def body_fun(i, m3):
        Img = PCs[i, :]
        PC_dot = jnp.einsum('i,j,k->ijk', Img, Img, Img)
        return m3 + w[i] * PC_dot
    
    m3 = jax.lax.fori_loop(0, n, body_fun, m3)
    
    C3 = m3 - m3_emp
    C3_conj = jnp.conj(C3)

    cost = norm(C3.flatten())**2 
    return cost / l3


@jit 
def find_cost_grad_m3(x, w_so3, Phi, Psi, m3_emp, l3):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0]), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m3 = jnp.zeros((d, d, d), dtype=jnp.complex128)
    
    def body_fun(i, m3):
        Img = PCs[i, :]
        PC_dot = jnp.einsum('i,j,k->ijk', Img, Img, Img)
        return m3 + w[i] * PC_dot
    
    m3 = jax.lax.fori_loop(0, n, body_fun, m3)
    
    C3 = m3 - m3_emp
    C3_conj = jnp.conj(C3)
    C3_conj_mat = C3_conj.reshape(d, d**2)

    cost = norm(C3.flatten())**2 
    
    grad_a = jnp.zeros(Phi.shape[2])
    grad_rho = jnp.zeros(n)
    
    def grad_body_fun(i, val):
        grad_a, grad_rho = val
        Img = PCs[i, :]
        Img2 = jnp.einsum('i,j->ij', Img, Img)
        tmp = C3_conj_mat @ Img2.flatten()
        grad_a += 6 * w[i] * jnp.real(Phi[i, :, :].T @ tmp)
        PC_dot = jnp.einsum('i,j,k->ijk', Img, Img, Img)  # Compute PC_dot on-the-fly
        grad_rho = grad_rho.at[i].set(2 * w_so3[i] * jnp.real(jnp.sum(PC_dot * C3_conj)))
        return grad_a, grad_rho
    
    grad_a, grad_rho = jax.lax.fori_loop(0, n, grad_body_fun, (grad_a, grad_rho))
    
    grad_b = jnp.conj(Psi).T @ grad_rho
    grad = jnp.concatenate([grad_a, grad_b[1:]])
    
    return cost / l3 , jnp.real(grad) / l3
    


def precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid):

    U2 = subspaces['m2']
    U3 = subspaces['m3']

    euler_nodes2, _ = quadrature_rules['m2']
    euler_nodes3, _ = quadrature_rules['m3']

    Phi_precomps, Psi_precomps = {}, {}

    Phi_precomps['m2'] = precomp_sphFB_all(U2, ell_max_vol, k_max, r0, indices_vol, euler_nodes2, grid)
    Phi_precomps['m3'] = precomp_sphFB_all(U3, ell_max_vol, k_max, r0, indices_vol, euler_nodes3, grid)

    Psi_precomps['m2'] = precomp_wignerD_all(ell_max_half_view, euler_nodes2)
    Psi_precomps['m3'] = precomp_wignerD_all(ell_max_half_view, euler_nodes3)

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


    return jnp.array(Phi_precomp)


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
    
    return jnp.array(Psi_precomp)