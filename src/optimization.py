import numpy as np 
import numpy.linalg as LA 
import jax
from jax import jit
import jax.numpy as jnp
from utils import * 
from viewing_direction import * 
from volume import * 
import math 
import time 
from scipy.optimize import minimize
from scipy.linalg import svd
from aspire.volume import Volume 
from aspire.numeric import fft
from aspire.source.simulation import Simulation
from tqdm import tqdm 
from tqdm import trange



def sequential_moment_matching(m1_emp,m2_emp,m3_emp,U2,U3,ds_res,ell_max_vol,ell_max_half_view,L2=None,L3=None):
    subspaces = {}
    subspaces['m2'] = U2 
    subspaces['m3'] = U3 
    
    quadrature_rules = {} 
    if L2 is None:
        L2 = 2*ell_max_vol
    if L3 is None:
        L3 = 3*ell_max_vol
    quadrature_rules['m2'] = load_so3_quadrature(L2, 2*ell_max_half_view)
    quadrature_rules['m3'] = load_so3_quadrature(L3, 2*ell_max_half_view)
    
    grid = get_2d_unif_grid(ds_res,1/ds_res)
    grid = Grid_3d(xs=grid.xs, ys=grid.ys, zs=np.zeros(grid.ys.shape))

    k_max, r0 = calc_k_max(ell_max_vol,ds_res,3)
    indices_vol = {}
    i = 0 
    for ell in range(ell_max_vol+1):
        for k in range(k_max[ell]):
            for m in range(-ell,ell+1):
                indices_vol[(ell,k,m)] = i
                i += 1 

    ell_max_view = 2*ell_max_half_view
    i = 0 
    indices_view = {}
    for ell in np.arange(ell_max_view+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices_view[(ell,m)] = i
              i += 1


    print('precomputation')
    t_precomp = time.time()
    Phi_precomps, Psi_precomps = precomputation(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid)
    
    
    na = len(indices_vol)
    nb = len(indices_view)-1
    view_constr, rhs, _ = get_linear_ineqn_constraint(ell_max_half_view)
    A_constr = np.zeros([len(rhs), na+nb])
    A_constr[:,na:] = view_constr 
    t_precomp = time.time() - t_precomp


    sphFB_r_t_c, sphFB_c_t_r = get_sphFB_r_t_c_mat(ell_max_vol, k_max, indices_vol)
    sph_r_t_c , sph_c_t_r =  get_sph_r_t_c_mat(ell_max_half_view)

    
    l1 = LA.norm(m1_emp.flatten())**2
    l2 = LA.norm(m2_emp.flatten())**2
    l3 = LA.norm(m3_emp.flatten())**2

    nc = 10
    np.random.seed(42)
    a0 = 1e-6*np.random.normal(0,1,na)
    centers = np.random.normal(0,1,size=(nc,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,nc)
    w_vmf = w_vmf/np.sum(w_vmf)
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return 4*np.pi*vMF_density(centers,w_vmf,2,grid)[0]
    sph_coef, _ = sph_harm_transform(my_fun, ell_max_half_view)
    rot_coef = sph_t_rot_coef(sph_coef, ell_max_half_view)
    rot_coef[0] = 1
    b0 = sph_c_t_r @ rot_coef
    b0 = jnp.real(b0[1:])
    x0 = jnp.concatenate([a0,b0])

    # fit m1 
    t_m1 = time.time()
    res1 = moment_LS(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l1=l1,l2=0,l3=0)
    x1 = res1.x
    a_est = x1[:na]
    vol_coef_est_m1 = sphFB_r_t_c @ a_est
    t_m1  = time.time()-t_m1

    # fit m1 and m2 
    t_m2 = time.time()
    res2 = moment_LS(x1, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs, l1=l1,l2=l2,l3=0)
    x2 = res2.x
    t_m2  = time.time()-t_m2
    a_est = x2[:na]
    vol_coef_est_m2 = sphFB_r_t_c @ a_est

    # fit m1, m2 and m3
    t_m3 = time.time()
    res3 = moment_LS(x2, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, rhs)
    t_m3 = time.time()-t_m3 
    x3 = res3.x 
    a_est = x3[:na]
    b_est = x3[na:]
    vol_coef_est_m3 = sphFB_r_t_c @ a_est

    res = {}
    res['a_est'] = a_est 
    res['b_est'] = b_est 
    res['vol_coef_est_m1'] = vol_coef_est_m1
    res['vol_coef_est_m2'] = vol_coef_est_m2
    res['vol_coef_est_m3'] = vol_coef_est_m3
    res['t_precomp'] = t_precomp 
    res['x1'] = x1
    res['x2'] = x2
    res['x3'] = x3
    res['t_m1'] = t_m1
    res['t_m2'] = t_m2
    res['t_m3'] = t_m3

    return res 



def moment_LS(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, b_constr, l1=None, l2=None, l3=None):
    
    if l1 is None:
        l1 = LA.norm(m1_emp.flatten())**2
    if l2 is None:
        l2 = LA.norm(m2_emp.flatten())**2
    if l3 is None:
        l3 = LA.norm(m3_emp.flatten())**2
    
    linear_constraint = {'type': 'ineq', 'fun': lambda x: b_constr - A_constr @ x}
    objective = lambda x: find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3)
    result = minimize(objective, x0, method='SLSQP', jac=True, constraints=[linear_constraint],
        options={'disp': True,'maxiter':1000, 'ftol':1e-8, 'iprint':2, 'eps': 1e-4})
    
    return result 



def moment_LS_analytical_test(x0, quadrature_rules, Phi_precomps, Psi_precomps, 
                              m1_emp, m2_emp, m3_emp, 
                              A_constr, b_constr, 
                              l1=None, l2=None, l3=None):
    
    if l1 is None:
        l1 = LA.norm(m1_emp.flatten())**2
    if l2 is None:
        l2 = LA.norm(m2_emp.flatten())**2
    if l3 is None:
        l3 = LA.norm(m3_emp.flatten())**2
    
    linear_constraint = {'type': 'ineq', 'fun': lambda x: b_constr - A_constr @ x}
    objective = lambda x: find_cost_grad_analytical_test(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3)
    result = minimize(objective, x0, method='SLSQP', jac=True, constraints=[linear_constraint],
        options={'disp': True,'maxiter':1000, 'ftol':1e-8, 'iprint':2, 'eps': 1e-4})


    return result 


def find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):
    
    _, w_so3_m2 = quadrature_rules['m2']
    _, w_so3_m3 = quadrature_rules['m3']

    # covert to jax array 
    x, w_so3_m2, w_so3_m3 = jnp.array(x), jnp.array(w_so3_m2), jnp.array(w_so3_m3)

    # compute the cost and gradient from the three moments
    if l1>0:
       cost1, grad1 = find_cost_grad_m1_einsum(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m1_emp, l1)
    else:
        cost1, grad1 = 0, np.zeros(x.shape)
    if l2>0:
        cost2, grad2 = find_cost_grad_m2_einsum(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m2_emp, l2)
    else:
        cost2, grad2 = 0, np.zeros(x.shape)
    if l3>0:
        cost3, grad3 = find_cost_grad_m3_einsum(x, w_so3_m3, Phi_precomps['m3'], Psi_precomps['m3'], m3_emp, l3)
    else:
        cost3, grad3 = 0, np.zeros(x.shape)

    cost = cost1+cost2+cost3 
    grad = grad1+grad2+grad3 
    return np.array(cost), np.array(grad)


def find_cost_grad_analytical_test(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):

    _, w_so3_m1 = quadrature_rules['m1']
    _, w_so3_m2 = quadrature_rules['m2']
    _, w_so3_m3 = quadrature_rules['m3']


    # covert to jax array 
    x, w_so3_m1, w_so3_m2, w_so3_m3 = jnp.array(x), jnp.array(w_so3_m1), jnp.array(w_so3_m2), jnp.array(w_so3_m3)

    # compute the cost and gradient from the three moments
    if l1>0:
        cost1, grad1 = find_cost_grad_m1_einsum(x, w_so3_m1, Phi_precomps['m1'], Psi_precomps['m1'], m1_emp, l1)
    else:
        cost1, grad1 = 0, np.zeros(x.shape)
    if l2>0:
        cost2, grad2 = find_cost_grad_m2_einsum(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m2_emp, l2)
    else:
        cost2, grad2 = 0, np.zeros(x.shape)
    if l3>0:
        cost3, grad3 = find_cost_grad_m3_einsum(x, w_so3_m3, Phi_precomps['m3'], Psi_precomps['m3'], m3_emp, l3)
    else:
        cost3, grad3 = 0, np.zeros(x.shape)

    cost = cost1+cost2+cost3 
    grad = grad1+grad2+grad3 
    return np.array(cost), np.array(grad)






@jit 
def find_cost_grad_m1(x, w_so3, Phi, Psi, m1_emp, l1):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0],dtype=b.dtype), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    m1_emp = m1_emp.flatten()
    m1 = jnp.zeros(PCs.shape[1], dtype=m1_emp.dtype)
    
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
    


@jit 
def find_cost_grad_m2(x, w_so3, Phi, Psi, m2_emp, l2):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0], dtype=b.dtype), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m2 = jnp.zeros((d, d), dtype=m2_emp.dtype)
    
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



@jit 
def find_cost_grad_m3(x, w_so3, Phi, Psi, m3_emp, l3):
    na = Phi.shape[2]
    a, b = x[:na], x[na:]
    b1 = jnp.concatenate([jnp.array([1.0],dtype=b.dtype), b])
    n = len(w_so3)
    PCs = jnp.einsum('ijk,k->ij', Phi, a)
    w = w_so3 * jnp.real(Psi @ b1)

    d = PCs.shape[1]
    m3 = jnp.zeros((d, d, d), dtype=m3_emp.dtype)
    
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



@jit
def find_cost_grad_m1_einsum(x, w_so3, Phi, Psi, m1_emp, l1):

    na    = Phi.shape[2]
    a, b  = x[:na], x[na:]
    b1    = jnp.concatenate([jnp.array([1.0], dtype=b.dtype), b])  
    PCs = jnp.einsum('ijk,k->ij', Phi, a, optimize='greedy')       
    w   = w_so3 * jnp.real(Psi @ b1)           
    m1  = jnp.einsum('i,ip->p', w, PCs, optimize='greedy')         

    C1   = m1 - m1_emp.flatten()                
    cost = jnp.linalg.norm(C1)**2 / l1
    grad_a = 2 * jnp.real(
        jnp.einsum('i,j,ijk->k', w, C1, jnp.conj(Phi), optimize='greedy')
    )                                           
    tmp     = jnp.einsum('ij,j->i', PCs, jnp.conj(C1), optimize='greedy') 
    grad_rho = 2 * w_so3 * jnp.real(tmp)              
    grad_b = jnp.conj(Psi).T @ grad_rho               
    grad   = jnp.concatenate([grad_a, grad_b[1:]]) / l1

    return cost, grad
    


@jit
def find_cost_grad_m2_einsum(x, w_so3, Phi, Psi, m2_emp, l2):

    na    = Phi.shape[2]
    a, b  = x[:na], x[na:]
    b1    = jnp.concatenate([jnp.array([1.0], dtype=b.dtype), b])

    PCs = jnp.einsum('ijk,k->ij', Phi, a, optimize='greedy')         
    w   = w_so3 * jnp.real(Psi @ b1)             
    m2 = jnp.einsum('i,ip,iq->pq', w, PCs, jnp.conj(PCs), optimize='greedy')

    C2  = m2 - m2_emp
    cost = jnp.linalg.norm(C2)**2 / l2
    C2c = jnp.conj(C2)
    v = jnp.einsum('rq,iq->ir', C2c, jnp.conj(PCs), optimize='greedy') 

    grad_a = 4 * jnp.real(
        jnp.einsum('ir,irk->k', w[:,None] * v, Phi, optimize='greedy')
    )                                                
    grad_rho = 2 * w_so3 * jnp.real(
        jnp.einsum('pq,ip,iq->i', C2c, PCs, jnp.conj(PCs), optimize='greedy')
    )                                                
    grad_b = jnp.conj(Psi).T @ grad_rho              
    grad   = jnp.concatenate([grad_a, grad_b[1:]]) / l2

    return cost, grad


@jit 
def find_cost_grad_m3_einsum(x, w_so3, Phi, Psi, m3_emp, l3):
    na      = Phi.shape[2]
    a, b    = x[:na], x[na:]
    b1      = jnp.concatenate([jnp.array([1.0]), b])
    PCs     = jnp.einsum('ijk,k->ij', Phi, a, optimize='greedy')       
    w       = w_so3 * jnp.real(Psi @ b1)            

    m3      = jnp.einsum('i,ip,iq,ir->pqr', w, PCs, PCs, PCs, optimize='greedy')
    C3      = m3 - m3_emp
    C3c     = jnp.conj(C3)
    cost    = jnp.linalg.norm(C3)**2 / l3
    tmp     = jnp.einsum('pqr,ip,iq->ir', C3c, PCs, PCs, optimize='greedy')   

    grad_a  = 6 * jnp.real(
                  jnp.einsum('i,ir,irk->k', w, tmp, Phi, optimize='greedy')
              )
    grad_rho = 2 * w_so3 * jnp.real(jnp.sum(tmp * PCs, axis=1))
    grad_b  = jnp.conj(Psi).T @ grad_rho
    grad    = jnp.concatenate([grad_a, grad_b[1:]]) / l3

    return cost, grad



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



def precomputation_analytical_test(ell_max_vol, k_max, r0, indices_vol, ell_max_half_view, subspaces, quadrature_rules, grid):

    U1 = subspaces['m1']
    U2 = subspaces['m2']
    U3 = subspaces['m3']

    euler_nodes1, _ = quadrature_rules['m1']
    euler_nodes2, _ = quadrature_rules['m2']
    euler_nodes3, _ = quadrature_rules['m3']

    Phi_precomps, Psi_precomps = {}, {}

    Phi_precomps['m1'] = precomp_sphFB_all(U1, ell_max_vol, k_max, r0, indices_vol, euler_nodes1, grid)
    Phi_precomps['m2'] = precomp_sphFB_all(U2, ell_max_vol, k_max, r0, indices_vol, euler_nodes2, grid)
    Phi_precomps['m3'] = precomp_sphFB_all(U3, ell_max_vol, k_max, r0, indices_vol, euler_nodes3, grid)

    Psi_precomps['m1'] = precomp_wignerD_all(ell_max_half_view, euler_nodes1)
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
    Phi_precomp = np.zeros((n_so3, ndim, n_basis), dtype=np.complex64)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(grid.ths))
    lpall = np.array(lpall / np.sqrt(4*np.pi), dtype=np.complex64)

    exp_all = np.zeros((2*ell_max+1,n_grid), dtype=np.complex64)
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
        yl = np.zeros((n_grid, 2*ell+1), dtype=np.complex64)
        for m in range(-ell,ell+1):
            lpmn = lpall[ell,abs(m),:]
            if m<0:
                lpmn = (-1)**m * lpmn
            yl[:,m+ell] = lpmn*exp_all[m+ell_max,:]
        Yl[ell] = yl  


    for i in tqdm(range(n_so3)):
        alpha, beta, gamma = euler_nodes[i,:]
        for ell in range(0,ell_max+1):
            D_l = wignerD(ell, alpha, beta, gamma)
            Yl_rot = Yl[ell] @ np.conj(D_l).T 
            for k in range(0,k_max[ell]):
                Flk = np.einsum('i,ij->ij', jlk[(ell,k)], Yl_rot)
                Phi_precomp[i,:,indices[(ell,k,-ell)]:indices[(ell,k,ell)]+1] = np.conj(U).T @ Flk

        Phi_precomp[i,:,:] = Phi_precomp[i,:,:] @ sphFB_r_t_c


    return jnp.array(Phi_precomp, dtype=jnp.complex64)




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
    Psi_precomp = np.zeros((n_grid, n_coef), dtype=np.complex64)
    for i in tqdm(range(n_grid)):
        alpha,beta,gamma = euler_nodes[i,:]
        for ell in range(ell_max+1):
            if ell%2 == 0:
                Dl = wignerD(ell,alpha,beta,gamma)
                Psi_precomp[i,indices[(ell,-ell)]:indices[(ell,ell)]+1] = Dl[:,ell]
    
    Psi_precomp = np.real(Psi_precomp @ sph_r_t_c)
    
    return jnp.array(Psi_precomp, dtype=jnp.float32)




def get_linear_ineqn_constraint(ell_max_half):
    
    ell_max = 2*ell_max_half
    n_coef = 0 
    indices = {}
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1

    data = np.genfromtxt('../data/sphere_rules/N030_M322_C4.dat',skip_header=2)
    nodes = data[:,0:3]
    _, betas, alphas = cart2sph(nodes[:,0], nodes[:,1], nodes[:,2])
    n_nodes = nodes.shape[0]

    lpall = norm_assoc_legendre_all(ell_max, np.cos(betas))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(alphas)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*alphas)

    sph_r_t_c, _  = get_sph_r_t_c_mat(ell_max_half)
    Psi = np.zeros((n_nodes, n_coef), dtype=np.complex128)
    for ell in range(0,ell_max+1):
        if ell % 2 ==0:
          for m in range(-ell,ell+1):
              lpmn = lpall[ell,abs(m),:]
              exps = exp_all[m+ell_max,:]
              if m<0:
                  lpmn = lpmn*(-1)**m
              ylm = lpmn*exps*np.sqrt(4*np.pi/(2*ell+1))
              ylm = np.conj(ylm)
              Psi[:,indices[(ell,m)]] = ylm 
    Psi = Psi @ sph_r_t_c
    A = -Psi[:,1:]
    b = Psi[:,0]
    return jnp.real(A), jnp.real(b), Psi 

