import numpy as np 
import numpy.linalg as LA 
import jax
from jax import jit
import jax.numpy as jnp
from jax.numpy.linalg import norm
from utils import * 
from viewing_direction import * 
from volume import * 
import e3x
import scipy 
import math 
import time 
from scipy.optimize import minimize, BFGS
from scipy.linalg import svd
#from aspire.volume import Volume
#from aspire.utils.rotation import Rotation


def momentPCA_rNLA(vol, rots, params):  
    t_start = time.time() 
    Ntot = rots.shape[0]
    Nbat = 1000
    r2_max = params['r2_max']
    r3_max = params['r3_max']
    tol2 = params['tol2']
    tol3 = params['tol3']
    ds_res = params['ds_res']
    ds_res2 = ds_res**2
    
    G = np.random.normal(0,1,(ds_res2, r2_max))
    G1 = np.random.normal(0,1,(ds_res2, r3_max))
    G2 = np.random.normal(0,1,(ds_res2, r3_max))
    nstream = math.ceil(Ntot/Nbat)

    # Vol = Volume(vol)
    M2 = np.zeros((ds_res**2,r2_max))
    M3 = np.zeros((ds_res**2,r3_max))
    for i in range(nstream):
        t1 = time.time()
        print('sketching stream '+str(i+1)+' out of '+str(nstream)+' streams')
        _rots = rots[((nstream-1)*Nbat):min((nstream*Nbat),Ntot),:,:]     
        # Rots = Rotation(_rots)
        # imags = Vol.project(Rots).downsample(ds_res=ds_res, zero_nyquist=False).asnumpy()
        imags = vol_proj(vol, _rots)
        imags = image_downsample(imags, ds_res)
        
        for imag in imags:
            I = imag.reshape(ds_res2, 1, order='F').astype(np.float64)
            I_trans = I.T
            M2 = M2 + I @ (I_trans @ G)/Ntot
            M3 = M3 + I @ ((I_trans @ G1) * (I_trans @ G2))/Ntot 
        t2 = time.time() 
        print('spent '+str(t2-t1)+' seconds')
    
    U2, S2, _ = svd(M2, full_matrices=False)
    r2 = np.argmax(np.cumsum(S2**2) / np.sum(S2**2) > (1 - tol2))
    U2 = U2[:,0:r2]
    U3, S3, _ = svd(M3, full_matrices=False)
    r3 = np.argmax(np.cumsum(S3**2) / np.sum(S3**2) > (1 - tol3))
    U3 = U3[:,0:r3]
    
    U2_fft = np.zeros(U2.shape, dtype=np.complex128)
    for i in range(r2):
        img = U2[:,i].reshape(ds_res, ds_res, order='F')
        img_fft = centered_fft2(img)/ds_res 
        U2_fft[:,i] = img_fft.flatten(order='F')
        
    U3_fft = np.zeros(U3.shape, dtype=np.complex128)
    for i in range(r3):
        img = U3[:,i].reshape(ds_res, ds_res, order='F')
        img_fft = centered_fft2(img)/ds_res 
        U3_fft[:,i] = img_fft.flatten(order='F')
        
    t_end = time.time() 
    
    return U2, U3, U2_fft, U3_fft, t_end-t_start 


def form_subspace_moments(vol, rots, U2, U3):
    t_start = time.time() 
    ds_res2, r2 = U2.shape
    ds_res = int(math.sqrt(ds_res2))
    r3 = U3.shape[1]
    Ntot = rots.shape[0]
    Nbat = 1000
    nstream = math.ceil(Ntot/Nbat)
    
    # Vol = Volume(vol)
    m1 = np.zeros((r2,1))
    m2 = np.zeros((r2,r2))
    m3 = np.zeros((r3,r3,r3))
    for i in range(nstream):
        t1 = time.time()
        print('forming from stream '+str(i+1)+' out of '+str(nstream)+' streams')
        _rots = rots[((nstream-1)*Nbat):min((nstream*Nbat),Ntot),:,:]     
        # Rots = Rotation(_rots)
        # imags = Vol.project(Rots).downsample(ds_res=ds_res, zero_nyquist=False).asnumpy()
        imags = vol_proj(vol, _rots)
        imags = image_downsample(imags, ds_res)
        
        for imag in imags:
            I = imag.reshape(ds_res2, 1, order='F').astype(np.float64)
            I2 = U2.T @ I 
            I3 = U3.T @ I 
            I3 = I3.flatten()
            m1 = m1+I2/Ntot 
            m2 = m2+(I2@I2.T)/Ntot 
            m3 = m3+np.einsum('i,j,k->ijk',I3,I3,I3)/Ntot 
        t2 = time.time() 
        print('spent '+str(t2-t1)+' seconds')
    
    
    m1 = m1*ds_res 
    m2 = m2*ds_res**2 
    m3 = m3*ds_res**3 
    t_end = time.time() 
    
    return m1,m2,m3,t_end-t_start 
    


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
    M1 = np.zeros([n_grid,1], dtype=np.complex128)
    print('getting the first moment')
    for i in range(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M1 += weights[i]*rot_density[i]*fft_Img
        fft_Img = fft_Img.flatten()

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


    U2, S2, _ = svd(M2, full_matrices=False)
    cumulative_energy = np.cumsum(S2**2) / np.sum(S2**2)
    r2 = np.argmax(cumulative_energy > (1 - tol2)) + 1
    U2 = U2[:,:r2]


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
        M3 += (weights[i]*rot_density[i]*fft_Img) @ ((fft_Img.T @ G1) * (fft_Img.T @ G2))

    U3, S3, _ = svd(M3, full_matrices=False)
    cumulative_energy = np.cumsum(S3**2) / np.sum(S3**2)
    r3 = np.argmax(cumulative_energy > (1 - tol3)) + 1
    U3 = U3[:,:r3]


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
    subMoMs['S2'] = S2
    subMoMs['S3'] = S3
    return subMoMs


def moment_LS(x0, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, A_constr, b_constr, l1=None, l2=None, l3=None):
    
    if l1 is None:
        l1 = LA.norm(m1_emp.flatten())**2
    if l2 is None:
        l2 = LA.norm(m2_emp.flatten())**2
    if l3 is None:
        l3 = LA.norm(m3_emp.flatten())**2
    
    linear_constraint = {'type': 'ineq', 'fun': lambda x: b_constr - A_constr @ x}
    objective = lambda x: find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3)
    result = minimize(objective, x0, method='SLSQP', jac=True, constraints=[linear_constraint], options={'disp': True,'maxiter':5000, 'ftol':1e-9, 'iprint':2, 'eps': 1e-4})
    # result = minimize(objective, x0, method='trust-constr', jac=True, hess=BFGS(), constraints=[linear_constraint], options={'disp': True,'maxiter':5000, 'verbose':3, 'initial_tr_radius':0.1})
    
    return result 



def find_cost(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):
    
    _, w_so3_m2 = quadrature_rules['m2']
    _, w_so3_m3 = quadrature_rules['m3']
    # Phi = Phi_precomps['m2']
    # Psi = Psi_precomps['m3']

    # covert to jax array 
    x, w_so3_m2, w_so3_m3 = jnp.array(x), jnp.array(w_so3_m2), jnp.array(w_so3_m3)

    # compute the cost and gradient from the three moments
    if l1>0:
       cost1 = find_cost_m1(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m1_emp, l1)
    else:
        cost1 = 0
    if l2>0:
        cost2 = find_cost_m2(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m2_emp, l2)
    else:
        cost2 = 0
    if l3>0:
        cost3 = find_cost_m3(x, w_so3_m3, Phi_precomps['m3'], Psi_precomps['m3'], m3_emp, l3)
    else:
        cost3 = 0

    cost = cost1+cost2+cost3 
    return cost 


def find_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):
    
    _, w_so3_m2 = quadrature_rules['m2']
    _, w_so3_m3 = quadrature_rules['m3']
    # Phi = Phi_precomps['m2']
    # Psi = Psi_precomps['m3']

    # covert to jax array 
    x, w_so3_m2, w_so3_m3 = jnp.array(x), jnp.array(w_so3_m2), jnp.array(w_so3_m3)

    # compute the cost and gradient from the three moments
    if l1>0:
       grad1 = find_grad_m1(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m1_emp, l1)
    else:
        grad1 = jnp.zeros(x.shape)
    if l2>0:
        grad2 = find_grad_m2(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m2_emp, l2)
    else:
        grad2 = jnp.zeros(x.shape)
    if l3>0:
        cost3, grad3 = find_grad_m3(x, w_so3_m3, Phi_precomps['m3'], Psi_precomps['m3'], m3_emp, l3)
    else:
        grad3 = jnp.zeros(x.shape)


    grad = grad1+grad2+grad3 
    return grad


def find_cost_grad(x, quadrature_rules, Phi_precomps, Psi_precomps, m1_emp, m2_emp, m3_emp, l1, l2, l3):
    
    _, w_so3_m2 = quadrature_rules['m2']
    _, w_so3_m3 = quadrature_rules['m3']
    # Phi = Phi_precomps['m2']
    # Psi = Psi_precomps['m3']

    # covert to jax array 
    x, w_so3_m2, w_so3_m3 = jnp.array(x), jnp.array(w_so3_m2), jnp.array(w_so3_m3)

    # compute the cost and gradient from the three moments
    if l1>0:
       cost1, grad1 = find_cost_grad_m1(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m1_emp, l1)
    else:
        cost1, grad1 = 0, np.zeros(x.shape)
    if l2>0:
        cost2, grad2 = find_cost_grad_m2(x, w_so3_m2, Phi_precomps['m2'], Psi_precomps['m2'], m2_emp, l2)
    else:
        cost2, grad2 = 0, np.zeros(x.shape)
    if l3>0:
        cost3, grad3 = find_cost_grad_m3(x, w_so3_m3, Phi_precomps['m3'], Psi_precomps['m3'], m3_emp, l3)
    else:
        cost3, grad3 = 0, np.zeros(x.shape)

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
def find_grad_m1(x, w_so3, Phi, Psi, m1_emp, l1):
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
    
    return jnp.real(grad) / l1
    
    


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
def find_grad_m2(x, w_so3, Phi, Psi, m2_emp, l2):
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
    
    return jnp.real(grad) / l2


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
    cost = norm(C3.flatten())**2 
    return cost / l3


@jit 
def find_grad_m3(x, w_so3, Phi, Psi, m3_emp, l3):
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
    
    return jnp.real(grad) / l3


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
            # js[r_idx] = 0
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






