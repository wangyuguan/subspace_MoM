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

def image_subspace_moments_invariant():
    return


def ctf_image_subspace_moments_CUR(vol, rotmats, h_ctfs, opts):
    tol2 = opts['tol2'] 
    tol3 = opts['tol3'] 
    nI2 = opts['nI2'] 
    nI3 = opts['nI3']
    nJ = opts['nJ']
    r2_max = opts['r2_max']
    r3_max = opts['r3_max'] 
    ds_res = opts['ds_res'] 
    defocus_ct = opts['defocus_ct'] 
    batch_size = opts['batch_size'] 
    var = opts['var'] 
    img_size = vol.shape[0]
    # plan = plan_vol_ds_proj_ft(img_size)
    img_size2 = img_size**2 
    num_img = rotmats.shape[0]
    ds_res2 = ds_res**2 
    ind = get_subindices(img_size, ds_res)
    I = jnp.array(sample_idxs(img_size,ds_res,60))
    I2 = jnp.array(sample_idxs(img_size,ds_res,nI2))
    I3 = jnp.array(sample_idxs(img_size,ds_res,nI3))
    J1 = jnp.array(sample_idxs(img_size,ds_res,nJ))
    J2 = jnp.array(sample_idxs(img_size,ds_res,nJ))
    M1_ctf = 0 
    M2_ctf_I2 = 0 
    # M2_ctf = 0
    M3_ctf_I3 = 0 
    M3_ctf_J = 0 
    M2_ctf_I = 0 
    M3_ctf_I = 0 
    mask_M1 = 0 
    mask_I2 = 0 
    # mask_M2 = 0 
    mask_I3 = 0 
    mask_J = 0 
    mask_II = 0 
    mask_III = 0
    t_form = 0 
    for i in trange(defocus_ct):
        np.random.seed(i)
        _rotmats = rotmats[(i*batch_size):(i+1)*batch_size,:,:]    
        H_full =  h_ctfs[i].evaluate_grid(img_size)
        projs = vol_proj(vol, _rotmats)
        imgs = fft.centered_ifft2(fft.centered_fft2(projs)*H_full).real
        H = H_full.flatten(order='F')[ind]
        H2 = H**2
        imgs_fft = fft.centered_fft2(projs)
        if var>0:
            noise = np.sqrt(var)*np.random.normal(0,1,(batch_size,img_size,img_size))
            imgs = imgs+noise 
        t1 = time.time()
        imgs_fft = image_downsample(imgs, ds_res, False, False)
        imgs_fft = jnp.array(imgs_fft.reshape((batch_size, ds_res2), order='F')*H[None,:], dtype=jnp.complex64)
        imgs_fft_I = imgs_fft[:,I]
        imgs_fft_I2 = imgs_fft[:,I2]
        imgs_fft_I3 = imgs_fft[:,I3]
        imgs_fft_J1 = imgs_fft[:,J1]
        imgs_fft_J2 = imgs_fft[:,J2]
        
        M1_ctf = M1_ctf+jnp.einsum('ni->i',imgs_fft,optimize='greedy')/num_img
        M2_ctf_I2 = M2_ctf_I2+jnp.einsum('ni,nj->ij',imgs_fft,jnp.conj(imgs_fft_I2),optimize='greedy')/num_img
        # M2_ctf = M2_ctf+jnp.einsum('ni,nj->ij',imgs_fft,jnp.conj(imgs_fft),optimize='greedy')/num_img
        M3_ctf_I3 = M3_ctf_I3+jnp.einsum('ni,nj,nk->ijk',imgs_fft_I3,imgs_fft_I3,imgs_fft_I3,optimize='greedy')/num_img
        M3_ctf_J = M3_ctf_J+jnp.einsum('ni,nj,nk->ijk',imgs_fft,imgs_fft_J1,imgs_fft_J2,optimize='greedy')/num_img

        M2_ctf_I = M2_ctf_I+jnp.einsum('ni,nj->ij',imgs_fft_I,jnp.conj(imgs_fft_I),optimize='greedy')/num_img
        M3_ctf_I = M3_ctf_I+jnp.einsum('ni,nj,nk->ijk',imgs_fft_I,imgs_fft_I,imgs_fft_I,optimize='greedy')/num_img

        mask_M1 = mask_M1+H2/defocus_ct
        mask_I2 = mask_I2+jnp.einsum('i,j->ij',H2,H2[I2],optimize='greedy')/defocus_ct
        # mask_M2 = mask_M2+jnp.einsum('i,j->ij',H2,H2,optimize='greedy')/defocus_ct
        H2_I3 = H2[I3]
        mask_I3 = mask_I3+jnp.einsum('i,j,k->ijk',H2_I3,H2_I3,H2_I3,optimize='greedy')/defocus_ct
        mask_J = mask_J+jnp.einsum('i,j,k->ijk',H2,H2[J1],H2[J2],optimize='greedy')/defocus_ct

        H2_I = H2[I]
        mask_II = mask_II+jnp.einsum('i,j->ij',H2_I,H2_I,optimize='greedy')/defocus_ct
        mask_III = mask_III+jnp.einsum('i,j,k->ijk',H2_I,H2_I,H2_I,optimize='greedy')/defocus_ct
        t2 = time.time()
        t_form = t_form+t2-t1 

    t1 = time.time()
    if var>0:
        
        _M1_ctf = M1_ctf.flatten(order='F')

        p = np.arange(img_size2)
        i =  p %  img_size
        j =  p // img_size  
        i_flip = (-i) % img_size
        j_flip = (-j) % img_size
        p_flip_full = i_flip + img_size * j_flip  
        rev = np.full(img_size2, -1, dtype=int)
        rev[ind] = np.arange(ds_res2)             
        p_flip_sub = p_flip_full[ind]               
        P_sub_idx = rev[p_flip_sub]                
        P_sub = np.zeros((ds_res2, ds_res2), dtype=int)
        rows = np.arange(ds_res2)
        idx = (P_sub_idx >= 0)
        P_sub[rows[idx], P_sub_idx[idx]] = 1
        scale = (ds_res**4/img_size**2)


        _M1_ctf_I  = _M1_ctf[I]       
        _M1_ctf_I3 = _M1_ctf[I3]
        _M1_ctf_J1 = _M1_ctf[J1]
        _M1_ctf_J2 = _M1_ctf[J2]
        
        P_sub_I = P_sub[np.ix_(I, I)]
        P_sub_I3 = P_sub[np.ix_(I3, I3)]
        P_sub_J1J2  = P_sub[np.ix_(J1, J2)]
        P_sub_J2 = P_sub[:,J2]    
        P_sub_J1 = P_sub[:,J1] 

        B2_I = 0 
        B2_I2 = 0
        B3_J = 0
        B3_I = 0
        B3_I3 = 0
        _scale = (scale*var/defocus_ct)
        for i in trange(defocus_ct):
            H = h_ctfs[i].evaluate_grid(img_size)
            H = H.flatten(order='F')[ind]

            H_I = H[I]       
            H_I3 = H[I3] 
            H_J1 = H[J1]
            H_J2 = H[J2]   

            FF_trans_I = H_I[:, None]*P_sub_I *H_I[None, :]
            FF_trans_I3 = H_I3[:, None]*P_sub_I3 *H_I3[None, :]

            B2_I = B2_I + _scale *jnp.diag(H_I**2)
            B2_I2 = B2_I2 + _scale *(jnp.diag(H)@jnp.diag(H)[:,I2])
            B3_I = B3_I + _scale *jnp.einsum('i,jk->ijk',_M1_ctf_I,FF_trans_I) 
            B3_I3 = B3_I3 + _scale *jnp.einsum('i,jk->ijk',_M1_ctf_I3,FF_trans_I3)
        
            
            A = H_J1[:, None]*P_sub_J1J2*H_J2[None, :]
            B = H[:, None]*P_sub_J2*H_J2[None, :]  
            C = H[:, None]*P_sub_J1*H_J1[None, :] 

            term1 = _M1_ctf[:, None, None]*A[None, :, :]
            term2 = _M1_ctf_J1[None, :, None]*B[:, None, :]
            term3 = C[:, :, None]*_M1_ctf_J2[None, None, :]

            B3_J = B3_J + _scale *(term1+term2+term3) 


        B3_I = B3_I + B3_I.transpose(1,0,2) + B3_I.transpose(1,2,0)
        B3_I3 = B3_I3 + B3_I3.transpose(1,0,2) + B3_I3.transpose(1,2,0) 
        M2_ctf_I = M2_ctf_I-B2_I
        M2_ctf_I2 = M2_ctf_I2-B2_I2
        M3_ctf_I = M3_ctf_I-B3_I
        M3_ctf_I3 = M3_ctf_I3-B3_I3
        M3_ctf_J = M3_ctf_J-B3_J

        
    # deconvolve and compress
    M2_I2 = M2_ctf_I2/mask_I2
    U2_cur = M2_I2    
    m2_cur = LA.pinv(M2_I2[I2,:],1e-5)
    m2, U2, r2, _ = trim_symmetric_lowrank(m2_cur, U2_cur, tol2, r2_max)
    print('r2='+str(r2))

    M3_I3 = M3_ctf_I3/mask_I3
    M3_J = M3_ctf_J/mask_J
    M3_J = M3_J.reshape(ds_res2,-1,order='F')
    U3_cur = M3_J @ LA.pinv(M3_J[I3,:],1e-5)
    m3_cur = M3_I3
    m3, U3, r3, _ = trim_symmetric_tucker(m3_cur, U3_cur, tol3, r3_max)
    print('r3='+str(r3))

    M1 = M1_ctf/mask_M1
    m1 = jnp.conj(U2).T @ M1 
    t2 = time.time()
    t_form = t_form+t2-t1 

    M2_est = M2_ctf_I/mask_II
    M3_est = M3_ctf_I/mask_III
    M2_approx = U2[I,:] @ m2 @ jnp.conj(U2[I,:]).T
    M3_approx = jnp.einsum('abc,ia,jb,kc->ijk',m3,U3[I,:],U3[I,:],U3[I,:])
    relerr2 = tns_norm(M2_approx-M2_est)/tns_norm(M2_est)
    relerr3 = tns_norm(M3_approx-M3_est)/tns_norm(M3_est)


    print(relerr2)
    print(relerr3)

    subMoMs = {}
    subMoMs['relerr2'] = relerr2
    subMoMs['relerr3'] = relerr3
    subMoMs['m1'] = np.array(m1,dtype=np.complex64)
    subMoMs['m2'] = np.array(m2,dtype=np.complex64)
    subMoMs['m3'] = np.array(m3,dtype=np.complex64)
    subMoMs['U2'] = np.array(U2,dtype=np.complex64)
    subMoMs['U3'] = np.array(U3,dtype=np.complex64)
    subMoMs['M1'] = np.array(M1,dtype=np.complex64)
    subMoMs['m2_cur'] = np.array(m2_cur,dtype=np.complex64)
    subMoMs['m3_cur'] = np.array(m3_cur,dtype=np.complex64)
    subMoMs['U2_cur'] = np.array(U2_cur,dtype=np.complex64)
    subMoMs['U3_cur'] = np.array(U3_cur,dtype=np.complex64)
    subMoMs['I'] = np.array(I)
    subMoMs['M3_est'] = np.array(M3_est,dtype=np.complex64)
    subMoMs['M2_est'] = np.array(M2_est,dtype=np.complex64)
    subMoMs['t_form'] = t_form
    return subMoMs


def image_subspace_moments_gaussian(vol, rotmats, opts):
    r2_max = opts['r2_max'] 
    r3_max = opts['r3_max']
    tol2 = opts['tol2']
    tol3 = opts['tol3'] 
    ds_res = opts['ds_res'] 
    var = opts['var'] 
    img_size = vol.shape[0]
    num_img = rotmats.shape[0]
    batch_size = 1000
    num_stream = math.ceil(num_img/batch_size)
    ds_res2 = ds_res**2 

    G = jnp.array(np.random.normal(0,1,(ds_res2, r2_max)))
    G1 = jnp.array(np.random.normal(0,1,(ds_res2, r3_max)))
    G2 = jnp.array(np.random.normal(0,1,(ds_res2, r3_max)))

    M1 = 0
    M2_G = 0 
    M3_G = 0

    for i in tqdm(range(num_stream)):
        np.random.seed(i)
        _rotmats = rotmats[(i*batch_size):min((i+1)*batch_size,num_img),:,:]    
        projs_fft = jnp.array(vol_ds_proj_ft(vol, _rotmats, ds_res),dtype=jnp.complex64)
        noise = jnp.array(np.random.normal(0,np.sqrt(var),(projs_fft.shape[0],img_size,img_size)),dtype=jnp.float32)
        imgs_fft = projs_fft + image_downsample(noise, ds_res, False)

        
        imgs_fft = imgs_fft.reshape(imgs_fft.shape[0],ds_res**2,order='F')
        imgs_fft_G = jnp.conj(imgs_fft) @ G
        imgs_fft_G1 = imgs_fft @ G1
        imgs_fft_G2 = imgs_fft @ G2
        M1 = M1+jnp.einsum('ni->i',imgs_fft)/num_img
        M2_G = M2_G+jnp.einsum('ni,nj->ij',
                                       imgs_fft,imgs_fft_G,
                                       optimize='greedy')/num_img
        M3_G = M3_G+jnp.einsum('ni,nj,nj->ij',
                                       imgs_fft,imgs_fft_G1,imgs_fft_G2,
                                       optimize='greedy')/num_img
    
    
    if var>0:
        F = get_centered_fft2_submtx(img_size, row_id=get_subindices(img_size, ds_res))*(ds_res**2/img_size**2)
        F = jnp.array(F)
        B2_G = var*F@((jnp.conj(F).T)@G)
        M2_G = M2_G-B2_G

        X = G1.T @ F            
        Y = G2.T @ F           
        M = M1.flatten()  
        A = G1.T @ M            
        B = G2.T @ M   
        term1 = jnp.einsum('p,kn,kn->pk',M,X,Y)   
        term2 = jnp.einsum('pn,kn,k->pk',F,Y,A)    
        term3 = jnp.einsum('pn,kn,k->pk',F,X,B)  
        B3_G = var * (term1 + term2 + term3) 
        M3_G = M3_G-B3_G 

    U2,S2,_ = LA.svd(M2_G,False)
    r2 = np.argmax(np.cumsum(S2**2) / np.sum(S2**2) > (1 - tol2))+1
    U2 = U2[:,:r2]
    U3,S3,_ = LA.svd(M3_G,False)
    r3 = np.argmax(np.cumsum(S3**2) / np.sum(S3**2) > (1 - tol3))+1
    U3 = U3[:,:r3]

    m1 = np.conj(U2.T)@M1 
    m2 = 0 
    m3 = 0
    U2 = jnp.array(U2)
    U3 = jnp.array(U3)
    for i in tqdm(range(num_stream)):
        np.random.seed(i)
        _rotmats = rotmats[(i*batch_size):min((i+1)*batch_size,num_img),:,:] 
        projs_fft = jnp.array(vol_ds_proj_ft(vol, _rotmats, ds_res),dtype=jnp.complex64)
        noise = jnp.array(np.random.normal(0,np.sqrt(var),(projs_fft.shape[0],img_size,img_size)),dtype=jnp.float32)
        imgs_fft = projs_fft + image_downsample(noise, ds_res, False)


        
        imgs_fft = imgs_fft.reshape(imgs_fft.shape[0],ds_res**2,order='F')
        imgs_fft_U2 = jnp.einsum('ni,ij->nj',
                                 imgs_fft,jnp.conj(U2),
                                 optimize='greedy')
        m2 = m2+jnp.einsum('ni,nj->ij',
                                   imgs_fft_U2,jnp.conj(imgs_fft_U2),
                                   optimize='greedy')/num_img
    
        imgs_fft_U3 = jnp.einsum('ni,ij->nj',
                                 imgs_fft,jnp.conj(U3),
                                 optimize='greedy')
        m3 = m3+jnp.einsum('ni,nj,nk->ijk',
                                   imgs_fft_U3,imgs_fft_U3,imgs_fft_U3,
                                   optimize='greedy')/num_img
    
    if var>0:
        b2 = var*(jnp.conj(U2.T)@F)@((jnp.conj(F).T)@U2)
        m2 = m2-b2 

        U3_H = jnp.conj(U3).T        
        X = U3_H @ F             
        m = (U3_H @ M1).ravel() 
        term1 = jnp.einsum('a,bi,ci->abc',m,X,X) 
        term2 = term1.transpose(1,0,2)
        term3 = term1.transpose(1,2,0)
        b3 = var * (term1 + term2 + term3)  
        m3 = m3-b3  
        
            
    m1 = np.array(m1)
    m2 = np.array(m2)
    m3 = np.array(m3)
    U2 = np.array(U2)
    U3 = np.array(U3)

    return m1, m2, m3, U2, U3



def translated_image_subspace_moments_gaussian(vol, rotmats, opts):
    r2_max = opts['r2_max'] 
    r3_max = opts['r3_max']
    tol2 = opts['tol2']
    tol3 = opts['tol3'] 
    batch_size = opts['batch_size'] 
    batch_number = opts['batch_number'] 
    ds_res = opts['ds_res'] 
    var = opts['var'] 
    img_size = vol.shape[0]
    num_img = rotmats.shape[0]
    ds_res2 = ds_res**2 

    G = jnp.array(np.random.normal(0,1,(ds_res2, r2_max)))
    G1 = jnp.array(np.random.normal(0,1,(ds_res2, r3_max)))
    G2 = jnp.array(np.random.normal(0,1,(ds_res2, r3_max)))

    M1 = 0
    M2_G = 0 
    M3_G = 0

    offsets = np.sqrt(var)*np.random.normal(0,1,(num_img,2))

    sim = Simulation(L=img_size,
                    n=num_img,
                    amplitudes=1.0,
                    vols=Volume(vol),
                    offsets=offsets,
                    dtype=np.float32)
    sim.rotations = rotmats
    

    for i in trange(batch_number):   
        imgs = sim.images[(i*batch_size):min((i+1)*batch_size,num_img)].asnumpy()*ds_res
        # imgs = vol_proj(vol, rotmats[(i*batch_size):min((i+1)*batch_size,num_img),:,:])
        # imgs_fft = jnp.array(image_downsample(imgs, ds_res, if_real=False),dtype=jnp.complex64)
        imgs_fft = centered_fft2(imgs)
        imgs_fft = imgs_fft.reshape(imgs_fft.shape[0],ds_res**2,order='F')
        imgs_fft_G = jnp.conj(imgs_fft) @ G
        imgs_fft_G1 = imgs_fft @ G1
        imgs_fft_G2 = imgs_fft @ G2
        M1 = M1+jnp.einsum('ni->i',imgs_fft)/num_img
        M2_G = M2_G+jnp.einsum('ni,nj->ij',
                                       imgs_fft,imgs_fft_G,
                                       optimize='greedy')/num_img
        M3_G = M3_G+jnp.einsum('ni,nj,nj->ij',
                                       imgs_fft,imgs_fft_G1,imgs_fft_G2,
                                       optimize='greedy')/num_img


    U2,S2,_ = LA.svd(M2_G,False)
    r2 = np.argmax(np.cumsum(S2**2) / np.sum(S2**2) > (1 - tol2))+1
    if r2>r2_max:
        r2 = r2_max
    print('r2 =',str(r2))
    U2 = U2[:,:r2]
    
    U3,S3,_ = LA.svd(M3_G,False)
    r3 = np.argmax(np.cumsum(S3**2) / np.sum(S3**2) > (1 - tol3))+1
    if r3>r3_max:
        r3 = r3_max
    print('r3 =',str(r3))
    U3 = U3[:,:r3]

    m1 = np.conj(U2.T)@M1 
    m2 = 0 
    m3 = 0
    U2 = jnp.array(U2)
    U3 = jnp.array(U3)
    for i in trange(batch_number):
        imgs = sim.images[(i*batch_size):min((i+1)*batch_size,num_img)].asnumpy()*ds_res
        # imgs = vol_proj(vol, rotmats[(i*batch_size):min((i+1)*batch_size,num_img),:,:])
        # imgs_fft = jnp.array(image_downsample(imgs, ds_res, if_real=False, zero_nyquist=True),dtype=jnp.complex64)
        imgs_fft = centered_fft2(imgs)
        imgs_fft = imgs_fft.reshape(imgs_fft.shape[0],ds_res**2,order='F')
        imgs_fft_U2 = jnp.einsum('ni,ij->nj',
                                 imgs_fft,jnp.conj(U2),
                                 optimize='greedy')
        m2 = m2+jnp.einsum('ni,nj->ij',
                                   imgs_fft_U2,jnp.conj(imgs_fft_U2),
                                   optimize='greedy')/num_img
    
        imgs_fft_U3 = jnp.einsum('ni,ij->nj',
                                 imgs_fft,jnp.conj(U3),
                                 optimize='greedy')
        m3 = m3+jnp.einsum('ni,nj,nk->ijk',
                                   imgs_fft_U3,imgs_fft_U3,imgs_fft_U3,
                                   optimize='greedy')/num_img
            
            
    m1 = np.array(m1)
    m2 = np.array(m2)
    m3 = np.array(m3)
    U2 = np.array(U2)
    U3 = np.array(U3)

    return m1, m2, m3, U2, U3



def analytical_subspace_moments_gaussian(vol_coef, ell_max_vol, k_max, r0, indices_vol, rot_coef, ell_max_half_view, opts):
    np.random.seed(1)
    c = 0.5 
    r2_max = opts['r2_max']
    r3_max = opts['r3_max']
    tol2 = opts['tol2']
    tol3 = opts['tol3']
    grid = opts['grid']
    img_size = opts['img_size']
    ds_res = opts['ds_res']
    n_grid = len(grid.xs)
    n_basis = len(indices_vol)
    ind = get_subindices(img_size, ds_res)
    I = jnp.array(sample_idxs(img_size,ds_res,60))
    
    precomp_vol_basis = precompute_sphFB_basis(ell_max_vol, k_max, r0, indices_vol, grid)

    # form the uncompressed first moment  
    t_m1 = time.time()
    euler_nodes, weights = load_so3_quadrature(ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    U1 = np.random.normal(0,1,[n_grid, min(400,n_grid)])
    U1, _ = np.linalg.qr(U1, mode='reduced')
    m1 = 0
    print('getting the first moment')
    for i in trange(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        m1 += weights[i]*rot_density[i]*(U1.T@fft_Img)
    t_m1 = time.time()-t_m1

    # form the projected second moment 
    t_m2 = time.time()
    euler_nodes, weights = load_so3_quadrature(2*ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M2_G = 0 
    M2_I = 0 
    G = np.random.normal(0,1,[n_grid, r2_max])
    print('getting the second moment')
    for i in trange(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M2_G += (weights[i]*rot_density[i]*fft_Img) @ (np.conj(fft_Img).T @ G)
        fft_Img = fft_Img[I,:]
        M2_I = M2_I + weights[i]*rot_density[i]*fft_Img @ np.conj(fft_Img).T


    U2, S2, _ = svd(M2_G, full_matrices=False)
    cumulative_energy = np.cumsum(S2**2) / np.sum(S2**2)
    r2 = np.argmax(cumulative_energy > (1 - tol2)) + 1
    U2 = U2[:,:r2]
    print('r2 =',str(r2))


    m2 = 0 
    print('from the second subspace moment')
    for i in trange(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        fft_Img = np.conj(U2).T @ fft_Img
        m2 += weights[i]*rot_density[i]*(fft_Img @ np.conj(fft_Img).T)
    t_m2 = time.time()-t_m2
        

    # form the projected third moment 
    t_m3 = time.time()
    print('sketch the third moment')
    euler_nodes, weights = load_so3_quadrature(3*ell_max_vol, 2*ell_max_half_view)
    precomp_view_basis = precompute_rot_density(rot_coef, ell_max_half_view, euler_nodes)
    rot_density = precomp_view_basis @ rot_coef
    M3_G = 0 
    M3_I = 0 
    G1 = np.random.normal(0,1,[n_grid, r3_max])
    G2 = np.random.normal(0,1,[n_grid, r3_max])
    for i in trange(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = fft_Img.reshape(-1,1)
        M3_G += (weights[i]*rot_density[i]*fft_Img) @ ((fft_Img.T @ G1) * (fft_Img.T @ G2))
        fft_Img = fft_Img.flatten()[I]
        M3_I += (weights[i]*rot_density[i]) * np.einsum('i,j,k->ijk', fft_Img,fft_Img,fft_Img)

    U3, S3, _ = svd(M3_G, full_matrices=False)
    cumulative_energy = np.cumsum(S3**2) / np.sum(S3**2)
    r3 = np.argmax(cumulative_energy > (1 - tol3)) + 1
    U3 = U3[:,:r3]
    print('r3 =',str(r3))

    print('from the third subspace moment')
    m3 = 0
    for i in trange(len(weights)):
        vol_coef_rot = rotate_sphFB(vol_coef, ell_max_vol, k_max, indices_vol, euler_nodes[i,:])
        fft_Img = precomp_vol_basis @ vol_coef_rot
        fft_Img = np.conj(U3).T @ fft_Img
        m3 += weights[i]*rot_density[i]*np.einsum('i,j,k->ijk', fft_Img, fft_Img, fft_Img)
    t_m3 = time.time()-t_m3

    M2_I_approx = U2[I,:] @ m2 @ np.conj(U2[I,:]).T
    M3_I_approx = np.einsum('abc,ia,jb,kc->ijk',m3,U3[I,:],U3[I,:],U3[I,:],optimize='greedy')

    relerr2 = tns_norm(M2_I_approx-M2_I) / tns_norm(M2_I)
    relerr3 = tns_norm(M3_I_approx-M3_I) / tns_norm(M3_I)
    print('relerr2 =',str(relerr2))
    print('relerr3 =',str(relerr3))

    subMoMs = {}
    subMoMs['G'] = G 
    subMoMs['G1'] = G1 
    subMoMs['G2'] = G2 
    subMoMs['m1'] = m1
    subMoMs['M2_G'] = M2_G 
    subMoMs['m2'] = m2 
    subMoMs['M3_G'] = M3_G 
    subMoMs['m3'] = m3 
    subMoMs['U1'] = U1 
    subMoMs['U2'] = U2 
    subMoMs['U3'] = U3 
    subMoMs['S2'] = S2
    subMoMs['S3'] = S3
    subMoMs['t_m1'] = t_m1
    subMoMs['t_m2'] = t_m2
    subMoMs['t_m3'] = t_m3
    subMoMs['relerr2'] = relerr2
    subMoMs['relerr3'] = relerr3
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
    result = minimize(objective, x0, method='SLSQP', jac=True, constraints=[linear_constraint],
        options={'disp': True,'maxiter':5000, 'ftol':1e-10, 'iprint':2, 'eps': 1e-4})
    
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
        options={'disp': True,'maxiter':5000, 'ftol':1e-10, 'iprint':2, 'eps': 1e-4})


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

