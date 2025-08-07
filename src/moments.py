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


