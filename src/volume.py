import numpy as np 
import numpy.linalg as LA 
import finufft 
from aspire.basis.basis_utils import all_besselj_zeros, lgwt 
from scipy.special import spherical_jn
from scipy.io import savemat
from utils import *




def sphFB_transform(vol, ell_max):
    """
    Project the volume into spherical Bessel basis 
    :param vol: The volume of size nxnxn
    :param ell_max: The truncation limit for spherical harmonics 
    :return vol_coef: The spherical Bessel coefficient
    :return k_max: The truncation limit of the spherical Bessel function 
    :return r0: Roots of the spherical Bessel function
    :return indices: The 3D indices in linear order  

    """
    n = vol.shape[0]
    k_max, r0 = calc_k_max(ell_max,n,3)
    nb = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))
    vol_coef = np.zeros(nb, dtype=np.complex128)

    # grid points in real domain 
    grid = get_3d_unif_grid(n)
    
    X = 2*np.pi*grid.xs
    Y = 2*np.pi*grid.ys
    Z = 2*np.pi*grid.zs
        

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    # generate grid point in fourier domain 
    nr = int(1.5*n)
    nth = int(1.5*n)
    nph = int(1.5*n)

    c = 0.5
    _grid = get_3dballquad(nr,nth,nph,c)
    w = _grid.w

    kx = _grid.xs
    ky = _grid.ys
    kz = _grid.zs

    kx = kx.astype(np.float64)
    ky = ky.astype(np.float64)
    kz = kz.astype(np.float64)

    # map volume to Fourier quadrature points 
    f = vol.flatten(order='F').astype(np.complex128)
    vol_ft = finufft.nufft3d3(X,Y,Z,f,kx,ky,kz,isign=-1,eps=1e-12)


    # evaluate inner product 
    r_unique, r_indices = np.unique(_grid.rs, return_inverse=True)
    th_unique, th_indices = np.unique(_grid.ths, return_inverse=True)
    ph_unique, ph_indices = np.unique(_grid.phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(th_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(ph_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*ph_unique)

    # lpall = jnp.asarray(lpall)
    # exp_all = jnp.asarray(exp_all)

    indices = {}

    i = 0 
    for ell in range(ell_max+1):
        for k in range(k_max[ell]):
            z0k = r0[ell][k]
            js = spherical_jn(ell, r_unique*z0k/c)
            djs = spherical_jn(ell, z0k, True)
            js = js*np.sqrt(2/c**3)/abs(djs)
            js[r_unique>c] = 0

            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                vol_coef[i] = np.sum(np.conj(js[r_indices]*lpmn[th_indices]*exps[ph_indices])*w*vol_ft)

                indices[(ell,k,m)] = i
                i += 1 

    return vol_coef, k_max, r0, indices


def sphFB_eval(vol_coef, ell_max, k_max, r0, indices, grid):

    r_unique, r_indices = np.unique(grid.rs, return_inverse=True)
    th_unique, th_indices = np.unique(grid.ths, return_inverse=True)
    ph_unique, ph_indices = np.unique(grid.phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(th_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(ph_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*ph_unique)

    vol = 0 
    c = 0.5  
    for ell in range(ell_max+1):
        for k in range(k_max[ell]):
            z0k = r0[ell][k]
            js = spherical_jn(ell, r_unique*z0k/c)
            djs = spherical_jn(ell, z0k, True)
            js = js*np.sqrt(2/c**3)/abs(djs)
            js[r_unique>c] = 0

            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                vol += vol_coef[indices[ell,k,m]]*js[r_indices]*lpmn[th_indices]*exps[ph_indices]
    
    return vol 


def precompute_sphFB_basis(ell_max, k_max, r0, indices, grid):
    """
    precompute Phi[(l,k,m),i] = Ri^T phi_{l,k,m} (r_i,th_i,ph_i) 
    """

    n_coef = len(indices)
    n_grid = len(grid.rs)
    r_unique, r_indices = np.unique(grid.rs, return_inverse=True)
    th_unique, th_indices = np.unique(grid.ths, return_inverse=True)
    ph_unique, ph_indices = np.unique(grid.phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(th_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(ph_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*ph_unique)

    vol = 0 
    c = 0.5  
    Phi = np.zeros([n_grid,n_coef], dtype=np.complex128)
    for ell in range(0,ell_max+1):
        for k in range(0,k_max[ell]):
            z0k = r0[ell][k]
            js = spherical_jn(ell, r_unique*z0k/c)
            djs = spherical_jn(ell, z0k, True)
            js = js*np.sqrt(2/c**3)/abs(djs)
            js[r_unique>c] = 0

            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                Phi[:,indices[(ell,k,m)]] =  js[r_indices]*lpmn[th_indices]*exps[ph_indices]
    return Phi



def coef_t_vol(vol_coef, ell_max, n, k_max, r0, indices):
    """
    Map the spherical Bessel coefficient into the volume 
    :param vol_coef: The spherical Bessel coefficient
    :param ell_max: The truncation limit for spherical harmonics 
    :param n: The resolution of the volume 
    :param k_max: The truncation limit for spherical Bessel function 
    :param r0: Roots of spherical Bessel function 
    :return: The nxnxn volume represented by the spherical Bessel coefficient

    """
    nb = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))
    assert nb == len(vol_coef), 'length of vector is wrong' 

    grid = get_3d_unif_grid(n)
    X = 2*np.pi*grid.xs
    Y = 2*np.pi*grid.ys
    Z = 2*np.pi*grid.zs

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    # generate grid point in fourier domain 
    nr = int(1.5*n)
    nth = int(1.5*n)
    nph = int(1.5*n)

    c = 0.5
    _grid = get_3dballquad(nr,nth,nph,c)
    kx = _grid.xs
    ky = _grid.ys
    kz = _grid.zs

    kx = kx.astype(np.float64)
    ky = ky.astype(np.float64)
    kz = kz.astype(np.float64)

    # evaluate 
    vol_expand_ft = sphFB_eval(vol_coef, ell_max, k_max, r0, indices, _grid)

    # map into real domain 
    f = vol_expand_ft*_grid.w 
    f = f.astype(np.complex128)
    vol_expand = finufft.nufft3d3(kx,ky,kz,f,X,Y,Z,isign=1)
    vol_expand = np.real(vol_expand).astype(np.float32)



    return vol_expand
    

    


def get_sphFB_r_t_c_mat(ell_max, k_max, indices):
    nb = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))

    sphFB_r_t_c = np.zeros([nb,nb], dtype=np.complex128)
    sphFB_c_t_r = np.zeros([nb,nb], dtype=np.complex128)

    i = 0
    for ell in range(ell_max+1):
        for k in range(k_max[ell]):
            for m in range(-ell,ell+1):
                if m>0:
                    sphFB_r_t_c[i,i] = 1
                    sphFB_r_t_c[i,indices[(ell,k,-m)]] = -1j*((-1)**(ell+m))
                elif m==0:
                    sphFB_r_t_c[i,i] = 1j**ell 
                else:
                    sphFB_r_t_c[i,i] = 1j 
                    sphFB_r_t_c[i,indices[(ell,k,-m)]] = (-1)**(ell+m)
                i += 1 
                
    sphFB_c_t_r = LA.inv(sphFB_r_t_c)
                
    return sphFB_r_t_c, sphFB_c_t_r



def rotate_sphFB(vol_coef, ell_max, k_max, indices, euler_angles):
    
    vol_coef_rot = np.zeros(vol_coef.shape, dtype=np.complex128)
    alpha, beta, gamma = euler_angles

    for ell in range(0,ell_max+1):
        Dl = wignerD(ell,alpha,beta,gamma)
        for k in range(0,k_max[ell]):
            istart = indices[(ell,k,-ell)]
            iend = indices[(ell,k,ell)]
            vol_coef_rot[istart:iend+1] = np.conj(Dl).T @ vol_coef[istart:iend+1]

    return vol_coef_rot

def calc_k_max(ell_max,nres,ndim):
    """
    Generate zeros of Bessel functions
    """
    # get upper_bound of zeros of Bessel functions
    upper_bound = min(ell_max + 1, 2 * nres + 1)

    # List of number of zeros
    n = []
    # List of zero values (each entry is an ndarray; all of possibly different lengths)
    zeros = []

    for ell in range(upper_bound):
        # for each ell, num_besselj_zeros returns the zeros of the
        # order ell Bessel function which are less than 2*pi*c*R = nres*pi/2,
        # the truncation rule for the Fourier-Bessel expansion
        if ndim == 2:
            bessel_order = ell
        elif ndim == 3:
            bessel_order = ell + 1 / 2
        _n, _zeros = all_besselj_zeros(bessel_order, nres * np.pi / 2)
        if _n == 0:
            break
        else:
            n.append(_n)
            zeros.append(_zeros)

    #  get maximum number of ell
    ell_max = len(n) - 1

    #  set the maximum of k for each ell
    k_max = np.array(n, dtype=int)

    # set the zeros for each ell
    # this is a ragged list of 1d ndarrays, where the i'th element is of size self.k_max[i]
    r0 = zeros

    return k_max, r0



