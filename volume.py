import numpy as np 
import numpy.linalg as LA 
import finufft 
from aspire.basis.basis_utils import all_besselj_zeros, lgwt 
from scipy.special import spherical_jn

def vol_t_coef(vol, ell_max, coef_real=True):
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
    x,y,z = get_3d_unif_grid(n)

    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z 

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    # generate grid point in fourier domain 
    nr = int(1.5*n)
    nth = int(1.5*n)
    nph = int(1.5*n)

    c = 0.5
    xr,xth,xph,w = get_spherequad(nr,nth,nph,c)

    kx = xr*np.sin(xth)*np.cos(xph)
    ky = xr*np.sin(xth)*np.sin(xph)
    kz = xr*np.cos(xth)

    kx = kx.astype(np.float64)
    ky = ky.astype(np.float64)
    kz = kz.astype(np.float64)

    # map volume to Fourier quadrature points 
    f = vol.flatten().astype(np.complex128)
    vol_ft = finufft.nufft3d3(X,Y,Z,f,kx,ky,kz,isign=-1)

    # evaluate inner product 
    r_unique, r_indices = np.unique(xr, return_inverse=True)
    th_unique, th_indices = np.unique(xth, return_inverse=True)
    ph_unique, ph_indices = np.unique(xph, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(th_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(ph_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*ph_unique)

    indices = {}

    i = 0 
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
                vol_coef[i] = np.sum(np.conj(js[r_indices]*lpmn[th_indices]*exps[ph_indices])*w*vol_ft)

                indices[(ell,k,m)] = i
                i += 1 

    if coef_real:
        _, vol_c_t_r = get_vol_r_t_c_mat(ell_max, k_max, indices)
        vol_coef = np.real(vol_c_t_r @ vol_coef)

    return vol_coef, k_max, r0, indices


def coef_t_vol(vol_coef, ell_max, n, k_max, r0, indices, coef_real=True):
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

    if coef_real:
        vol_r_t_c, _ = get_vol_r_t_c_mat(ell_max, k_max, indices)
        vol_coef = vol_r_t_c @ vol_coef

    x,y,z = get_3d_unif_grid(n)

    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z 

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    # generate grid point in fourier domain 
    nr = int(1.5*n)
    nth = int(1.5*n)
    nph = int(1.5*n)

    c = 0.5
    xr,xth,xph,w = get_spherequad(nr,nth,nph,c)


    kx = xr*np.sin(xth)*np.cos(xph)
    ky = xr*np.sin(xth)*np.sin(xph)
    kz = xr*np.cos(xth)

    kx = kx.astype(np.float64)
    ky = ky.astype(np.float64)
    kz = kz.astype(np.float64)

    # evaluate inner product 
    r_unique, r_indices = np.unique(xr, return_inverse=True)
    th_unique, th_indices = np.unique(xth, return_inverse=True)
    ph_unique, ph_indices = np.unique(xph, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(th_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(ph_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*ph_unique)

    # evaluate in frequency space 
    vol_expand_ft = 0 
    i = 0 
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
                vol_expand_ft += vol_coef[i]*js[r_indices]*lpmn[th_indices]*exps[ph_indices]
                i += 1 

    # map into real domain 
    f = vol_expand_ft*w 
    f = f.astype(np.complex128)
    vol_expand = finufft.nufft3d3(kx,ky,kz,f,X,Y,Z,isign=1)
    vol_expand = np.real(vol_expand).astype(np.float32)



    return vol_expand
    

    
def get_spherequad(nr,nth,nph,c):
    """
    Get the spherical rule in 3D 
    :param nr: The order of discretization for radial part 
    :param nth: The order of discretization for polar part 
    :param nph: The order of discretization for azimuthal  part 
    :return: The quadrature points under spherical coordinate and the weights

    """
    [xr,wr] = lgwt(nr,0,c)
    [xth,wth] = lgwt(nth,-1,1)
    xth = np.arccos(xth)
    xph = phis = 2*np.pi*np.arange(0,nph)/nph
    wph = 2*np.pi*np.ones(nph)/nph


    [xr,xth,xph] = np.meshgrid(xr,xth,xph,indexing='ij')
    [wr,wth,wph] = np.meshgrid(wr,wth,wph,indexing='ij')

    w = wr*wth*wph*(xr**2)
    w = w.flatten()

    xr = xr.flatten()
    xth = xth.flatten()
    xph = xph.flatten()

    return xr,xth,xph,w


def get_3d_unif_grid(n):
    """
    Get the equispace quadrature points in 3D space 
    :param n: The order of discretization in each dimension 
    :return: The 3d grid points given in x,y,z of size n**3 for each 

    """
    if n%2==0:
        x = np.arange(-n/2,n/2)
    else:
        x = np.arange(-(n-1)/2,(n-1)/2+1)
 
    [x,y,z] = np.meshgrid(x,x,x,indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten() 

    return x,y,z 

    
    



def norm_assoc_legendre_all(nmax, x):
    """
    Evaluate the normalized associated Legendre polynomial
    as  Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
        for n=0,...,nmax and m=0,...,n
    :param j: The order of the associated Legendre polynomial
    :param x: A 1D array of values between -1 and +1 on which to evaluate.
    :return: The normalized associated Legendre polynomial evaluated at corresponding x.

    """

    x = x.flatten()
    nx = len(x)
    y = np.zeros((nmax+1,nmax+1,nx))

    u = -np.sqrt((1-x)*(1+x))
    y[0,0,:] = 1 

    for m in range(0,nmax+1):
        if m>0:
            y[m,m,:] = y[m-1,m-1,:]*u*np.sqrt((2.0*m-1)/(2.0*m))
        if m<nmax:
            y[m+1,m,:] = x*y[m,m,:]*np.sqrt((2.0*m+1)) 

        for n in range(m+2,nmax+1):
            y[n,m,:] = ((2*n-1)*x*y[n-1,m,:]-np.sqrt((n+m-1)*(n-m-1))*y[n-2,m,:])/np.sqrt((n-m)*(n+m))
        
    for n in range(0,nmax+1):
        for m in range(0,n+1):
            y[n,m,:] = y[n,m,:]*np.sqrt(2*n+1.0)

    return y



def get_vol_r_t_c_mat(ell_max, k_max, indices):
    nb = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))

    vol_r_t_c = np.zeros([nb,nb], dtype=np.complex128)
    vol_c_t_r = np.zeros([nb,nb], dtype=np.complex128)

    i = 0
    for ell in range(0,ell_max+1):
        for k in range(0,k_max[ell]):
            for m in range(-ell,ell+1):
                if m>0:
                    vol_r_t_c[i,i] = 1
                    vol_r_t_c[i,indices[(ell,k,-m)]] = -1j*((-1)**(ell+m))
                elif m==0:
                    vol_r_t_c[i,i] = 1j**ell 
                else:
                    vol_r_t_c[i,i] = 1j 
                    vol_r_t_c[i,indices[(ell,k,-m)]] = (-1)**(ell+m)
                i += 1 
                
    vol_c_t_r = LA.inv(vol_r_t_c)
                
    return vol_r_t_c, vol_c_t_r

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



def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters:
        x (float or array): x-coordinate(s)
        y (float or array): y-coordinate(s)
        z (float or array): z-coordinate(s)

    Returns:
        tuple: (r, theta, phi)
            r (float or array): Radial distance
            theta (float or array): Polar angle (in radians)
            phi (float or array): Azimuthal angle (in radians)
    """
    r = np.sqrt(x**2 + y**2 + z**2)               # Radial distance
    theta = np.arccos(z / r) if r != 0 else 0    # Polar angle
    phi = np.arctan2(y, x)                       # Azimuthal angle
    return r, theta, phi




# def get_indices(ell_max, k_max):
#     """
#     Create the indices for each basis function
#     """
#     count = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))
#     indices_ells = np.zeros(count, dtype=int)
#     indices_ms = np.zeros(count, dtype=int)
#     indices_ks = np.zeros(count, dtype=int)

#     ind = 0
#     for ell in range(ell_max + 1):
#         ks = range(0, k_max[ell])
#         for m in range(-ell, ell + 1):
#             rng = range(ind, ind + len(ks))
#             indices_ells[rng] = ell
#             indices_ms[rng] = m
#             indices_ks[rng] = ks

#             ind += len(ks)

#     return {"ells": indices_ells, "ms": indices_ms, "ks": indices_ks}