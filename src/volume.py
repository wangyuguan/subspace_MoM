import numpy as np 
import numpy.linalg as LA 
import finufft 
from aspire.basis.basis_utils import all_besselj_zeros
from scipy.special import spherical_jn
from utils import *
import pymanopt
from scipy.interpolate import PchipInterpolator
from tqdm import trange

def get_sphFB_indices(n,ell_max):
    
    k_max, r0 = calc_k_max(ell_max,n,3)
    indices = {}

    i = 0 
    for ell in range(ell_max+1):
        for k in range(k_max[ell]):
            for m in range(-ell,ell+1):
                indices[(ell,k,m)] = i
                i += 1
    return k_max, r0, indices
        


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
            js = js[r_indices] 
            js[_grid.rs>c] = 0

            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                vol_coef[i] = np.sum(np.conj(js*lpmn[th_indices]*exps[ph_indices])*w*vol_ft)

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
            js = js[r_indices] 
            js[grid.rs>c] = 0

            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                vol += vol_coef[indices[ell,k,m]]*js*lpmn[th_indices]*exps[ph_indices]
    
    return vol 


def precompute_sphFB_basis(ell_max, k_max, r0, indices, grid):
    """
    precompute Phi[(l,k,m),i] = Ri^T phi_{l,k,m} (r_i,th_i,ph_i) 
    :param ell_max: The truncation limit for spherical harmonics 
    :param k_max: The truncation limit for spherical Bessel function 
    :param r0: Roots of spherical Bessel function 
    :param indices: Indices mapping of spherical Bessel basis 
    :param grid: A 3d grid object
    :return: The precomputed spherical Bessel basis on a 3D grid 

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

    c = 0.5  
    Phi = np.zeros([n_grid,n_coef], dtype=np.complex128)
    for ell in range(0,ell_max+1):
        for k in range(0,k_max[ell]):
            z0k = r0[ell][k]
            js = spherical_jn(ell, r_unique*z0k/c)
            djs = spherical_jn(ell, z0k, True)
            js = js*np.sqrt(2/c**3)/abs(djs)
            js = js[r_indices] 
            js[grid.rs>c] = 0
            for m in range(-ell,ell+1):
                lpmn = lpall[ell,abs(m),:]
                if m<0:
                    lpmn = (-1)**m * lpmn 
                exps = exp_all[m+ell_max,:]
                Phi[:,indices[(ell,k,m)]] =  js*lpmn[th_indices]*exps[ph_indices]

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
    """
    Get the linear transformation between real and complex coefficients
    :param ell_max: The truncation limit for spherical harmonics 
    :param k_max: The truncation limit for spherical Bessel function 
    :param r0: Roots of spherical Bessel function 
    :param indices: Indices mapping of spherical Bessel basis 
    :return sphFB_r_t_c: linear transform from real to complex
    :return sphFB_r_t_c: linear transform from complex to real
    """
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
    """
    Rotating a volume under spherical Bessel representation
    :param vol_coef: The spherical Bessel coefficient
    :param ell_max: The truncation limit for spherical harmonics 
    :param k_max: The truncation limit for spherical Bessel function 
    :param indices: Indices mapping of spherical Bessel basis 
    :param euler_angles: Euler angle representation of the rotation
    :return vol_coef_rot: The rotated spherical Bessel coefficients
    """
    vol_coef_rot = np.zeros(vol_coef.shape, dtype=np.complex128)
    alpha, beta, gamma = euler_angles

    for ell in range(0,ell_max+1):
        Dl = wignerD(ell,alpha,beta,gamma)
        for k in range(0,k_max[ell]):
            istart = indices[(ell,k,-ell)]
            iend = indices[(ell,k,ell)]
            vol_coef_rot[istart:iend+1] = np.conj(Dl).T @ vol_coef[istart:iend+1]

    return vol_coef_rot

def reflect_sphFB(vol_coef, ell_max, k_max, indices):
    
    vol_coef_ref = np.zeros(vol_coef.shape, dtype=np.complex128)
    for ell in range(0,ell_max+1):
        for k in range(0,k_max[ell]):
            for m in range(-ell,ell+1):
                vol_coef_ref[indices[(ell,k,m)]] = vol_coef[indices[(ell,k,m)]]*((-1)**(ell-m))
            
    return vol_coef_ref


def align_vol_coef(vol_coef, vol_coef_est, ell_max, k_max, indices):
    
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3) 
    @pymanopt.function.numpy(manifold)
    def cost(Rot):
        alpha, beta, gamma  = rot_t_euler(Rot)
        vol_coef_rot =  rotate_sphFB(vol_coef_est, ell_max, k_max, indices, (alpha, beta, gamma))
        return LA.norm(vol_coef-vol_coef_rot)**2 / LA.norm(vol_coef)**2
    
    
    vol_coef_est_ref = reflect_sphFB(vol_coef_est, ell_max, k_max, indices)
    @pymanopt.function.numpy(manifold)
    def cost_ref(Rot):
        alpha, beta, gamma  = rot_t_euler(Rot)
        vol_coef_rot =  rotate_sphFB(vol_coef_est_ref, ell_max, k_max, indices, (alpha, beta, gamma))
        return LA.norm(vol_coef-vol_coef_rot)**2 / LA.norm(vol_coef)**2
    
    @pymanopt.function.numpy(manifold)
    def grad(Rot):
        return two_point_fd(cost, Rot, h=1e-6)
    
    @pymanopt.function.numpy(manifold)
    def grad_ref(Rot):
        return two_point_fd(cost_ref, Rot, h=1e-6)
    
    n = 30 
    alpha = np.arange(n)*2*np.pi/n 
    beta = np.arange(n)*np.pi/n 
    gamma = np.arange(n)*2*np.pi/n 
    alpha,beta,gamma= np.meshgrid(alpha,beta,gamma,indexing='xy')
    alpha = alpha.flatten(order='F')
    beta = beta.flatten(order='F')
    gamma = gamma.flatten(order='F')
    N_rot = len(gamma)
    
    Rots = np.zeros((N_rot,3,3))
    cost_initial = 100000000
    cost_ref_initial = 100000000
    Rot0 = None 
    Rot0_ref = None 

    print('brute forcing...')
    for i in trange(N_rot):
        Rots[i] = Rz(alpha[i])@Ry(beta[i])@Rz(gamma[i])
        cost_curr = cost(Rots[i])
        cost_ref_curr = cost_ref(Rots[i])
        if cost_curr<cost_initial:
            cost_initial = cost_curr
            Rot0 = Rots[i]
        if cost_ref_curr<cost_ref_initial:
            cost_ref_initial = cost_ref_curr
            Rot0_ref = Rots[i]
    
    
    print('refining...')
    problem = pymanopt.Problem(manifold=manifold, cost=cost, euclidean_gradient=grad)
    optimizer = pymanopt.optimizers.ConjugateGradient()
    result = optimizer.run(problem,initial_point=Rot0)
    Rot = result.point
    cost_val = cost(Rot)
    
    
    problem_ref = pymanopt.Problem(manifold=manifold, cost=cost_ref, euclidean_gradient=grad_ref)
    result_ref = optimizer.run(problem_ref,initial_point=Rot0_ref)
    Rot_ref = result_ref.point
    cost_val_ref = cost_ref(Rot_ref)
    
    
    if cost_val<cost_val_ref:
        reflect= False 
        alpha, beta, gamma  = rot_t_euler(Rot)
        vol_coef_rot =  rotate_sphFB(vol_coef_est, ell_max, k_max, indices, (alpha, beta, gamma))
        return vol_coef_rot,Rot,cost_val,reflect 
    else:
        reflect= True
        alpha, beta, gamma  = rot_t_euler(Rot_ref)
        vol_coef_rot =  rotate_sphFB(vol_coef_est_ref, ell_max, k_max, indices, (alpha, beta, gamma))
        return vol_coef_rot,Rot_ref,cost_val_ref,reflect 
        

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

def FSCorr(m1, m2):
    # Assume m1 and m2 are 3D numpy arrays of shape (n, n, n)
    n, n1, n2 = m1.shape

    # Define the origin
    ctr = (n + 1) / 2
    origin = np.array([ctr, ctr, ctr])

    # Create a 3D grid and compute radius matrix R
    x, y, z = np.meshgrid(
        np.arange(1, n+1) - origin[0],
        np.arange(1, n+1) - origin[1],
        np.arange(1, n+1) - origin[2],
        indexing='ij'
    )
    R = np.sqrt(x**2 + y**2 + z**2)
    eps = 1e-4

    # Fourier-transform the maps
    f1 = np.fft.fftshift(np.fft.fftn(m1))
    f2 = np.fft.fftshift(np.fft.fftn(m2))

    # Initialize correlation array
    c = np.zeros(n // 2)

    # Initial shell
    d0 = R < 0.5 + eps

    for i in range(n // 2):
        d1 = R < 0.5 + i + 1 + eps
        ring = np.logical_and(d1, np.logical_not(d0))

        r1 = ring * f1
        r2 = ring * f2

        num = np.real(np.sum(r1 * np.conj(r2)))
        den = np.sqrt(np.sum(np.abs(r1)**2) * np.sum(np.abs(r2)**2))

        c[i] = num / den if den != 0 else 0
        d0 = d1

    return c


def fscres(fsc, cutoff):
    n = len(fsc)
    r = n
    x = np.arange(1, n + 1)
    xx = np.arange(1, n, 0.01)

    # Use PCHIP interpolation (monotonic, shape-preserving)
    interp = PchipInterpolator(x, fsc)
    y = interp(xx)

    # Find the first index where FSC drops below cutoff
    below_cutoff = np.where(y < cutoff)[0]
    if below_cutoff.size > 0:
        r = xx[below_cutoff[0]]

    return r



def get_fsc(V1, V2, pixelsize):

    fsc = FSCorr(V1, V2) 
    fsc[0] = 1
    n = len(fsc)

    # Plot FSC
    # plt.figure()
    # plt.plot(range(1, n+1), fsc, 'r-*', linewidth=2)
    # plt.xlim([1, n])
    # plt.ylim([-0.1, 1.05])
    # plt.grid(True)

    # # Plot 0.5 reference line
    # plt.plot(range(1, n+1), [0.5]*n, 'k--', linewidth=1.5)

    # Resolution estimate
    j = fscres(fsc, 0.5)  

    res = 2 * pixelsize * n / j

    # # Update x-ticks with frequency labels
    # df = 1 / (2 * pixelsize * n)
    # xticks = plt.xticks()[0]
    # xtick_labels = [f"{x*df:.3f}" for x in xticks]
    # plt.xticks(xticks, xtick_labels)

    # # Axis labels and legend
    # plt.xlabel(r'$1/\mathrm{\AA}$', fontsize=20)
    # plt.tick_params(labelsize=20)
    # plt.legend([fr'{res:.2f} $\mathrm{{\AA}}$'], loc='best', fontsize=20)

    # plt.tight_layout()
    # plt.show()

    return res


def compare_fscs(V1, V_list, pixelsize, labels=None, colors=None, savepath=None):
    """
    Compare multiple volumes against a reference and plot FSCs.

    Parameters:
        V1 : ndarray
            Reference volume.
        V_list : list of ndarrays
            Volumes to compare with V1.
        pixelsize : float
            Pixel size in Angstroms.
        labels : list of str, optional
            Legend labels.
        colors : list of str, optional
            Plot colors.
        savepath : str, optional
            If provided, saves the plot to this file (e.g., 'fsc_plot.pdf').

    Returns:
        resolutions : list of float
            List of FSC=0.5 resolution estimates.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    n = None
    df = None
    resolutions = []

    if labels is None:
        labels = [f'Volume {i+1}' for i in range(len(V_list))]
    if colors is None:
        colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k'] * 5

    for V2, label, color in zip(V_list, labels, colors):
        fsc = FSCorr(V1, V2)
        fsc[0] = 1
        n = len(fsc)
        df = 1 / (2 * pixelsize * n)
        freq = np.arange(1, n+1) * df

        j = fscres(fsc, 0.5)
        res = 2 * pixelsize * n / j
        resolutions.append(res)

        plt.plot(freq, fsc, f'{color}-*', label=fr'{label}: {res:.2f} $\mathrm{{\AA}}$', linewidth=2)

    # Plot FSC threshold line
    plt.axhline(0.5, color='k', linestyle='--', linewidth=1.5)

    plt.xlabel(r'$1/\mathrm{\AA}$', fontsize=20)
    plt.ylabel('FSC', fontsize=20)
    plt.ylim([-0.1, 1.05])
    plt.grid(True)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, format='pdf', bbox_inches='tight')

    plt.show()
    return resolutions
