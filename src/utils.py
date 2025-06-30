import numpy as np 
import finufft 
from aspire.basis.basis_utils import lgwt 
from aspire.numeric import fft
import e3x
from scipy.linalg import eigh
import numpy.linalg as LA 
import jax.numpy as jnp
from jax import jit

class Grid_2d:
    
    def __init__(self, type='euclid', xs=None, ys=None, rs=None, phs=None, w=None, rescale=1):
          
          if type=='euclid':
              self.xs = xs*rescale 
              self.ys = ys*rescale 

              self.rs, self.phs = cart2pol(self.xs, self.ys)

          if type=='spherical':
              
              self.rs = rs*rescale  
              self.phs = phs 

              self.xs = self.rs*np.cos(phs)
              self.ys = self.rs*np.sin(phs)

          if w is None:
            self.w = np.ones(len(self.xs))
          else:
            self.w = w 

class Grid_3d:
    
    def __init__(self, type='euclid', xs=None, ys=None, zs=None, rs=None, ths=None, phs=None, w=None, rescale=1):
        
        if type=='euclid':
            
            self.xs = xs*rescale 
            self.ys = ys*rescale 
            self.zs = zs*rescale 


            self.rs, self.ths, self.phs = cart2sph(self.xs, self.ys, self.zs)


        if type=='spherical':
            
            self.ths = ths 
            self.phs = phs 

            if rs is None:
                self.rs = np.ones(len(ths))
            else:
                self.rs = rs
            self.rs = self.rs*rescale  


            self.xs = self.rs*np.sin(self.ths)*np.cos(self.phs)
            self.ys = self.rs*np.sin(self.ths)*np.sin(self.phs)
            self.zs = self.rs*np.cos(self.ths)


        if w is None:
            self.w = np.ones(len(self.xs))
        else:
            self.w = w 

    def get_xyz_combined(self):
        return np.column_stack((self.xs, self.ys, self.zs))
    
    def get_rotated_grid(self, rot):
        xyz_rot = np.column_stack((self.xs, self.ys, self.zs)) @ rot.T
        return Grid_3d(type='euclid', xs=xyz_rot[:,0], ys=xyz_rot[:,1], zs=xyz_rot[:,2])

def load_so3_quadrature(ell_max_1, ell_max_2):
    """
    quadrature for (1/8/pi^2) \int_{so3} f(R) dR = \sum_i wi f(alpha_i,beta_i,gamma_i)
    """
    
    sphere_grids =  load_sph_gauss_quadrature(ell_max_1+ell_max_2)
    
    alphas = sphere_grids.phs
    betas = sphere_grids.ths 
    n_sph = len(alphas)
    w_sph = sphere_grids.w 

    n_circ = ell_max_1+1 
    gammas = 2*np.pi*np.arange(n_circ)/n_circ
    w_circ = np.ones(n_circ)/n_circ
    
    n_so3 = n_sph*n_circ
    
    euler_nodes = np.zeros([n_so3,3])
    weights = np.zeros(n_so3)

    for i in range(n_sph):
        for j in range(n_circ):
            euler_nodes[i*n_circ+j,0] = alphas[i]
            euler_nodes[i*n_circ+j,1] = betas[i]
            euler_nodes[i*n_circ+j,2] = gammas[j]
            weights[i*n_circ+j] = w_sph[i]*w_circ[j]

    return euler_nodes, weights 


def rot_t_euler(Rot):
    if Rot[2, 2] < 1:  
        if Rot[2, 2] > -1:
            beta = np.arccos(Rot[2, 2])
            alpha = np.arctan2(Rot[1, 2], Rot[0, 2])
            gamma = np.arctan2(Rot[2, 1], -Rot[2, 0])
        else:
            beta = np.pi
            alpha = -np.arctan2(Rot[1, 0], Rot[1, 1])
            gamma = 0
    else:
        beta = 0
        alpha = np.arctan2(Rot[1, 0], Rot[1, 1])
        gamma = 0

    # Ensure angles are in the range [0, 2*pi]
    if alpha < 0:
        alpha += 2 * np.pi

    if gamma < 0:
        gamma += 2 * np.pi

    return alpha, beta, gamma


def load_sph_gauss_quadrature(N):

    is_gauss = True 
    
    if N==1:
        data = np.genfromtxt('../data/sphere_rules/N001_M2_Inv.dat',skip_header=2)
    elif N==2:
        data = np.genfromtxt('../data/sphere_rules/N002_M4_Tetra.dat',skip_header=2)
    elif N==3:
        data = np.genfromtxt('../data/sphere_rules/N003_M6_Octa.dat',skip_header=2)
    elif N==4:
        data = np.genfromtxt('../data/sphere_rules/N004_M10_C4.dat',skip_header=2)
    elif N==5:
        data = np.genfromtxt('../data/sphere_rules/N005_M12_Ico.dat',skip_header=2)
    elif N==6:
        data = np.genfromtxt('../data/sphere_rules/N006_M18_C4.dat',skip_header=2)
    elif N==7:
        data = np.genfromtxt('../data/sphere_rules/N007_M22_C5.dat',skip_header=2)
    elif N==8:
        data = np.genfromtxt('../data/sphere_rules/N008_M28_Tetra.dat',skip_header=2)
    elif N==9:
        data = np.genfromtxt('../data/sphere_rules/N009_M32_Ico.dat',skip_header=2)
    elif N==10:
        data = np.genfromtxt('../data/sphere_rules/N010_M42_C4.dat',skip_header=2)
    elif N==11:
        data = np.genfromtxt('../data/sphere_rules/N011_M48_Octa.dat',skip_header=2)
    elif N==12:
        data = np.genfromtxt('../data/sphere_rules/N012_M58_C4.dat',skip_header=2)
    elif N==13:
        data = np.genfromtxt('../data/sphere_rules/N013_M64_Inv.dat',skip_header=2)
    elif N==14:
        data = np.genfromtxt('../data/sphere_rules/N014_M72_Ico.dat',skip_header=2)
    elif N==15:
        data = np.genfromtxt('../data/sphere_rules/N015_M82_C5.dat',skip_header=2)
    elif N==16:
        data = np.genfromtxt('../data/sphere_rules/N016_M98_C4.dat',skip_header=2)
    elif N==17:
        data = np.genfromtxt('../data/sphere_rules/N017_M104_C3.dat',skip_header=2)
    elif N==18:
        data = np.genfromtxt('../data/sphere_rules/N018_M122_C4.dat',skip_header=2)
    elif N==19:
        data = np.genfromtxt('../data/sphere_rules/N019_M130_Inv.dat',skip_header=2)
    elif N==20:
        data = np.genfromtxt('../data/sphere_rules/N020_M148_Tetra.dat',skip_header=2)
    elif N==21:
        data = np.genfromtxt('../data/sphere_rules/N021_M156_C3.dat',skip_header=2)
    elif N==22:
        data = np.genfromtxt('../data/sphere_rules/N022_M178_C4.dat',skip_header=2)
    elif N==23:
        data = np.genfromtxt('../data/sphere_rules/N023_M186_C3.dat',skip_header=2)
    elif N==24:
        data = np.genfromtxt('../data/sphere_rules/N024_M210_C4.dat',skip_header=2)
    elif N==25:
        data = np.genfromtxt('../data/sphere_rules/N025_M220_Inv.dat',skip_header=2)
    elif N==26:
        data = np.genfromtxt('../data/sphere_rules/N026_M244_Tetra.dat',skip_header=2)
    elif N==27:
        data = np.genfromtxt('../data/sphere_rules/N027_M254_C3.dat',skip_header=2)
    elif N==28:
        data = np.genfromtxt('../data/sphere_rules/N028_M282_C4.dat',skip_header=2)
    elif N==29:
        data = np.genfromtxt('../data/sphere_rules/N029_M292_C5.dat',skip_header=2)
    elif N==30:
        data = np.genfromtxt('../data/sphere_rules/N030_M322_C4.dat',skip_header=2)
    elif N==32:
        data = np.genfromtxt('../data/sphere_rules/N032_M364_Tetra.dat',skip_header=2)
    elif N==34:
        data = np.genfromtxt('../data/sphere_rules/N034_M410_C4.dat',skip_header=2)
    elif N==35:
        data = np.genfromtxt('../data/sphere_rules/N035_M422_C5.dat',skip_header=2)
    elif N==36:
        data = np.genfromtxt('../data/sphere_rules/N036_M458_C4.dat',skip_header=2)
    elif N==37:
        data = np.genfromtxt('../data/sphere_rules/N037_M472_C5.dat',skip_header=2)
    elif N==38:
        data = np.genfromtxt('../data/sphere_rules/N038_M508_Tetra.dat',skip_header=2)
    elif N==39:
        data = np.genfromtxt('../data/sphere_rules/N039_M522_C5.dat',skip_header=2)
    elif N==44: 
        data = np.genfromtxt('../data/sphere_rules/N044_M672_Ico.dat',skip_header=2)
    else:
        nodes, weights = e3x.so3.quadrature.lebedev_quadrature(N)
        nodes = np.array(nodes,dtype=np.float64)
        weights = np.array(weights,dtype=np.float64)
        is_gauss = False 

    if is_gauss:
        nodes = data[:,0:3]
        weights = data[:,3]

    phs = np.pi - np.arctan2(nodes[:,0],nodes[:,1])
    ths = np.arccos(nodes[:,2])
      
    return Grid_3d(type='spherical',ths=ths,phs=phs,w=weights)

def get_spherequad(nth, nph):
    """
    Get the quadrature rule under polor coord on unit sphere such that 
    
    \int_{||x||=1} f dS  =  \sum_i w(i) * f(th(i),ph(i))  

    :param nr: The order of discretization for radial part 
    :param nth: The order of discretization for polar part 
    :param nph: The order of discretization for azimuthal  part 
    :param R: The radius of the ball 
    :return: The 3d grid points object

    """
    ths, wths = lgwt(nth,-1,1)
    ths = np.arccos(ths)

    phs = 2*np.pi*np.arange(nph)/nph 
    wphs =  2*np.pi*np.ones(nph)/nph 
  
    ths, phs = np.meshgrid(ths, phs, indexing='xy')
    ths, phs = ths.flatten(order='F'), phs.flatten(order='F')

    wths, wphs = np.meshgrid(wths, wphs, indexing='xy')
    wths, wphs = wths.flatten(order='F'), wphs.flatten(order='F')
    w = wths*wphs 

    return Grid_3d(type = 'spherical', ths=ths, phs=phs, w=w)



def get_3dballquad(nr,nth,nph,R):
    """
    Get the quadrature rule under spherical coord in a ball of radius R such that 

    \int_{||x||<=R} f dV  =  \sum_i w(i) * f(r(i),th(i),ph(i))  

    :param nr: The order of discretization for radial part 
    :param nth: The order of discretization for polar part 
    :param nph: The order of discretization for azimuthal  part 
    :param R: The radius of the ball 
    :return: The 3d grid points object

    """
    # [r,wr] = lgwt(nr,0,R)
    # [th,wth] = lgwt(nth,-1,1)
    # th = np.arccos(th)
    # ph = phis = 2*np.pi*np.arange(0,nph)/nph
    # wph = 2*np.pi*np.ones(nph)/nph


    # [r,th,ph] = np.meshgrid(r,th,ph,indexing='ij')
    # [wr,wth,wph] = np.meshgrid(wr,wth,wph,indexing='ij')

    # w = wr*wth*wph*(r**2)
    # w = w.flatten()

    # r = r.flatten()
    # th = th.flatten()
    # ph = ph.flatten()
    r,th,ph,w = spherequad(nr, nth, nph, R)

    return Grid_3d(type='spherical',rs=r,ths=th,phs=ph,w=w)

        
        
def get_3d_unif_grid(n,rescale=1):
    """
    Get the equispace quadrature points in 3D space 
    :param n: The order of discretization in each dimension 
    :return: The 3d grid points object

    """
    if n%2==0:
        x = np.arange(-n/2,n/2)
    else:
        x = np.arange(-(n-1)/2,(n-1)/2+1)

    [x,y,z] = np.meshgrid(x,x,x,indexing='xy')
    grid = Grid_3d(type='euclid', xs=x.flatten(order='F'), ys=y.flatten(order='F'), zs=z.flatten(order='F'), rescale=rescale)

    return grid


def get_2d_unif_grid(n,rescale=1):
    """
    Get the equispace quadrature points in 3D space 
    :param n: The order of discretization in each dimension 
    :return: The 2d grid points object

    """
    if n%2==0:
        x = np.arange(-n/2,n/2)
    else:
        x = np.arange(-(n-1)/2,(n-1)/2+1)
 
    [x,y] = np.meshgrid(x,x,indexing='xy')
    grid = Grid_2d(type='euclid', xs=x.flatten(order='F'), ys=y.flatten(order='F'), rescale=rescale)

    return grid



def cart2pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Parameters:
        x (float or array-like): x-coordinate(s)
        y (float or array-like): y-coordinate(s)

    Returns:
        tuple: (r, theta)
            r (float or array): Radial distance
            phi (float or array): Angle in radians (range [0, 2*pi])
    """
    x = np.asarray(x)  # Convert inputs to NumPy arrays if they are not already
    y = np.asarray(y)

    r = np.sqrt(x**2 + y**2)      # Compute radial distance
    phi = np.arctan2(y, x)      # Azimuthal angle
    phi = np.mod(phi, 2*np.pi)  # Ensure theta is in [0, 2*pi]

    return r, phi



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
    theta = np.zeros(r.shape)
    theta[r!=0] = np.arccos(z[r!=0] / r[r!=0])         # Polar angle
    phi = np.arctan2(y, x)                       # Azimuthal angle
    phi = np.mod(phi, 2*np.pi)  # Ensure theta is in [0, 2*pi]
    return r, theta, phi



    

def image_downsample(images, ds_res, if_real=True, zero_nyquist=True):
    L = images.shape[1]
    start_idx = (L // 2) - (ds_res // 2)
    slice_idx = slice(start_idx, start_idx + ds_res)
    images_fft = fft.centered_fft2(images)
    images_fft = images_fft[:,slice_idx,slice_idx]
    if zero_nyquist:
        images_fft[:,0,:] = 0 
        images_fft[:,:,0] = 0
    if if_real:
        images =  np.array(fft.centered_ifft2(images_fft).real, dtype=np.float32)*(ds_res**2/L**2)
        return images
    else:
        return np.array(images_fft, dtype=np.complex64)*(ds_res**2/L**2)
        




def centered_fft2(img):
    """
    Compute the centered 2D Fast Fourier Transform (FFT).

    Parameters:
        img (numpy array): 2D input array (image or matrix).

    Returns:
        numpy array: Centered FFT of the input.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))



def centered_ifft2(fft_img):
    """
    Compute the inverse centered 2D Fast Fourier Transform (IFFT).

    Parameters:
        fft_img (numpy array): 2D centered FFT array.

    Returns:
        numpy array: Reconstructed spatial domain array.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft_img)))  


def centered_fftn(vol):
    """
    Compute the centered nD Fast Fourier Transform (FFT).

    Parameters:
        img (numpy array): nD input array (image or matrix).

    Returns:
        numpy array: Centered FFT of the input.
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol)))



def centered_ifftn(fft_vol):
    """
    Compute the inverse centered nD Fast Fourier Transform (IFFT).

    Parameters:
        fft_img (numpy array): nD centered FFT array.

    Returns:
        numpy array: Reconstructed spatial domain array.
    """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fft_vol)))  



def wignerD(j, alpha, beta, gamma):
    """
    Evaluate the Wigner-D matrix of order j 
    :param j: The order of angular momentum 
    :param alpha: The first euler angle under zyz convention
    :param beta: The second euler angle under zyz convention
    :param gamma: The third euler angle under zyz convention
    :return: The (2j+1)x(2j+1) complex orthornormal Wigner-D matrix
    """
    fctrl = np.ones(2*j+1)
    for i in range(1,2*j+1):
        fctrl[i] = fctrl[i-1]*i 
    
    Dj = np.zeros([2*j+1,2*j+1], dtype=np.complex128)

    eps = np.finfo(np.float32).eps
    if beta < eps:
        Dj = np.diag(np.exp(-1j*(alpha+gamma)*np.arange(-j,j+1))) 
    else:
        for mp in range(-j,j+1):
            for m in range(-j,j+1):
                s = np.arange(max(0,m-mp), min(j+m,j-mp)+1).T 
                m1_t = (-1)**s
                fact_t = fctrl[j+mp]*fctrl[j-mp]*fctrl[j+m]*fctrl[j-m]
                fact_t = np.sqrt(fact_t)
                fact_t = fact_t/(fctrl[j+m-s]*fctrl[mp-m+s]*fctrl[j-mp-s]*fctrl[s])
                cos_beta = np.cos(beta/2)**(2*j+m-mp-2*s)
                sin_beta = np.sin(beta/2)**(mp-m+2*s)
                d_l_mn = (-1)**(mp-m)*np.sum(m1_t*fact_t*cos_beta*sin_beta)
                Dj[mp+j,m+j] = np.exp(-1j*alpha*mp)*d_l_mn*np.exp(-1j*gamma*m) 
        
    return Dj


def norm_assoc_legendre_all(nmax, x):
    """
    Evaluate the normalized associated Legendre polynomial
    as  Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
        for n=0,...,nmax and m=0,...,n
    :param j: The order of the associated Legendre polynomial
    :param x: A 1D array of values between -1 and +1 on which to evaluate.
    :return: The normalized associated Legendre polynomial evaluated at corresponding x.

    """

    x = x.flatten(order='F')
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



def Rz(th):
    """
    Rotation around the z axis 
    :param th: The rotation angle 
    :return: The 3x3 rotation matrix rotating a vector around z axis by th
    """
    return np.array([
        [np.cos(th), -np.sin(th), 0], 
        [np.sin(th), np.cos(th), 0],
        [0, 0, 1]
    ])


def Ry(th):
    """
    Rotation around the y axis 
    :param th: The rotation angle 
    :return: The 3x3 rotation matrix rotating a vector around y axis by th
    """    
    return np.array([
        [np.cos(th), 0, np.sin(th)], 
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]
    ])



def vMF_density(centers,w,kappa,grid):
    """
    Evaluate the von-Mises-Fisher density on a sphere 
    :param centers: mx3 centers 
    :param w: weights of length m
    :param kappa: concentration parameter 
    :param xs: nx3 locations on a sphere    
    :return: The density at the n locations.
    """

    if kappa ==0:
        # rs, ths, phs = cart2sph(xs[:,0],xs[:,1],xs[:,2])
        return 1/(4*np.pi)
    
    xyz = grid.get_xyz_combined()

    f = np.exp(kappa * centers @ xyz.T)
    C = kappa/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    f = C*f 
    f = np.sum(np.diag(w) @ f, 0)
    
    return f 





def spherequad(nr, nt, np_, rad):
    """
    Generate Gauss quadrature nodes and weights for spherical volume integrals.

    Parameters
    ----------
    nr : int
         Number of radial nodes.
    nt : int
         Number of theta nodes in [0, pi].
    np_ : int
         Number of phi nodes in [0, 2*pi].
    rad : float
         Sphere radius. Set to np.inf for infinite domain.

    Returns
    -------
    r : 1D array
        Radial nodes (flattened).
    t : 1D array
        Theta nodes (flattened).
    p : 1D array
        Phi nodes (flattened).
    w : 1D array
        Quadrature weights (flattened).
    """
    # Radial quadrature (mapped Jacobi) for k = 2
    r, wr = rquad(nr, 2)
    r = np.clip(r, 0, 1)  # ensure r in [0,1]
    if np.isinf(rad):    # Infinite radius sphere
        wr = wr / (1 - r)**4
        r = r / (1 - r)
    else:                # Finite sphere: scale nodes and weights
        wr = wr * (rad**3)
        r = r * rad

    # Theta quadrature (mapped Legendre) for k = 0
    x, wt = rquad(nt, 0)
    x = np.clip(x, 0, 1)
    # Compute theta nodes: t = arccos(2*x - 1); ensure argument is in [-1,1]
    t = np.arccos(np.clip(2 * x - 1, -1, 1))
    wt = 2 * wt

    # Phi nodes (Gauss-Fourier)
    p = 2 * np.pi * np.arange(np_) / np_
    wp = 2 * np.pi * np.ones(np_) / np_

    # Create product grid using MATLAB-style meshgrid:
    # MATLAB: [rr,tt,pp] = meshgrid(r, t, p) produces arrays of shape (len(t), len(r), len(p))
    rr, tt, pp = np.meshgrid(r, t, p, indexing='xy')
    # Flatten arrays in Fortran (column-major) order to mimic MATLAB's (rr(:), etc.)
    r_flat = rr.ravel(order='F')
    t_flat = tt.ravel(order='F')
    p_flat = pp.ravel(order='F')

    # Combine the weights. In MATLAB:
    #    w = reshape( reshape(wt*wr', nr*nt, 1) * wp', nr*nt*np, 1);
    # In Python, first form the outer product of wt (theta weights) and wr (radial weights).
    W_rt = np.outer(wt, wr)         # shape: (nt, nr)
    W_rt_flat = W_rt.ravel(order='F') # flatten in column-major order
    # Then form the outer product with the phi weights
    W = np.outer(W_rt_flat, wp).ravel(order='F')

    return r_flat, t_flat, p_flat, W

def rquad(N, k):
    """
    Compute Gauss quadrature nodes and weights for a Jacobi-type weight.
    
    Parameters
    ----------
    N : int
        Number of quadrature points.
    k : int or float
        Parameter for the weight function.
        
    Returns
    -------
    x : 1D array
        Quadrature nodes mapped to [0,1].
    w : 1D array
        Quadrature weights.
    """
    k1 = k + 1
    k2 = k + 2
    n = np.arange(1, N + 1)      # n = 1,2,...,N
    nnk = 2 * n + k            # vector of length N

    # First column A: [k/k2,  k^2/( (2*n+k)*(2*n+k+2) ) for n=1:N]
    A0 = k / k2
    A_rest = (k**2) / (nnk * (nnk + 2))
    A = np.concatenate(([A0], A_rest))  # length = N+1

    # For n = 2:N, update
    n2 = np.arange(2, N + 1)     # length N-1
    nnk_n2 = nnk[1:]           # corresponding nnk for n>=2
    B1 = 4 * k1 / (k2**2 * (k + 3))
    nk = n2 + k                # length N-1
    nnk2 = nnk_n2 ** 2
    B = 4 * (n2 * nk)**2 / (nnk2**2 - nnk2)

    # Construct matrix 'ab'. MATLAB does:
    #   ab = [A'  [ (2^k1)/k1; B1; B'] ];
    col2 = np.concatenate(([2**k1 / k1], [B1], B))
    # We need only the first N rows (MATLAB uses ab(1:N,:))
    ab = np.column_stack((A[:N], col2[:N]))
    
    # Compute s = sqrt(ab(2:N,2)) for rows 2 to N (MATLAB 2-indexed)
    s = np.sqrt(ab[1:N, 1])
    
    # Build symmetric tridiagonal matrix T (size N x N)
    d = ab[:N, 0]  # main diagonal
    T = np.diag(d)
    if N > 1:
        T += np.diag(s, k=-1) + np.diag(s, k=1)
    
    # Compute eigenvalues and eigenvectors
    # eigh returns eigenvalues in ascending order
    X, V = eigh(T)
    # Map eigenvalues: x = (X + 1) / 2
    x = (X + 1) / 2
    # Compute weights: w = (1/2)^(k1)*ab(1,2)*(first row of V)^2
    w = (0.5)**(k1) * ab[0, 1] * (V[0, :]**2)
    
    return x, w




def two_point_fd(f, x, h=1e-6):
    """
    Computes the gradient of a function using two-point finite difference method.
    
    Args:
    - f: A function that takes a 3x3 matrix as input and returns a scalar output.
    - x: A 3x3 matrix at which to compute the gradient.
    - h: A small perturbation value for finite difference (default is 1e-6).
    
    Returns:
    - grad: The 3x3 gradient matrix.
    """
    grad = np.zeros_like(x)
    
    # Loop over each element in the 3x3 matrix
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Perturb the (i, j) element
            x_perturb_pos = x.copy()
            x_perturb_neg = x.copy()
            
            x_perturb_pos[i, j] += h  # Perturb positively
            x_perturb_neg[i, j] -= h  # Perturb negatively
            
            # Apply the function to both perturbed matrices
            f_pos = f(x_perturb_pos)
            f_neg = f(x_perturb_neg)
            
            # Compute the finite difference for the gradient at position (i, j)
            grad[i, j] = (f_pos - f_neg) / (2 * h)
    
    return grad

def vol_downsample(vol, ds_res):
    L = vol.shape[1]
    start_idx = (L // 2) - (ds_res // 2)
    slice_idx = slice(start_idx, start_idx + ds_res)
    vol_fft = centered_fftn(vol)
    vol_fft_ds = vol_fft[slice_idx,slice_idx,slice_idx]
    return np.real(centered_ifftn(vol_fft_ds))*ds_res**3/L**3

def vol_upsample(vol_ds, L):
    ds_res = vol_ds.shape[0]
    start_idx = (L // 2) - (ds_res // 2)
    slice_idx = slice(start_idx, start_idx + ds_res)
    vol_ds_fft = centered_fftn(vol_ds)
    vol_fft = np.zeros([L,L,L],dtype=np.complex128)
    vol_fft[slice_idx,slice_idx,slice_idx] = vol_ds_fft 
    return np.real(centered_ifftn(vol_fft))*L**3/ds_res**3
    


def vol_proj(vol, rots, offsets=None):
    
    nrot = rots.shape[0]
    n = vol.shape[0]
    if n % 2 == 0:
        k = np.arange(-n/2,n/2)/n 
    else:
        k = np.arange(-(n-1)/2,(n-1)/2+1)/n
    kx, ky = np.meshgrid(k, k, indexing='xy')
    kx = kx.flatten(order='F')
    ky = ky.flatten(order='F')
    
    # rotated_grids = np.zeros((3,n**2,nrot))
    # for i in range(nrot):
    #     rot = rots[i]
    #     rotated_grids[:,:,i] = rot[:,0].reshape(-1, 1) @ kx.reshape(1,n**2)+rot[:,1].reshape(-1, 1) @ ky.reshape(1,n**2)
    K = np.vstack((kx, ky))        
    R2 = rots[:, : , :2]      
    rot_sub = R2 @ K
    rotated_grids = rot_sub.transpose(1, 2, 0) 
        
    s = 2*np.pi*rotated_grids[0].flatten(order='F')
    t = 2*np.pi*rotated_grids[1].flatten(order='F')
    u = 2*np.pi*rotated_grids[2].flatten(order='F')
    
    vol = np.array(vol, dtype=np.complex128)
    # vol = np.transpose(vol, (2,1,0))
    vol = np.ascontiguousarray(vol)

    
    images_fft = finufft.nufft3d2(s,t,u,vol)
    images_fft = images_fft.reshape(n,n,nrot,order='F').transpose(2,0,1)
    if n//2==0:
        images_fft[:,0,:] = 0 
        images_fft[:,:,0] = 0 
    if offsets is not None:
        phase1 = np.einsum('i,j->ij',offsets[:,0].flatten(order='F'),kx)
        phase1 = phase1.reshape(nrot,n,n,order='F')
        phase2 = np.einsum('i,j->ij',offsets[:,1].flatten(order='F'),ky)
        phase2 = phase2.reshape(nrot,n,n,order='F')
        phase = np.exp(1j*2*np.pi*(phase1+phase2))
        images_fft = images_fft*phase
        
    images = np.array(fft.centered_ifft2(images_fft).real , dtype=np.float32)
        
    return images



def vol_proj1(vol, rots, offsets=None):
    """
    Compute 2D projections of a 3D volume with optional translation in Fourier space,
    using zero-padding to eliminate periodic wrap-around artifacts.

    Parameters
    ----------
    vol : ndarray, shape (n, n, n)
        Input 3D volume.
    rots : ndarray, shape (nrot, 3, 3)
        Rotation matrices for each projection.
    offsets : ndarray, shape (nrot, 2), optional
        (dx, dy) shifts (in pixels) to apply to each 2D projection.

    Returns
    -------
    images : ndarray, shape (nrot, n, n)
        Real-valued 2D projections, cropped to original size.
    """
    n = vol.shape[0]
    # pad by full size to avoid wrap-around
    pad = n
    n_pad = n + 2 * pad

    # zero-pad the 3D volume on all sides
    vol_padded = np.pad(vol, pad_width=pad, mode='constant')

    # frequency grid for padded images
    k = np.fft.fftfreq(n_pad)  # cycles per sample in [-0.5, 0.5)
    kx, ky = np.meshgrid(k, k, indexing='xy')
    kx = kx.flatten(order='F')
    ky = ky.flatten(order='F')

    # build rotated frequency coordinates
    K = np.vstack((kx, ky))  # shape: (2, n_pad^2)
    R2 = rots[:, :2, :2]     # take top-left 2x2 of each 3x3 rotation
    # apply each rotation to (kx, ky)
    # result shape: (nrot, 2, n_pad^2)
    rot_k = np.einsum('rij,jk->rik', R2, K)

    # phase coordinates for NUFFT: scale to [0, 2pi)
    s = 2 * np.pi * rot_k[:, 0, :].ravel(order='F')
    t = 2 * np.pi * rot_k[:, 1, :].ravel(order='F')
    # third dimension is zero: projection along z
    u = np.zeros_like(s)

    # ensure contiguous complex volume
    vol_padded = np.ascontiguousarray(vol_padded.astype(np.complex128))

    # NUFFT: sample 3D FT on each rotated 2D plane
    images_fft = finufft.nufft3d2(s, t, u, vol_padded)
    # reshape to (nrot, n_pad, n_pad)
    images_fft = images_fft.reshape(n_pad, n_pad, -1, order='F').transpose(2, 0, 1)

    # apply translation in Fourier domain if requested
    if offsets is not None:
        # offsets in pixels; use negative sign for forward shift
        dx = offsets[:, 0][:, None, None]
        dy = offsets[:, 1][:, None, None]
        # shape kx2d: (n_pad, n_pad)
        kx2d = kx.reshape(n_pad, n_pad, order='F')
        ky2d = ky.reshape(n_pad, n_pad, order='F')
        # build phase per image
        phase = np.exp(-2j * np.pi * (dx * kx2d + dy * ky2d))
        images_fft *= phase

    # inverse FFT back to spatial domain and take real part
    images_full = centered_ifft2(images_fft).real  # shape: (nrot, n_pad, n_pad)

    # crop to original size
    start = pad
    end = pad + n
    images = images_full[:, start:end, start:end].astype(np.float32)

    return images


def vol_ds_proj_ft(vol, rots, ds_res):
    

    nrot = rots.shape[0]
    n = vol.shape[0]
    rots = rots.astype(np.float32)

    ds_start = np.floor(n / 2) - np.floor(ds_res / 2)
    ds_idx = np.arange(int(ds_start) + 1, int(ds_start) + ds_res + 1)  
    ind1, ind2 = np.meshgrid(ds_idx, ds_idx, indexing='ij')  
    ind1 = ind1.ravel(order='F')  
    ind2 = ind2.ravel(order='F')
    ind = np.ravel_multi_index((ind1 - 1, ind2 - 1), (n, n), order='F') 

    
    if n % 2 == 0:
        k = np.arange(-n/2,n/2)/n 
    else:
        k = np.arange(-(n-1)/2,(n-1)/2+1)/n

    kx, ky = np.meshgrid(k, k, indexing='xy')
    kx = kx.flatten(order='F').astype(np.float32)
    ky = ky.flatten(order='F').astype(np.float32)

    
    K = np.vstack((kx, ky)) 
    K_sub = K[:, ind]          
    R2 = rots[:, : , :2]      
    rot_sub = np.matmul(R2, K_sub) 
    rotated_grids = rot_sub.transpose(1, 2, 0) 
        
    s = 2*np.pi*rotated_grids[0].flatten(order='F')
    t = 2*np.pi*rotated_grids[1].flatten(order='F')
    u = 2*np.pi*rotated_grids[2].flatten(order='F')

    vol = np.array(vol, dtype=np.complex64)
    
    vol = np.ascontiguousarray(vol)

    Img_fft_rot = finufft.nufft3d2(s,t,u,vol)
    Img_fft_rot = Img_fft_rot.reshape(ds_res**2,nrot,order='F').T
    return Img_fft_rot.reshape(nrot,ds_res,ds_res,order='F')*ds_res**2/n**2


def plan_vol_ds_proj_ft(img_size):
    return finufft.Plan(nufft_type=2,                       
    n_modes_or_dim=(img_size,img_size,img_size),           
    n_trans=1,eps=1e-4,isign=+1,dtype='complex64')



def vol_ds_proj_ft_planned(vol, rots, ds_res, plan):
    
    nrot = rots.shape[0]
    n = vol.shape[0]
    rots = rots.astype(np.float32)

    ds_start = np.floor(n / 2) - np.floor(ds_res / 2)
    ds_idx = np.arange(int(ds_start) + 1, int(ds_start) + ds_res + 1)  
    ind1, ind2 = np.meshgrid(ds_idx, ds_idx, indexing='ij')  
    ind1 = ind1.ravel(order='F')  
    ind2 = ind2.ravel(order='F')
    ind = np.ravel_multi_index((ind1 - 1, ind2 - 1), (n, n), order='F') 

    
    if n % 2 == 0:
        k = np.arange(-n/2,n/2)/n 
    else:
        k = np.arange(-(n-1)/2,(n-1)/2+1)/n

    kx, ky = np.meshgrid(k, k, indexing='xy')
    kx = kx.flatten(order='F').astype(np.float32)
    ky = ky.flatten(order='F').astype(np.float32)

    
    K = np.vstack((kx, ky)) 
    K_sub = K[:, ind]          
    R2 = rots[:, : , :2]      
    rot_sub = np.matmul(R2, K_sub) 
    rotated_grids = rot_sub.transpose(1, 2, 0) 
        
    s = 2*np.pi*rotated_grids[0].flatten(order='F')
    t = 2*np.pi*rotated_grids[1].flatten(order='F')
    u = 2*np.pi*rotated_grids[2].flatten(order='F')
    plan.setpts(s, t, u)

    vol = np.array(vol, dtype=np.complex64)
    vol = np.ascontiguousarray(vol)

    Img_fft_rot = plan.execute(vol)
    Img_fft_rot = Img_fft_rot.reshape(ds_res**2,nrot,order='F').T
    return Img_fft_rot.reshape(nrot,ds_res,ds_res,order='F')*ds_res**2/n**2





def get_subindices(D, d):
    ds_start = np.floor(D / 2) - np.floor(d / 2)
    ds_idx = np.arange(int(ds_start) + 1, int(ds_start) + d + 1)  

    ind1, ind2 = np.meshgrid(ds_idx, ds_idx, indexing='ij')  
    ind1 = ind1.ravel(order='F')  
    ind2 = ind2.ravel(order='F')

    ind = np.ravel_multi_index((ind1 - 1, ind2 - 1), (D, D), order='F') 

    return ind



def get_centered_fft2_submtx(n, row_id=None, col_id=None):
    if n % 2 == 0:
        k = np.arange(-n//2, n//2) / n
    else:
        k = np.arange(-(n-1)//2, (n-1)//2 + 1) / n
    
    k1, k2 = np.meshgrid(k, k, indexing='xy')  # Ensure MATLAB-like meshgrid behavior
    k1 = k1.ravel(order='F')  # Flatten in column-major order
    k2 = k2.ravel(order='F')

    if row_id is not None and len(row_id) > 0:
        k1 = k1[row_id]
        k2 = k2[row_id]
    
    if n % 2 == 0:
        x = 2 * np.pi * np.arange(-n//2, n//2)
    else:
        x = 2 * np.pi * np.arange(-(n-1)//2, (n-1)//2 + 1)

    x1, x2 = np.meshgrid(x, x, indexing='xy')  # Again, use MATLAB-like behavior
    x1 = x1.ravel(order='F')  # Flatten in column-major order
    x2 = x2.ravel(order='F')

    if col_id is not None and len(col_id) > 0:
        x1 = x1[col_id]
        x2 = x2[col_id]

    F2 = np.exp(-1j * (np.outer(k1, x1) + np.outer(k2, x2)))

    return F2

def sample_idxs(n,d,n_sample):
    start_idx = (n // 2) - (d // 2)
    slice_idx = slice(start_idx, start_idx + d)
    if n % 2 == 0:
        k = np.arange(-n/2,n/2)/n 
    else:
        k = np.arange(-(n-1)/2,(n-1)/2+1)/n
    k = k[slice_idx]
    kx,ky = np.meshgrid(k, k, indexing='xy')
    kx = kx.flatten(order='F')
    ky = ky.flatten(order='F')
    k2 = kx**2+ky**2
    zero_idx = int(np.argwhere(k2 == 0)[0][0])
    probs = np.zeros(k2.shape)
    probs[k2!=0] = 1/k2[k2!=0]
    probs[k2!=0] = probs[k2!=0]/sum(probs[k2!=0])
    idxs = np.random.choice(np.arange(d**2), size=n_sample-1, replace=False, p=probs)
    idxs = np.sort(np.append(idxs,zero_idx))
    return idxs 

    

def get_preprocessing_matrix(D,d):
    ind = get_subindices(D, d)
    F = get_centered_fft2_submtx(D, row_id=ind)
    f = get_centered_fft2_submtx(d)
    return np.conj(f.T) @ F / D**2 
    # return F 


# def get_estimated_std(vol, SNR):
#     n_rot = 100 
#     rots = np.zeros((n_rot,3,3))
    
#     for i in range(n_rot):
#         alpha = np.random.uniform(0,2*np.pi)
#         beta = np.random.uniform(0,np.pi)
#         gamma = np.random.uniform(0,2*np.pi)
#         R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
#         rots[i,:,:] = R

#     images = vol_proj(vol, rots, 0, 1)
#     gauss_noise = np.random.normal(0,1,images.shape)

#     s = 0 
#     for i in range(n_rot):
#       s = s+np.sqrt(LA.norm(images[i],'fro')**2/SNR/LA.norm(gauss_noise[i],'fro')**2)/n_rot 

#     return s 
    

def tns_norm(T):
    return LA.norm(T.flatten())


def trim_symmetric_lowrank(m2,U2,tol,r_max):
    Q, Rmat = np.linalg.qr(U2, mode='reduced')    # Q:(n,R), Rmat:(R,R)
    H_small = Rmat @ m2 @ Rmat.conj().T           # (R, R)
    eigvals, eigvecs = np.linalg.eigh(H_small)    
    idx      = np.argsort(np.abs(eigvals))[::-1]
    eigvals  = eigvals[idx]
    eigvecs  = eigvecs[:, idx]
    R_full   = eigvals.size
    sq       = np.abs(eigvals)**2
    cumeng   = np.cumsum(sq)
    total    = cumeng[-1]
    if tol <= np.finfo(float).eps:
        r = R_full
    else:
        k = np.searchsorted(cumeng, (1 - tol)*total, side='left')
        r = min(k + 1, R_full)
    if r>r_max:
        r = r_max
    V_r      = eigvecs[:, :r]                   # (R, r)
    U2_trim  = Q @ V_r                          # (n, r), orthonormal columns
    m2_trim  = np.diag(eigvals[:r])             # (r, r) diagonal core

    return m2_trim, U2_trim, r, eigvals

def trim_symmetric_tucker(core, U, tol, r_max):
    """
    Trim a symmetric Tucker decomposition
        X ≈ core ×₁ U ×₂ U … ×ₙ U
    down to a smaller rank r chosen so that the retained energy ≥ (1-tol).

    Args:
      core : np.ndarray, shape (R, R, ..., R)   # symmetric core, N modes
      U    : np.ndarray, shape (I, R)           # factor, not necessarily orthonormal
      tol  : float in [0,1), e.g. 0.01 for 99% energy retained

    Returns:
      core_trim : np.ndarray, shape (r, ..., r)
      U_trim    : np.ndarray, shape (I, r)
      r         : int, the chosen multilinear rank
    """

    Q, Rmat = np.linalg.qr(U, mode='reduced')     # Q: (I,R), Rmat: (R,R)

    G = core
    N = core.ndim
    for mode in range(N):
        G = np.tensordot(Rmat, G, axes=(1, mode))
        G = np.moveaxis(G, 0, mode)

    Rfull = G.shape[0]
    G1    = G.reshape(Rfull, -1)                  # (R, R^(N-1))
    Uc, S, Vh = np.linalg.svd(G1, full_matrices=False)

    energy = np.cumsum(S**2)
    energy /= energy[-1]
    r = int(np.argmax(energy >= (1 - tol))) + 1
    if r>r_max:
        r = r_max
    Qr     = Uc[:, :r]                             # (R, r)
    U_trim = Q @ Qr                                # (I, r)
    QrH = Qr.conj().T
    core_trim = G
    for mode in range(N):
        core_trim = np.tensordot(QrH, core_trim, axes=(1, mode))
        core_trim = np.moveaxis(core_trim, 0, mode)  # now shape (r,...,r)
    return core_trim, U_trim, r, S