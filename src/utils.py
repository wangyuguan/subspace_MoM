import numpy as np 
from aspire.basis.basis_utils import lgwt 
import e3x
from scipy.linalg import eigh


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


def load_sph_gauss_quadrature(N):
    
    is_gauss = True 
    N
    
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

    phs = np.pi - np.arctan2(data[:,0],data[:,1])
    ths = np.arccos(data[:,2])
      
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