import numpy as np 
from aspire.basis.basis_utils import lgwt 

class Grid_3d:
    
    def __init__(self, type='euclid', xs=None, ys=None, zs=None, rs=None, ths=None, phs=None, w=None):
        
        if type=='euclid':
            
            self.xs = xs 
            self.ys = ys 
            self.zs = zs 


            self.s, self.ths, self.phs = cart2sph(self.xs, self.ys, self.zs)


        if type=='spherical':
            
            self.ths = ths 
            self.phs = phs 

            if rs is None:
                self.rs = np.ones(len(ths))
            else:
                self.rs = rs 


            self.xs = self.rs*np.sin(self.ths)*np.cos(self.phs)
            self.ys = self.rs*np.sin(self.ths)*np.sin(self.phs)
            self.zs = self.rs*np.cos(self.ths)


        if w is None:
            self.w = np.ones(len(self.xs))
        else:
            self.w = w 

    def get_xyz_combined(self):
        return np.column_stack((self.xs, self.ys, self.zs))
    

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
  
    ths, phs = np.meshgrid(ths, phs, indexing='ij')
    ths, phs = ths.flatten(), phs.flatten()

    wths, wphs = np.meshgrid(wths, wphs, indexing='ij')
    wths, wphs = wths.flatten(), wphs.flatten()
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
    [r,wr] = lgwt(nr,0,R)
    [th,wth] = lgwt(nth,-1,1)
    th = np.arccos(th)
    ph = phis = 2*np.pi*np.arange(0,nph)/nph
    wph = 2*np.pi*np.ones(nph)/nph


    [r,th,ph] = np.meshgrid(r,th,ph,indexing='ij')
    [wr,wth,wph] = np.meshgrid(wr,wth,wph,indexing='ij')

    w = wr*wth*wph*(r**2)
    w = w.flatten()

    r = r.flatten()
    th = th.flatten()
    ph = ph.flatten()

    return Grid_3d(type='spherical',rs=r,ths=th,phs=ph,w=w)

        
        
def get_3d_unif_grid(n):
    """
    Get the equispace quadrature points in 3D space 
    :param n: The order of discretization in each dimension 
    :return: The 3d grid points object

    """
    if n%2==0:
        x = np.arange(-n/2,n/2)
    else:
        x = np.arange(-(n-1)/2,(n-1)/2+1)
 
    [x,y,z] = np.meshgrid(x,x,x,indexing='ij')
    grid = Grid_3d(type='euclid', xs=x.flatten(), ys=y.flatten(), zs=z.flatten())

    return grid



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
    theta = np.arccos(z[r!=0] / r[r!=0])         # Polar angle
    phi = np.arctan2(y, x)                       # Azimuthal angle
    return r, theta, phi
