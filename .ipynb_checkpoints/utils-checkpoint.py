import numpy as np 
from aspire.basis.basis_utils import all_besselj_zeros, sph_bessel 

def sph_bessel_derivative(ell, r):
    """
    Compute derivatives of spherical Bessel function values.

    :param ell: The order of the spherical Bessel function.
    :param r: The coordinates where the function is to be evaluated.
    :return: The value of j_ell at r.
    """
    scalar = np.isscalar(r)
    len_r = 1 if scalar else len(r)

    j = np.zeros(len_r)
    j[r == 0] = 1 if ell == 0 else 0

    r_mask = r != 0
    j[r_mask] = sph_bessel(ell, r[r_mask])

    j1 = np.zeros(len_r)
    if ell==0:
        j1[r_mask] = sph_bessel(1, r[r_mask])
        j[r_mask] = -j1[r_mask]+ell*j[r_mask]/r[r_mask]
        j[r==0] = 0 
    elif ell==1:
        j1[r_mask] = sph_bessel(0, r[r_mask])
        j[r_mask] = j1[r_mask]-(ell+1)*j[r_mask]/r[r_mask]
        j[r==0] = 1 
    else:
        j1[r_mask] = sph_bessel(ell-1, r[r_mask])
        j[r_mask] = j1[r_mask]-(ell+1)*j[r_mask]/r[r_mask]
        j[r==0] = 1 

    if scalar:
        j = j.item()
        
    return j


    

