# utility functions extracted from aspire 
import numpy as np 
from aspire.basis.basis_utils import all_besselj_zeros, sph_bessel 






def basis_norm_3d(self, ell, k, nres, r0):
    """
    Calculate the normalized factor of a specified basis function.
    """
    return (
        np.abs(sph_bessel(ell + 1, r0[ell][k - 1]))
        / np.sqrt(2)
        * np.sqrt((self.nres / 2) ** 3)
        )