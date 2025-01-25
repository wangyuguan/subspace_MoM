# utility functions extracted from aspire 
import numpy as np 
from aspire.basis.basis_utils import all_besselj_zeros, sph_bessel 


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


def get_indices(ell_max, k_max):
    """
    Create the indices for each basis function
    """
    count = sum(k_max * (2 * np.arange(0, ell_max + 1) + 1))
    indices_ells = np.zeros(count, dtype=int)
    indices_ms = np.zeros(count, dtype=int)
    indices_ks = np.zeros(count, dtype=int)

    ind = 0
    for ell in range(ell_max + 1):
        ks = range(0, k_max[ell])
        for m in range(-ell, ell + 1):
            rng = range(ind, ind + len(ks))
            indices_ells[rng] = ell
            indices_ms[rng] = m
            indices_ks[rng] = ks

            ind += len(ks)

    return {"ells": indices_ells, "ms": indices_ms, "ks": indices_ks}



def basis_norm_3d(self, ell, k, nres, r0):
    """
    Calculate the normalized factor of a specified basis function.
    """
    return (
        np.abs(sph_bessel(ell + 1, r0[ell][k - 1]))
        / np.sqrt(2)
        * np.sqrt((self.nres / 2) ** 3)
        )