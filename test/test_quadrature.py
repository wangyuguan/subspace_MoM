import sys
from pathlib import Path

# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))

import numpy as np 
import numpy.linalg as LA 
from utils import * 
from viewing_direction import * 
from volume import * 

ell_max = 20
grids = load_sph_gauss_quadrature(ell_max)

integrals = []
lpall = norm_assoc_legendre_all(ell_max, np.cos(grids.ths))
lpall = lpall/np.sqrt(4*np.pi)
exp_all = np.zeros((2*ell_max+1,len(grids.phs)), dtype=complex)
for m in range(-ell_max,ell_max+1):
    exp_all[m+ell_max,:] = np.exp(1j*m*grids.phs)


for ell in range(ell_max+1):
    for m in range(-ell,ell+1):
        lp = lpall[ell,abs(m),:]
        if m<0:
            lp = (-1)**m*lp
        vals  = lp*exp_all[m+ell_max,:]
        integrals.append(np.sum(vals*grids.w))

integrals = np.array(integrals, dtype=np.complex128)


# integrate all spherical harmonics, except that the integral of Y00 
# is 1/sqrt(4*pi), the others are zeros

print(integrals[0]/np.sqrt(4*np.pi), LA.norm(integrals[1:], np.inf))

ell_max_1 = 5 
ell_max_2 = 2 
euler_nodes, weights  = load_so3_quadrature(ell_max_1, ell_max_2)
np.sum(weights)