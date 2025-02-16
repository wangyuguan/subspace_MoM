import sys
from pathlib import Path

# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))


import numpy as np 
import numpy.linalg as LA 
from viewing_direction import *
from utils import *
from aspire.basis.basis_utils import lgwt
from volume import *
import matplotlib.pyplot as plt


j = 5 
alpha = 2 
beta = 3 
gamma = 2

Rot = Rz(alpha) @ Ry(beta) @ Rz(gamma)


Dj = wignerD(j,alpha,beta,gamma)

# should be close to zero 

err = LA.norm(np.eye(2*j+1) - Dj @ np.conj(Dj).T, 'fro')
print(err)

err = LA.norm(np.eye(2*j+1) - np.conj(Dj).T @ Dj , 'fro')
print(err)

# should agree 

lp = norm_assoc_legendre_all(j, np.cos(beta))


for m in range(-j,j+1):
    lpjm = lp[j,abs(m)]*np.exp(-1j*m*alpha)/np.sqrt(2*j+1)

    if m<0:
        lpjm = lpjm*(-1)**m 


    print(Dj[m+j,j],lpjm,abs(lpjm-Dj[m+j,j])/abs(lpjm))

# generate rotated vectors 

x = np.array([1,2,3])
rx, thx, phx = cart2sph(x[0], x[1], x[2])

y = Rot @ x 
ry, thy, phy = cart2sph(y[0], y[1], y[2])


# evaluate rotated spherical harmonics 



lpy = norm_assoc_legendre_all(j, np.cos(thy))
lpx = norm_assoc_legendre_all(j, np.cos(thx))


for m in range(-j,j+1):
    yjm = lpy[j,abs(m)]*np.exp(1j*m*phy)/np.sqrt(4*np.pi)
    if m<0:
        yjm = yjm*(-1)**m 

    _yjm = 0 

    for mp in range(-j,j+1):
        yjmp = lpx[j,abs(mp)]*np.exp(1j*mp*phx)/np.sqrt(4*np.pi)
        if mp<0:
            yjmp = yjmp*(-1)**mp
            
        _yjm += np.conj(Dj[m+j,mp+j])*yjmp


    print(_yjm,yjm,abs(_yjm-yjm)/abs(yjm))



# test wigner-D transform 

ell_max = 4 
j = 4
mp = 1
m = -2 


def my_fun(alpha,beta,gamma):
    Dj = wignerD(j,alpha,beta,gamma)
    return (1+3j)*Dj[mp+j,m+j]


coef, indices = wignerD_transform(my_fun, ell_max)


# print(coef)
print(coef[indices[j,mp,m]])
print(np.sort(abs(coef)))


# test sph harm transform 


ell_max = 4 
j = 4
m = -1 
def my_fun(th,ph):
    lpall = norm_assoc_legendre_all(j,np.cos(th))
    lpall /= np.sqrt(4*np.pi)
    if m<0:
        lpall = lpall*(-1)**m
    exp_m = np.exp(1j*m*ph)
    return (0.3+0.5*1j)*lpall[j,abs(m),:]*exp_m


coef, indices = full_sph_harm_transform(my_fun, ell_max)

# print(coef)
print(coef[indices[j,m]])
print(np.sort(abs(coef)))


# plot vMF density 

c = 10
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)


ngrid = 50 
_ths = np.pi*np.arange(ngrid)/ngrid
_phs = 2*np.pi*np.arange(ngrid)/ngrid

ths, phs = np.meshgrid(_ths,_phs,indexing='ij')
ths, phs = ths.flatten(), phs.flatten()


# grid = {}
# grid['ths'] = ths
# grid['phs'] = phs 

grid = Grid_3d(type='spherical', ths=ths, phs=phs)


kappa = 3
f_vmf = vMF_density(centers,w_vmf,kappa,grid)
f_vmf = f_vmf.reshape((ngrid,ngrid))

plt.imshow(f_vmf, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')  # Add a colorbar for reference
plt.title("vMF density")
plt.xlabel("thetas")
plt.ylabel("phis")
plt.show()


# test the if the integral over sphere is 1 

xlege, wlege = lgwt(ngrid, -1, 1)
_ths = np.arccos(xlege)

_ths, _phs = np.meshgrid(_ths,_phs,indexing='ij')
_ths, _phs = _ths.flatten(), _phs.flatten()

_wphs = 2*np.pi*np.ones(ngrid)/ngrid

wths, wphs = np.meshgrid(wlege,_wphs,indexing='ij')
wths, wphs = wths.flatten(), wphs.flatten()
wsph = wths*wphs


# _grid = {}
# _grid['ths'] = _ths 
# _grid['phs'] = _phs 
_grid = Grid_3d(type='spherical', ths=_ths, phs=_phs)


f = vMF_density(centers,w_vmf,kappa,_grid)
print(np.sum(f*wsph))


# transform the vMF distribution 


def my_fun(th,ph):
    grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
    return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
# my_fun(1,2)

ell_max = 8 
sph_coef, indices = full_sph_harm_transform(my_fun, ell_max)

f_vmf_expand = full_sph_harm_eval(sph_coef, ell_max, grid)
f_vmf_expand = np.real(f_vmf_expand.reshape([ngrid, ngrid]))


plt.imshow(f_vmf_expand, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')  # Add a colorbar for reference
plt.title("vMF density")
plt.xlabel("thetas")
plt.ylabel("phis")
plt.show()


print(LA.norm(f_vmf-f_vmf_expand/4/np.pi,'fro')/LA.norm(f_vmf,'fro'))


# use only the even degree spherical harmonics 
ell_max_half = 3 

sph_coef, indices = sph_harm_transform(my_fun, ell_max_half)
f_vmf_expand = sph_harm_eval(sph_coef, ell_max_half, grid)
f_vmf_expand = np.real(f_vmf_expand.reshape([ngrid, ngrid]))

plt.imshow(f_vmf_expand, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')  # Add a colorbar for reference
plt.title("vMF density")
plt.xlabel("thetas")
plt.ylabel("phis")
plt.show()


# check real and complex transform

sph_r_t_c, sph_c_t_r = get_sph_r_t_c_mat(ell_max_half)
sph_coef_r = sph_c_t_r @ sph_coef

print(LA.norm(np.imag(sph_coef_r)))


# map to rotation coefficient 
rot_coef = sph_t_rot_coef(sph_coef, ell_max_half)


# integrate over SO3 to get 1 
ell_max = 2*ell_max_half 
euler_nodes, weights = load_so3_quadrature(ell_max, 0)

evals = np.zeros(len(weights), dtype=np.complex128)
for i in range(len(weights)):
    alpha = euler_nodes[i,0]
    beta = euler_nodes[i,1]
    gamma = euler_nodes[i,2]
    for ell in range(ell_max+1):
        if ell % 2 == 0:
            Dl = wignerD(ell,alpha,beta,gamma)
            for m in range(-ell,ell+1):
                evals[i] += weights[i]*rot_coef[indices[(ell,m)]]*Dl[m+ell,ell]

                
print(np.sum(evals))


# effectively doing the same thing as above 

Psi = precompute_rot_density(rot_coef, ell_max_half, euler_nodes)
evals = Psi@rot_coef
print(np.dot(Psi@rot_coef, weights))
