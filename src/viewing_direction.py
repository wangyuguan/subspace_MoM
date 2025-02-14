import numpy as np 
import numpy.linalg as LA 
from utils import *
from aspire.basis.basis_utils import lgwt



def sample_vmf(N,centers,w,kappa,C=5.0):
    
    ncount = 0 
    samples = np.zeros((N,3))
    while ncount<N:
        x = np.random.normal(0,1,size=(3,))
        x /= LA.norm(x)

        f = np.exp(kappa * centers @ x)
        f /= (2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
        f = np.sum(np.diag(w) @ f)

        u = np.random.uniform(0,1,1)[0] 
        if u<f/C:
            samples[ncount,:] = x 
            ncount += 1 

    return samples 


def sample_sph_coef(N, spham_coef, ell_max_half, C=5.0):

    
    samples = np.zeros((N,3))
    ell_max = 2*ell_max_half

    ncount = 0 
    while ncount<N:
        x = np.random.normal(0,1,size=(3,N))
        grid = Grid_3d(type='euclid', xs=x[0,:], ys=x[1,:], zs=x[2,:])
        fs = sph_harm_eval(spham_coef, ell_max_half, grid)
        ratios = fs/C 

        for i in range(N):
            u = np.random.uniform(0,1,1)[0] 
            if u<ratios[i]:
                samples[ncount,:] = x[:,i]
                ncount += 1 

                if ncount == N:
                    break  
    return samples
    

def wignerD_transform(fun, ell_max):
    
    _alpha = np.pi*np.arange(2*ell_max+2)/(ell_max+1)
    _gamma = np.pi*np.arange(2*ell_max+2)/(ell_max+1)

    _walpha = 2*np.pi*np.ones(2*ell_max+2)/(2*ell_max+2)
    _wgamma = 2*np.pi*np.ones(2*ell_max+2)/(2*ell_max+2)

    _beta, _wbeta = lgwt(2*(ell_max+1),-1,1)
    _beta =  np.arccos(_beta)

    [alpha,beta,gamma] = np.meshgrid(_alpha,_beta,_gamma)
    [walpha,wbeta,wgamma] = np.meshgrid(_walpha,_wbeta,_wgamma)

    alpha, beta, gamma = alpha.flatten(), beta.flatten(), gamma.flatten()
    w = walpha*wbeta*wgamma
    w = w.flatten()

    indices = {}
    j = 0 
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                indices[(ell,mp,m)] = j 
                j += 1 


    
    coef =  np.zeros(int(2*(ell_max*(ell_max+1)*(2*ell_max+1))/3+2*(ell_max*(ell_max+1))+ell_max+1), dtype=np.complex128)
    for i in range(len(w)):
        fi = fun(alpha[i],beta[i],gamma[i])
        wi = w[i]
        for ell in range(ell_max+1):
            Dl = wignerD(ell,alpha[i],beta[i],gamma[i])
            C = 8*np.pi**2/(2*ell+1)
            for mp in range(-ell,ell+1):
                for m in range(-ell,ell+1):
                    coef[indices[(ell,mp,m)]] += np.conj(Dl[mp+ell,m+ell])*fi*wi/C

    
    return coef, indices


def vmf_t_rot_coef(centers,w_vmf,kappa,ell_max_half):
    ell_max = 2*ell_max_half 
    def _fun(alpha, beta, gamma):
        grid = Grid_3d(type='spherical', ths=np.array([beta]),phs=np.array([alpha]))
        return 4*np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    _rot_coef , _ = wignerD_transform(_fun, ell_max)
    indices = {}
    rot_coef = []
    i = 0 
    for ell in range(ell_max+1):    
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                if ell % 2 == 0 and m==0:
                    indices[(ell,mp)] = i 
                    rot_coef.append(_rot_coef[i])
                i += 1 
    rot_coef = np.array(rot_coef)
    rot_coef = rot_coef/rot_coef[0]
    return rot_coef, indices
    
    


def full_sph_harm_transform(fun, ell_max):
    
    grid = get_spherequad(ell_max+1, 2*ell_max+1)

    ths = grid.ths 
    phs = grid.phs 
    w = grid.w

    ths_unique, ths_indices = np.unique(ths, return_inverse=True)
    phs_unique, phs_indices = np.unique(phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(ths_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(phs_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*phs_unique)

    f = np.zeros(len(w),dtype=np.complex128)
    for i in range(len(w)):
        f[i] = fun(ths[i],phs[i])

    coef = np.zeros((ell_max+1)**2,dtype=np.complex128)
    i = 0 
    indices = {}
    for ell in range(0,ell_max+1):
        for m in range(-ell,ell+1):
            lpmn = lpall[ell,abs(m),:]
            exps = exp_all[m+ell_max,:]
            if m<0:
                lpmn = lpmn*(-1)**m
            coef[i] += np.sum(np.conj(lpmn[ths_indices]*exps[phs_indices])*w*f)
            indices[(ell,m)] = i 
            i += 1 
                
    return coef, indices


def full_sph_harm_eval(sph_coef, ell_max, grid):
    
    indices = {}
    n_coef = 0 
    for ell in np.arange(ell_max+1):
        for m in range(-ell, ell+1):
            indices[(ell,m)] = n_coef 
            n_coef += 1
    
    ths = grid.ths
    phs = grid.phs

    ths_unique, ths_indices = np.unique(ths, return_inverse=True)
    phs_unique, phs_indices = np.unique(phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(ths_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(phs_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*phs_unique)

    evals = 0
    for ell in range(0,ell_max+1):
        for m in range(-ell,ell+1):
            lpmn = lpall[ell,abs(m),:]
            exps = exp_all[m+ell_max,:]
            if m<0:
                lpmn = lpmn*(-1)**m
            evals += sph_coef[indices[(ell,m)]]*lpmn[ths_indices]*exps[phs_indices]
    
    return evals
            
def sph_harm_transform(fun, ell_max_half):
    
    ell_max = 2*ell_max_half
    
    grid = get_spherequad(ell_max+1, 2*ell_max+1)

    ths = grid.ths 
    phs = grid.phs 
    w = grid.w 

    ths_unique, ths_indices = np.unique(ths, return_inverse=True)
    phs_unique, phs_indices = np.unique(phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(ths_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(phs_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*phs_unique)

    f = np.zeros(len(w),dtype=np.complex128)
    for i in range(len(w)):
        f[i] = fun(ths[i],phs[i])

    
    n_coef = 0 
    indices = {}
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1
              
    coef = np.zeros(n_coef, dtype=np.complex128)
    for ell in range(0,ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell,ell+1):
              lpmn = lpall[ell,abs(m),:]
              exps = exp_all[m+ell_max,:]
              if m<0:
                  lpmn = lpmn*(-1)**m
              coef[indices[(ell,m)]] += np.sum(np.conj(lpmn[ths_indices]*exps[phs_indices])*w*f)

                
    return coef, indices



def sph_harm_eval(spham_coef, ell_max_half, grid):
    
    ell_max = 2*ell_max_half
    indices = {}
    n_coef = 0 
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1
    
    ths = grid.ths
    phs = grid.phs

    ths_unique, ths_indices = np.unique(ths, return_inverse=True)
    phs_unique, phs_indices = np.unique(phs, return_inverse=True)

    lpall = norm_assoc_legendre_all(ell_max, np.cos(ths_unique))
    lpall /= np.sqrt(4*np.pi)

    exp_all = np.zeros((2*ell_max+1,len(phs_unique)), dtype=complex)
    for m in range(-ell_max,ell_max+1):
        exp_all[m+ell_max,:] = np.exp(1j*m*phs_unique)

    evals = 0
    for ell in range(0,ell_max+1):
        if ell % 2 ==0:
          for m in range(-ell,ell+1):
              lpmn = lpall[ell,abs(m),:]
              exps = exp_all[m+ell_max,:]
              if m<0:
                  lpmn = lpmn*(-1)**m
              evals += spham_coef[indices[(ell,m)]]*lpmn[ths_indices]*exps[phs_indices]
    
    return evals


def get_sph_r_t_c_mat(ell_max_half):
    
    ell_max = 2*ell_max_half
    n_coef = 0 
    indices = {}
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1

    sph_r_t_c = np.zeros((n_coef, n_coef), dtype=np.complex128)
    for ell in range(ell_max+1):
        if ell % 2 == 0:
            for m in range(-ell,ell+1):
                i = indices[(ell,m)]
                _i = indices[(ell,-m)]
                if m==0:
                    sph_r_t_c[i,i] = 1 
                elif m>0:
                    sph_r_t_c[i,i] = 1 
                    sph_r_t_c[i,_i] = (-1)**m*1j 
                else:
                    sph_r_t_c[i,i] = -1j 
                    sph_r_t_c[i,_i] = (-1)**m
    
    sph_c_t_r = LA.inv(sph_r_t_c)

    return sph_r_t_c, sph_c_t_r 


def sph_t_rot_coef(sph_coef, ell_max_half):
    
    ell_max = 2*ell_max_half
    n_coef = 0 
    indices = {}
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1

    rot_coef = np.zeros(sph_coef.shape, dtype=np.complex128)
    for ell in range(ell_max+1):
        if ell % 2 ==0:
            for m in range(-ell,ell+1):
                rot_coef[indices[(ell,m)]] = (-1)**m*sph_coef[indices[(ell,-m)]]/np.sqrt(4*np.pi/(2*ell+1))

    return rot_coef




def precompute_rot_density(rot_coef, ell_max_half, euler_nodes):
    """
    precompute Psi[i,(l,m)] = Dl_{m,0}(Ri) 
    """
    ell_max = 2*ell_max_half
    n_coef = 0 
    indices = {}
    for ell in np.arange(ell_max+1):
        if ell % 2 == 0:
          for m in range(-ell, ell+1):
              indices[(ell,m)] = n_coef 
              n_coef += 1

    n_nodes = euler_nodes.shape[0]
    Psi = np.zeros([n_nodes, n_coef], dtype=np.complex128)

    for i in range(n_nodes):
        
        alpha = euler_nodes[i,0]
        beta = euler_nodes[i,1]
        gamma = euler_nodes[i,2]

        for ell in range(ell_max+1):
            if ell % 2 == 0:
                Dl = wignerD(ell,alpha,beta,gamma)
                for m in range(-ell, ell+1):
                    Psi[i,indices[(ell,m)]] = Dl[m+ell,ell]
    
    return Psi

    
    
    


