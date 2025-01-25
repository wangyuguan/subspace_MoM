import numpy as np 
from volume import cart2sph, norm_assoc_legendre_all
from aspire.basis.basis_utils import lgwt

def vMF_density(centers,w,kappa,xs):
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
    
    f = np.exp(kappa * centers @ xs.T)
    C = kappa/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    f = C*f 
    f = np.sum(np.diag(w) @ f, 0)
    return f 

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

def wignerD_transform(fun, ell_max):
    
    _alpha = 2*np.pi*np.arange(2*ell_max+1)/(2*ell_max+1)
    _gamma = 2*np.pi*np.arange(2*ell_max+1)/(2*ell_max+1)

    _walpha = 2*np.pi*np.ones(2*ell_max+1)/(2*ell_max+1)
    _wgamma = 2*np.pi*np.ones(2*ell_max+1)/(2*ell_max+1)

    _beta, _wbeta = lgwt(2*(ell_max+1),-1,1)
    _beta =  np.arccos(_beta)

    [alpha,beta,gamma] = np.meshgrid(_alpha,_beta,_gamma)
    [walpha,wbeta,wgamma] = np.meshgrid(_walpha,_wbeta,_wgamma)

    alpha, beta, gamma = alpha.flatten(), beta.flatten(), gamma.flatten()
    w = walpha*wbeta*wgamma
    w = w.flatten()

    
    coef =  np.zeros(int(2*(ell_max*(ell_max+1)*(2*ell_max+1))/3+2*(ell_max*(ell_max+1))+ell_max+1), dtype=np.complex128)
    for i in range(len(w)):
        fi = fun(alpha[i],beta[i],gamma[i])
        wi = w[i]
        j = 0 
        for ell in range(ell_max+1):
            Dl = wignerD(ell,alpha[i],beta[i],gamma[i])
            C = 8*np.pi**2/(2*ell+1)
            for mp in range(-ell,ell+1):
                for m in range(-ell,ell+1):
                    coef[j] += np.conj(Dl[mp+ell,m+ell])*fi*wi/C
                    j += 1 

    indices = {}
    j = 0 
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                indices[(ell,mp,m)] = j 
                j += 1 

    return coef, indices


def sph_harm_transform(fun, ell_max):
    
    ths, phs, w = get_spherequad(ell_max+1, 2*ell_max+1)

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


def get_spherequad(nth, nph):
    """
    Get the quadrature rule under polor coord on unit sphere such that 
    
    \int_{||x||=1} f dS  =  \sum_i w(i) * f(th(i),ph(i))  

    :param nr: The order of discretization for radial part 
    :param nth: The order of discretization for polar part 
    :param nph: The order of discretization for azimuthal  part 
    :param R: The radius of the ball 
    :return: The quadrature points under spherical coordinate and the weights

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

    return ths, phs, w


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

