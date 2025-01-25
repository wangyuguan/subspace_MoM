import numpy as np 

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

