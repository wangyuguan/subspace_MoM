import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))


from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.basis import FBBasis3D
from utils import * 
from viewing_direction import * 
from moments import *
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mrcfile 
from scipy.io import savemat 
from utils_BO import align_BO



with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())
    
rots = np.zeros((10,3,3))
for i in range(10):
    alpha = 2
    beta = 1.3
    gamma = np.random.uniform(0,2*np.pi)
    rots[i] = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    
I_projs = np.real(vol_proj(vol, rots, 0, 1))


fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(I_projs[i], cmap='gray')
plt.tight_layout()
plt.show()

    



