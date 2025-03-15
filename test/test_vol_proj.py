import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))

from utils import * 
from moments import *
import numpy as np 
import numpy.linalg as LA 
import matplotlib.pyplot as plt
import mrcfile 
import finufft 
from scipy.io import savemat
import time 
import jax
import jax.numpy as jnp
from aspire.volume import Volume 
from aspire.image import Image
from aspire.utils import Rotation
from aspire.operators import RadialCTFFilter
from aspire.ctf import estimate_ctf
from aspire.denoising.denoiser_cov2d import DenoiserCov2D
from aspire.source import Simulation

with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())

np.random.seed(1)    
nrot = 10
alpha = np.random.uniform(0,2*np.pi,nrot)
beta = np.random.uniform(0,np.pi,nrot)
gamma = np.random.uniform(0,2*np.pi,nrot)
rots = np.zeros((nrot,3,3),dtype=np.float32)
for i in range(nrot):
    rot = Rz(alpha[i]) @ Ry(beta[i]) @ Rz(gamma[i])
    rots[i,:,:] = rot
# savemat('Rots.mat',{'Rots':rots})

Rots = Rotation(rots) 
Vol = Volume(vol)
Images = Vol.project(Rots)



images = Images.asnumpy()*vol.shape[0]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images[i], cmap='gray')
    # fig.colorbar(im, ax=ax) 

plt.tight_layout()
plt.show()



radial_ctf_filter = RadialCTFFilter(
    pixel_size=1,  # angstrom
    voltage=200,  # kV
    defocus=10000,  # angstrom, 10000 A = 1 um
    Cs=2.26,  # Spherical aberration constant
    alpha=0.07,  # Amplitude contrast phase in radians
    B=0,  # Envelope decay in inverse square angstrom (default 0)
)




Images_ctf = Images.filter(radial_ctf_filter)
images_ctf = Images_ctf.asnumpy()*vol.shape[0]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images_ctf[i], cmap='gray')
    # fig.colorbar(im, ax=ax) 

plt.tight_layout()
plt.show()


imageSource = ImageSource(L=vol.shape[0],n=nrot,dtype="double")
# denoiser = DenoiserCov2D(Images_ctf)


'''
images1 = np.real(vol_proj(vol, rots, 0, 1))
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images1[i], cmap='gray')
    fig.colorbar(im, ax=ax) 

plt.tight_layout()
plt.show()
'''
