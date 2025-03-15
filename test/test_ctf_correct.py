import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))


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
from aspire.basis import FFBBasis2D
from aspire.covariance import RotCov2D
from aspire.source import Simulation
from aspire.denoising.denoiser_cov2d import DenoiserCov2D

np.random.seed(42)


# %% load volume data 
with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())

img_size = vol.shape[0]
Vol = Volume(vol)



# %% generate euler angles for rotation 
nimage  = 1000
angles = np.zeros((nimage,3))
angles[:,0] = np.random.uniform(0,2*np.pi,nimage)
angles[:,1] = np.arccos(np.random.uniform(-1,1,nimage))
angles[:,2] = np.random.uniform(0,2*np.pi,nimage)


# %% ctf functions
pixel_size = 1 
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast
ctf_filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %% create image generator 
sim = Simulation(L=img_size, n=nimage, vols=Vol, angles=angles, offsets = 0, 
                 unique_filters=ctf_filters)


# %% visualize images with CTF effect 
Images = sim.images[:]
images_ctf = Images.asnumpy()
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images_ctf[i], cmap='gray')
plt.tight_layout()
plt.show()

# %% expand images using basis 
ffbbasis = FFBBasis2D((img_size, img_size), dtype = np.float32)

# Assign the CTF information and index for each image
h_idx = sim.filter_indices

# Evaluate CTF in the 8X8 FB basis
h_ctf_fb = [ffbbasis.filter_to_basis_mat(filt) for filt in ctf_filters]


covar_opt = {
    "shrinker": "frobenius_norm",
    "verbose": 0,
    "max_iter": 250,
    "iter_callback": [],
    "store_iterates": False,
    "rel_tolerance": 1e-12,
    "precision": "float64",
    "preconditioner": "identity",
}

coef = ffbbasis.evaluate_t(sim.images[:])


# %% cov2d denoise 
cov2d = RotCov2D(ffbbasis)
mean_coef_est = cov2d.get_mean(coef, h_ctf_fb, h_idx)
covar_coef_est = cov2d.get_covar(
    coef,
    h_ctf_fb,
    h_idx,
    mean_coef_est,
    noise_var=0,
    covar_est_opt=covar_opt,
)

