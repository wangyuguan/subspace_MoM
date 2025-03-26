import sys
from pathlib import Path
import os 


# Add the 'src' directory to the Python path
src_path = Path('../../fast-cryoEM-PCA').resolve()
sys.path.append(str(src_path))
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))


from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.source.simulation import Simulation
from aspire.operators import RadialCTFFilter
from aspire.noise import WhiteNoiseAdder
from utils import * 
from viewing_direction import * 
from moments import *
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mrcfile 
from scipy.io import savemat 
from utils_BO import align_BO
import utils_cwf_fast_batch as utils
from scipy.sparse import block_diag
from scipy.sparse.linalg import svds

np.random.seed(42)
jax.config.update('jax_platform_name', 'cpu')

with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    
    
# %% preprocess the volume 
img_size = vol.shape[0]
ds_res = 64
vol_ds = vol_downsample(vol, ds_res)
ell_max_vol = 5
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
vol_ds = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol_ds = vol_ds.reshape(ds_res,ds_res,ds_res,order='F')
vol = vol_upsample(vol_ds,img_size)
vol = vol/LA.norm(vol.flatten())
Vol= Volume(np.array(vol, dtype=np.float32))



# %% set up viewing direction distribution 
c = 10
centers = np.random.normal(0,1,size=(c,3))
centers /= LA.norm(centers, axis=1, keepdims=True)
w_vmf = np.random.uniform(0,1,c)
w_vmf = w_vmf/np.sum(w_vmf)
kappa = 5
def my_fun(th,ph):
    grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
    return np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
ell_max_half_view = 2
sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)



# %% generate angles 
batch_size = 5000
defocus_ct = 10
N = batch_size*defocus_ct
angles = np.zeros((N,3),dtype=np.float32)
rotations = np.zeros((N,3,3),dtype=np.float32)
print('sampling viewing directions')
samples = sample_sph_coef(N, sph_coef, ell_max_half_view)
print('done')
gamma = np.random.uniform(0,2*np.pi,N)
for i in range(N):
    _, beta, alpha = cart2sph(samples[i,0], samples[i,1], samples[i,2])
    angles[i,0] = alpha 
    angles[i,1] = beta 
    angles[i,2] = gamma[i]
    rotations[i,:,:] = Rz(alpha) @ Ry(beta) @ Rz(gamma[i])


# %% generate CTF 
pixel_size = 1.04  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1e4  # Minimum defocus value (in angstroms)
defocus_max = 3e4  # Maximum defocus value (in angstroms)
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# create CTF indices for each image, e.g. h_idx[0] returns the CTF index (0 to 99 if there are 100 CTFs) of the 0-th image
# h_idx = utils.create_ordered_filter_idx(N, defocus_ct)
h_idx = []
for i in range(defocus_ct):
    h_idx += [i]*batch_size

dtype = np.float32

h_ctf = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]


source_ctf_clean = Simulation(
    L=img_size,
    n=N,
    vols=Vol,
    offsets=0.0,
    amplitudes=1.0,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    dtype=dtype,
)

#  determine noise variance to create noisy images with certain SNR
sn_ratio = 1000
noise_var = utils.get_noise_var_batch(source_ctf_clean, sn_ratio, batch_size)

# %% create noise filter
noise_adder = WhiteNoiseAdder(var=noise_var)


source = Simulation(
    L=img_size,
    n=N,
    vols=Vol,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    offsets=0.0,
    amplitudes=1.0,
    dtype=dtype,
    noise_adder=noise_adder,
)
source.rotations = rotations

# %% fast PCA
eps = 1e-3
fle = FLEBasis2D(img_size, img_size, eps=eps)
noise_var = source.noise_adder.noise_var
options = {
    "whiten": False,
    "single_pass": True, # whether estimate mean and covariance together (single pass over data), not separately
    "noise_var": noise_var, # noise variance
    "batch_size": batch_size,
    "dtype": np.float64
}
fast_pca = FastPCA(source, fle, options)
defocus_ct = int(N/batch_size)

print('estimate mean and covariance ...')
mean_est, covar_est = fast_pca.estimate_mean_covar()

# %% make the second moment non-centered

mu0 = mean_est.reshape((-1,1))
mu0 = mu0[:covar_est[0].shape[0]]
covar_est[0] += mu0 @ np.conj(mu0.T)



# %% find the eigen-image 
covar_est_mat = block_diag(covar_est)
k = 300
U,S,Vh = svds(covar_est_mat,k)


# %% transform into basis in pixel space 
U2 = np.zeros((img_size**2,k))
for i in range(k):
    U2[:,i] = fle.evaluate(U[:,i]).flatten()
Q, _ = LA.qr(U2)


# %% compress an image using the PCs
image = source.projections[0].asnumpy().squeeze()
image = image.flatten()
image_proj = Q @ (Q.T @ image)

image = image.reshape((img_size,img_size))
image_proj = image_proj.reshape((img_size,img_size))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Image 1")
axs[0].axis('off')

axs[1].imshow(image_proj, cmap='gray')
axs[1].set_title("Image 2")
axs[1].axis('off')

plt.tight_layout()
plt.savefig('eigen_image_test.pdf')
plt.show()