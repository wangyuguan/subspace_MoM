#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 10:45:44 2025

@author: yuguan
"""

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


import utils_cwf_fast_batch as utils
import matplotlib.pyplot as plt
import mrcfile

from fle_2d_single import FLEBasis2D
from fast_cryo_pca import FastPCA
import logging

import numpy as np
import scipy.linalg as LA
from aspire.source.simulation import Simulation
from aspire.volume import Volume
# from aspire.operators import ScalarFilter
from aspire.operators import RadialCTFFilter
from aspire.noise import WhiteNoiseAdder
import time 

batch_size = 100
num_imgs = 1000


eps = 1e-3


# %% load volume data 
with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    vol = vol/LA.norm(vol.flatten())

img_size = vol.shape[0]
Vol = Volume(vol)


# %% Specify the CTF parameters
pixel_size = 1.04  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1e4  # Minimum defocus value (in angstroms)
defocus_max = 3e4  # Maximum defocus value (in angstroms)
defocus_ct = 100 # the number of defocus groups
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# create CTF indices for each image, e.g. h_idx[0] returns the CTF index (0 to 99 if there are 100 CTFs) of the 0-th image
h_idx = utils.create_ordered_filter_idx(num_imgs, defocus_ct)
dtype = np.float32

h_ctf = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]


# %% determine noise level 
source_ctf_clean = Simulation(
    L=img_size,
    n=num_imgs,
    vols=Vol,
    offsets=0.0,
    amplitudes=1.0,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    dtype=dtype,
)



# determine noise variance to create noisy images with certain SNR
sn_ratio = 10 
noise_var = utils.get_noise_var_batch(source_ctf_clean, sn_ratio, batch_size)

# create noise filter
# noise_filter = ScalarFilter(dim=2, value=noise_var)
noise_adder = WhiteNoiseAdder(var=noise_var)

# %% create simulation object for noisy images
source = Simulation(
    L=img_size,
    n=num_imgs,
    vols=Vol,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    offsets=0.0,
    amplitudes=1.0,
    dtype=dtype,
    noise_adder=noise_adder,
)




# %% create fast PCA 

print('building fast pca ... ')
fle = FLEBasis2D(img_size, img_size, eps=eps)
# options for covariance estimation
options = {
    "whiten": False,
    "single_pass": True, # whether estimate mean and covariance together (single pass over data), not separately
    "noise_var": noise_var, # noise variance
    "batch_size": batch_size,
    "dtype": dtype
}

# create fast PCA object
fast_pca = FastPCA(source, fle, options)
# options for denoising




print('estmating mean and covariance ...')
t1 = time.time()
mean_est, covar_est = fast_pca.estimate_mean_covar()
t2 = time.time()
print(t2-t1)



# %% denoise per defocus group 
n = 0
for i in range(defocus_ct):

    denoise_options = {
        "denoise_df_id": [i], # denoise 0-th, 30-th, 60-th, 90-th defocus groups
        "denoise_df_num": [int(num_imgs/defocus_ct)+1], # for each defocus group, respectively denoise the first 10, 15, 1, 100 images
                                            # 240 exceed the number of images (100) per defocus group, so only 100 images will be returned
        "return_denoise_error": True,
        "store_images": True,
    }
    
    
    print('perform denoising ...')
    t1 = time.time()
    results = fast_pca.denoise_images(mean_est=mean_est, covar_est=covar_est, denoise_options=denoise_options)
    print(int(num_imgs/defocus_ct)+1, results["denoised_images"].shape)
    t2 = time.time()
    print(t2-t1)
    
    n += len(results["denoised_images"])
print(n)

# %% visualization 
imgs_gt = results["clean_images"]
imgs_raw = results["raw_images"]
imgs_est = results["denoised_images"]


_, frc_vec = utils.compute_frc(imgs_est, imgs_gt, fle.n_angular, fle.eps, dtype=dtype)

im = 0
plt.subplot(2,2,1)
plt.imshow(imgs_gt[im], 'gray')
plt.title("clean")
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(imgs_raw[im], 'gray')
plt.title("noisy")
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(imgs_est[im], 'gray')
plt.title("denoised")
plt.axis("off")
plt.subplot(2,2,4)
plt.plot(frc_vec, '-r')
plt.title("FRC")

plt.show()



'''fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(imgs_est[i], cmap='gray')
plt.tight_layout()
plt.show()
'''


