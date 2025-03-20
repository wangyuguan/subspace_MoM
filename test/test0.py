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

from utils import * 
from viewing_direction import * 
from moments import *
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
from scipy.io import savemat, loadmat 


np.random.seed(42)



with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    # vol = vol/LA.norm(vol.flatten())
    
# preprocess the volume 
L = vol.shape[0]
ds_res = 64
vol_ds = vol_downsample(vol, ds_res)
ell_max_vol = 5
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
vol_ds = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol_ds = vol_ds.reshape(ds_res,ds_res,ds_res,order='F')
vol = vol_upsample(vol_ds,L)
vol = vol/LA.norm(vol.flatten())
Vol = Volume(np.array(vol, dtype=np.float32))



# set up viewing direction distribution 
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


# form moments 
N = 5000
batch_size = 100
angles = np.zeros((N,3),dtype=np.float32)
rots = np.zeros((N,3,3),dtype=np.float32)

print('sampling viewing directions')
samples = sample_sph_coef(N, sph_coef, ell_max_half_view)
print('done')
gamma = np.random.uniform(0,2*np.pi,N)
for i in range(N):
    _, beta, alpha = cart2sph(samples[1,0], samples[1,1], samples[1,2])
    angles[i,0] = alpha 
    angles[i,1] = beta 
    angles[i,2] = gamma[i]
    rots[i,:,:] = Rz(alpha) @ Ry(beta) @ Rz(gamma[i])
    
images = Vol.project(Rotation(rots[0:100])).asnumpy()*L
    
    

    
    
pixel_size = 1.04  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1e4  # Minimum defocus value (in angstroms)
defocus_max = 3e4  # Maximum defocus value (in angstroms)
defocus_ct = 100 # the number of defocus groups
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# create CTF indices for each image, e.g. h_idx[0] returns the CTF index (0 to 99 if there are 100 CTFs) of the 0-th image
h_idx = utils.create_ordered_filter_idx(N, defocus_ct)
dtype = np.float32

h_ctf = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]
noise_adder = WhiteNoiseAdder(var=0)


source = Simulation(
    L=L,
    n=N,
    vols=Vol,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    offsets=0.0,
    amplitudes=1.0,
    dtype=dtype,
    noise_adder=noise_adder,
)

source.rotations = rots


images1 = source.images[0:100].asnumpy()*L

    


fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images[i], cmap='gray')
plt.tight_layout()
plt.show()




fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(images1[i], cmap='gray')
plt.tight_layout()
plt.show()


# %%





