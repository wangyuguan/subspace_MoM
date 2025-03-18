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


# form moments 
N = 5000
batch_size = 100
angles = np.zeros((N,3),dtype=np.float32)

print('sampling viewing directions')
samples = sample_sph_coef(N, sph_coef, ell_max_half_view)
print('done')
gamma = np.random.uniform(0,2*np.pi,N)
for i in range(N):
    _, beta, alpha = cart2sph(samples[i,0], samples[i,1], samples[i,2])
    angles[i,0] = alpha 
    angles[i,1] = beta 
    angles[i,2] = gamma[i]
    
    
    
    
    

