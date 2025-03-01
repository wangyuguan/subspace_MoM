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

ds_res = 64
with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
# savemat('vol.mat',{'vol':vol})
L = vol.shape[0];
vol = vol/LA.norm(vol.flatten())
vol_ds = vol_downsample(vol, ds_res)
ell_max_vol = 3
vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
vol_ds = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
vol_ds = vol_ds.reshape(ds_res,ds_res,ds_res,order='F')
vol = vol_upsample(vol_ds,L)
vol = np.array(vol, dtype=np.float32)
with mrcfile.new('vol.mrc', overwrite=True) as mrc:
    mrc.set_data(vol)  # Set the volume data
    mrc.voxel_size = 1.0 
vol_ds_ds = vol_downsample(vol,ds_res)
print(LA.norm(vol_ds.flatten()-vol_ds_ds.flatten())/LA.norm(vol_ds.flatten()))
    
np.random.seed(1)    
nrot = 1000
alpha = np.random.uniform(0,2*np.pi,nrot)
beta = np.random.uniform(0,np.pi,nrot)
gamma = np.random.uniform(0,2*np.pi,nrot)
rots = np.zeros((nrot,3,3),dtype=np.float32)
for i in range(nrot):
    rot = Rz(alpha[i]) @ Ry(beta[i]) @ Rz(gamma[i])
    rots[i,:,:] = rot
# savemat('Rots.mat',{'Rots':rots})
    
ds_res2 = ds_res**2
images = vol_proj(vol, rots)
images = vol_downsample(images, ds_res)
r2 = 200
r3 = 50
G = np.random.normal(0,1,(ds_res2,r2))
G1 = np.random.normal(0,1,(ds_res2,r3))
G2 = np.random.normal(0,1,(ds_res2,r3))
M2 = 0 
M3 = 0 
Ntot  = 1000

t1 = time.time()
for imag in images:
    I = imag.reshape(ds_res2, 1, order='F').astype(np.float64)
    I_trans = I.T
    M2 = M2 + I @ (I_trans @ G)/Ntot
    M3 = M3 + I @ ((I_trans @ G1) * (I_trans @ G2))/Ntot 
t2 = time.time()
print(t2-t1)


t1 = time.time()
_M2 = 0 
_M3 = 0 
nimag = images.shape[0]
images = jnp.array(images)
I_all = jnp.transpose(images, (0, 2, 1)).reshape(nimag, ds_res2).astype(jnp.float64)
I_all = I_all[..., None]
I_all_T = jnp.transpose(I_all, (0, 2, 1))
_M2 = _M2 +  jnp.sum(jnp.matmul(I_all, jnp.matmul(I_all_T, G)), axis=0) / Ntot
_M3 = _M3 +  jnp.sum(jnp.matmul(I_all, (jnp.matmul(I_all_T, G1) * jnp.matmul(I_all_T, G2))), axis=0) / Ntot
t2 = time.time()
print(t2-t1)


U2 = np.random.normal(0,1,(ds_res2,r2))
U3 = np.random.normal(0,1,(ds_res2,r3))
m1 = np.zeros((r2,1))
m2 = np.zeros((r2,r2))
m3 = np.zeros((r3,r3,r3))
t1 = time.time()
for imag in images:
    I = imag.reshape(ds_res2, 1, order='F').astype(np.float64)
    I2 = U2.T @ I 
    I3 = U3.T @ I 
    I3 = I3.flatten()
    m1 = m1+I2/Ntot 
    m2 = m2+(I2@I2.T)/Ntot 
    m3 = m3+np.einsum('i,j,k->ijk',I3,I3,I3)/Ntot 
t2 = time.time()
print(t2-t1)

_m1 = jnp.zeros((r2,1))
_m2 = jnp.zeros((r2,r2))
_m3 = jnp.zeros((r3,r3,r3))
t1 = time.time()
I_all = jnp.transpose(images, (0, 2, 1)).reshape(images.shape[0], ds_res2).astype(jnp.float64)
I2_all = jnp.tensordot(I_all, U2, axes=([1], [0]))  
I3_all = jnp.tensordot(I_all, U3, axes=([1], [0]))  
_m1 = _m1 + jnp.sum(I2_all, axis=0).reshape(r2,1) / Ntot  
_m2 = _m2 + jnp.einsum('ni,nj->ij', I2_all, I2_all) / Ntot  
_m3 = _m3 + jnp.einsum('ni,nj,nk->ijk', I3_all, I3_all, I3_all) / Ntot
t2 = time.time()
print(t2-t1)


'''
n = vol.shape[0]
if n % 2 == 0:
    k = np.arange(-n/2,n/2)/n 
else:
    k = np.arange(-(n-1)/2,(n-1)/2+1)/n
kx, ky = np.meshgrid(k, k, indexing='xy')
kx = kx.flatten(order='F')
ky = ky.flatten(order='F')

rotated_grids = np.zeros((3,n**2,nrot))
for i in range(nrot):
    rot = rots[i]
    rotated_grids[:,:,i] = rot[:,0].reshape(-1, 1) @ kx.reshape(1,n**2)+rot[:,1].reshape(-1, 1) @ ky.reshape(1,n**2)
    
s = 2*np.pi*rotated_grids[0].flatten(order='F')
t = 2*np.pi*rotated_grids[1].flatten(order='F')
u = 2*np.pi*rotated_grids[2].flatten(order='F')


# savemat('coords.mat',{'ss':s,'tt':t, 'uu':u})

vol = np.array(vol, dtype=np.complex128)
# vol1 = np.transpose(vol, (1, 0, 2))
# savemat('vol1.mat',{'vol1':vol1})
Img_fft_rot = finufft.nufft3d2(s,t,u,np.transpose(vol, (1, 0, 2)))
# savemat('Img_fft_rot.mat',{'Img_fft_rot':Img_fft_rot})

Img_fft_rot = Img_fft_rot.reshape(n**2,nrot,order='F')
fig, axes = plt.subplots(1, nrot, figsize=(15, 5))
for i, ax in enumerate(axes):
    I = np.real(centered_ifft2(Img_fft_rot[:,i].reshape(n,n)))
    ax.imshow(I, cmap='gray') 
    ax.axis('off')

plt.tight_layout()
plt.show()
'''



