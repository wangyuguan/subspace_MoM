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

with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data


# savemat('vol.mat',{'vol':vol})
np.random.seed(1)
    
nrot = 5 
alpha = np.random.uniform(0,2*np.pi,nrot)
beta = np.random.uniform(0,np.pi,nrot)
gamma = np.random.uniform(0,2*np.pi,nrot)
rots = np.zeros((nrot,3,3),dtype=np.float32)
for i in range(nrot):
    rot = Rz(alpha[i]) @ Ry(beta[i]) @ Rz(gamma[i])
    rots[i,:,:] = rot
# savemat('Rots.mat',{'Rots':rots})
    
images = vol_proj(vol, rots)
fig, axes = plt.subplots(1, nrot, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray') 
    ax.axis('off')

plt.tight_layout()
plt.show()


ds_res = 64
images_ds = image_downsample(images, ds_res)
fig, axes = plt.subplots(1, nrot, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(images_ds[i], cmap='gray') 
    ax.axis('off')

plt.tight_layout()
plt.show()

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



