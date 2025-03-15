import sys
from pathlib import Path
import os 


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))


from aspire.volume import Volume 
from aspire.image import Image
from aspire.utils.rotation import Rotation
from aspire.operators import CTFFilter, RadialCTFFilter
from aspire.ctf import estimate_ctf
from utils import * 
from viewing_direction import * 
from moments import *
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mrcfile 
from scipy.io import savemat 
from utils_BO import align_BO
from tempfile import TemporaryDirectory

np.random.seed(42)


# %% load data 
with mrcfile.open('../data/emd_34948.map') as mrc:
    vol = mrc.data
    
    
IMG_SIZE = vol.shape[0]



# %% generate CTF 


radial_ctf_filter = RadialCTFFilter(
    pixel_size=1,  # angstrom
    voltage=200,  # kV
    defocus=10000,  # angstrom, 10000 A = 1 um
    Cs=2.26,  # Spherical aberration constant
    alpha=0.07,  # Amplitude contrast phase in radians
    B=0,  # Envelope decay in inverse square angstrom (default 0)
)

rctf_fn = radial_ctf_filter.evaluate_grid(IMG_SIZE)
plt.imshow(rctf_fn)
plt.colorbar()
plt.show()



# %% generate rotations 

num_img = 1
rots = np.zeros((num_img,3,3))
for i in range(num_img):
    alpha = np.random.uniform(0,2*np.pi)
    beta = np.arccos(np.random.uniform(-1,1))
    gamma = np.random.uniform(0,2*np.pi)
    rots[i,:,:] = Rz(alpha) @ Ry(beta) @ Rz(gamma)



# %% plot the test image before CTF damage  

images = np.real(vol_proj(vol, rots, 0, 1))

plt.imshow(images[0])
plt.axis('off')  # Hide the axes if desired
plt.show()

# %% plot the test image after CTF damage  


images_ctf = Image(images).filter(radial_ctf_filter)
plt.imshow(images_ctf.asnumpy()[0])
plt.axis('off')  # Hide the axes if desired
plt.show()

# %% estimate CTF 

with TemporaryDirectory() as d:
    images_ctf.save(os.path.join(d, "test_img.mrc"))
    radial_ctf_est = estimate_ctf(
        data_folder=d,
        pixel_size=radial_ctf_filter.pixel_size,
        cs=radial_ctf_filter.Cs,
        amplitude_contrast=radial_ctf_filter.alpha,
        voltage=radial_ctf_filter.voltage,
        psd_size=IMG_SIZE,
        num_tapers=1,
        dtype=np.float64,
    )
    


est = radial_ctf_est["test_img.mrc"]

# Take an average defocus for radial case.
defocus = (est["defocus_u"] + est["defocus_v"]) / 2.0
print(f"defocus = {defocus}")

# Create a filter and evaluate.
est_ctf = RadialCTFFilter(
    pixel_size=est["pixel_size"],
    voltage=est["voltage"],
    defocus=defocus,  # Modeled CTF was 10000
    Cs=est["cs"],
    alpha=est["amplitude_contrast"],
    B=0,
)



# %% plot the estimated CTF function 

est_ctf_fn = est_ctf.evaluate_grid(IMG_SIZE)
plt.imshow(est_ctf_fn)
plt.colorbar()
plt.show()


# %% correct CTF and plot  





