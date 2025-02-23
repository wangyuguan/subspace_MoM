import sys
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))

import mrcfile 
from aspire.volume import Volume
from aspire.utils import Rotation
from utils_BO import align_BO
import numpy as np 


with mrcfile.open('cleanimag_test/vol_ds.mrc') as mrc:
    vol_ds = mrc.data
    
with mrcfile.open('cleanimag_test/vol_est_m2.mrc') as mrc:
    vol2 = mrc.data
    
    
with mrcfile.open('cleanimag_test/vol_est_m3.mrc') as mrc:
    vol3 = mrc.data
    
    
Vol_ds = Volume(vol_ds)
Vol2 = Volume(vol2)
Vol3 = Volume(vol3)

para =['wemd',64,300,True] 
[R02,R2]=align_BO(Vol_ds,Vol2,para,reflect=True) 
[R03,R3]=align_BO(Vol_ds,Vol3,para,reflect=True) 

vol_ds = Vol_ds.asnumpy()
vol2_aligned = Vol2.rotate(Rotation(R2)).asnumpy()
vol3_aligned = Vol3.rotate(Rotation(R3)).asnumpy()

with mrcfile.new('cleanimag_test/vol_ds.mrc', overwrite=True) as mrc:
    mrc.set_data(vol_ds[0])  # Set the volume data
    mrc.voxel_size = 1.0  

with mrcfile.new('cleanimag_test/vol_est_m2_aligned.mrc', overwrite=True) as mrc:
    mrc.set_data(vol2_aligned[0])  # Set the volume data
    mrc.voxel_size = 1.0  

with mrcfile.new('cleanimag_test/vol_est_m3_aligned.mrc', overwrite=True) as mrc:
    mrc.set_data(vol3_aligned[0])  # Set the volume data
    mrc.voxel_size = 1.0  