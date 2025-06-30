import sys
import os 
import shutil
from pathlib import Path


# Add the 'src' directory to the Python path
src_path = Path('../src').resolve()
sys.path.append(str(src_path))
# Add the 'BOTalign' directory to the Python path
src_path = Path('../../BOTalign').resolve()
sys.path.append(str(src_path))

from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.source.simulation import Simulation
from aspire.operators import RadialCTFFilter
from aspire.noise import WhiteNoiseAdder
from aspire.numeric import fft
import mrcfile
import numpy as np 
import numpy.linalg as LA 
from viewing_direction import *
from utils import *
from volume import *
from moments import *
from utils_BO import align_BO
from tqdm import trange 
import matplotlib.pyplot as plt 



def run_subspace_MoM(vol_path, snr, batch_size, defocus_ct, out_file_name=None):
    np.random.seed(42)
    # with mrcfile.open('../data/emd_34948.map') as mrc:
    out = {}
    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    
    # preprocess the volume 
    vol = vol/tns_norm(vol)
    img_size = vol.shape[0]
    ds_res = 64
    vol_ds = vol_downsample(vol, ds_res)
    ell_max_vol = 10
    vol_coef, k_max, r0, indices_vol = sphFB_transform(vol_ds, ell_max_vol)
    vol_ds = coef_t_vol(vol_coef, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_ds = vol_ds.reshape(ds_res,ds_res,ds_res,order='F')
    with mrcfile.new('vol_ds.mrc', overwrite=True) as mrc:
        mrc.set_data(vol_ds)  # Set the volume data
        mrc.voxel_size = 1.0  
    # vol = np.array(vol_upsample(vol_ds, img_size),dtype=np.float32)
    # with mrcfile.new('vol.mrc', overwrite=True) as mrc:
    #     mrc.set_data(vol)  # Set the volume data
    #     mrc.voxel_size = 1.0  
    vol = np.array(vol,dtype=np.float32)
    # expanded volume 
    out['vol_ds'] = vol_ds
    out['vol_coef'] = vol_coef

    # set up viewing direction distribution 
    c = 10
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    kappa = 6
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    ell_max_half_view = 3
    sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)

    # set up ctfs 
    pixel_size = 1.04  # Pixel size of the images (in angstroms)
    voltage = 300  # Voltage (in KV)
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    defocus_min = 1e4  # Minimum defocus value (in angstroms)
    defocus_max = 3e4 
    num_img = batch_size*defocus_ct
    h_ctfs = [
        RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]

    num_img = batch_size*defocus_ct
    rotmats = np.zeros((num_img,3,3),dtype=np.float32)


    # sample rotation matrix 
    print('sampling viewing directions')
    sphere_samples = sample_sph_coef(num_img, sph_coef, ell_max_half_view)
    print('done')
    inplane_angles = np.random.uniform(0,2*np.pi,num_img)
    for i in trange(num_img):
        _, beta, alph = cart2sph(sphere_samples[i,0], sphere_samples[i,1], sphere_samples[i,2])
        rotmats[i,:,:] = Rz(alph) @ Ry(beta) @ Rz(inplane_angles[i])

    # viewing distribution 
    out['sph_coef'] = sph_coef
    out['sphere_samples'] = sphere_samples
    out['inplane_angles'] = inplane_angles

    # compute variance of white noise 
    images = vol_proj(vol, rotmats[:defocus_ct])
    signal_norm2 = 0
    noise_norm2 = 0 
    print('compute variance')
    for i in trange(defocus_ct):
        H = h_ctfs[i].evaluate_grid(img_size)
        signal_norm2 = signal_norm2+tns_norm(H*centered_fft2(images[i]))**2 
        noise_norm2 = noise_norm2+tns_norm(centered_fft2(np.random.normal(0,1,(img_size,img_size))))**2
    var = signal_norm2 / noise_norm2 / snr 
    print(var)

    # get compressed moments 
    opts = {}
    opts['nI2'] = 400
    opts['nI3'] = 400
    opts['nJ'] = 120
    opts['r2_max'] = 200
    opts['r3_max'] = 120
    opts['tol2'] = 1e-8
    opts['tol3'] = 1e-6
    opts['ds_res'] = ds_res
    opts['var'] = var
    opts['batch_size'] = batch_size
    opts['defocus_ct'] = defocus_ct

    moms_out = ctf_image_subspace_moments_CUR(vol, rotmats, h_ctfs, opts)
    m1_emp = moms_out['m1']
    m2_emp = moms_out['m2']
    m3_emp = moms_out['m3'] 
    U2 = moms_out['U2']
    U3 = moms_out['U3']
    # moments and subspaces
    out['m1_emp'] = m1_emp
    out['m2_emp'] = m2_emp
    out['m3_emp'] = m3_emp
    out['U2'] = U2
    out['U3'] = U3
    out['relerr2'] = moms_out['relerr2'] 
    out['relerr3'] = moms_out['relerr3'] 

    if out_file_name:
        np.savez(out_file_name, **out)


    # run reconstruction 
    res = sequential_moment_matching(
        m1_emp,m2_emp,m3_emp,
        U2,U3,ds_res,ell_max_vol,ell_max_half_view,
        15,15)
    out['x1'] = res['x1']
    out['x2'] = res['x2']
    out['x3'] = res['x3']
    
    if out_file_name:
        np.savez(out_file_name, **out)

    vol_coef_est_m1 = res['vol_coef_est_m1']
    vol_coef_est_m2 = res['vol_coef_est_m2']
    vol_coef_est_m3 = res['vol_coef_est_m3']
    vol_est_m2 = coef_t_vol(vol_coef_est_m2, ell_max_vol, ds_res, k_max, r0, indices_vol)
    vol_est_m3 = coef_t_vol(vol_coef_est_m3, ell_max_vol, ds_res, k_max, r0, indices_vol) 
    vol_est_m2 = vol_est_m2.reshape([ds_res,ds_res,ds_res])
    vol_est_m3 = vol_est_m3.reshape([ds_res,ds_res,ds_res])
    # reconstructed volumes
    out['a_est'] = res['a_est']
    out['b_est'] = res['b_est']
    out['vol_coef_est_m1'] = vol_coef_est_m1
    out['vol_coef_est_m2'] = vol_coef_est_m2
    out['vol_coef_est_m3'] = vol_coef_est_m3

    if out_file_name:
        np.savez(out_file_name, **out)
    

    # align volume
    Vol_ds = Volume(vol_ds)
    Vol2 = Volume(vol_est_m2)
    Vol3 = Volume(vol_est_m3)

    para =['wemd',64,300,True] 
    [R02,R2]=align_BO(Vol_ds,Vol2,para,reflect=True) 
    [R03,R3]=align_BO(Vol_ds,Vol3,para,reflect=True) 


    vol_ds = Vol_ds.asnumpy()
    vol_ds = vol_ds[0]
    vol2_aligned = Vol2.rotate(Rotation(R2)).asnumpy()
    vol2_aligned = vol2_aligned[0]
    vol3_aligned = Vol3.rotate(Rotation(R3)).asnumpy()
    vol3_aligned = vol3_aligned[0]
    with mrcfile.new('vol_est_m3_aligned.mrc', overwrite=True) as mrc:
        mrc.set_data(vol3_aligned)  # Set the volume data
        mrc.voxel_size = 1.0  
    resolution = get_fsc(vol_ds, vol3_aligned, pixel_size*img_size/ds_res)
    # alignment
    out['R2'] = R2
    out['R3'] = R3
    out['vol2_aligned'] = vol2_aligned
    out['vol3_aligned'] = vol3_aligned
    
    out['resolution'] = resolution


    # timings
    out['t_form'] = moms_out['t_form']
    out['t_precomp'] = res['t_precomp']
    out['t_m1'] = res['t_m1']
    out['t_m2'] = res['t_m2']
    out['t_m3'] = res['t_m3']


    if out_file_name:
        np.savez(out_file_name, **out)

    return out 


def generate_particles(vol_path, snr, batch_size, defocus_ct):
    np.random.seed(42)
    
    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    vol = vol/tns_norm(vol)
    # preprocess the volume 
    img_size = vol.shape[0]
    # set up viewing direction distribution 
    c = 10
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    kappa = 6
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    ell_max_half_view = 3
    sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)

    # set up ctfs 
    pixel_size = 1.04  # Pixel size of the images (in angstroms)
    voltage = 300  # Voltage (in KV)
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    defocus_min = 1e4  # Minimum defocus value (in angstroms)
    defocus_max = 3e4 
    num_img = batch_size*defocus_ct
    h_ctfs = [
        RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]
    h_idx = []
    for i in range(defocus_ct):
        h_idx += [i]*batch_size

    num_img = batch_size*defocus_ct
    rotmats = np.zeros((num_img,3,3),dtype=np.float32)


    # sample rotation matrix 
    print('sampling viewing directions')
    sphere_samples = sample_sph_coef(num_img, sph_coef, ell_max_half_view)
    print('done')
    # alphs = np.random.uniform(0,2*np.pi,num_img)
    # betas = np.arccos(np.random.uniform(-1,1,num_img))
    inplane_angles = np.random.uniform(0,2*np.pi,num_img)
    print('computing rotation matrices')
    for i in trange(num_img):
        _, beta, alph = cart2sph(sphere_samples[i,0], sphere_samples[i,1], sphere_samples[i,2])
        rotmats[i,:,:] = Rz(alph) @ Ry(beta) @ Rz(inplane_angles[i])
        # rotmats[i,:,:] = Rz(alphs[i]) @ Ry(betas[i]) @ Rz(inplane_angles[i])

    # compute variance of white noise 
    images = vol_proj(vol, rotmats[:defocus_ct])
    signal_norm2 = 0
    noise_norm2 = 0 
    print('compute variance')
    for i in trange(defocus_ct):
        H = h_ctfs[i].evaluate_grid(img_size)
        signal_norm2 = signal_norm2+tns_norm(H*centered_fft2(images[i]))**2 
        noise_norm2 = noise_norm2+tns_norm(centered_fft2(np.random.normal(0,1,(img_size,img_size))))**2
    var = signal_norm2 / noise_norm2 / snr 
    print(var)


    # save the information of ctfs
    DefocusU = []
    DefocusV = []
    DefocusAngle = []
    Voltage = [] 
    SphericalAberration = []
    AmplitudeContrast = []
    particle_filenames = [] 

    # for background normalization 
    bg_radius = img_size / 2
    y, x = np.indices((img_size,img_size))
    center_y, center_x = img_size / 2, img_size / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    bg_mask = distance >= bg_radius


    # create particles 
    par_dir = "particles"
    if os.path.exists(par_dir) and os.path.isdir(par_dir):
        shutil.rmtree(par_dir)
    os.makedirs(par_dir)


    img_idx = 0 
    print('generating images')
    for i in trange(defocus_ct):
        _rotmats = rotmats[(i*batch_size):(i+1)*batch_size,:,:]   
        # ctf operator  
        H = h_ctfs[i].evaluate_grid(img_size)
        # get downsampled projections
        projs = vol_proj(vol, _rotmats)
        imgs = fft.centered_ifft2(fft.centered_fft2(projs) * H)
        imgs = imgs.real
        
        # white noise 
        np.random.seed(i)
        noise = np.sqrt(var)*np.random.normal(0,1,
                      (batch_size, img_size, img_size))
        noise = np.array(noise, dtype=np.float32)
        imgs = imgs+noise
        
        for img in imgs:
            # background normalization
            bg_pixels = img[bg_mask]
            bg_mean = np.mean(bg_pixels)
            bg_std = np.std(bg_pixels)
            img = (img - bg_mean) / bg_std

	          # save an example image 
            if img_idx==0:
                plt.imsave('example_noisy_projection.pdf', img, cmap='gray')
	
            # extract the ctf information 
            DefocusU.append(h_ctfs[i].defocus_u)
            DefocusV.append(h_ctfs[i].defocus_v)
            DefocusAngle.append(h_ctfs[i].defocus_ang)
            Voltage.append(h_ctfs[i].voltage)
            SphericalAberration.append(h_ctfs[i].Cs)
            AmplitudeContrast.append(h_ctfs[i].alpha)
            # save particle 
            img_idx = img_idx+1 
            filename = f"particles/particle_{img_idx:04d}.mrc"
            particle_filenames.append(filename)
            with mrcfile.new(filename, overwrite=True) as mrc:
                mrc.set_data(img.astype(np.float32))
    
    # write star file 
    current_dir = os.getcwd()
    star_file = 'particles.star'
    if os.path.exists(star_file) and os.path.isfile(star_file):
        os.remove(star_file)

    with open(star_file, 'w') as file:
        # Write the optics block using integer optics groups
        file.write("data_optics\n\n")
        file.write("loop_\n")
        file.write("_rlnOpticsGroup            #1\n")
        file.write("_rlnVoltage                #2\n")
        file.write("_rlnSphericalAberration    #3\n")
        file.write("_rlnAmplitudeContrast      #4\n")
        file.write("_rlnImagePixelSize         #5\n")
        file.write("_rlnImageSize              #6\n")
        file.write("_rlnImageDimensionality    #7\n")
        file.write(f"1                          {voltage}    {Cs}   {alpha}   {pixel_size}   {img_size}   2\n\n")
      
        # Write the particles block with extra columns for ab initio modeling
        file.write("data_particles\n\n")
        file.write("loop_\n")
        file.write("_rlnImageName      #1\n")
        file.write("_rlnDefocusU       #2\n")
        file.write("_rlnDefocusV       #3\n")
        file.write("_rlnDefocusAngle   #4\n")
        file.write("_rlnOpticsGroup    #5\n")
        file.write("_rlnAngleRot       #6\n")
        file.write("_rlnAngleTilt      #7\n")
        file.write("_rlnAnglePsi       #8\n")
        file.write("_rlnOriginX        #9\n")
        file.write("_rlnOriginY        #10\n")
        
      
        for j in range(len(DefocusU)):
            particle_file = os.path.join(current_dir, par_dir, f"particle_{j+1:04d}.mrc")
            line = (
                f"{particle_file}  {DefocusU[j]}  {DefocusV[j]}  {DefocusAngle[j]}  "
                f"1 0 0 0 0 0"
            )
            file.write(" ".join(line.split()) + "\n")


def generate_example_images(vol_path, snr, batch_size, defocus_ct, fname):
    np.random.seed(42)
    
    with mrcfile.open(vol_path) as mrc:
        vol = mrc.data
    vol = vol/tns_norm(vol)
    # preprocess the volume 
    img_size = vol.shape[0]
    # set up viewing direction distribution 
    c = 10
    centers = np.random.normal(0,1,size=(c,3))
    centers /= LA.norm(centers, axis=1, keepdims=True)
    w_vmf = np.random.uniform(0,1,c)
    w_vmf = w_vmf/np.sum(w_vmf)
    kappa = 6
    def my_fun(th,ph):
        grid = Grid_3d(type='spherical', ths=np.array([th]),phs=np.array([ph]))
        return np.pi*vMF_density(centers,w_vmf,kappa,grid)[0]
    ell_max_half_view = 3
    sph_coef, indices_view = sph_harm_transform(my_fun, ell_max_half_view)

    # set up ctfs 
    pixel_size = 1.04  # Pixel size of the images (in angstroms)
    voltage = 300  # Voltage (in KV)
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    defocus_min = 1e4  # Minimum defocus value (in angstroms)
    defocus_max = 3e4 
    num_img = batch_size*defocus_ct
    h_ctfs = [
        RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]
    h_idx = []
    for i in range(defocus_ct):
        h_idx += [i]*batch_size

    num_img = batch_size*defocus_ct
    rotmats = np.zeros((num_img,3,3),dtype=np.float32)


    # sample rotation matrix 
    print('sampling viewing directions')
    sphere_samples = sample_sph_coef(num_img, sph_coef, ell_max_half_view)
    print('done')
    # alphs = np.random.uniform(0,2*np.pi,num_img)
    # betas = np.arccos(np.random.uniform(-1,1,num_img))
    inplane_angles = np.random.uniform(0,2*np.pi,num_img)
    print('computing rotation matrices')
    for i in trange(num_img):
        _, beta, alph = cart2sph(sphere_samples[i,0], sphere_samples[i,1], sphere_samples[i,2])
        rotmats[i,:,:] = Rz(alph) @ Ry(beta) @ Rz(inplane_angles[i])
        # rotmats[i,:,:] = Rz(alphs[i]) @ Ry(betas[i]) @ Rz(inplane_angles[i])

    # compute variance of white noise 
    images = vol_proj(vol, rotmats[:defocus_ct])
    signal_norm2 = 0
    noise_norm2 = 0 
    print('compute variance')
    for i in trange(defocus_ct):
        H = h_ctfs[i].evaluate_grid(img_size)
        signal_norm2 = signal_norm2+tns_norm(H*centered_fft2(images[i]))**2 
        noise_norm2 = noise_norm2+tns_norm(centered_fft2(np.random.normal(0,1,(img_size,img_size))))**2
    var = signal_norm2 / noise_norm2 / snr 
    print(var)


    # save the information of ctfs
    DefocusU = []
    DefocusV = []
    DefocusAngle = []
    Voltage = [] 
    SphericalAberration = []
    AmplitudeContrast = []
    particle_filenames = [] 

    # for background normalization 
    bg_radius = img_size / 2
    y, x = np.indices((img_size,img_size))
    center_y, center_x = img_size / 2, img_size / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    bg_mask = distance >= bg_radius



    img_idx = 0 
    print('generating images')
    for i in trange(defocus_ct):
        _rotmats = rotmats[(i*batch_size):(i+1)*batch_size,:,:]   
        # ctf operator  
        H = h_ctfs[i].evaluate_grid(img_size)
        # get downsampled projections
        projs = vol_proj(vol, _rotmats)
        imgs = fft.centered_ifft2(fft.centered_fft2(projs) * H)
        imgs = imgs.real
        plt.imsave('ctf_image_clean.pdf', imgs[0], cmap='gray')
        
        # white noise 
        np.random.seed(i)
        noise = np.sqrt(var)*np.random.normal(0,1,
                      (batch_size, img_size, img_size))
        noise = np.array(noise, dtype=np.float32)
        imgs = imgs+noise
        
        for img in imgs:
            # background normalization
            bg_pixels = img[bg_mask]
            bg_mean = np.mean(bg_pixels)
            bg_std = np.std(bg_pixels)
            img = (img - bg_mean) / bg_std

	          # save an example image 
            if img_idx==0:
                plt.imsave(fname, img, cmap='gray')
                return 
	

if __name__ == "__main__":
    
    vol_path = '../data/emd_34948.map'
    
    batch_size = 1000
    defocus_ct = 100

    snr = 1/4
    # generate_example_images(vol_path, snr, batch_size, defocus_ct, 'ctf_image_snr_1_4.pdf')
    # generate_particles(vol_path, snr, batch_size, defocus_ct)
    run_subspace_MoM(vol_path, snr, batch_size, defocus_ct, 'ctf_snr_1_4.npz')


    snr = 1/16
    # generate_example_images(vol_path, snr, batch_size, defocus_ct, 'ctf_image_snr_1_16.pdf')
    # generate_particles(vol_path, snr, batch_size, defocus_ct)
    run_subspace_MoM(vol_path, snr, batch_size, defocus_ct, 'ctf_snr_1_16.npz')

    
    snr = 1/64
    # generate_example_images(vol_path, snr, batch_size, defocus_ct, 'ctf_image_snr_1_64.pdf')
    # generate_particles(vol_path, snr, batch_size, defocus_ct)
    run_subspace_MoM(vol_path, snr, batch_size, defocus_ct, 'ctf_snr_1_64.npz')

   
          


      

      

