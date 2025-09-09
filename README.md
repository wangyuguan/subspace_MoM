# Subspace method of moments for cryo-EM reconstruction

This repository contains the research code for the paper:  
**"Subspace Method of Moments for Ab Initio 3-D Single-Particle Cryo-EM Reconstruction"**  
[arXiv:2410.06889](https://arxiv.org/abs/2410.06889)

The reproducible code for the results in the paper is provided in the `examples/` folder.

**Paper authors:** Jeremy Hoskins, Yuehaw Khoo, Oscar Mickelin, Amit Singer and Yuguan Wang  

**Version:** 2.0  (The first version was in [MATLAB](https://github.com/wangyuguan/subspace_MoM_matlab]))

**Release Date:** June 30, 2025  

**Language:** Python 3.13.2


## Third-Party Packages, Code, and Data

This package makes use of the following third-party software packages:

1. [ASPIRE](https://github.com/PrincetonUniversity/aspire) — Python package for a comprehensive cryo-EM image analysis and reconstruction.
2. [BOTalign](https://github.com/RuiyiYang/BOTalign) — Python package for alignment of 3-D density maps.
3. [Pymanopt](https://pymanopt.org/) — Python package for optimization on manifolds.
4. [FINUFFT](https://finufft.readthedocs.io/en/latest/) — Fast nonuniform FFT library.

This package makes use of the following third-party data:

1. [Electron Microscopy Data Bank (EMDB)](https://www.ebi.ac.uk/emdb/) - A public repository of 3-D EM maps.
2. [Quadrature Rules on Manifolds](https://www-user.tu-chemnitz.de/~potts/workgroup/graef/quadrature/index.php.en) - A repository containing high order quadrature rules on the unit sphere and 3-D rotation group.

## Usage

To generate the particle stacks in Section 4.2.2 of the paper:

1. Open the file `run_noisy_images_example.py`.
2. Uncomment the line:
   ```python
   generate_particles(vol_path, snr, batch_size, defocus_ct)
3. Run the script. This will:
   - Create a `.star` file that can be used as input for RELION.
   - Generate a folder named `particles/` containing the individual particle images.

Make sure to configure the parameters such as vol_path, snr, and defocus_ct according to your dataset and desired settings.

The particle stack can be downloaded from https://zenodo.org/records/15815808.
