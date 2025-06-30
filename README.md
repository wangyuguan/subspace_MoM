# Subspace method of moments for cryo-EM reconstruction

This repository contains the complementary software for the paper:  
**"Subspace Method of Moments for Ab Initio 3-D Single-Particle Cryo-EM Reconstruction"**  
[arXiv:2410.06889](https://arxiv.org/abs/2410.06889)

The reproducible code for the results in the paper is provided in the `examples/` folder.

**Paper authors:** Jeremy Hoskins, Yuehaw Khoo, Oscar Mickelin, Amit Singer, and Yuguan Wang  
**Version:** 2.0  
**Release Date:** June 30, 2025  
**Language:** Python 3.13.2


## Third-Party Packages, Code, and Data

This package makes use of the following third-party software:

1. [ASPIRE](https://github.com/PrincetonUniversity/aspire) — Python package for a comprehensive cryo-EM image analysis and reconstruction.
2. [BOTalign](https://github.com/RuiyiYang/BOTalign) — Python package for alignment of 3-D density maps.
3. [Pymanopt](https://pymanopt.org/) — Python package for optimization on manifolds.
4. [FINUFFT](https://finufft.readthedocs.io/en/latest/) — Fast nonuniform FFT library.

This package makes use of the following third-party data:

1. [Electron Microscopy Data Bank (EMDB)](https://www.ebi.ac.uk/emdb/) - A public repository of 3D EM maps.
2. [Quadrature Rules on Manifolds](https://www-user.tu-chemnitz.de/~potts/workgroup/graef/quadrature/index.php.en) - Higher order quandrature on the unit sphere and 3-D rotation group. 
