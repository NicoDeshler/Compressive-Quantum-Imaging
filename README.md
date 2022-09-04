# Compressive Quantum Imaging
Author: Nicolas Deshler

[The Paper](https://www.overleaf.com/read/pffyxrhkqvfw) can be found here.

This package unifies ideas from compressive sensing and quantum parameter estimation to passively image incoherent distributed scenes at multiple resolution levels extending beyond the diffraction limit. We take inspiration from [[1]](https://iopscience.iop.org/article/10.1088/1367-2630/aa60ee) and use an adaptive bayesian approach to estimate parameters of the scene. Since natural images are generally compressible when transformed to a wavelet basis, or algorithm adaptively estimates wavelet coefficients while enforcing a sparsity prior. The bayesian framework for quantum parameter estimation is detailed in [[2]](https://ieeexplore.ieee.org/document/1054643).

Our algorithm employs Spatial Mode Demultiplexing (SPADE) detailed in [[3]](https://iopscience.iop.org/article/10.1088/1367-2630/aa60ee) to decompose the optical field at the focal plane of the imaging system into an orthogonal modal basis. Our quantum measurement consists of counting the number of photons that appear in each mode.


# Algorithm Features

- Employs photon counting measurements on transverse spatial modes to outperform direct imaging
- Parameters to be estimated are surrogates to the wavelet coefficients of the image related by a linear transformation that preserve the trace 1 norm of the density operator and the non-negativity requirement of the object intensity distribution.
- Uses adaptive Bayesian framework founded on Personick quantum parameter estimation theory to update joint measurement operator
- Sparsity prior imposed on the surrogate coefficients
- Markov-Chain-Monte-Carlo methods used to sample from the posterior distribution

# Installation and Setup
To donwload the repository, open up a Git terminal and enter the following command
```
git clone https://github.com/NicoDeshler/Compressive-Quantum-Imaging.git
```

The algorithm can be shown to work in simulation by running it on a provided target scene. The main script `PersonickWavletEstimation.m` reads a target image from the working directory. The entire image is assumed to be sparse in the 'db1' wavelet basis. Its spatial extent resides within the Rayleigh width of a direct imaging system. Photon counting on SPADE measurements are simulated and the Bayesian adaptive measurement scheme is applied to recover the image.

- $\texttt{imgFile}$ : Filename for the target image (must be a grayscale image in the working directory)
- $\texttt{Npho}$ : Number of photons collected between adaptations
- $\texttt{Nsamples}$ : Number of posterior samples taken

```
matlab [img_out,] PersonickWaveletEstimation(imgFile,Npho,Nsamples)
```

# Test Examples



# References
1) K. K. Lee, S. Guha, and A. Ashok, "Quantum-inspired Optical Super-resolution Adaptive Imaging," In: OSA Imaging and Applied Optics Congress (2021)

2) S Personick. “Application of quantum estimation theory to analog commu-
nication over quantum channels”. In: IEEE Transactions on Information
Theory 17.3 (1971), pp. 240–246.


3) Mankei Tsang. “Subdiffraction incoherent optical imaging via spatial-mode
demultiplexing”. In: New Journal of Physics 19.2 (Feb. 2017), p. 023054.
doi: 10.1088/1367- 2630/aa60ee. url: https://doi.org/10.1088/
1367-2630/aa60ee.

4) Jesus Rubio and Jacob Dunningham. “Bayesian multiparameter quantum
metrology with limited data”. In: Physical Review A 101.3 (Mar. 2020).
doi: 10.1103/physreva.101.032114. url: https://doi.org/10.1103%
2Fphysreva.101.032114.
