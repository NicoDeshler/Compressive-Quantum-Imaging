# Compressive Quantum Imaging

[The Paper](https://www.overleaf.com/read/pffyxrhkqvfw) can be found here.


This package unifies ideas from compressive sensing and quantum parameter estimation to passively image incoherent distributed scenes at multiple resolution levels extending beyond the diffraction limit. We take inspiration from [[1]](https://iopscience.iop.org/article/10.1088/1367-2630/aa60ee) and use an adaptive bayesian approach to estimate parameters of the scene. Since natural images are generally compressible when transformed to a wavelet basis, or algorithm adaptively estimates wavelet coefficients while enforcing a sparsity prior. The bayesian framework for quantum parameter estimation is detailed in [[2]](https://ieeexplore.ieee.org/document/1054643).

Our algorithm employs Spatial Mode Demultiplexing (SPADE) detailed in [[3]](https://iopscience.iop.org/article/10.1088/1367-2630/aa60ee) to decompose the optical field at the focal plane of the imaging system into the Hermite-Gauss modes. Our quantum measurement consists of counting the number of photons that appear in each mode.


# Algorithm Features

- Employs pre-detection spatial mode sorting to outperform direct imaging
- Parameters to be estimated are surrogates to the wavelet coefficients of the image related by a linear transformation that preserves the trace 1 norm of the density operator describing the image state quantum mechanically.
- Uses adaptive Bayesian framework founded on Personick quantum parameter estimation theory to update joint measurement operator
- Sparsity prior imposed on the surrogate coefficients
- Markov-Chain-Monte-Carlo methods used to sample from the posterior distribution

# Theory

We describe the quantum mechanical state of a normalized intensity distribution $f(\mathbf{R})$ at the object plane with coordinates $\mathbf{R} = [X,Y]$ as a single-photon density operator (mixed state),

$$\hat{\rho} = \int f(\mathbf{R}) \ket{\psi(\mathbf{R})}\bra{\psi(\mathbf{R})}d^{2}\mathbf{R}$$


where

$$\ket{\psi(\mathbf{R})} = \int_ \psi(\mathbf{r}-\mathbf{R}) \ket{\mathbf{r}} d^{2}\mathbf{r}$$

and $\ket{\mathbf{r}} = \hat{a}^{\dagger} \delta(\mathbf{r}-\mathbf{r}')\ket{0}$ is the single-photon excitation state at the position $\mathbf{r}$ on the image plane (assuming 0 magnification).  

In the Bayesian framework, we assume that the class of objects that we are imaging come from some distribution. We express our initial beliefs about the object $f(\mathbf{R})$ through a prior distribution. Natural scenes have been shown to be compressible in the wavelet domain. In general, the scene intensty distribution can be represented as a linear combination of wavelet functions $\{\Upsilon_i(\mathbf{R})\}$

$$f(\mathbf{R}) = \sum_{i}^{N} \theta_i \Upsilon_i(\mathbf{R})$$

In this work we leverage compressive sensing by imposing sparsity priors on the wavelet coefficients $\{\theta_i\}$. The purpose of this algorithm is to estimate these coefficients by performing jointly-optimal quantum modal measurements on the optical field at the image plane.


Applying the normalization constraint $\int f(\mathbf{R}) d^2 \mathbf{R} = 1$ and non-negativity constraint $f(\mathbf{R}) \geq 0$ on the object intensity distribution, we can transform the wavelet estimation problem into the estimation of a elements from a simplex $\mathbb{S}^N$. First, vectorize the object intensity distribution $f(\mathbf{R}) \rightarrow \mathbf{f}$. With this we can write the relation.

$$\mathbf{f} = \mathbf{W} \mathbf{M} \mathbf{V} \mathbf{x}$$

where,

- $\mathbf{W}$ is the inverse wavelet transform matrix 
- $\mathbf{M}$ is a quasi-diagonal matrix that enforces the normalization constraint
- $\mathbf{V}$ is the polytope vertex matrix that enforces the non-negativity constraint




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
