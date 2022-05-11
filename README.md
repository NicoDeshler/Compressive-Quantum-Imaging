# Compressive Subdiffraction Quantum Imaging

This package unifies ideas from compressive sensing and quantum parameter estimation to passively image incoherent distributed scenes at subdiffraction resolutions.


- Employs modal measurements pre-detection to outperform direct imaging
- Parameters to be estimated are surrogates to the wavelet coefficients of the image related by a linear transformation that preserves the trace 1 norm of the density operator describing the image state quantum mechanically.
- Uses adaptive Bayesian framework predicated on Personick quantum parameter estimation theory to update joint measurement operator
- Sparsity prior imposed on the desired parameters
- Markov-Chain-Monte-Carlo methods used to sample from the posterior distribution
- 