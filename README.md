# Compressive Subdiffraction Quantum Imaging

[The Paper](https://www.overleaf.com/read/pffyxrhkqvfw) can be found here.


This package unifies ideas from compressive sensing and quantum parameter estimation to passively image incoherent distributed scenes at subdiffraction resolutions. Taking inspiration from [Reference 1](https://iopscience.iop.org/article/10.1088/1367-2630/aa60ee) our algorithm employs Spatial Mode Demultiplexing (SPADE) to decompose the optical field at the focal plane of the imaging system into the Hermite-Gauss modes. Our quantum measurement consists of counting the number of photons that appear in each mode.


- Employs modal measurements pre-detection to outperform direct imaging
- Parameters to be estimated are surrogates to the wavelet coefficients of the image related by a linear transformation that preserves the trace 1 norm of the density operator describing the image state quantum mechanically.
- Uses adaptive Bayesian framework predicated on Personick quantum parameter estimation theory to update joint measurement operator
- Sparsity prior imposed on the desired parameters
- Markov-Chain-Monte-Carlo methods used to sample from the posterior distribution


# References
1) Mankei Tsang. “Subdiffraction incoherent optical imaging via spatial-mode
demultiplexing”. In: New Journal of Physics 19.2 (Feb. 2017), p. 023054.
doi: 10.1088/1367- 2630/aa60ee. url: https://doi.org/10.1088/
1367-2630/aa60ee.

2) S Personick. “Application of quantum estimation theory to analog commu-
nication over quantum channels”. In: IEEE Transactions on Information
Theory 17.3 (1971), pp. 240–246.

3) K. K. Lee, S. Guha, and A. Ashok, "Quantum-inspired Optical Super-resolution Adaptive Imaging," In: OSA Imaging and Applied Optics Congress (2021)
