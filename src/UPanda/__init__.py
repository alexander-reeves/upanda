"""UPanda: Unified cosmological likelihood framework for CMB and LSS data.

UPanda provides a unified framework for computing cosmological likelihoods
from various CMB and large-scale structure (LSS) datasets. It supports
both NumPy and JAX backends for flexible computation, including JIT compilation
and automatic differentiation capabilities.

The package includes:
- Likelihood classes for various CMB and LSS experiments
- Integrated emulator models for fast cosmological predictions
- MCMC sampling utilities with multiple sampler backends
- Provider class for sharing CMB spectra between likelihoods
- Configuration management for JAX/NumPy switching

Example:
    Basic usage:
    ```python
    from UPanda import Provider, config
    from UPanda.likelihoods import desi_bao_dr2

    # Configure JAX backend
    config.set_jax_enabled(True)

    # Initialize likelihood
    likelihood = desi_bao_dr2.desi_bao_dr2(...)
    ```
"""

__author__ = "Alexander Reeves"
__email__ = "areeves@phys.ethz.ch"
__credits__ = "ETH Zurich, Institute for Particle Physics and Astrophysics"

__version__ = "0.2.0"

from . import config  # noqa: F401
from . import utils  # noqa: F401
from .provider import Provider  # noqa: F401
