"""Likelihood modules for various cosmological datasets.

This package contains likelihood classes for computing cosmological
likelihoods from various CMB and large-scale structure experiments,
including Planck, ACT, DESI BAO, Pantheon+, WMAP, and others.

Each likelihood module provides a class that inherits from the base
Likelihood class and implements experiment-specific data loading,
prediction, and likelihood computation.
"""

from . import likelihood, utils_likelihood  # noqa
from .likelihood import Likelihood  # noqa: F401

__author__ = "Alexander Reeves"
__email__ = "areeves@phys.ethz.ch"
__credits__ = "ETH Zurich, Institute for Particle Physics and Astrophysics"

__version__ = "0.1.0"
