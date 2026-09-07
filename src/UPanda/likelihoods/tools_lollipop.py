"""LoLLiPoP binning and offset Hamimeche–Lewis numerical helpers.

Adapted from https://github.com/planck-npipe/lollipop. Bin boundaries are
initialization metadata; spectra and covariance operations support NumPy and
JAX, including differentiation with respect to the spectra.
"""

import numpy as np

from UPanda.module import *
from .tools_hillipop import Bins as _NumpyBins


class Bins(_NumpyBins):
    """Inclusive multipole bins with the historical LoLLiPoP call signatures.

    Boundaries are fixed before JIT compilation. The common implementation
    maintains bin metadata after cuts; numerical products use the active backend.
    """

    def _bin_operators(self, lmins, lmaxs, Dl=False, cov=False):
        bins = _NumpyBins(lmins, lmaxs)
        p, q = bins._bin_operators(Dl=Dl, cov=cov)
        return asarray(p), asarray(q)

    def bin_spectra(self, spectra, lmins, lmaxs):
        """Average C_ell in each inclusive bin (no D_ell weighting)."""
        spectra = asarray(spectra)
        p, _ = self._bin_operators(lmins, lmaxs)
        length = min(spectra.shape[-1], p.shape[1])
        return dot(spectra[..., :length], p.T[:length])

    def bin_covariance(self, clcov, lmins, lmaxs):
        """Propagate a multipole covariance through the averaging operator."""
        p, q = self._bin_operators(lmins, lmaxs, cov=True)
        return p @ asarray(clcov) @ q


def compute_offsets(ell, varcl, clref, fsky=1.0, iter=10):
    """Iteratively estimate the effective noise offsets used by LoLLiPoP."""
    Nl = sqrt(abs(varcl - (2.0 / (2.0 * ell + 1) * clref**2) / fsky))
    for _ in range(iter):
        Nl = sqrt(
            abs(varcl - 2.0 / (2.0 * ell + 1) / fsky * (clref**2 + 2.0 * Nl * clref))
        )
    return Nl * sqrt((2.0 * ell + 1) / 2.0)


def read_dl(datafile):
    """Read ell, EE, BB, EB columns onto a zero-padded integer ell grid.

    Preserve the historical minimum length of 301 while accepting longer files.
    """
    data = np.loadtxt(datafile, ndmin=2)
    ell = data[:, 0].astype(int)
    if np.any(ell < 0) or np.any(ell != data[:, 0]):
        raise ValueError("Multipoles must be nonnegative integers")
    dl = np.zeros((3, max(301, int(ell.max()) + 1)))
    dl[:, ell] = data[:, 1:4].T
    return asarray(dl)


def get_binning(lmin, lmax):
    """Unit bins through ell=35, then complete bins of width ten up to lmax."""
    if lmin < 2 or lmax > 200 or lmax < lmin:
        raise ValueError("Require 2 <= lmin <= lmax <= 200")
    if lmin >= 36:
        lmins = list(range(lmin, lmax - 8, 10))
        lmaxs = [ell + 9 for ell in lmins]
    else:
        lmins = list(range(lmin, min(lmax, 35) + 1))
        lmaxs = lmins.copy()
        if lmax > 35:
            high = list(range(36, lmax - 8, 10))
            lmins += high
            lmaxs += [ell + 9 for ell in high]
    return Bins(lmins, lmaxs), lmins, lmaxs


def _bin_cov_block(clcov, binc, lmins, lmaxs, row, col):
    """Extract one EE/BB/EB covariance block whose input grid starts at ell=2."""
    nell = len(clcov) // 3
    length = binc.lmax - 1
    if len(clcov) % 3 or nell < length:
        raise ValueError("Covariance does not cover the requested polarization bins")
    part = asarray(clcov)[row * nell : row * nell + length,
                         col * nell : col * nell + length]
    return binc.bin_covariance(pad(part, ((2, 0), (2, 0))), lmins, lmaxs)


def bin_covEB(clcov, binc, lmins, lmaxs):
    """Bin all EE, BB, EB covariance blocks, retaining their correlations."""
    return block([[_bin_cov_block(clcov, binc, lmins, lmaxs, i, j)
                   for j in range(3)] for i in range(3)])


def bin_covBB(clcov, binc, lmins, lmaxs):
    """Bin the BB covariance block."""
    return _bin_cov_block(clcov, binc, lmins, lmaxs, 1, 1)


def bin_covEE(clcov, binc, lmins, lmaxs):
    """Bin the EE covariance block."""
    return _bin_cov_block(clcov, binc, lmins, lmaxs, 0, 0)


def vec2mat(vect):
    """Convert [EE, BB, EB] (or [EE, BB]) into a symmetric 2x2 matrix."""
    cross = vect[2] if len(vect) == 3 else 0.0
    return array([[vect[0], cross], [cross, vect[1]]])


def mat2vec(mat):
    """Convert a symmetric polarization matrix into [EE, BB, EB]."""
    return array([mat[0, 0], mat[1, 1], mat[0, 1]])


def ghl(x):
    """Signed Hamimeche–Lewis transform, with roundoff clipped at the minimum."""
    return sign(x - 1) * sqrt(2.0 * clip(x - log(x) - 1, 0, None))
