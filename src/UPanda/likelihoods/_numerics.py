"""Shared numerical kernels; experiment-specific data conventions stay in likelihoods."""

import numpy as np
from UPanda.module import asarray, einsum, linalg


def projected_precision(precision, projection):
    """Return the precision of compressed data ``y = B x``.

    The compressed covariance is ``B C B.T``. Projecting ``C^-1`` directly
    is generally wrong, even for bin averages. Solve against the original
    precision to avoid materializing the full covariance.
    """
    projection = asarray(projection)
    covariance = projection @ linalg.solve(asarray(precision), projection.T)
    return linalg.inv(covariance)

def gaussian_loglike(residual, precision):
    """Return ``-0.5 r^T P r`` without a normalization constant.

    The last residual axis indexes data; leading axes index samples. A single
    vector returns a scalar, a batch returns one value per row. No matrix of
    cross-sample products is allocated. Both NumPy and JAX/JIT are supported.
    """
    residual = asarray(residual)
    precision = asarray(precision)
    if residual.ndim < 1 or precision.ndim != 2:
        raise ValueError("Expected a residual vector/batch and a 2D precision matrix")
    if precision.shape != (residual.shape[-1], residual.shape[-1]):
        raise ValueError("Precision matrix shape does not match residual data axis")
    return -0.5 * einsum("...i,ij,...j->...", residual, precision, residual)


def marginal_precision(precision, mask):
    """Precision of retained variables after integrating out discarded ones.

    A sub-block of the precision instead conditions on discarded data. Use
    the Schur complement to obtain the inverse of the retained covariance.
    This is a NumPy initialization helper, outside the JIT evaluation path.
    """
    precision = np.asarray(precision)
    mask = np.asarray(mask, dtype=bool)
    kept = precision[np.ix_(mask, mask)]
    if mask.all():
        return kept
    cross = precision[np.ix_(mask, ~mask)]
    dropped = precision[np.ix_(~mask, ~mask)]
    return kept - cross @ np.linalg.solve(dropped, cross.T)
