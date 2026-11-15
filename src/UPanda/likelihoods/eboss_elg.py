"""eBOSS DR16 ELG (z_eff = 0.85) likelihoods from the official SDSS products, differentiable under JAX.

The ELG BAO+RSD result is published only as a non-Gaussian likelihood (no covariance matrix), so a Gaussian
f sigma8 = 0.315 +- 0.095 (Table III of Alam et al. 2021) is only a summary. This module evaluates the official files
(data/eboss_dr16_elg/README.txt):

* ``ELGfs8Table`` -- the 1D f sigma8 likelihood used for the "RSD-only" fits of Alam et al. 2021 (eBOSS CosmoMC dataset
  DR16ELGrsd). Natural cubic spline in ln L through the 100 tabulated points down to ln L = -8; beyond that (and the
  table edges) a quadratic continuation (value and slope matched, curvature -1/sigma^2, sigma the table's standard
  deviation), because the far tail of the table is histogram noise (non-monotone beyond 4 sigma).
* ``ELGGrid`` -- the 3D BAO+full-shape grid likelihood in (D_M/r_d, D_H/r_d, f sigma8), as used by Alam et al. 2021 and
  cobaya's ``bao.sdss_dr16_baoplus_elg``; mode='2d' gives its f sigma8-marginalised (D_M/r_d, D_H/r_d) likelihood for
  distance-only analyses. The grid is an interpolated MCMC histogram with cell-to-cell noise (rms 0.1-0.15 in ln L), so
  L is lightly smoothed (Gaussian, 0.7 cells; widens the 68% half-widths by <= 1.6%), ln L is floored at -60 and
  interpolated with a C1 tricubic (bicubic) Catmull-Rom kernel; outside the grid the point is clamped to the edge and
  a quadratic wall is added. Cobaya's trilinear ln L on the raw grid agrees to rms 0.04 in ln L where ln L > -4.5.

All log-likelihoods are peak-normalised (0 at the maximum), like ``-0.5 chi^2`` of the Gaussian terms they are added to.
Mock mode: ``shift`` translates the likelihood so that its mean moves to the mock value (noiseless mocks keep the real
shape). Works with the NumPy and the JAX backend (UPanda.config).
"""
import os

import numpy as np

import UPanda.config as config

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eboss_dr16_elg")
TABLE_FILE = os.path.join(DATA, "sdss_DR16_ELG_rsd_consensus.dat")
GRID_FILE = os.path.join(DATA, "sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt")
CACHE_FILE = os.path.join(DATA, "eboss_elg_grid_processed.npz")
Z_EFF = 0.85
SMOOTH_CELLS = 0.7
LNL_FLOOR = -60.0


def _xp():
    if config.get_jax_enabled():
        import jax.numpy as jnp
        return jnp
    return np


# --------------------------------------------------------------------------------------------- 1D f sigma8 table
class ELGfs8Table:
    """Official eBOSS DR16 ELG RSD-only f sigma8 likelihood (1D)."""

    TAIL_CUT = -8.0

    def __init__(self, shift=0.0):
        from scipy.interpolate import CubicSpline
        x, L = np.loadtxt(TABLE_FILE, unpack=True)
        assert np.allclose(np.diff(x), x[1] - x[0], rtol=1e-6), "ELG f sigma8 table is not uniformly spaced"
        lnL = np.log(L / L.max())
        w = L / L.sum()
        self.mean = float(np.sum(w * x))
        self.sd = float(np.sqrt(np.sum(w * (x - self.mean) ** 2)))
        cs = CubicSpline(x, lnL, bc_type="natural")
        self.x0, self.dx, self.n = float(x[0]), float(x[1] - x[0]), len(x)
        self.c = np.asarray(cs.c)                                   # (4, n-1), polynomial in (x - x_k)
        # Beyond ln L = TAIL_CUT the table is MCMC-histogram noise (non-monotone at f sigma8 > 0.67, ln L < -9, i.e.
        # > 4 sigma; < 1e-4 in L): the spline is used between the outermost nodes above the cut, the quadratic beyond.
        above = np.where(lnL >= self.TAIL_CUT)[0]
        a, b = x[above[0]], x[above[-1]]
        self.edges = (float(a), float(b))
        self.lo = (float(cs(a)), float(cs(a, 1)))
        self.hi = (float(cs(b)), float(cs(b, 1)))
        assert self.lo[1] > 0 and self.hi[1] < 0, "ELG table: spline slope at the cut has the wrong sign"
        self.curv = -1.0 / self.sd ** 2
        self.shift = float(shift)

    def lnL(self, fs8):
        xp = _xp()
        x = xp.asarray(fs8) - self.shift
        k = xp.clip(xp.floor((x - self.x0) / self.dx).astype(int), 0, self.n - 2)
        d = x - (self.x0 + k * self.dx)
        c = xp.asarray(self.c)
        inside = ((c[0, k] * d + c[1, k]) * d + c[2, k]) * d + c[3, k]
        dlo, dhi = x - self.edges[0], x - self.edges[1]
        below = self.lo[0] + self.lo[1] * dlo + 0.5 * self.curv * dlo ** 2
        above = self.hi[0] + self.hi[1] * dhi + 0.5 * self.curv * dhi ** 2
        return xp.where(x < self.edges[0], below, xp.where(x > self.edges[1], above, inside))


# --------------------------------------------------------------------------------------------- 3D / 2D grid
def build_grid_cache(force=False):
    """Read the official 100^3 grid once and store the processed arrays next to it."""
    if os.path.exists(CACHE_FILE) and not force:
        return dict(np.load(CACHE_FILE))
    from scipy import ndimage
    raw = np.loadtxt(GRID_FILE)
    axes = [np.unique(raw[:, i]) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    assert np.prod(shape) == len(raw), "ELG grid is not a full rectangular grid"
    for a in axes:
        assert np.allclose(np.diff(a), a[1] - a[0], rtol=1e-5), "ELG grid axis is not uniform"
    # column 1 varies slowest: reshape (dm, dh, fs8); check against the stored coordinates
    L = raw[:, 3].reshape(shape)
    assert np.allclose(raw[:, 0].reshape(shape)[:, 0, 0], axes[0]) and np.allclose(raw[:, 2].reshape(shape)[0, 0, :], axes[2])
    L = L / L.max()
    w = L / L.sum()
    mean = np.array([np.sum(w * axes[0][:, None, None]), np.sum(w * axes[1][None, :, None]), np.sum(w * axes[2][None, None, :])])
    Ls = ndimage.gaussian_filter(L, SMOOTH_CELLS, mode="nearest")
    ln3 = np.log(np.clip(Ls / Ls.max(), 1e-300, None))
    L2 = Ls.sum(axis=2)
    ln2 = np.log(np.clip(L2 / L2.max(), 1e-300, None))
    out = dict(dm=axes[0], dh=axes[1], fs8=axes[2], ln3=np.maximum(ln3, LNL_FLOOR), ln2=np.maximum(ln2, LNL_FLOOR),
               mean=mean, smooth_cells=SMOOTH_CELLS, floor=LNL_FLOOR)
    try:
        np.savez(CACHE_FILE, **out)
    except OSError:
        pass
    return out


def _catmull_rom(xp, u):
    u2, u3 = u * u, u * u * u
    return xp.stack([(-u3 + 2 * u2 - u), (3 * u3 - 5 * u2 + 2), (-3 * u3 + 4 * u2 + u), (u3 - u2)], axis=-1) / 2.0


class ELGGrid:
    """Official eBOSS DR16 ELG BAO+full-shape likelihood: mode '3d' in (DM/rd, DH/rd, fs8), '2d' in (DM/rd, DH/rd)."""

    def __init__(self, mode="3d", shift=None):
        if mode not in ("3d", "2d"):
            raise ValueError("mode must be '3d' or '2d'")
        g = build_grid_cache()
        self.mode = mode
        self.mean = np.asarray(g["mean"])                          # (DM/rd, DH/rd, fs8) means of the raw grid
        axes = [g["dm"], g["dh"]] + ([g["fs8"]] if mode == "3d" else [])
        self.grid = np.asarray(g["ln3"] if mode == "3d" else g["ln2"])
        self.lo = np.array([a[0] for a in axes]); self.hi = np.array([a[-1] for a in axes])
        self.step = np.array([a[1] - a[0] for a in axes]); self.nmax = np.array(self.grid.shape) - 3
        self.shift = np.zeros(len(axes)) if shift is None else np.asarray(shift, dtype=float)[: len(axes)]

    def lnL(self, dm_rd, dh_rd, fs8=None):
        """Peak-normalised ln L; inputs broadcast to a common shape (any batch shape)."""
        xp = _xp()
        cols = [dm_rd, dh_rd] + ([fs8] if self.mode == "3d" else [])
        p = xp.stack([xp.asarray(c) for c in xp.broadcast_arrays(*[xp.asarray(c) for c in cols])], axis=-1)   # (..., nd)
        p = p - xp.asarray(self.shift)
        lo, hi, step = xp.asarray(self.lo), xp.asarray(self.hi), xp.asarray(self.step)
        pc = xp.clip(p, lo, hi)
        wall = -0.5 * xp.sum(((p - pc) / (0.02 * (hi - lo))) ** 2, axis=-1)
        t = (pc - lo) / step
        i = xp.clip(xp.floor(t).astype(int), 1, xp.asarray(self.nmax))
        u = t - i
        o = xp.arange(-1, 3)
        G = xp.asarray(self.grid)
        w0, w1 = _catmull_rom(xp, u[..., 0]), _catmull_rom(xp, u[..., 1])
        if self.mode == "3d":
            w2 = _catmull_rom(xp, u[..., 2])
            sub = G[(i[..., 0, None, None, None] + o[:, None, None]), (i[..., 1, None, None, None] + o[None, :, None]),
                    (i[..., 2, None, None, None] + o[None, None, :])]
            val = xp.einsum("...ijk,...i,...j,...k->...", sub, w0, w1, w2)
        else:
            sub = G[(i[..., 0, None, None] + o[:, None]), (i[..., 1, None, None] + o[None, :])]
            val = xp.einsum("...ij,...i,...j->...", sub, w0, w1)
        return val + wall
