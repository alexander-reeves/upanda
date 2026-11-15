"""BOSS/eBOSS DR16 RSD + BAO joint likelihood module.

Implements the Gaussian approximation to the joint BAO+full-shape (RSD)
likelihood from Alam et al. 2021 (arXiv:2007.08991).  Four galaxy/quasar
samples are supported, matching the files distributed at
https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-plus/

Samples and their data vectors
-------------------------------
mgs  : SDSS DR7 MGS, z_eff=0.15
       [f*sigma8, DV/rs]
boss_dr12 : BOSS DR12 LRG, z_eff=0.38 and 0.51
       [DM/rs, DH/rs, f*sigma8]  x2 bins  (6-dim)
eboss_lrg : eBOSS DR16 LRG (CMASS+eBOSS), z_eff=0.698
       [DM/rs, DH/rs, f*sigma8]  (3-dim)
eboss_qso : eBOSS DR16 QSO, z_eff=1.48
       [DM/rs, DH/rs, f*sigma8]  (3-dim)

The emulator (emu_model_list[0]) must return the theoretical prediction for
the chosen sample's full data vector in the same ordering as above.

Usage
-----
The ``samples`` constructor argument selects which galaxy samples to include.
Pass any subset of ``{'mgs', 'boss_dr12', 'eboss_lrg', 'eboss_qso'}`` (the
default), optionally plus ``'eboss_elg'``: the eBOSS DR16 ELG at z_eff = 0.85,
[DM/rs, DH/rs, f*sigma8], evaluated with the official non-Gaussian 3D grid
likelihood (eboss_elg.ELGGrid; ordered last). The individual blocks are
combined as independent terms; no cross-sample covariance is applied (they are
independent surveys/redshifts), as in Alam et al. 2021. ``drop_fsigma8()``
marginalises f*sigma8 out of every sample for distance-only analyses.

References
----------
Alam et al. 2021, Phys. Rev. D 103, 083533  (arXiv:2007.08991)
    — combined eBOSS DR16 cosmological results
Ross et al. 2015, MNRAS 449, 835            (arXiv:1409.3242)
    — MGS measurements
Alam et al. 2017, MNRAS 470, 2617           (arXiv:1607.03155)
    — BOSS DR12 measurements
Bautista et al. 2021, MNRAS 500, 736        (arXiv:2007.08993)
    — eBOSS DR16 LRG measurements
Neveux et al. 2020, MNRAS 499, 210          (arXiv:2007.08999)
    — eBOSS DR16 QSO measurements
"""

import logging
import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

logger = logging.getLogger(__name__)

from .likelihood import Likelihood

# Canonical ordering of quantities within each sample block.
# The emulator must produce predictions in this same order.
_SAMPLE_META = {
    "mgs": {
        "data_file": "sdss_MGS_FSBAO_DVfs8.txt",
        "cov_file": "sdss_MGS_FSBAO_DVfs8_covtot.txt",
        "z_eff": [0.15],
        # quantities: f_sigma8, DV_over_rs
        "quantities": ["f_sigma8", "DV_over_rs"],
        "n_dim": 2,
    },
    "boss_dr12": {
        "data_file": "sdss_DR12_LRG_FSBAO_DMDHfs8.txt",
        "cov_file": "sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt",
        "z_eff": [0.38, 0.51],
        # quantities per bin: DM_over_rs, DH_over_rs, f_sigma8
        "quantities": ["DM_over_rs", "DH_over_rs", "f_sigma8",
                       "DM_over_rs", "DH_over_rs", "f_sigma8"],
        "n_dim": 6,
    },
    "eboss_lrg": {
        "data_file": "sdss_DR16_LRG_FSBAO_DMDHfs8.txt",
        "cov_file": "sdss_DR16_LRG_FSBAO_DMDHfs8_covtot.txt",
        "z_eff": [0.698],
        "quantities": ["DM_over_rs", "DH_over_rs", "f_sigma8"],
        "n_dim": 3,
    },
    "eboss_qso": {
        "data_file": "sdss_DR16_QSO_FSBAO_DMDHfs8.txt",
        "cov_file": "sdss_DR16_QSO_FSBAO_DMDHfs8_covtot.txt",
        "z_eff": [1.48],
        "quantities": ["DM_over_rs", "DH_over_rs", "f_sigma8"],
        "n_dim": 3,
    },
}

_ALL_SAMPLES = ("mgs", "boss_dr12", "eboss_lrg", "eboss_qso")   # default selection (Gaussian blocks)

# Optional: eBOSS DR16 ELG, z_eff = 0.85, distributed only as a non-Gaussian likelihood on a 3D grid in
# (DM/rs, DH/rs, f*sigma8) (eboss_elg.ELGGrid). Opt-in via samples=(..., 'eboss_elg'); always ordered last.
_SAMPLE_META["eboss_elg"] = {"z_eff": [0.85], "quantities": ["DM_over_rs", "DH_over_rs", "f_sigma8"], "n_dim": 3,
                             "grid": True}
_ORDER = _ALL_SAMPLES + ("eboss_elg",)


class boss_dr16_rsd(Likelihood):
    """Joint BOSS/eBOSS DR16 BAO+RSD (fsigma8) likelihood.

    Implements the Gaussian approximation to the combined BAO and full-shape
    (growth-rate) likelihood from Alam et al. 2021.  Each galaxy sample
    contributes an independent block to the total chi-squared.

    The emulator (``emu_model_list[0]``) must return the concatenated
    theoretical prediction vector in the same quantity ordering as the data
    files — see module docstring for per-sample ordering.

    Attributes:
        samples (tuple): Galaxy samples included in this instance.
        data_vector (ndarray): Concatenated observed data vector.
        inv_cov (ndarray): Block-diagonal inverse covariance matrix.
        z_eff (list): Effective redshifts in data-vector order.
        quantities (list): Quantity labels in data-vector order.
    """

    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        dtype=None,
        compression_vecs=None,
        cosmology_params_fixed=None,
        additional_args=None,
        external_spectra=None,
        samples=None,
    ):
        """Initialise the likelihood.

        Parameters
        ----------
        emu_model_list : list
            Single-element list containing the emulator that predicts the
            concatenated data vector for all selected samples.
        samples : iterable of str, optional
            Which galaxy samples to include.  Any subset of
            ``{'mgs', 'boss_dr12', 'eboss_lrg', 'eboss_qso'}``.
            Defaults to all four samples.
        dtype : numpy dtype or None
            Parameter dtype used by ``utils_likelihood.prepare_params_dict``.
        cosmology_params_fixed : list, optional
            Names of parameters held fixed during sampling.
        additional_args : dict, optional
            May contain ``'mock_data'`` (array) and ``'mock_cov'`` (array)
            to override the bundled data files.
        external_spectra : dict or None
            If provided, ``external_spectra['boss_dr16_rsd_data']`` is used
            as the predicted data vector instead of calling the emulator.
        """
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        self.emu_model_list = emu_model_list
        self.external_spectra = external_spectra
        self.dtype = dtype
        self.cosmology_params_fixed = (
            cosmology_params_fixed if cosmology_params_fixed is not None else []
        )
        self.additional_args = additional_args if additional_args is not None else {}

        # Resolve which samples to load
        if samples is None:
            samples = _ALL_SAMPLES
        samples = tuple(samples)
        if not samples:
            raise ValueError("Select at least one BOSS/eBOSS sample")
        unknown = set(samples) - set(_ORDER)
        if unknown:
            raise ValueError(
                f"Unknown sample(s): {unknown}. "
                f"Valid choices are: {_ORDER}"
            )
        self.samples = tuple(s for s in _ORDER if s in set(samples))
        self.has_elg = "eboss_elg" in self.samples
        # data-vector bookkeeping (both modes): z_eff, quantities, which entries are Gaussian / the ELG grid
        self.z_eff, self.quantities = [], []
        for name in self.samples:
            meta = _SAMPLE_META[name]
            per_bin = len(meta["quantities"]) // len(meta["z_eff"])
            for z in meta["z_eff"]:
                self.z_eff.extend([z] * per_bin)
            self.quantities.extend(meta["quantities"])
        n_gauss = len(self.quantities) - (3 if self.has_elg else 0)

        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "boss_dr16_rsd",
        )

        mock_data = self.additional_args.get("mock_data", None)

        if mock_data is not None:
            # --- mock mode: bypass file loading ---
            self.data_vector = array(mock_data)
            mock_cov = self.additional_args.get("mock_cov", None)
            if mock_cov is not None:
                cov = array(mock_cov)
            else:
                n = len(self.data_vector)
                cov = array(np.eye(n))
            self.inv_cov = array(np.linalg.inv(np.asarray(cov)))
            self._full_cov = np.asarray(cov, dtype=np.float64)
        else:
            # --- normal mode: load from bundled data files ---
            data_blocks = []
            cov_blocks = []

            for name in self.samples:
                meta = _SAMPLE_META[name]
                if meta.get("grid"):
                    # ELG: placeholder entries (grid means; diagonal Table III-like variances) keep the vector and
                    # inv_cov the same size as the prediction; compute_like uses the grid, never this block.
                    from .eboss_elg import ELGGrid
                    data_blocks.append(np.asarray(ELGGrid("3d").mean))
                    cov_blocks.append(np.diag([1.0 ** 2, 2.2 ** 2, 0.095 ** 2]))
                    continue
                dv = np.loadtxt(
                    os.path.join(data_dir, meta["data_file"]),
                    usecols=(1,),
                    comments="#",
                )
                cov_block = np.loadtxt(
                    os.path.join(data_dir, meta["cov_file"]),
                    comments="#",
                )
                if cov_block.ndim == 1:
                    # 2x2 stored as flat 4-element row or already shaped (2,2)
                    side = int(np.sqrt(len(cov_block)))
                    cov_block = cov_block.reshape(side, side)

                # The MGS data file lists f_sigma8 first, DV/rs second —
                # match the ordering in _SAMPLE_META['mgs']['quantities'].
                data_blocks.append(dv)
                cov_blocks.append(cov_block)

            # Concatenate into a single data vector and block-diagonal covariance
            self.data_vector = array(np.concatenate(data_blocks))
            full_cov = self._block_diagonal(cov_blocks)
            self.inv_cov = array(np.linalg.inv(full_cov))
            self._full_cov = full_cov

        # Gaussian part vs ELG grid (ELG entries are the last three: DM/rs, DH/rs, f*sigma8)
        self.elg_mode = "3d" if self.has_elg else None
        if self.has_elg:
            from .eboss_elg import ELGGrid
            self.gauss_index = np.arange(n_gauss)
            self.elg_index = np.arange(n_gauss, n_gauss + 3)
            self.inv_cov_gauss = array(np.linalg.inv(self._full_cov[np.ix_(self.gauss_index, self.gauss_index)]))
            g3 = ELGGrid("3d")
            self._elg_shift = (np.zeros(3) if mock_data is None
                               else np.asarray(mock_data, dtype=np.float64)[self.elg_index] - g3.mean)
            self.elg_grid = ELGGrid("3d", shift=self._elg_shift)

        logger.debug(
            "boss_dr16_rsd: loaded %d data points from samples %s",
            len(np.asarray(self.data_vector)),
            self.samples,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_diagonal(blocks):
        """Stack a list of 2-D arrays into a block-diagonal matrix."""
        sizes = [b.shape[0] for b in blocks]
        # int(np.sum(...)), not the builtin sum: `from UPanda.module import *`
        # rebinds `sum` to jnp.sum under set_jax_enabled(True), which rejects
        # a Python list of ints.
        n = int(np.sum(sizes))
        out = np.zeros((n, n))
        offset = 0
        for b in blocks:
            s = b.shape[0]
            out[offset : offset + s, offset : offset + s] = b
            offset += s
        return out

    # ------------------------------------------------------------------
    # Likelihood interface
    # ------------------------------------------------------------------

    def set_external_spectra(self, spectra):
        """Provide a pre-computed prediction vector instead of calling the emulator.

        Parameters
        ----------
        spectra : dict
            Must contain key ``'boss_dr16_rsd_data'`` mapping to the
            predicted data vector (array of same length as ``self.data_vector``).
        """
        logger.debug("boss_dr16_rsd: using external prediction vector")
        self.external_spectra = spectra

    def get_prediction(self, p):
        """Return the theoretical prediction for the concatenated data vector.

        Parameters
        ----------
        p : array-like, shape (n_samples, n_params) or structured array
            Cosmological parameter samples.

        Returns
        -------
        array, shape (n_samples, n_data)
        """
        if self.external_spectra is not None:
            return self.external_spectra["boss_dr16_rsd_data"]

        self.params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )
        return self.emu_model_list[0].predict(self.params_dict)

    def drop_fsigma8(self):
        """Keep only the distance observables (DM/rs, DH/rs, DV/rs), marginalising f*sigma8 out.

        Gaussian blocks: the sub-block of the COVARIANCE (exact marginalisation, carries the f*sigma8-distance
        correlation into the distance errors). ELG: the f*sigma8-marginalised 2D grid likelihood.
        Returns the indices of the kept entries in the original vector (to slice the theory prediction)."""
        from .eboss_elg import ELGGrid
        q = list(self.quantities)
        keep = np.array([i for i, x in enumerate(q) if x != "f_sigma8"], dtype=int)
        cov = self._full_cov[np.ix_(keep, keep)]
        self.data_vector = array(np.asarray(self.data_vector)[keep])
        self.inv_cov = array(np.linalg.inv(cov))
        self._full_cov = cov
        self.quantities = [q[i] for i in keep]
        self.z_eff = [self.z_eff[i] for i in keep]
        if self.has_elg:
            n_gauss = len(keep) - 2
            self.gauss_index = np.arange(n_gauss)
            self.elg_index = np.arange(n_gauss, n_gauss + 2)
            self.inv_cov_gauss = array(np.linalg.inv(cov[np.ix_(self.gauss_index, self.gauss_index)]))
            self.elg_mode = "2d"
            self.elg_grid = ELGGrid("2d", shift=self._elg_shift[:2])
        return keep

    def compute_like(self, params):
        """Compute the log-likelihood for the BOSS/eBOSS DR16 RSD+BAO data.

        Parameters
        ----------
        params : array-like, shape (n_samples, n_params)

        Returns
        -------
        ndarray, shape (n_samples,)
            Log-likelihood values  ``-0.5 * chi2``.
        """
        pred = self.get_prediction(params)
        diff = self.data_vector - pred
        if not self.has_elg:      # unchanged legacy path (callers may slice data_vector / inv_cov themselves)
            return utils_likelihood.gaussian_loglike(diff, self.inv_cov)
        g, e = self.gauss_index, self.elg_index
        ll = utils_likelihood.gaussian_loglike(diff[..., g], self.inv_cov_gauss)
        if self.elg_mode == "3d":
            return ll + self.elg_grid.lnL(pred[..., e[0]], pred[..., e[1]], pred[..., e[2]])
        return ll + self.elg_grid.lnL(pred[..., e[0]], pred[..., e[1]])
