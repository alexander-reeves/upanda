"""BOSS/eBOSS DR16 fsigma8-only likelihood.

Implements a diagonal Gaussian likelihood for the six consensus fsigma8
measurements from Table III of Alam et al. 2021 (arXiv:2007.08991,
Phys. Rev. D 103, 083533).

Data
----
Six independent measurements from the combined BAO+RSD analysis:
  MGS        z=0.15 : fsigma8 = 0.530 ± 0.160
  BOSS DR12  z=0.38 : fsigma8 = 0.497 ± 0.045
  BOSS DR12  z=0.51 : fsigma8 = 0.459 ± 0.038
  eBOSS LRG  z=0.70 : fsigma8 = 0.473 ± 0.041
  eBOSS ELG  z=0.85 : fsigma8 = 0.315 ± 0.095
  eBOSS QSO  z=1.48 : fsigma8 = 0.462 ± 0.045

Likelihood
----------
Diagonal Gaussian: each measurement is treated as independent.
The inverse covariance is diagonal: diag(1/sigma_i^2).

References
----------
Alam et al. 2021, Phys. Rev. D 103, 083533  (arXiv:2007.08991)
"""

import logging
import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

logger = logging.getLogger(__name__)

from .likelihood import Likelihood


class boss_dr16_fsigma8(Likelihood):
    """Diagonal Gaussian fsigma8 likelihood from BOSS/eBOSS DR16.

    Uses the six consensus BAO+RSD fsigma8 values from Table III of
    Alam et al. 2021.  All cross-sample covariances are set to zero
    (independent surveys).

    The emulator (``emu_model_list[0]``) must return a 6-element vector
    [fsigma8(z) for z in [0.15, 0.38, 0.51, 0.70, 0.85, 1.48]].

    Attributes
    ----------
    z_eff : ndarray, shape (6,)
        Effective redshifts.
    data_vector : ndarray, shape (6,)
        Observed fsigma8 values.
    inv_cov : ndarray, shape (6, 6)
        Diagonal inverse covariance (1/sigma_i^2 on diagonal).
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
    ):
        """Initialise the likelihood.

        Parameters
        ----------
        emu_model_list : list, optional
            Single-element list with the emulator predicting the 6-element
            fsigma8 vector.  Not needed when ``external_spectra`` is used.
        additional_args : dict, optional
            May contain ``'mock_data'`` (6-element array) to override the
            bundled Table III data vector.
        external_spectra : dict, optional
            If provided, ``external_spectra['boss_dr16_fsigma8_data']`` is
            used as the predicted data vector instead of calling the emulator.
        """
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        self.emu_model_list = emu_model_list
        self.external_spectra = external_spectra
        self.dtype = dtype
        self.cosmology_params_fixed = (
            cosmology_params_fixed if cosmology_params_fixed is not None else []
        )
        self.additional_args = additional_args if additional_args is not None else {}

        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "boss_dr16_fsigma8", "fsigma8_alam2021_tableIII.txt",
        )

        # Load Table III values
        table = np.loadtxt(data_file, comments="#")   # shape (6, 3)
        self.z_eff = table[:, 0]                      # [0.15, 0.38, 0.51, 0.70, 0.85, 1.48]
        sigma = table[:, 2]

        mock_data = self.additional_args.get("mock_data", None)
        if mock_data is not None:
            self.data_vector = array(np.asarray(mock_data))
        else:
            self.data_vector = array(table[:, 1])

        self.inv_cov = array(np.diag(1.0 / sigma**2))   # 6×6 diagonal

        logger.debug(
            "boss_dr16_fsigma8: loaded %d data points at z_eff=%s",
            len(np.asarray(self.data_vector)),
            self.z_eff.tolist(),
        )

    def set_external_spectra(self, spectra):
        """Provide a pre-computed prediction vector.

        Parameters
        ----------
        spectra : dict
            Must contain ``'boss_dr16_fsigma8_data'`` mapping to the
            predicted fsigma8 vector, shape ``(batch, 6)`` or ``(6,)``.
        """
        self.external_spectra = spectra

    def get_prediction(self, p):
        """Return the theoretical fsigma8 prediction.

        Parameters
        ----------
        p : array-like or None
            Cosmological parameters.  Ignored when ``external_spectra`` is set.

        Returns
        -------
        array, shape (batch, 6) or (6,)
        """
        if self.external_spectra is not None:
            return self.external_spectra["boss_dr16_fsigma8_data"]

        self.params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )
        return self.emu_model_list[0].predict(self.params_dict)

    def compute_like(self, params):
        """Compute the log-likelihood.

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

        return utils_likelihood.gaussian_loglike(diff, self.inv_cov)
