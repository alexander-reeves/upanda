"""Pantheon+ supernova likelihood module.

This module provides the likelihood class for Pantheon+ supernova dataset,
including optional SH0ES (SH0ES) calibration. The likelihood computes
the chi-squared between observed and predicted supernova magnitudes.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class pantheon_plus(Likelihood):
    """Likelihood class for Pantheon+ supernova dataset.

    This class computes the likelihood for Pantheon+ supernova measurements,
    comparing observed supernova magnitudes with theoretical predictions from
    cosmological models. Supports optional SH0ES calibration.

    Attributes:
        emu_model_list (list): List of emulator models for predictions.
        external_spectra: External spectra provider (optional).
        use_shoes (bool): Whether to use SH0ES calibration.
        mag (ndarray): Observed supernova magnitudes.
        zcmb (ndarray): CMB-frame redshifts.
        zhel (ndarray): Heliocentric redshifts.
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
        use_shoes=False,
        external_spectra=None,
    ):
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        self.emu_model_list = emu_model_list
        self.external_spectra = external_spectra

        additional_args = additional_args or {}

        # Check for mock data flag (for testing)
        if additional_args.get("mock_data", None) is not None:
            self.mag = additional_args["mock_data"]
            self.zcmb = additional_args.get(
                "mock_zcmb", np.linspace(0.01, 2.0, len(self.mag))
            )
            self.zhel = additional_args.get("mock_zhel", self.zcmb)  # Approximation
            self.cov = additional_args.get("mock_cov", np.eye(len(self.mag)))
            self.nsn = len(self.mag)
        else:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_directory, "data/pantheon_plus")

            # Read data file
            data_file = os.path.join(self.data_dir, "Pantheon+SH0ES.dat")
            self._read_data(data_file)

            # Read covariance matrix
            cov_file = os.path.join(self.data_dir, "Pantheon+SH0ES_STAT+SYS.cov")
            self.cov = np.loadtxt(cov_file)

        if self.cov.ndim == 1:
            # Determine the size of the square matrix
            n = int(np.sqrt(len(self.cov)))

            # Check that the length matches a perfect square
            if n * n != len(self.cov):
                print("not a perfect square-take the [1:] array as in cobaya")
                self.cov = self.cov[1:]
                n = int(np.sqrt(len(self.cov)))

            # Reshape to n x n matrix
            self.cov = self.cov.reshape(n, n)
        self.inv_cov = np.linalg.inv(self.cov)

        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []
        self.additional_args = additional_args if additional_args is not None else {}

        # Apply standard Pantheon+ mask (no SH0ES) - skip if using mock data
        if additional_args.get("mock_data", None) is None:
            mask = self.zcmb > 0.01
            self._apply_mask(mask)

        # Always marginalize over M (DESI-like analysis)
        self._marginalize_abs_mag()

    # From Cobaya's pantheonplus.py
    def _read_data(self, data_file):

        with open(data_file, "r") as f:
            header = f.readline().strip()
            columns = header.split()

        m_b_idx = columns.index("m_b_corr")
        zhd_idx = columns.index("zHD")
        zhel_idx = columns.index("zHEL")

        self.cols = ["mag", "zcmb", "zhel"]
        data = np.loadtxt(
            data_file,
            skiprows=1,  # Skip header row
            usecols=(
                m_b_idx,
                zhd_idx,
                zhel_idx,
            ),  # Load only m_b_corr, zhd, zhel columns
        )
        self.mag = data[:, 0]  # mags
        self.zcmb = data[:, 1]  # zhd
        self.zhel = data[:, 2]  # zhel

    def _apply_mask(self, mask):
        """Apply a mask to the data and covariance"""
        self.mag = self.mag[mask]
        self.zcmb = self.zcmb[mask]
        self.zhel = self.zhel[mask]
        self.cov = self.cov[np.ix_(mask, mask)]
        self.inv_cov = np.linalg.inv(self.cov)
        self.nsn = len(self.mag)

    def _marginalize_abs_mag(self):
        """Marginalize over absolute magnitude M"""
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.inv_cov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.inv_cov = self.inv_cov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def set_external_spectra(self, spectra):
        """Set external distance modulus predictions (bypassing emulator)"""
        print(
            "using external data for pantheon_plus, this should be the distance modulus array"
        )
        self.external_spectra = spectra

    def get_prediction(self, p):
        """Get model prediction using the emulator or external data"""
        if self.external_spectra is not None:
            return self.external_spectra
        else:
            # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
            params_dict = utils_likelihood.prepare_params_dict(
                p, self.dtype, self.cosmology_params_fixed, self.additional_args
            )

            # Get angular diameter distances from emulator
            pantheon_plus_model = self.emu_model_list[0]
            distances = pantheon_plus_model.predict(params_dict)

            # Convert to magnitude prediction: 5 * log10(D_L)
            mag_prediction = 5 * log10((1 + self.zhel) * (1 + self.zcmb) * distances)

            return mag_prediction

    def compute_like(self, params):
        """Compute the Pantheon+ likelihood"""
        # Get model prediction
        mag_prediction = self.get_prediction(params)

        diff = self.mag - mag_prediction

        # Compute chi2 using marginalized inverse covariance

        return utils_likelihood.gaussian_loglike(diff, self.inv_cov)
