"""DESI BAO DR2 likelihood module.

This module provides the likelihood class for DESI (Dark Energy Spectroscopic
Instrument) Baryon Acoustic Oscillation (BAO) Data Release 2. The likelihood
computes the chi-squared between observed and predicted BAO measurements.
"""

import logging
import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

logger = logging.getLogger(__name__)

from .likelihood import Likelihood


class desi_bao_dr2(Likelihood):
    """Likelihood class for DESI BAO Data Release 2.

    This class computes the likelihood for DESI BAO DR2 measurements,
    comparing observed BAO scale measurements with theoretical predictions
    from emulator models.

    Attributes:
        emu_model_list (list): List of emulator models for predictions.
        external_spectra: External spectra provider (optional).
        data_dir (str): Directory containing DESI BAO DR2 data files.
        desi_bao_data (ndarray): Observed BAO data vector.
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
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        additional_args = additional_args if additional_args is not None else {}
        self.emu_model_list = emu_model_list
        self.external_spectra = external_spectra

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/desi_bao_dr2")

        if additional_args.get("mock_data", None) is not None:
            self.desi_bao_data = array(additional_args["mock_data"])
            mock_cov = additional_args.get("mock_cov", None)
            if mock_cov is not None:
                self.desi_bao_covariance = array(mock_cov)
            else:
                # Create identity covariance if not provided
                n = len(self.desi_bao_data)
                self.desi_bao_covariance = array(np.eye(n))
        else:
            self.desi_bao_data = array(np.loadtxt(self.data_dir + "/mean.txt", usecols=(1)))
            self.desi_bao_covariance = array(np.loadtxt(self.data_dir + "/cov.txt"))
        self.inv_c_bao = np.linalg.inv(self.desi_bao_covariance)
        self.inv_c_bao = array(self.inv_c_bao)

        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []

        self.additional_args = additional_args if additional_args is not None else {}

    def set_external_spectra(self, spectra):
        """Set external BAO data instead of loading from bundled files.

        Args:
            spectra (dict): Dictionary with a single key ``'desi_bao_data'``
                mapping to the external data vector.
        """
        logger.debug("Using external BAO data vector (dict key: 'desi_bao_data')")
        self.external_spectra = spectra

    def get_prediction(self, p):
        if self.external_spectra is not None:
            return self.external_spectra["desi_bao_data"]
        else:
            # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
            self.params_dict = utils_likelihood.prepare_params_dict(
                p, self.dtype, self.cosmology_params_fixed, self.additional_args
            )

            desi_bao_emu_model = self.emu_model_list[0]

            desi_bao_dv = desi_bao_emu_model.predict(self.params_dict)

            return desi_bao_dv

    def compute_like(self, params):
        desi_bao_dv_predict = self.get_prediction(params)

        diff = self.desi_bao_data - desi_bao_dv_predict

        return utils_likelihood.gaussian_loglike(diff, self.inv_c_bao)
