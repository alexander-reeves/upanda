"""Planck 2018 lensing likelihood module.

This module provides the likelihood class for Planck 2018 CMB lensing
reconstruction. The likelihood computes the chi-squared between observed
and predicted lensing potential power spectrum C_L^kk.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class planck_2018_lensing(Likelihood):
    """Likelihood class for Planck 2018 CMB lensing.

    This class computes the likelihood for Planck 2018 lensing potential
    power spectrum measurements, comparing observed C_L^kk with theoretical
    predictions from cosmological models.

    Attributes:
        emu_model_list (list): List of emulator models for predictions.
        cl_hat (ndarray): Observed lensing power spectrum.
        bin_mat_lens (ndarray): Binning matrix for lensing spectrum.
        covariance (ndarray): Covariance matrix for lensing data.
        theory_offset (ndarray): Theory offset correction.
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
    ):
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        self.emu_model_list = emu_model_list

        # Check for mock data flag (for testing)
        if additional_args.get("mock_data", None) is not None:
            self.cl_hat = additional_args["mock_data"]
            n_bins = len(self.cl_hat)
            self.bin_mat_lens = additional_args.get(
                "mock_binning_matrix", np.eye(n_bins)
            )
            self.covariance = additional_args.get("mock_cov", np.eye(n_bins))
            self.theory_offset = additional_args.get(
                "mock_theory_offset", np.zeros(n_bins)
            )
        else:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_directory, "data/planck_2018_lensing")

            self.bin_mat_lens = np.loadtxt(
                self.data_dir + "/binning_plus_correction_matrix.txt"
            ).reshape(9, -1)
            self.cl_hat = np.loadtxt(self.data_dir + "/cl_hat.txt")
            self.covariance = np.loadtxt(self.data_dir + "/covariance.txt").reshape(
                9, -1
            )
            self.theory_offset = np.loadtxt(self.data_dir + "/theory_offset.txt")

        self.inv_c_lens = linalg.inv(self.covariance)
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []

        self.additional_args = additional_args if additional_args is not None else {}

    def get_prediction(self, p):
        ell_min = 2
        ell_max = 2500

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        kk_emu_model = self.emu_model_list[0]

        # DO YOU NEED AN EXP HERE!?
        if self.additional_args.get("integrated_model") is True:
            if self.additional_args.get("lens_exp") is True:
                Cl_kk = exp(kk_emu_model.predict(params_dict)) * 2 / pi
            else:
                Cl_kk = kk_emu_model.predict(params_dict) * 2 / pi

        else:
            if self.additional_args.get("lens_exp") is True:
                Cl_kk = exp(kk_emu_model.predictions_np(params_dict)) * 2 / pi
            else:
                Cl_kk = kk_emu_model.predictions_np(params_dict) * 2 / pi

        cl_binned = Cl_kk[
            :, ell_min - 2 : ell_max - 2 + 1
        ]  # note that ellmin=2 for this emulator
        cl_binned = dot(self.bin_mat_lens, cl_binned.T).T
        cl_theory_cmbkk = (
            cl_binned - self.theory_offset
        )  # remove the lensing correction following lensing paper- make sure this needs to be removed not added

        return cl_theory_cmbkk

    def compute_like(self, params):
        cl_theory_cmbkk = self.get_prediction(params)

        diff = self.cl_hat - cl_theory_cmbkk

        chi2 = matmul(self.inv_c_lens, transpose(diff))
        chi2 = matmul(diff, chi2)

        chi2 = diag(chi2)

        loglkl = -0.5 * chi2

        return loglkl
