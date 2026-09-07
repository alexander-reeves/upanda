"""Planck 2018 CMB likelihood for ACT analysis.

This module provides the likelihood class for Planck 2018 CMB measurements
optimized for use in combination with ACT data. The likelihood computes
the chi-squared between observed and predicted CMB power spectra.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood

"""
As described in https://arxiv.org/abs/2109.04451 the ACT team uses the TT-ony spectra from planck up to a maximum multipole of ell=650. They use the
full plik likelihood whereas here we make an approximation getting as close to this ell-cut as possible with the broader plik_lite likelihood bins (ends up being
up to ellmax=645).

"""


class planck_2018_for_act(Likelihood):
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
        additional_args = additional_args if additional_args is not None else {}

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/planck_2018_highell")

        blmin_file = self.data_dir + "/blmin.dat"
        blmax_file = self.data_dir + "/blmax.dat"
        binw_file = self.data_dir + "/bweight.dat"
        self.blmin = np.loadtxt(blmin_file).astype(int)
        self.blmax = np.loadtxt(blmax_file).astype(int)
        self.binw = np.loadtxt(binw_file)
        data_dir_lowl = os.path.join(current_directory, "data/planck_2018_lowl")

        # EE
        data_dir_lowl = os.path.join(current_directory, "data/planck_2018_lowl")

        (
            self.lmin_list_EE,
            self.lmax_list_EE,
            self.mu_LN_EE,
            self.sig_LN_EE,
            self.loc_LN_EE,
        ) = np.loadtxt(
            data_dir_lowl + "/lognormal_fit_" + "3" + "bins_EE.txt", unpack=True
        )

        # TT
        (
            self.lmin_list_TT,
            self.lmax_list_TT,
            self.mu_LN_TT,
            self.sig_LN_TT,
        ) = np.loadtxt(
            data_dir_lowl + "/lognormal_fit_" + "2" + "bins_TT.txt", unpack=True
        )

        self.compression_vecs = compression_vecs
        self.emu_model_list = emu_model_list
        self.dtype = dtype

        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []

        self.additional_args = additional_args if additional_args is not None else {}

        # read covariance and dv from file
        bval, self.data_vector, X_sig = np.genfromtxt(
            self.data_dir + "/cl_cmb_plik_v22.dat", unpack=True
        )

        if additional_args.get("mock_data") is not None:
            self.data_vector = additional_args["mock_data"]

        self.covariance_matrix = utils_likelihood.load_covmat_fortran(
            self.data_dir + "/c_matrix_plik_v22.dat", 613
        )  # 215 bins of data

        self.data_vector = self.data_vector[0:215]

        self.apply_planck_temperature_rescaling = additional_args.get(
            "apply_planck_temperature_rescaling", False
        )

    def get_prediction(self, p):
        units_factor = 1e12  # units factor to convert to microKelvin
        # defining binning set-up
        nbintt = 215  # 30-2508   #used when getting covariance matrix
        plmin = 30
        plmax = 2508
        ellmin = 2

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        tt_emu_model, te_emu_model, ee_emu_model = self.emu_model_list

        if self.additional_args.get("integrated_model") is True:
            self.Cltt = tt_emu_model.predict(params_dict) * units_factor
            self.Clee = ee_emu_model.predict(params_dict) * units_factor

        else:
            self.Cltt = exp(tt_emu_model.predictions_np(params_dict)) * units_factor
            self.Clee = exp(ee_emu_model.predictions_np(params_dict)) * units_factor

        begintt = self.blmin + plmin - ellmin
        endtt = self.blmax + plmin + 1 - ellmin

        beginwtt = self.blmin
        endwtt = self.blmax + 1

        indicestt = []
        windowtt = []
        indices_reptt = []

        for i in range(nbintt):
            idxreptt = repeat(i, len(arange(begintt[i], endtt[i])))
            indicestt.append(arange(begintt[i], endtt[i]))
            windowtt.append(self.binw[beginwtt[i] : endwtt[i]])
            indices_reptt.append(idxreptt)

        flat_list = [item for sublist in indices_reptt for item in sublist]
        indices_reptt = array(flat_list)
        flat_list = [item for sublist in indicestt for item in sublist]
        indicestt = array(flat_list)
        flat_list = [item for sublist in windowtt for item in sublist]
        windowtt = array(flat_list)

        indices = indicestt

        Cl = self.Cltt[:, 0 : plmax - 1]

        indices_rep = indices_reptt

        if is_jax:
            if self.apply_planck_temperature_rescaling:
                Cl_bin = array(
                    [
                        bincount(
                            indices_rep,
                            weights=Cl[index, indices] * windowtt,
                            length=613,
                        )  # hardcode length for JIT compilation
                        for index in range(Cl.shape[0])
                    ]
                ) * ((1 + params_dict["m_t_planck"][:, None]) ** 2)
            else:
                Cl_bin = array(
                    [
                        bincount(
                            indices_rep,
                            weights=Cl[index, indices] * windowtt,
                            length=613,
                        )  # hardcode length for JIT compilation
                        for index in range(Cl.shape[0])
                    ]
                ) / (array(params_dict["A_planck"])[:, None] ** 2)

        else:
            if self.apply_planck_temperature_rescaling:
                Cl_bin = (
                    array(
                        [
                            bincount(indices_rep, weights=Cl[index, indices] * windowtt)
                            for index in range(Cl.shape[0])
                        ]
                    )
                    * (1 + array(params_dict["m_t_planck"])[:, None]) ** 2
                )

            else:
                Cl_bin = array(
                    [
                        bincount(indices_rep, weights=Cl[index, indices] * windowtt)
                        for index in range(Cl.shape[0])
                    ]
                ) / (array(params_dict["A_planck"])[:, None] ** 2)

        return Cl_bin

    def compute_like(self, params):
        predictions = self.get_prediction(params)
        y_obs = self.data_vector

        # make the lmax cut at 645
        ellmax = 645

        idx = 77  # hardcoded for jit compilation

        predictions = predictions[:, 0 : idx + 1]
        y_obs = y_obs[0 : idx + 1]
        self.covariance_matrix = self.covariance_matrix[0 : idx + 1, 0 : idx + 1]

        # Assuming the covariance matrix is already inverted. If not, we'll need to invert it.
        inv_C = linalg.inv(self.covariance_matrix)

        diff = y_obs - predictions
        chi2 = matmul(inv_C, transpose(diff))
        chi2 = matmul(diff, chi2)
        chi2 = diag(chi2)

        like = -0.5 * chi2

        # Add the lognormal Lowl part of the likelihood if included in aditional_args

        # always add the lowell TT likelihood here

        Cls_tt_2_29 = self.Cltt[:, 0:28]
        Cls_ee_2_29 = self.Clee[:, 0:28]

        lognorm_tt = utils_likelihood.planck_lowT_binned_loglike(
            Cls_tt_2_29,
            self.mu_LN_TT,
            self.sig_LN_TT,
            self.lmin_list_TT,
            self.lmax_list_TT,
        )

        like += lognorm_tt

        if self.additional_args.get("add_lowell_ee") is True:
            print("Adding planck lowell EE")
            lognorm_ee = utils_likelihood.planck_lowE_binned_loglike(
                Cls_ee_2_29,
                self.mu_LN_EE,
                self.sig_LN_EE,
                self.loc_LN_EE,
                self.lmin_list_EE,
                self.lmax_list_EE,
            )

            like += lognorm_ee

        return like
