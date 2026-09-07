"""Planck 2018 high-ℓ and low-ℓ CMB likelihood module.

This module provides the likelihood class for Planck 2018 high-ℓ (TT, TE, EE)
and low-ℓ (TT, EE) CMB temperature and polarization measurements. The
likelihood computes the chi-squared between observed and predicted CMB
power spectra.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class planck_2018_highell_lowell(Likelihood):
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

        if additional_args.get("mock_data", None) is not None:
            self.data_vector = additional_args["mock_data"]

        self.covariance_matrix = utils_likelihood.load_covmat_fortran(
            self.data_dir + "/c_matrix_plik_v22.dat", 613
        )  # 613 bins of data

        self.apply_planck_temperature_rescaling = self.additional_args.get(
            "apply_planck_temperature_rescaling", False
        )
        self.apply_planck_pol_rescaling = self.additional_args.get(
            "apply_planck_pol_rescaling", False
        )

    def get_prediction(self, p):
        units_factor = 1e12  # units factor to convert to microKelvin
        if self.additional_args.get("use_cpj", False):
            TCMB0 = 2.7255
            units_factor = units_factor * (TCMB0**2)
        # defining binning set-up
        nbintt = 215  # 30-2508   #used when getting covariance matrix
        nbinte = 199  # 30-1996
        nbinee = 199  # 30-1996
        plmin = 30
        plmax = 2508
        ellmin = 2

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )
        self.params_dict = params_dict

        tt_emu_model, te_emu_model, ee_emu_model = self.emu_model_list

        if self.additional_args.get("integrated_model") is True:
            self.Cltt = tt_emu_model.predict(params_dict) * units_factor
            Clte = te_emu_model.predict(params_dict) * units_factor
            self.Clee = ee_emu_model.predict(params_dict) * units_factor

        else:
            self.Cltt = exp(tt_emu_model.predictions_np(params_dict)) * units_factor
            Clte = te_emu_model.predictions_np(params_dict) * units_factor
            self.Clee = exp(ee_emu_model.predictions_np(params_dict)) * units_factor

        begintt = self.blmin + plmin - ellmin
        endtt = self.blmax + plmin + 1 - ellmin
        beginte = self.blmin + plmin - ellmin
        endte = self.blmax + plmin + 1 - ellmin
        beginee = self.blmin + plmin - ellmin
        endee = self.blmax + plmin + 1 - ellmin

        beginwtt = self.blmin
        endwtt = self.blmax + 1
        beginwte = self.blmin
        endwte = self.blmax + 1
        beginwee = self.blmin
        endwee = self.blmax + 1

        indicestt = []
        windowtt = []
        indices_reptt = []
        indiceste = []
        windowte = []
        indices_repte = []
        indicesee = []
        windowee = []
        indices_repee = []

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

        for i in range(nbinte):
            idxrepte = repeat(nbintt + i, len(arange(beginte[i], endte[i])))
            indiceste.append(plmax - 1 + arange(beginte[i], endte[i]))
            windowte.append(self.binw[beginwte[i] : endwte[i]])
            indices_repte.append(idxrepte)

        flat_list = [item for sublist in indices_repte for item in sublist]
        indices_repte = array(flat_list)
        flat_list = [item for sublist in indiceste for item in sublist]
        indiceste = array(flat_list)
        flat_list = [item for sublist in windowte for item in sublist]
        windowte = array(flat_list)

        for i in range(nbinee):
            idxrepee = repeat(nbintt + nbinte + i, len(arange(beginee[i], endee[i])))
            indicesee.append(plmax - 1 + plmax - 1 + arange(beginee[i], endee[i]))
            windowee.append(self.binw[beginwee[i] : endwee[i]])
            indices_repee.append(idxrepee)

        flat_list = [item for sublist in indices_repee for item in sublist]
        indices_repee = array(flat_list)
        flat_list = [item for sublist in indicesee for item in sublist]
        indicesee = array(flat_list)
        flat_list = [item for sublist in windowee for item in sublist]
        windowee = array(flat_list)

        indices_rep = concatenate([indices_reptt, indices_repte, indices_repee], axis=0)
        indices = concatenate([indicestt, indiceste, indicesee], axis=0)

        window_ttteee = hstack((windowtt, windowte, windowee))

        self.window_ttteee = window_ttteee
        self.indices = indices
        self.indices_rep = indices_rep

        def ensure_2d(arr):
            # Check if the array is 1D
            if arr.ndim == 1:
                # Add an extra dimension
                return arr[newaxis, :]  # Alternatively, arr.reshape(1, -1)
            return arr  # Return unchanged if it's not 1D

        self.Cltt = ensure_2d(self.Cltt)
        Clte = ensure_2d(Clte)
        self.Clee = ensure_2d(self.Clee)

        if self.apply_planck_temperature_rescaling and self.apply_planck_pol_rescaling:
            self.Cltt = self.Cltt[:, 0 : (plmax - 1)] * (
                (1 + params_dict["m_t_planck"][:, None]) ** 2
            )
            Clte = Clte[:, 0 : (plmax - 1)] * (
                (1 + params_dict["m_t_planck"][:, None])
                * (1 + params_dict["m_e_planck"][:, None])
            )
            self.Clee = self.Clee[:, : (plmax - 1)] * (
                (1 + params_dict["m_e_planck"][:, None]) ** 2
            )

        # apply planck calibration parameter
        self.Cltt = self.Cltt / (array(params_dict["A_planck"])[:, None] ** 2)
        Clte = Clte / (array(params_dict["A_planck"])[:, None] ** 2)
        self.Clee = self.Clee / (array(params_dict["A_planck"])[:, None] ** 2)

        Cl = hstack(
            (
                self.Cltt[:, 0 : plmax - 1],
                Clte[:, 0 : plmax - 1],
                self.Clee[:, : plmax - 1],
            )
        )

        if is_jax:
            Cl_bin = array(
                [
                    bincount(
                        indices_rep,
                        weights=Cl[index, indices] * window_ttteee,
                        length=613,
                    )  # hardcode length for compilation
                    for index in range(Cl.shape[0])
                ]
            )

        else:
            Cl_bin = array(
                [
                    bincount(indices_rep, weights=Cl[index, indices] * window_ttteee)
                    for index in range(Cl.shape[0])
                ]
            )

        return Cl_bin

    def compute_like(self, params):
        predictions = self.get_prediction(params)
        y_obs = self.data_vector

        # Assuming the covariance matrix is already inverted. If not, we'll need to invert it.
        inv_C = linalg.inv(self.covariance_matrix)

        diff = y_obs - predictions
        chi2 = matmul(inv_C, transpose(diff))
        chi2 = matmul(diff, chi2)
        chi2 = diag(chi2)

        like = -0.5 * chi2

        Cls_tt_2_29 = self.Cltt[:, 0:28]
        Cls_ee_2_29 = self.Clee[:, 0:28]

        if self.additional_args.get("add_lowell") is True:

            self.lognorm_ee = utils_likelihood.planck_lowE_binned_loglike(
                Cls_ee_2_29,
                self.mu_LN_EE,
                self.sig_LN_EE,
                self.loc_LN_EE,
                self.lmin_list_EE,
                self.lmax_list_EE,
            )

            self.lognorm_tt = utils_likelihood.planck_lowT_binned_loglike(
                Cls_tt_2_29,
                self.mu_LN_TT,
                self.sig_LN_TT,
                self.lmin_list_TT,
                self.lmax_list_TT,
            )

            like += self.lognorm_tt
            like += self.lognorm_ee

        if self.additional_args.get("with_planck") is True:

            self.lognorm_tt = utils_likelihood.planck_lowT_binned_loglike(
                Cls_tt_2_29,
                self.mu_LN_TT,
                self.sig_LN_TT,
                self.lmin_list_TT,
                self.lmax_list_TT,
            )

            # special case combining ACT with planck- want just lowell tt
            like += self.lognorm_tt

        return like
