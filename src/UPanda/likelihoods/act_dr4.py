"""ACT DR4 CMB likelihood module.

This module provides the likelihood class for Atacama Cosmology Telescope
(ACT) Data Release 4 CMB temperature and polarization measurements.
The likelihood computes the chi-squared between observed and predicted
CMB power spectra.
"""

from UPanda.module import *

is_jax = config.get_jax_enabled()

import os

import numpy as np
import scipy
from scipy.io import FortranFile  # need this to read the fortran data format

from UPanda.likelihoods import utils_likelihood

from .likelihood import Likelihood


class act_dr4(Likelihood):
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

        self.nbin = 260  # total bins
        # defining binning set-up
        self.nbintt = 40
        self.nbinte = 45
        self.nbinee = 45
        self.lmax_win = 7925  # total ell in window functions
        self.bmax = 52
        self.b0 = 5  # first bin in TT theory selection
        self.nbinw = 130

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/act_dr4")

        like_file = os.path.join(self.data_dir, "cl_cmb_ap.dat")
        cov_file = os.path.join(self.data_dir, "c_matrix_ap.dat")
        bbldeep_file = os.path.join(self.data_dir, "coadd_bpwf_15mJy_191127_lmin2.npz")
        bblwide_file = os.path.join(self.data_dir, "coadd_bpwf_100mJy_191127_lmin2.npz")

        # read window functions
        bmax_win = 520
        lmax_win = 7925

        if is_jax:
            bbldeep = np.load(bbldeep_file)["bpwf"]
            win_func_d = zeros((bmax_win, lmax_win))
            self.win_func_d = win_func_d.at[:bmax_win, 1:lmax_win].set(
                bbldeep[:bmax_win, :lmax_win]
            )
            bblwide = np.load(bblwide_file)["bpwf"]
            win_func_w = zeros((bmax_win, lmax_win))
            self.win_func_w = win_func_w.at[:bmax_win, 1:lmax_win].set(
                bblwide[:bmax_win, :lmax_win]
            )

        else:
            bbldeep = np.load(bbldeep_file)["bpwf"]
            self.win_func_d = zeros((bmax_win, lmax_win))
            self.win_func_d[:bmax_win, 1:lmax_win] = bbldeep[:bmax_win, :lmax_win]
            bblwide = np.load(bblwide_file)["bpwf"]
            self.win_func_w = zeros((bmax_win, lmax_win))
            self.win_func_w[:bmax_win, 1:lmax_win] = bblwide[:bmax_win, :lmax_win]

        f = FortranFile(cov_file, "r")
        cov = f.read_reals(dtype=float).reshape((self.nbin, self.nbin))

        self.cov = array(cov)

        nbin = self.cov.shape[
            0
        ]  # Assuming cov is a square matrix and its shape is (nbin, nbin)

        for i_index in range(self.nbin):
            for j_index in range(i_index, self.nbin):

                if is_jax:
                    self.cov = self.cov.at[i_index, j_index].set(cov[j_index, i_index])
                else:
                    self.cov[i_index, j_index] = self.cov[j_index, i_index]

        # important: arrays in Fortran are 1-indexed,
        # but arrays in Python are 0-indexed. :(

        self.additional_args = additional_args if additional_args is not None else {}

        self.test = self.additional_args.get("test", False)
        self.input_predictions = self.additional_args.get("input_predictions", None)

        self.apply_act_temperature_rescaling = self.additional_args.get(
            "apply_act_temperature_rescaling", False
        )

        # selection if act alone or ACT + planck
        bmin = 0
        bmin_te = 0
        bmin_ee = 0

        if self.additional_args.get("with_planck") is True:
            bmin = 24

        if self.additional_args.get("remove_intermediate") is True:
            print("Removing first bins, setting default ellmin=1800")
            bmin = 24
            bmin_te = 29  # matching the ell cut @ ell=1800
            bmin_ee = 29

            if self.additional_args.get("input_bmin", None) is not None:
                print("Using input bmin")
                bmin = self.additional_args.get("input_bmin")
                bmin_te = self.additional_args.get("input_bmin") + 5
                bmin_ee = self.additional_args.get("input_bmin") + 5

        # cull lmin in TT if combining with Planck
        if bmin > 0:
            for i in range(bmin):

                if is_jax:
                    self.cov = self.cov.at[i, : self.nbintt].set(0.0)  # deep
                    self.cov = self.cov.at[: self.nbintt, i].set(0.0)  # deep
                    self.cov = self.cov.at[i, i].set(1e10)  # deep
                    self.cov = self.cov.at[
                        self.nbinw + i, self.nbinw : self.nbinw + self.nbintt
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[
                        self.nbinw : self.nbinw + self.nbintt, self.nbinw + i
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[self.nbinw + i, self.nbinw + i].set(
                        1e10
                    )  # wide

                else:
                    self.cov[i, : self.nbintt] = 0.0  # deep
                    self.cov[: self.nbintt, i] = 0.0  # deep
                    self.cov[i, i] = 1e10  # deep
                    self.cov[self.nbinw + i, self.nbinw : self.nbinw + self.nbintt] = (
                        0.0  # wide
                    )
                    self.cov[self.nbinw : self.nbinw + self.nbintt, self.nbinw + i] = (
                        0.0  # wide
                    )
                    self.cov[self.nbinw + i, self.nbinw + i] = 1e10  # wide

        # cull lmin in TE and EE if combining with Planck
        if bmin_te > 0:
            for i in range(bmin_te):
                if is_jax:
                    self.cov = self.cov.at[
                        self.nbintt + i, self.nbintt : self.nbintt + self.nbinte
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbintt : self.nbintt + self.nbinte, self.nbintt + i
                    ].set(0.0)
                    self.cov = self.cov.at[self.nbintt + i, self.nbintt + i].set(1e10)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + i,
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                        self.nbinw + self.nbintt + i,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + i, self.nbinw + self.nbintt + i
                    ].set(1e10)
                else:
                    self.cov[
                        self.nbintt + i, self.nbintt : self.nbintt + self.nbinte
                    ] = 0.0
                    self.cov[
                        self.nbintt : self.nbintt + self.nbinte, self.nbintt + i
                    ] = 0.0
                    self.cov[self.nbintt + i, self.nbintt + i] = 1e10
                    self.cov[
                        self.nbinw + self.nbintt + i,
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                        self.nbinw + self.nbintt + i,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt + i, self.nbinw + self.nbintt + i
                    ] = 1e10

        if bmin_ee > 0:
            for i in range(bmin_ee):
                if is_jax:
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte + i,
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                        self.nbintt + self.nbinte + i,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte + i, self.nbintt + self.nbinte + i
                    ].set(1e10)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ].set(1e10)
                else:
                    self.cov[
                        self.nbintt + self.nbinte + i,
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                    ] = 0.0
                    self.cov[
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                        self.nbintt + self.nbinte + i,
                    ] = 0.0
                    self.cov[
                        self.nbintt + self.nbinte + i, self.nbintt + self.nbinte + i
                    ] = 1e10
                    self.cov[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                    ] = 0.0
                    self.cov[
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ] = 1e10

        if self.additional_args.get("clip_cmb_index", None) is not None:
            maxbin = self.nbintt - self.additional_args.get("clip_cmb_index")
            maxbin_te = self.nbinte - self.additional_args.get("clip_cmb_index")

        else:
            maxbin = 0

        if maxbin > 0:
            for i in range(maxbin, self.nbintt):
                if is_jax:
                    self.cov = self.cov.at[i, : self.nbintt].set(0.0)  # deep
                    self.cov = self.cov.at[: self.nbintt, i].set(0.0)  # deep
                    self.cov = self.cov.at[i, i].set(1e12)  # deep
                    self.cov = self.cov.at[
                        self.nbinw + i, self.nbinw : self.nbinw + self.nbintt
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[
                        self.nbinw : self.nbinw + self.nbintt, self.nbinw + i
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[self.nbinw + i, self.nbinw + i].set(
                        1e12
                    )  # wide
                else:
                    self.cov[i, : self.nbintt] = 0.0  # deep
                    self.cov[: self.nbintt, i] = 0.0  # deep
                    self.cov[i, i] = 1e12  # deep
                    self.cov[self.nbinw + i, self.nbinw : self.nbinw + self.nbintt] = (
                        0.0  # wide
                    )
                    self.cov[self.nbinw : self.nbinw + self.nbintt, self.nbinw + i] = (
                        0.0  # wide
                    )
                    self.cov[self.nbinw + i, self.nbinw + i] = 1e12  # wide

            # for for te and ee
            for i in range(maxbin_te, self.nbinte):
                if is_jax:
                    # te
                    self.cov = self.cov.at[
                        self.nbintt + i, self.nbintt : self.nbintt + self.nbinte
                    ].set(
                        0.0
                    )  # deep
                    self.cov = self.cov.at[
                        self.nbintt : self.nbintt + self.nbinte, self.nbintt + i
                    ].set(
                        0.0
                    )  # deep
                    self.cov = self.cov.at[self.nbintt + i, self.nbintt + i].set(
                        1e12
                    )  # deep
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + i,
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                        self.nbinw + self.nbintt + i,
                    ].set(
                        0.0
                    )  # wide
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + i, self.nbinw + self.nbintt + i
                    ].set(1e12)
                    # ee
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte + i,
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                        self.nbintt + self.nbinte + i,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbintt + self.nbinte + i, self.nbintt + self.nbinte + i
                    ].set(1e12)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ].set(0.0)
                    self.cov = self.cov.at[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ].set(1e12)
                else:
                    # te
                    self.cov[
                        self.nbintt + i, self.nbintt : self.nbintt + self.nbinte
                    ] = 0.0
                    self.cov[
                        self.nbintt : self.nbintt + self.nbinte, self.nbintt + i
                    ] = 0.0
                    self.cov[self.nbintt + i, self.nbintt + i] = 1e12
                    self.cov[
                        self.nbinw + self.nbintt + i,
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt : self.nbinw + self.nbinte,
                        self.nbinw + self.nbintt + i,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt + i, self.nbinw + self.nbintt + i
                    ] = 1e12
                    # ee
                    self.cov[
                        self.nbintt + self.nbinte + i,
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                    ] = 0.0
                    self.cov[
                        self.nbintt + self.nbinte : self.nbintt + self.nbinee,
                        self.nbintt + self.nbinte + i,
                    ] = 0.0
                    self.cov[
                        self.nbintt + self.nbinte + i, self.nbintt + self.nbinte + i
                    ] = 1e12
                    self.cov[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                    ] = 0.0
                    self.cov[
                        self.nbinw
                        + self.nbintt
                        + self.nbinte : self.nbinw
                        + self.nbinee,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ] = 0.0
                    self.cov[
                        self.nbinw + self.nbintt + self.nbinte + i,
                        self.nbinw + self.nbintt + self.nbinte + i,
                    ] = 1e12

        self.fisher = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(self.cov), b=identity(self.nbin)
        )

        self.bval, self.data_vector, self.X_sig = np.genfromtxt(
            like_file, max_rows=self.nbin, delimiter=None, unpack=True
        )

        self.compression_vecs = compression_vecs
        self.emu_model_list = emu_model_list
        self.dtype = dtype

        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []

    def get_prediction(self, p):
        units_factor = 1e12  # units factor to convert to microKelvin

        # defining binning set-up
        nbin = self.nbin  # total bins
        nbintt = self.nbintt
        nbinte = self.nbinte
        nbinee = self.nbinee
        lmax_win = self.lmax_win  # total ell in window functions
        bmax = self.bmax
        b0 = self.b0  # first bin in TT theory selection

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        tt_emu_model, te_emu_model, ee_emu_model = self.emu_model_list

        if self.additional_args.get("integrated_model") is True:
            cltt = tt_emu_model.predict(params_dict) * units_factor
            clte = te_emu_model.predict(params_dict) * units_factor
            clee = ee_emu_model.predict(params_dict) * units_factor

        else:
            cltt = exp(tt_emu_model.predictions_np(params_dict)) * units_factor
            clte = te_emu_model.predictions_np(params_dict) * units_factor
            clee = exp(ee_emu_model.predictions_np(params_dict)) * units_factor

        num_zeros_to_append = lmax_win - cltt.shape[1]

        cltt = pad(cltt, ((0, 0), (0, num_zeros_to_append)), mode="constant").T
        clte = pad(clte, ((0, 0), (0, num_zeros_to_append)), mode="constant").T
        clee = pad(clee, ((0, 0), (0, num_zeros_to_append)), mode="constant").T

        cl_tt_d = (
            self.win_func_d[2 * bmax : 3 * bmax, 1:lmax_win] @ cltt[: lmax_win - 1]
        )
        cl_te_d = (
            self.win_func_d[6 * bmax : 7 * bmax, 1:lmax_win] @ clte[: lmax_win - 1]
        )
        cl_ee_d = (
            self.win_func_d[9 * bmax : 10 * bmax, 1:lmax_win] @ clee[: lmax_win - 1]
        )
        # use 150x150 windows
        cl_tt_w = (
            self.win_func_w[2 * bmax : 3 * bmax, 1:lmax_win] @ cltt[: lmax_win - 1]
        )
        cl_te_w = (
            self.win_func_w[6 * bmax : 7 * bmax, 1:lmax_win] @ clte[: lmax_win - 1]
        )
        cl_ee_w = (
            self.win_func_w[9 * bmax : 10 * bmax, 1:lmax_win] @ clee[: lmax_win - 1]
        )

        yp2 = array(params_dict["yp2"])

        if is_jax:
            X_model = zeros((nbin, cltt.shape[1]))
            if self.apply_act_temperature_rescaling:
                X_model = X_model.at[:nbintt, :].set(
                    cl_tt_d[b0 : b0 + nbintt] * (1 + params_dict["m_t_act"]) ** 2
                )  # TT
            else:
                X_model = X_model.at[:nbintt, :].set(cl_tt_d[b0 : b0 + nbintt])  # TT
            if self.apply_act_temperature_rescaling:
                X_model = X_model.at[nbintt : nbintt + nbinte].set(
                    cl_te_d[:nbinte] * yp2 * (1 + params_dict["m_t_act"])
                )  # TE
            else:
                X_model = X_model.at[nbintt : nbintt + nbinte].set(
                    cl_te_d[:nbinte] * yp2
                )  # TE

            X_model = X_model.at[nbintt + nbinte : nbintt + nbinte + nbinee].set(
                (cl_ee_d[:nbinee] * yp2**2)
            )  # EE
            X_model = X_model.at[
                nbintt + nbinte + nbinee : 2 * nbintt + nbinte + nbinee
            ].set(
                cl_tt_w[b0 : b0 + nbintt]
            )  # TT
            X_model = X_model.at[
                2 * nbintt + nbinte + nbinee : 2 * nbintt + 2 * nbinte + nbinee
            ].set(
                cl_te_w[:nbinte] * yp2
            )  # TE
            X_model = X_model.at[
                2 * nbintt + 2 * nbinte + nbinee : 2 * nbintt + 2 * nbinte + 2 * nbinee
            ].set(
                cl_ee_w[:nbinee] * yp2 * yp2
            )  # EE

        else:
            X_model = zeros((nbin, cltt.shape[1]))

            if self.apply_act_temperature_rescaling:
                X_model[:nbintt] = (
                    cl_tt_d[b0 : b0 + nbintt] * (1 + params_dict["m_t_act"]) ** 2
                )  # TT
            else:
                X_model[:nbintt] = cl_tt_d[b0 : b0 + nbintt]  # TT

            if self.apply_act_temperature_rescaling:
                X_model[nbintt : nbintt + nbinte] = (
                    cl_te_d[:nbinte] * yp2 * (1 + params_dict["m_t_act"])
                )  # TE
            else:
                X_model[nbintt : nbintt + nbinte] = cl_te_d[:nbinte] * yp2  # TE

            X_model[nbintt + nbinte : nbintt + nbinte + nbinee] = (
                cl_ee_d[:nbinee] * yp2**2
            )  # EE
            X_model[nbintt + nbinte + nbinee : 2 * nbintt + nbinte + nbinee] = cl_tt_w[
                b0 : b0 + nbintt
            ]  # TT
            X_model[2 * nbintt + nbinte + nbinee : 2 * nbintt + 2 * nbinte + nbinee] = (
                cl_te_w[:nbinte] * yp2
            )  # TE
            X_model[
                2 * nbintt + 2 * nbinte + nbinee : 2 * nbintt + 2 * nbinte + 2 * nbinee
            ] = (
                cl_ee_w[:nbinee] * yp2 * yp2
            )  # EE

        return X_model

    def compute_like(self, params):
        if self.test:
            predictions = self.input_predictions

        else:
            predictions = self.get_prediction(params)

        y_obs = self.data_vector
        diff_vec = predictions.T - y_obs
        ptemp = dot(self.fisher, diff_vec.T)
        like = -0.5 * sum(ptemp * diff_vec.T, axis=0)

        return like
