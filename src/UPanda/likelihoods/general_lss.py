"""General large-scale structure (LSS) likelihood module.

This module provides a general likelihood class for large-scale structure
observables, including galaxy clustering, weak lensing, and their cross-correlations.
The likelihood can handle multiple probes and redshift bins.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood

is_jax = config.get_jax_enabled()
if is_jax:
    pass  # JAX-specific imports handled elsewhere


class general_lss(Likelihood):
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

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir_wl = os.path.join(current_directory, "data/KiDS_1000")
        self.data_dir_clustering = os.path.join(current_directory, "data/BOSS")
        self.data_dir_cmb_lensing = os.path.join(
            current_directory, "data/planck_2018_lensing"
        )
        self.data_dir_act_lensing = os.path.join(
            current_directory, "data/act_dr6_lensing"
        )

        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []

        self.additional_args = additional_args if additional_args is not None else {}
        self.external_spectra = external_spectra

        # additional args for this likelihood
        self.bump_vector_list = additional_args.get("bump_vector_list", None)
        self.indices_list = additional_args.get("indices_list", None)
        self.kappa_indices_list = additional_args.get("kappa_indices_list", None)
        self.spectra = additional_args.get("spectra", None)
        self.ufalcon_database = additional_args.get("ufalcon_database", None)
        self.ufalcon_database_2 = additional_args.get("ufalcon_database_2", None)
        self.data_vector_database = additional_args.get("data_vector_database", None)
        self.apply_indices_to_dv = additional_args.get("apply_indices_to_dv", False)
        self.apply_indices_to_cov = additional_args.get("apply_indices_to_cov", False)
        self.cut_cmblensing_indices = additional_args.get(
            "cut_cmblensing_indices", False
        )
        self.indices_list_mg = additional_args.get("indices_list_mg", None)
        self.indices_list_k_cross = additional_args.get("indices_list_k_cross", None)
        self.indices_list_ka_cross = additional_args.get("indices_list_ka_cross", None)
        self.indices_list_t_cross = additional_args.get("indices_list_t_cross", None)
        self.apply_t_indices = additional_args.get("apply_t_indices", False)
        self.apply_kappa_rescaling = additional_args.get("apply_kappa_rescaling", False)
        self.mc_norm_file = additional_args.get("mc_norm_file", None)
        self.omh3 = additional_args.get("omh3", None)
        self.indices_shot = additional_args.get("indices_shot", None)

        # Check that "kk" and "ka" are not both in spectraß
        if "kk" in self.spectra and "kaka" in self.spectra:
            raise ValueError(
                "Cannot include both Planck ('kk') and ACT DR6 ('ka') lensing in the same analysis"
            )

        # Setup for ACT DR6 lensing corrections
        self.want_lensing_lklh_correction = self.additional_args.get(
            "want_lensing_lklh_correction", True
        )
        self.cmb_marginalized = self.additional_args.get("cmb_marginalized", False)

        if self.additional_args.get("input_cov", None) is None:
            cov = utils_likelihood.create_cov_lss(
                self.spectra, self.ufalcon_database, self.ufalcon_database_2
            )
        else:
            cov = additional_args["input_cov"]

        # Covariance readers intentionally return NumPy arrays. Convert once at
        # the backend boundary before the functional JAX block replacements below.
        cov = asarray(cov)

        if self.apply_indices_to_cov:
            cov = cov[tuple(self.indices_list), :][:, tuple(self.indices_list)]

        if self.additional_args.get("input_dv", None) is None:

            self.y_obs = utils_likelihood.create_datavector(
                self.spectra, self.data_vector_database, self.mc_norm_file
            )
        else:
            self.y_obs = additional_args["input_dv"]

        if self.apply_indices_to_dv:
            self.y_obs = self.y_obs[tuple(self.indices_list)]
            # Handle Planck CMB lensing
        if "kk" in self.spectra:
            if self.spectra[-1] != "kk":
                raise (
                    "WARNING: must have kk specified last in list of spectra when adding"
                )

            cov_kk_official = np.loadtxt(
                self.data_dir_cmb_lensing + "/covariance.txt"
            ).reshape(9, -1)
            # Check if mock kk data is provided, otherwise load from file
            mock_kk = self.additional_args.get("mock_kk", None)
            if mock_kk is not None:
                self.cl_hat_kk = mock_kk if is_jax else np.array(mock_kk)
                print("  Using mock kk (Planck CMB lensing) data")
            else:
                self.cl_hat_kk = np.loadtxt(self.data_dir_cmb_lensing + "/cl_hat.txt")
            self.theory_offset_kk = np.loadtxt(
                self.data_dir_cmb_lensing + "/theory_offset.txt"
            )
            self.bin_mat_lens = np.loadtxt(
                self.data_dir_cmb_lensing + "/binning_plus_correction_matrix.txt"
            ).reshape(9, -1)

            # replace last block of covariance
            if (additional_args.get("add_kk_official") is True) and (
                "kk" in self.spectra
            ):
                print("adding offical kk covariance to the full block covariance")
                if is_jax:
                    cov = cov.at[-9:, -9:].set(cov_kk_official)
                else:
                    cov[-9:, -9:] = cov_kk_official

            if self.cut_cmblensing_indices:
                # remove the first n bins of the cmb_lensing likelihood
                self.cutoff = additional_args.get("cmb_lensing_cutoff", 2)
                print("setting CMB lensing cut off:", self.cutoff)
                final_bin = 9 - self.cutoff
                cov = delete(
                    delete(cov, s_[-9:-final_bin], axis=0), s_[-9:-final_bin], axis=1
                )  # remove the first two bins from the cmb lensing cov

            self.ell_min_kk = 2
            self.ell_max_kk = 2500

            if self.cut_cmblensing_indices:
                self.cl_hat_kk = self.cl_hat_kk[self.cutoff :]

            self.y_obs = hstack((self.y_obs, self.cl_hat_kk))

        # Handle ACT DR6 CMB lensing
        if "kaka" in self.spectra:
            print("adding ACT DR6 lensing to the likelihood")
            if self.spectra[-1] != "kaka":
                raise (
                    "WARNING: must have kaka specified last in list of spectra when adding"
                )

            # Load ACT DR6 lensing data
            if self.cmb_marginalized:
                print("using CMB marginalized mode for act_dr6 lensing")
                # CMB marginalized mode - load marginalized binning and covariance
                self.bin_mat_act = np.loadtxt(
                    os.path.join(
                        self.data_dir_act_lensing, "binning_matrix_ACT-DR6_Clkk.txt"
                    )
                )
                cov_act_official = np.loadtxt(
                    os.path.join(
                        self.data_dir_act_lensing, "covmat_ACT-DR6_Clkk_cmbmarg.txt"
                    )
                )
            else:
                print("using full CMB mode for act_dr6 lensing")
                # Full CMB mode - load regular binning and covariance
                self.bin_mat_act = np.loadtxt(
                    os.path.join(
                        self.data_dir_act_lensing, "binning_matrix_ACT-DR6_Clkk.txt"
                    )
                )
                cov_act_official = np.loadtxt(
                    os.path.join(
                        self.data_dir_act_lensing, "covmat_ACT-DR6_Clkk_baseline.txt"
                    )
                )

            # Check if mock kaka data is provided, otherwise load from file
            mock_kaka = self.additional_args.get("mock_kaka", None)
            if mock_kaka is not None:
                self.cl_hat_act = mock_kaka if is_jax else np.array(mock_kaka)
                print("  Using mock kaka data")
            else:
                self.cl_hat_act = np.loadtxt(
                    os.path.join(self.data_dir_act_lensing, "clkk_bandpowers_act.txt")
                )
            self.y_obs = hstack((self.y_obs, self.cl_hat_act))

            # Replace last block of covariance with official ACT covariance
            if additional_args.get("add_ka_official", True):
                print("adding official ka covariance to the full block covariance")
                act_bins = len(self.cl_hat_act)
                if is_jax:
                    cov = cov.at[-act_bins:, -act_bins:].set(cov_act_official)
                else:
                    cov[-act_bins:, -act_bins:] = cov_act_official

        # Load fiducial data needed for corrections if want_lensing_lklh_correction is True
        if self.want_lensing_lklh_correction:
            # Load fiducial CMB spectra
            f_ls, f_tt, f_ee, f_bb, f_te = np.loadtxt(
                os.path.join(
                    self.data_dir_act_lensing, "cosmo2017_10K_acc3_lensedCls.dat"
                ),
                unpack=True,
            )

            # Standardize the fiducial spectra
            self.f_tt_act = utils_likelihood.standardize_input_cl(
                f_ls, f_tt / (f_ls * (f_ls + 1.0)) * 2.0 * np.pi
            )
            self.f_ee_act = utils_likelihood.standardize_input_cl(
                f_ls, f_ee / (f_ls * (f_ls + 1.0)) * 2.0 * np.pi
            )
            self.f_bb_act = utils_likelihood.standardize_input_cl(
                f_ls, f_bb / (f_ls * (f_ls + 1.0)) * 2.0 * np.pi
            )
            self.f_te_act = utils_likelihood.standardize_input_cl(
                f_ls, f_te / (f_ls * (f_ls + 1.0)) * 2.0 * np.pi
            )
            self.fiducial_cls = np.array(
                [self.f_tt_act, self.f_ee_act, self.f_bb_act, self.f_te_act]
            )

            # Load fiducial phi spectra
            fd_ls, f_dd = np.loadtxt(
                os.path.join(
                    self.data_dir_act_lensing, "cosmo2017_10K_acc3_lenspotentialCls.dat"
                ),
                unpack=True,
                usecols=[0, 5],
            )
            self.clkk_fid = utils_likelihood.standardize_input_cl(
                fd_ls, f_dd * 2.0 * np.pi / 4.0
            )

            # Load normalization and N1 data with exact filenames from original code
            self.dNorm_dCl = np.load(
                os.path.join(
                    self.data_dir_act_lensing,
                    "norm_correction_matrix_Lmin0_Lmax4000.npy",
                )
            )
            fAL_ls, fAL = np.loadtxt(
                os.path.join(
                    self.data_dir_act_lensing,
                    "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt",
                )
            )
            self.norm_fid = utils_likelihood.standardize_input_cl(fAL_ls, fAL)

            # Load dN1_dkk
            self.dN1_dClkk = np.loadtxt(
                os.path.join(
                    self.data_dir_act_lensing, "N1der_KK_lmin600_lmax3000_full.txt"
                )
            )

            # Load dN1_dCl for each mode
            dN1_dCl_data = []
            for spec in ["tt", "ee", "bb", "te"]:
                filename = f"N1der_{spec.upper()}_lmin600_lmax3000_full.txt"
                dN1_dCl_data.append(
                    np.loadtxt(os.path.join(self.data_dir_act_lensing, filename))
                )
            self.dN1_dCl = np.array(dN1_dCl_data)

            self.cl_hat_act = np.loadtxt(
                os.path.join(self.data_dir_act_lensing, "clkk_bandpowers_act.txt")
            )

        # ACT DR6 lensing bandpower range, as in act_dr6_lenslike: "baseline" = bins [2:-6] (40 <= L <= 763),
        # "extended" = bins [2:-3] (40 <= L <= 1300), "all" = all 18 bandpowers (the behaviour before 2026-10-06).
        # Applied to the data, the binning matrix and the full covariance (before inversion); kaka is always last.
        self.act_lensing_range = self.additional_args.get("act_lensing_range", "all")
        if "kaka" in self.spectra:
            act_slices = {"all": slice(None), "baseline": slice(2, -6), "extended": slice(2, -3)}
            if self.act_lensing_range not in act_slices:
                raise ValueError(f"act_lensing_range must be one of {list(act_slices)}, got {self.act_lensing_range}")
            n_act = np.asarray(self.bin_mat_act).shape[0]
            keep_act = np.arange(n_act)[act_slices[self.act_lensing_range]]
            if keep_act.size != n_act:
                n_tot = cov.shape[0]
                keep = np.r_[np.arange(n_tot - n_act), n_tot - n_act + keep_act]
                cov = cov[np.ix_(keep, keep)]
                self.y_obs = self.y_obs[keep]
                self.cl_hat_act = np.asarray(self.cl_hat_act)[keep_act]
                self.bin_mat_act = np.asarray(self.bin_mat_act)[keep_act]
            print(f"ACT DR6 lensing range '{self.act_lensing_range}': {keep_act.size} of {n_act} bandpowers")

        self.inv_C = array(np.linalg.inv(cov))

        # assert that inv cov @ cov = I
        try:
            assert np.allclose(np.dot(cov, self.inv_C), np.eye(cov.shape[0]))
        except:
            print("covariance matrix probably not well converged...")
            print(
                "diagonal of inv @ cov",
                np.diag(np.dot(cov, self.inv_C)),
                np.eye(cov.shape[0]),
            )

    def get_prediction(self, p):
        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, sigma8, etc.)
        self.params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        if self.omh3 is not None:
            o_nu = 3 * array(self.params_dict["m_nu"]) / 93.14
            omega_mh2 = (
                self.params_dict["omega_cdm"] + self.params_dict["omega_b"] + o_nu
            )
            self.params_dict["h"] = array(self.omh3) / array(omega_mh2)
            # print("params_dict h", self.params_dict["h"])

        predictions = []
        for i, emulator in enumerate(self.emu_model_list):
            if self.external_spectra is not None:
                y_theory = self.external_spectra[i]
            else:
                # Usual emulator call
                if self.additional_args.get("integrated_model") is True:
                    y_theory = emulator.predict(self.params_dict)
                else:
                    y_theory = exp(emulator.predictions_np(self.params_dict))

            # Handle Planck lensing (kk) if it's the last emulator
            if (i == len(self.emu_model_list) - 1) and ("kk" in self.spectra):
                y_theory = y_theory[
                    :, self.ell_min_kk - 2 : self.ell_max_kk - 2 + 1
                ] * (2 / pi)

                y_theory = dot(self.bin_mat_lens, y_theory.T).T
                cls_emu = (
                    y_theory - self.theory_offset_kk
                )  # remove the lensing correction following lensing paper- make sure this needs to be removed not added

                if self.cut_cmblensing_indices:
                    # the low ell bins are poorly modelled with simulations- remove these
                    cls_emu = cls_emu[:, self.cutoff :]

            # Handle ACT lensing (ka) if it's the last emulator
            elif (i == len(self.emu_model_list) - 1) and ("kaka" in self.spectra):
                # Get raw predictions
                y_theory_raw = y_theory

                # emulator starts at ell=2 so add some zeros at the front for ell=0, ell=1
                # add two extra zeros at the front for ell=0,1
                rows = y_theory.shape[0]
                zeros_column = zeros(rows)  # Single column of zeross
                y_theory_raw = column_stack((zeros_column, zeros_column, y_theory_raw))

                # Apply corrections if needed
                if self.want_lensing_lklh_correction:
                    # Get CMB spectra from provider or use fiducial spectra
                    if (
                        not self.cmb_marginalized
                        and hasattr(self, "provider")
                        and self.provider is not None
                        and self.provider.has_cmb_data
                    ):
                        cl = self.provider.get_Cl(ell_factor=False, units="FIRASmuK2")

                        # Robust shape handling: harmonize ell ranges and ensure proper array structure
                        min_ell = min(
                            cl["tt"].shape[-1],
                            cl["ee"].shape[-1],
                            cl["bb"].shape[-1],
                            cl["te"].shape[-1],
                        )

                        # Truncate all spectra to same ell range
                        tt_trunc = cl["tt"][..., :min_ell]
                        ee_trunc = cl["ee"][..., :min_ell]
                        bb_trunc = cl["bb"][..., :min_ell]
                        te_trunc = cl["te"][..., :min_ell]

                        # Stack into (4, ...) then transpose to (..., 4, min_ell) if needed
                        cl_TEB = array([tt_trunc, ee_trunc, bb_trunc, te_trunc])
                        if cl_TEB.ndim == 3 and cl_TEB.shape[0] == 4:
                            cl_TEB = cl_TEB.transpose(1, 0, 2)
                    else:
                        # Use fiducial spectra
                        if y_theory_raw.ndim == 2:  # Has batch dimension
                            batch_size = y_theory_raw.shape[0]
                            cl_TEB = repeat(
                                self.fiducial_cls[newaxis, :, :], batch_size, axis=0
                            )
                        else:
                            cl_TEB = self.fiducial_cls

                    # Compute and apply corrections to the unbinned spectrum
                    y_theory_corrected = (
                        utils_likelihood.compute_act_lensing_auto_corrections(
                            self, cl_TEB, y_theory_raw
                        )
                    )
                else:
                    # No corrections needed
                    y_theory_corrected = y_theory_raw

                # Apply binning to the (possibly corrected) spectrum
                ell_max_bin = self.bin_mat_act.shape[1] - 1
                if y_theory_corrected.shape[1] < ell_max_bin + 1:
                    # JAX-safe zero-padding: jnp arrays are immutable so indexed
                    # assignment (arr[:, :n] = x) is a silent no-op under JAX.
                    # Use pad() which is functional and JIT-compatible.
                    pad_width = ell_max_bin + 1 - y_theory_corrected.shape[1]
                    y_theory_corrected = pad(
                        y_theory_corrected, ((0, 0), (0, pad_width))
                    )

                # Apply binning matrix using einsum for efficiency
                cls_emu = einsum("ij,bj->bi", self.bin_mat_act,
                                 y_theory_corrected[:, :ell_max_bin + 1])

            else:
                # Nuisance operations must not mutate an external theory buffer.
                y_theory = copy(y_theory)
                # first apply mg_i nuisance parameters if specified
                if self.indices_list_mg is not None:
                    linear_list = self.indices_list_mg[i][0]
                    squared_list = self.indices_list_mg[i][1]
                    for j in range(5):
                        if len(linear_list[j]) != 0:
                            multiplier = (
                                1 + self.params_dict[f"m{j + 1}"][:, None]
                            )  # Shape: (batch_size, 1)
                            updated_values = y_theory[:, linear_list[j]] * multiplier
                            if is_jax:
                                y_theory = y_theory.at[:, linear_list[j]].set(
                                    updated_values
                                )
                            else:
                                y_theory[:, linear_list[j]] = updated_values

                        if len(squared_list[j]) != 0:
                            multiplier = (
                                1 + self.params_dict[f"m{j + 1}"][:, None]
                            ) ** 2  # Shape: (batch_size, 1)
                            updated_values = y_theory[:, squared_list[j]] * multiplier
                            if is_jax:
                                y_theory = y_theory.at[:, squared_list[j]].set(
                                    updated_values
                                )
                            else:
                                y_theory[:, squared_list[j]] = updated_values

                # now apply k indices (Planck CMB lensing cross-correlations)
                if self.indices_list_k_cross is not None:
                    k_indices = self.indices_list_k_cross[
                        i
                    ]  # Assuming this is a list or array of indices
                    if k_indices is not None and len(k_indices) != 0:
                        # Reshape m_k for proper broadcasting
                        multiplier = (
                            1 + self.params_dict["m_k"][:, None]
                        )  # Shape: (batch_size, 1)

                        # Multiply y_theory at k_indices by the multiplier
                        # y_theory[:, k_indices] has shape (batch_size, num_k_indices)
                        updated_values = (
                            y_theory[:, k_indices] * multiplier
                        )  # Broadcasting over columns
                        if is_jax:
                            y_theory = y_theory.at[:, k_indices].set(updated_values)
                        else:
                            y_theory[:, k_indices] = updated_values

                # Apply ka indices (ACT CMB lensing cross-correlations)
                if self.indices_list_ka_cross is not None:
                    ka_indices = self.indices_list_ka_cross[
                        i
                    ]  # Get indices for ACT lensing cross-correlations
                    # print("ka_indices", ka_indices)
                    if ka_indices is not None and len(ka_indices) != 0:
                        # Use the specific ACT nuisance parameter if available, otherwise fall back to m_k
                        m_k_act = self.params_dict.get(
                            "m_k_act", self.params_dict.get("m_k", array([0.0]))
                        )

                        # Reshape for broadcasting
                        multiplier = 1 + m_k_act[:, None]  # Shape: (batch_size, 1)

                        # Apply correction to cross-correlations
                        updated_values = y_theory[:, ka_indices] * multiplier
                        if is_jax:
                            y_theory = y_theory.at[:, ka_indices].set(updated_values)
                        else:
                            y_theory[:, ka_indices] = updated_values

                    # If we have lensing likelihood corrections enabled
                    if self.want_lensing_lklh_correction:
                        # Get CMB spectra (fiducial or from provider)
                        if (
                            not self.cmb_marginalized
                            and hasattr(self, "provider")
                            and self.provider is not None
                            and self.provider.has_cmb_data
                        ):
                            cl = self.provider.get_Cl(
                                ell_factor=False, units="FIRASmuK2"
                            )

                            # Robust shape handling: harmonize ell ranges
                            min_ell = min(
                                cl["tt"].shape[-1],
                                cl["ee"].shape[-1],
                                cl["bb"].shape[-1],
                                cl["te"].shape[-1],
                            )
                            tt_trunc = cl["tt"][..., :min_ell]
                            ee_trunc = cl["ee"][..., :min_ell]
                            bb_trunc = cl["bb"][..., :min_ell]
                            te_trunc = cl["te"][..., :min_ell]

                            cl_TEB = array([tt_trunc, ee_trunc, bb_trunc, te_trunc])
                            if cl_TEB.ndim == 3 and cl_TEB.shape[0] == 4:
                                cl_TEB = cl_TEB.transpose(1, 0, 2)
                        else:
                            cl_TEB = self.fiducial_cls

                        # Apply normalization correction to cross-correlations
                        lmax = self.dNorm_dCl.shape[-1] - 1

                        # Preserve each sample's CMB normalization correction.
                        # Reusing sample zero makes likelihoods batch-dependent.
                        if cl_TEB.ndim == 2:
                            cl_TEB = cl_TEB[newaxis, :, :]
                        clTEB_diff = cl_TEB[:, :, :lmax + 1] - self.fiducial_cls[newaxis, :, :lmax + 1]
                        norm_correction = einsum("snl,bsl->bn", self.dNorm_dCl, clTEB_diff)
                        cross_corr_factor = 1.0 + norm_correction[:, :len(ka_indices)]
                        updated = y_theory[:, ka_indices] * cross_corr_factor
                        if is_jax:
                            y_theory = y_theory.at[:, ka_indices].set(updated)
                        else:
                            y_theory[:, ka_indices] = updated

                # Now apply t indices
                if self.indices_list_t_cross is not None and self.apply_t_indices:
                    t_indices = self.indices_list_t_cross[i]
                    if t_indices is not None and len(t_indices) != 0:
                        try:
                            multiplier = (
                                1 + self.params_dict["m_t_planck"][:, None]
                            )  # Shape: (batch_size, 1)
                        except:
                            multiplier = 1.0
                        try:
                            updated_values = y_theory[:, t_indices] * multiplier
                            if is_jax:
                                y_theory = y_theory.at[:, t_indices].set(updated_values)
                            else:
                                y_theory[:, t_indices] = updated_values
                        except:
                            continue

                # finally if there is are shot noise nuisance parameters
                if self.indices_shot is not None and self.indices_shot[i] is not None:
                    for j in range(len(self.indices_shot[i])):
                        shot_indices = self.indices_shot[i][j]
                        updated_values = (
                            y_theory[:, shot_indices]
                            + (self.params_dict[f"sn_{j + 1}"] * 1e-6)[:, None]
                        )  # RECALL WEIGHTED BY 1e-6!!
                        if is_jax:
                            y_theory = y_theory.at[:, shot_indices].set(updated_values)
                        else:
                            y_theory[:, shot_indices] = updated_values

                # Select indices if specified, otherwise use y_theory as is
                cls_emu = (
                    utils_likelihood.select_indices(y_theory, self.indices_list[i])
                    if self.indices_list is not None
                    else y_theory
                )

                # print("CLS EMU SHAPE", cls_emu.shape)

                # Adjust for bump_vector if specified
                if self.bump_vector_list is not None:
                    cls_emu -= abs(
                        self.bump_vector_list[i][self.indices_list[i]]
                    )  # multiply bump vectors by 2 by default- check in your own case!

            predictions.append(cls_emu)  # or whatever the processed output is

        # The final emulator is the lensing auto-spectrum. Rescaling every
        # emulator also scales unrelated galaxy/shear bins (and cross terms twice).
        if self.apply_kappa_rescaling and ("kk" in self.spectra or "kaka" in self.spectra):
            calibration = self.params_dict.get("m_k", 0.0)
            if "kaka" in self.spectra:
                calibration = self.params_dict.get("m_k_act", calibration)
            calibration = asarray(calibration)
            multiplier = (1 + calibration) ** 2
            if multiplier.ndim:
                multiplier = multiplier[:, None]
            predictions[-1] = predictions[-1] * multiplier

        # Concatenate all processed predictions
        return concatenate(predictions, axis=1 if predictions[0].ndim > 1 else 0)

    def compute_like(self, params):
        cls_emu_out = self.get_prediction(params)

        diff = cls_emu_out - self.y_obs

        def compute_chi2_jax(delta_cl, invkll):
            # Matrix multiplication
            intermediate = dot(invkll, delta_cl.T)

            # Computing the diagonal elements efficiently
            # Calculate chi2 using einsum
            chi2 = einsum("ij,ji->i", delta_cl, intermediate)

            return chi2

        # loglkl = -0.5 * (diff.dot(self.inv_C) * diff).sum(axis=1)
        loglkl = -0.5 * compute_chi2_jax(diff, self.inv_C)

        if self.additional_args.get("kl_prior", False):

            if self.additional_args.get("need_sigma8", False):
                sigma8_samples = self.params_dict["sigma8"]
            else:
                index_sigma8 = self.dtype.names.index("sigma8")
                sigma8_samples = params[:, index_sigma8]
            index_omega_cdm = self.dtype.names.index("omega_cdm")
            omega_cdm_samples = params[:, index_omega_cdm]
            index_omega_b = self.dtype.names.index("omega_b")
            omega_b_samples = params[:, index_omega_b]
            try:
                index_m_nu = self.dtype.names.index("m_nu")
                m_nu_samples = params[:, index_m_nu]
            except:
                m_nu_samples = 0.02

            omega_nu_samples = 3 * m_nu_samples / 93.14
            index_h = self.dtype.names.index("h")
            h_samples = params[:, index_h]
            S8_samples = sigma8_samples * sqrt(
                (omega_cdm_samples + omega_b_samples + omega_nu_samples)
                / h_samples
                / h_samples
                / 0.3
            )
            S8_mu = 0.815
            S8_sigma = 0.019
            loglkl += -0.5 * ((S8_samples - S8_mu) / S8_sigma) ** 2

        if self.additional_args.get("kog_prior", False):

            if self.additional_args.get("need_sigma8", False):
                sigma8_samples = self.params_dict["sigma8"]
            else:
                index_sigma8 = self.dtype.names.index("sigma8")
                sigma8_samples = params[:, index_sigma8]
            index_omega_cdm = self.dtype.names.index("omega_cdm")
            omega_cdm_samples = params[:, index_omega_cdm]
            index_omega_b = self.dtype.names.index("omega_b")
            omega_b_samples = params[:, index_omega_b]
            try:
                index_m_nu = self.dtype.names.index("m_nu")
                m_nu_samples = params[:, index_m_nu]
            except:
                m_nu_samples = 0.02

            omega_nu_samples = 3 * m_nu_samples / 93.14
            index_h = self.dtype.names.index("h")
            h_samples = params[:, index_h]
            S8_samples = sigma8_samples * sqrt(
                (omega_cdm_samples + omega_b_samples + omega_nu_samples)
                / h_samples
                / h_samples
                / 0.3
            )
            S8_mu = 0.759
            S8_sigma = 0.022
            loglkl += -0.5 * ((S8_samples - S8_mu) / S8_sigma) ** 2

        return loglkl

    def set_external_spectra(self, spectra):
        self.external_spectra = spectra

    def set_provider(self, provider):
        """
        Set the provider for this likelihood to receive CMB spectra from other components.
        """
        self.provider = provider
