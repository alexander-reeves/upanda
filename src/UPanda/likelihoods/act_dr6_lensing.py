"""ACT DR6 lensing likelihood module.

This module provides the likelihood class for Atacama Cosmology Telescope
(ACT) Data Release 6 CMB lensing reconstruction. The likelihood computes
the chi-squared between observed and predicted lensing potential power
spectrum C_L^kk.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class act_dr6_lensing(Likelihood):
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

        self.emu_model_list = emu_model_list

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/act_dr6_lensing")

        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []
        self.additional_args = additional_args if additional_args is not None else {}

        # Set both flags separately
        self.want_lensing_lklh_correction = self.additional_args.get(
            "want_lensing_lklh_correction", True
        )
        self.cmb_marginalized = self.additional_args.get("cmb_marginalized", False)

        # Check for mock data flag (for testing)
        if additional_args.get("mock_data", None) is not None:
            self.data_vector = additional_args["mock_data"]
            n_bins = len(self.data_vector)
            self.binning_matrix = additional_args.get(
                "mock_binning_matrix", np.eye(n_bins)
            )
            mock_cov = additional_args.get("mock_cov", np.eye(n_bins))
            self.inv_cov = linalg.inv(mock_cov)
            # Skip loading fiducial data if mocking
            self.want_lensing_lklh_correction = False
        else:
            # Load data vector
            if data_vector is None:
                self.data_vector = np.loadtxt(
                    os.path.join(self.data_dir, "clkk_bandpowers_act.txt")
                )
            else:
                self.data_vector = data_vector

            # Load binning matrix and covariance based on the marginalization flag
            if self.cmb_marginalized:
                # CMB marginalized mode - load marginalized binning and covariance
                self.binning_matrix = np.loadtxt(
                    os.path.join(self.data_dir, "binning_matrix_ACT-DR6_Clkk.txt")
                )
                cov_matrix = np.loadtxt(
                    os.path.join(self.data_dir, "covmat_ACT-DR6_Clkk_cmbmarg.txt")
                )
                self.inv_cov = linalg.inv(cov_matrix)
            else:
                # Full CMB mode - load regular binning and covariance
                self.binning_matrix = np.loadtxt(
                    os.path.join(self.data_dir, "binning_matrix_ACT-DR6_Clkk.txt")
                )
                cov_matrix = np.loadtxt(
                    os.path.join(self.data_dir, "covmat_ACT-DR6_Clkk_baseline.txt")
                )
                self.inv_cov = np.linalg.inv(cov_matrix)

            # Always load fiducial data needed for corrections if want_lensing_lklh_correction is True
            if self.want_lensing_lklh_correction:
                # Load fiducial CMB spectra
                f_ls, f_tt, f_ee, f_bb, f_te = np.loadtxt(
                    os.path.join(self.data_dir, "cosmo2017_10K_acc3_lensedCls.dat"),
                    unpack=True,
                )

                # Standardize the fiducial spectra
                self.f_tt = utils_likelihood.standardize_input_cl(
                    f_ls, f_tt / (f_ls * (f_ls + 1.0)) * 2.0 * pi
                )
                self.f_ee = utils_likelihood.standardize_input_cl(
                    f_ls, f_ee / (f_ls * (f_ls + 1.0)) * 2.0 * pi
                )
                self.f_bb = utils_likelihood.standardize_input_cl(
                    f_ls, f_bb / (f_ls * (f_ls + 1.0)) * 2.0 * pi
                )
                self.f_te = utils_likelihood.standardize_input_cl(
                    f_ls, f_te / (f_ls * (f_ls + 1.0)) * 2.0 * pi
                )
                self.fiducial_cls = array([self.f_tt, self.f_ee, self.f_bb, self.f_te])

                # Load fiducial phi spectra
                fd_ls, f_dd = np.loadtxt(
                    os.path.join(
                        self.data_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"
                    ),
                    unpack=True,
                    usecols=[0, 5],
                )
                self.clkk_fid = utils_likelihood.standardize_input_cl(
                    fd_ls, f_dd * 2.0 * pi / 4.0
                )

                # Load normalization and N1 data with exact filenames from original code
                self.dNorm_dCl = load(
                    os.path.join(
                        self.data_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"
                    )
                )
                fAL_ls, fAL = np.loadtxt(
                    os.path.join(
                        self.data_dir,
                        "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt",
                    )
                )
                self.norm_fid = utils_likelihood.standardize_input_cl(fAL_ls, fAL)

                # Load dN1_dkk
                self.dN1_dClkk = np.loadtxt(
                    os.path.join(self.data_dir, "N1der_KK_lmin600_lmax3000_full.txt")
                )

                # Load dN1_dCl for each mode
                dN1_dCl_data = []
                for spec in ["tt", "ee", "bb", "te"]:
                    filename = f"N1der_{spec.upper()}_lmin600_lmax3000_full.txt"
                    dN1_dCl_data.append(
                        np.loadtxt(os.path.join(self.data_dir, filename))
                    )
                self.dN1_dCl = array(dN1_dCl_data)

        # Set ell range
        self.lmin = 0
        self.lmax = 3000

    def set_provider(self, provider):
        """
        Set the provider for this likelihood.
        """
        self.provider = provider

    def get_prediction(self, p):
        """
        Get the predicted binned CMB lensing power spectrum with corrections if needed.

        Parameters:
        -----------
        p : array_like
            Array of parameters

        Returns:
        --------
        binned_Cl_kk : array_like
            Binned CMB lensing power spectrum (with corrections if needed)
        """
        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        # Get the unbinned theoretical lensing power spectrum
        kk_emu_model = self.emu_model_list[0]

        if self.additional_args.get("integrated_model", False):
            fac = 1.0
            if self.additional_args.get("lens_exp", False):
                Cl_kk = exp(kk_emu_model.predict(params_dict)) * fac
            else:
                Cl_kk = kk_emu_model.predict(params_dict) * fac
        else:
            fac = 2.0 / pi
            if self.additional_args.get("lens_exp", False):
                Cl_kk = exp(kk_emu_model.predictions_np(params_dict)) * fac
            else:
                Cl_kk = kk_emu_model.predictions_np(params_dict) * fac

        # Apply corrections if requested
        if self.want_lensing_lklh_correction:
            # For non-marginalized case, try to get CMB spectra from provider
            if (
                not self.cmb_marginalized
                and hasattr(self, "provider")
                and self.provider is not None
                and self.provider.has_cmb_data
            ):
                # Get CMB spectra from provider
                cl = self.provider.get_Cl(ell_factor=False, units="FIRASmuK2")
                length = min(cl[s].shape[-1] for s in ["tt", "ee", "bb", "te"])
                cl_TEB = stack([asarray(cl[s])[..., :length].reshape(-1, length)
                                for s in ["tt", "ee", "bb", "te"]], axis=1)
            else:
                # If marginalized or provider not available, use fiducial spectra
                if not self.cmb_marginalized:
                    print("Warning: CMB provider not available, using fiducial spectra")
                # provide fiducial spectra with the same batch dimension as Cl_kk just fill with the same fiducial spectra
                print("original fiducial cls shape: ", self.fiducial_cls.shape)
                fid_cls = repeat(asarray(self.fiducial_cls)[None, :, :], axis=0, repeats=Cl_kk.shape[0])
                # reshape?
                fid_cls = fid_cls.reshape(Cl_kk.shape[0], 4, -1)
                print("fid_cls shape: ", fid_cls.shape)
                cl_TEB = fid_cls

            # Compute and apply corrections to the unbinned spectrum
            Cl_kk_corrected = utils_likelihood.compute_act_lensing_auto_corrections(self, cl_TEB, Cl_kk)
        else:
            # No corrections needed
            Cl_kk_corrected = Cl_kk

        print("Cl_kk_corrected shape: ", Cl_kk_corrected.shape)

        # Apply binning to the (possibly corrected) Cl_kk
        # Ensure the theoretical spectrum has the right shape for the binning matrix
        ell_max_bin = self.binning_matrix.shape[1] - 1
        Cl_kk_corrected = pad(
            Cl_kk_corrected[:, :ell_max_bin + 1],
            ((0, 0), (0, max(0, ell_max_bin + 1 - Cl_kk_corrected.shape[1]))),
        )

        binned_Cl_kk = einsum("ij,bj->bi", self.binning_matrix, Cl_kk_corrected)

        return binned_Cl_kk

    def compute_like(self, params):
        """
        Compute the likelihood for the given parameters.

        Parameters:
        -----------
        params : array_like
            Array of parameters

        Returns:
        --------
        loglkl : float
            Log-likelihood value
        """
        cl_theory_cmbkk = self.get_prediction(params)

        diff = self.data_vector - cl_theory_cmbkk

        chi2 = matmul(self.inv_cov, transpose(diff))
        chi2 = matmul(diff, chi2)

        chi2 = diag(chi2)

        loglkl = -0.5 * chi2

        return loglkl
