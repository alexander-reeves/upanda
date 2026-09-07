"""ACT DR6 CMB likelihood module.

This module provides the likelihood class for Atacama Cosmology Telescope
(ACT) Data Release 6 CMB temperature and polarization measurements.
The likelihood computes the chi-squared between observed and predicted
CMB power spectra.
"""

from UPanda.module import *

is_jax = config.get_jax_enabled()

import os

import numpy as np
import sacc  # Need SACC to read the new data format

from UPanda.likelihoods import utils_likelihood

from .likelihood import Likelihood


class act_dr6(Likelihood):
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

        # Configure file paths
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/act_dr6")
        self.file_base_name = "dr6_data_cmbonly"
        self.input_file = f"{self.file_base_name}.fits"

        # Default ell cuts (can be overridden in additional_args)

        self.ell_cuts = {"TT": [600, 6500], "TE": [600, 6500], "EE": [600, 6500]}

        # Process additional args
        self.additional_args = additional_args if additional_args is not None else {}
        if self.additional_args.get("ell_cuts", None):
            self.ell_cuts.update(self.additional_args["ell_cuts"])

        self.lmax_theory = self.additional_args.get("lmax_theory", 9000)
        self.test = self.additional_args.get("test", False)
        self.input_predictions = self.additional_args.get("input_predictions", None)
        self.apply_act_temperature_rescaling = self.additional_args.get(
            "apply_act_temperature_rescaling", False
        )

        # Load data from SACC file
        input_filename = os.path.join(self.data_dir, self.input_file)
        input_file = sacc.Sacc.load_fits(input_filename)

        # Dictionary to map polarization indices
        pol_dt = {"t": "0", "e": "e", "b": "b"}

        self.spec_meta = []
        self.cull = []
        idx_max = 0

        # Process each spectrum type (TT, TE, EE)
        for pol in ["TT", "TE", "EE"]:
            p1, p2 = pol.lower()
            t1, t2 = pol_dt[p1], pol_dt[p2]
            dt = f"cl_{t1}{t2}"

            tracers = input_file.get_tracer_combinations(dt)

            for tr1, tr2 in tracers:
                lmin, lmax = self.ell_cuts.get(pol, (np.inf, -np.inf))
                ls, mu, ind = input_file.get_ell_cl(dt, tr1, tr2, return_ind=True)
                mask = np.logical_and(ls >= lmin, ls <= lmax)

                if not np.all(mask):
                    print(f"Cutting {pol} data to the range [{lmin}-{lmax}].")
                    self.cull.append(ind[~mask])

                if np.any(mask):
                    window = input_file.get_bandpower_windows(ind[mask])

                    self.spec_meta.append(
                        {
                            "data_type": dt,
                            "tracer1": tr1,
                            "tracer2": tr2,
                            "pol": pol.lower(),
                            "ell": ls[mask],
                            "spec": mu[mask],
                            "idx": ind[mask],
                            "window": window,
                        }
                    )

                    idx_max = max(idx_max, max(ind))
                    self.lmax_theory = max(
                        self.lmax_theory, int(window.values.max()) + 1
                    )

        # Create data vector
        self.data_vector = np.zeros((idx_max + 1,))
        for m in self.spec_meta:
            self.data_vector[m["idx"]] = m["spec"]

        # Process covariance matrix.
        # SACC FITS stores the covmat big-endian (>f8); JAX only accepts
        # native byte order, so coerce to native float64 before array().
        self.cov = array(np.asarray(input_file.covariance.covmat, dtype=np.float64))
        for culls in self.cull:
            if is_jax:
                for c in culls:
                    self.cov = self.cov.at[c, :].set(0.0)
                    self.cov = self.cov.at[:, c].set(0.0)
                    self.cov = self.cov.at[c, c].set(1e10)
            else:
                self.cov[culls, :] = 0.0
                self.cov[:, culls] = 0.0
                self.cov[culls, culls] = 1e10

        self.fisher = array(linalg.inv(self.cov))

        # Save parameters for later use
        self.compression_vecs = compression_vecs
        self.emu_model_list = emu_model_list
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []
        self.external_cls = additional_args.get("input_cls", None)

        # Default calibration parameters
        self.A_act = 1.0
        self.P_act = 1.0

    def get_prediction(self, p):
        """Get theoretical predictions for the given parameters"""
        units_factor = 1e12  # units factor to convert to microKelvin

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        # Extract calibration parameters if present
        A_act = params_dict.get("A_act", self.A_act)
        P_act = params_dict.get("P_act", self.P_act)

        # Get emulators
        tt_emu_model, te_emu_model, ee_emu_model = self.emu_model_list

        if self.external_cls is not None:
            print("testing with external cls")
            cltt = self.external_cls["tt"]
            clte = self.external_cls["te"]
            clee = self.external_cls["ee"]

        else:
            # Get predictions from emulators
            if self.additional_args.get("integrated_model") is True:
                cltt = tt_emu_model.predict(params_dict) * units_factor
                clte = te_emu_model.predict(params_dict) * units_factor
                clee = ee_emu_model.predict(params_dict) * units_factor
            else:
                cltt = exp(tt_emu_model.predictions_np(params_dict)) * units_factor
                clte = te_emu_model.predictions_np(params_dict) * units_factor
                clee = exp(ee_emu_model.predictions_np(params_dict)) * units_factor

        # emulator ells start at ell=2 so we need to add ell=0 and ell=1 to the predictions which are just 0s remembering about the batch index

        ells_tt = np.arange(2, cltt.shape[1] + 2)
        ells_te = np.arange(2, clte.shape[1] + 2)
        ells_ee = np.arange(2, clee.shape[1] + 2)

        # multiply by ell factor
        ell_factor_tt = ells_tt * (ells_tt + 1) / (2 * np.pi)
        ell_factor_te = ells_te * (ells_te + 1) / (2 * np.pi)
        ell_factor_ee = ells_ee * (ells_ee + 1) / (2 * np.pi)
        cltt = cltt * ell_factor_tt
        clte = clte * ell_factor_te
        clee = clee * ell_factor_ee

        if self.apply_act_temperature_rescaling:
            cltt = cltt * (1 + params_dict["m_t_act"][:, newaxis]) ** 2
            clte = (
                clte
                * (1 + params_dict["m_e_act"][:, newaxis])
                * (1 + params_dict["m_t_act"][:, newaxis])
            )
            clee = clee * (1 + params_dict["m_e_act"][:, newaxis]) ** 2

        # # add two extra zeros at the front for ell=0,1
        zeros_column = zeros((cltt.shape[0], 2), dtype=cltt.dtype)

        cltt = column_stack((zeros_column, cltt))
        clte = column_stack((zeros_column, clte))
        clee = column_stack((zeros_column, clee))

        # Create spectra dictionary for easier reference in window function calculation
        # make sure these are 32 bit arrays
        cls = {"tt": cltt, "te": clte, "ee": clee}

        # Send CMB spectra to provider for lensing corrections
        if hasattr(self, "provider") and self.provider is not None:
            # Convert to compatible format (remove ell(ell+1)/(2pi) factor for provider)
            # Provider expects raw Cl values in FIRASmuK2 units
            # Compute ell_factor for each spectrum individually to handle different shapes

            ells_tt = arange(cltt.shape[1])
            ell_factor_tt = where(ells_tt > 0, ells_tt * (ells_tt + 1) / (2 * pi), 1.0)

            ells_ee = arange(clee.shape[1])
            ell_factor_ee = where(ells_ee > 0, ells_ee * (ells_ee + 1) / (2 * pi), 1.0)

            ells_te = arange(clte.shape[1])
            ell_factor_te = where(ells_te > 0, ells_te * (ells_te + 1) / (2 * pi), 1.0)

            cls_for_provider = {
                "tt": cltt / ell_factor_tt,
                "ee": clee / ell_factor_ee,
                "te": clte / ell_factor_te,
                "bb": zeros_like(cltt / ell_factor_tt),
            }
            self.provider.set_Cl(cls_for_provider)

        # Apply window functions to get binned spectra ps-spec has shape batch, data_vector shape
        batch = cltt.shape[0]
        ps_vec = zeros((batch, self.data_vector.shape[0]))

        for m in self.spec_meta:
            idx = m["idx"]
            # SACC window weights are big-endian (>f8) from FITS; coerce to
            # native float64 before array() so JAX accepts them.
            win = array(np.asarray(m["window"].weight.T, dtype=np.float64))
            ls = m["window"].values.astype(int32)
            pol = m["pol"]

            # Get right spectrum and apply calibration
            calibration = A_act * A_act
            dat = cls[pol].T[ls] / calibration
            if pol[0] == "e":
                dat /= P_act
            if pol[1] == "e":
                dat /= P_act

            if is_jax:
                ps_vec = ps_vec.at[:, idx].set(einsum("ij, jb -> bi", win, dat))
            else:
                ps_vec[:, idx] = einsum("ij, jb -> bi", win, dat)

        return ps_vec

    def compute_like(self, params):
        """Compute likelihood for given parameters"""
        if self.test:
            predictions = self.input_predictions
        else:
            predictions = self.get_prediction(params)

        y_obs = self.data_vector
        diff_vec = predictions - y_obs

        # Vectorized quadratic form calculation for batched inputs
        # diff_vec is shape (batch, data_dim)
        if is_jax:
            # For jax: batch @ fisher @ batch.T - properly vectorized
            like = -0.5 * einsum("bi,ij,bj->b", diff_vec, self.fisher, diff_vec)
        else:
            # For numpy: equivalent vectorized operation
            like = -0.5 * np.einsum("bi,ij,bj->b", diff_vec, self.fisher, diff_vec)

        return like

    def set_provider(self, provider):
        """
        Set the provider for this likelihood to share CMB spectra with other components.
        """
        self.provider = provider
