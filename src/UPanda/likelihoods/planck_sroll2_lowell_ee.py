"""Planck SROLL2 low-ℓ EE likelihood module.

This module provides the likelihood class for Planck SROLL2 low-ℓ E-mode
polarization measurements. The likelihood computes the chi-squared between
observed and predicted CMB E-mode power spectrum.
"""

import os

import jax.numpy as jnp
import numpy as np
from jax import vmap

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class planck_sroll2_lowell_ee(Likelihood):
    r"""
    Planck 2018 SRoll2 low‑ℓ EE likelihood (ℓ = 2 … 29) in the UPanda style.

    *Reads the SRoll2 probability look‑up table, converts it to JAX
    DeviceArrays and performs a nearest‑grid (no‑interpolation) lookup,
    identical to the Cobaya implementation.*

    Parameters
    ----------
    emu_model_list : list
        A single emulator that predicts *L(L+1)Cℓ/2π* for EE beginning at ℓ = 2.
    dtype : numpy.dtype
        Ordered dtype describing the parameter vector *p*.
    cosmology_params_fixed : list, optional
        Tuples ``(name, value)`` that are held fixed.
    additional_args : dict
        Recognised keys
        ----------------
        apply_planck_pol_rescaling : bool, default ``False``
        use_cpj                     : bool, default ``False``
            If ``True`` multiply by *T*₀² = 2.7255² when converting to µK².
    """

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        compression_vecs=None,
        dtype=None,
        cosmology_params_fixed=None,
        additional_args=None,
    ):
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        cosmology_params_fixed = cosmology_params_fixed or []
        additional_args = additional_args or {}

        # Check for mock data flag (for testing)
        if additional_args.get("mock_prob_table", None) is not None:
            prob_table_np = additional_args["mock_prob_table"]
        else:
            # --- SRoll2 probability table -----------------------------------
            current_directory = os.path.dirname(os.path.abspath(__file__))
            prob_file = os.path.join(
                current_directory,
                "data/planck_sroll2_lowell_ee",
                "sroll2_prob_table.txt",
            )
            prob_table_np = np.loadtxt(prob_file)

        # JAX copies -----------------------------------------------------
        # self.grid      = jnp.asarray(prob_table_np[:, 0])     # shape (Ngrid,)
        self.lnL_table = jnp.asarray(prob_table_np)  # (Ngrid, 28)
        self.grid_step = 0.0001  # 1e‑4 µK²

        # --- Book‑keeping ----------------------------------------------
        if emu_model_list and len(emu_model_list) > 0:
            self.emu_model = emu_model_list[0]
        else:
            self.emu_model = None
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed
        self.additional_args = additional_args

        self._units_factor = 1e12  # K² → µK²
        if additional_args.get("use_cpj", False):
            self._units_factor *= 2.7255**2  # TCMB²

    # ------------------------------------------------------------------
    # PREDICT Cℓ^{EE}  (n_batch, 28) for ℓ = 2…29
    # ------------------------------------------------------------------
    def get_prediction(self, p):
        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        # Emulator returns (n_batch, ℓ_max) in dimensionless units
        Cl_EE_full = self.emu_model.predict(params_dict) * self._units_factor  # → µK²

        # Keep ℓ = 2…29 (index 0…27 if emulator starts at ℓ = 2)
        Cl_EE = jnp.asarray(Cl_EE_full[:, :28])

        ells = arange(2, 30)
        Cl_EE *= (ells) * (ells + 1) / (2 * pi)

        # Polarisation‑efficiency rescaling
        if self.additional_args.get("apply_planck_pol_rescaling", False):
            Cl_EE = Cl_EE * (1 + jnp.asarray(params_dict["m_e_planck"])[:, None]) ** 2

        # Absolute calibration
        if "A_planck" in params_dict:
            Cl_EE = Cl_EE / (jnp.asarray(params_dict["A_planck"])[:, None]) ** 2

        return Cl_EE  # shape (n_batch, 28)

    # ------------------------------------------------------------------
    # LOG‑LIKELIHOOD  (vectorised over the batch dimension)
    # ------------------------------------------------------------------
    def compute_like(self, params):
        """
        Returns
        -------
        lnL : jnp.ndarray  shape (n_batch,)
        """

        # ----------------------------------------------------------------
        # 1.  Predictions
        # ----------------------------------------------------------------
        Cl_pred = self.get_prediction(params)  # (n_batch, 28)

        # ----------------------------------------------------------------
        # 2.  Define one‑sample lookup = Cobaya nearest‑bin rule
        # ----------------------------------------------------------------
        def _loglike_single(cl_row):
            """
            cl_row : (28,)  predicted L(L+1)Cℓ/2π in µK² for ℓ = 2…29
            """
            ngrid = self.lnL_table.shape[0]
            ridx = cl_row / self.grid_step  # real-valued table index per multipole

            # Clamp the lookup into the valid table range so it is always finite:
            # an out-of-range prediction reuses the nearest tabulated (edge) value.
            idx = jnp.clip(jnp.floor(ridx).astype(int32), 0, ngrid - 1)
            lnL_vals = take_along_axis(self.lnL_table, idx[None, :], axis=0)[0]  # (28,)

            # Replace the old hard `-inf` for out-of-table predictions with a smooth,
            # finite penalty. A hard -inf turns the whole (wide) prior box into a flat
            # -inf plateau, which starves gradient-free nested samplers such as
            # nautilus: every initial live point collapses onto the -1e9 floor, log Z
            # never moves, and the returned "chain" is all-identical (the
            # nautilus_w0wa_cmb_bao all -1e9 bug). A LINEAR distance penalty (in
            # grid-index units - linear, so no float overflow even for extreme
            # emulator extrapolations) keeps a monotone gradient that pulls the
            # predicted EE back into the tabulated range, and is exactly 0 whenever
            # the prediction is already in range so the in-range likelihood value is
            # unchanged.
            oob_dist = jnp.sum(
                jnp.maximum(ridx - (ngrid - 1), 0.0) + jnp.maximum(-ridx, 0.0)
            )
            return jnp.sum(lnL_vals) - oob_dist

        # ----------------------------------------------------------------
        # 3.  Vectorise over MCMC / NF batch dimension
        # ----------------------------------------------------------------
        lnL = vmap(_loglike_single)(Cl_pred)  # (n_batch,)

        return lnL
