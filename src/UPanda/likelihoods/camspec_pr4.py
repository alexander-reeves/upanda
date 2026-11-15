"""CamSpec PR4 CMB likelihood module.

This module provides the likelihood class for CamSpec Planck Release 4
high-ℓ CMB temperature and polarization measurements. The likelihood
computes the chi-squared between observed and predicted CMB power spectra.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood

"""
Planck PR4/NPIPE **CamSpec** high‑ℓ likelihood for UPanda — now with the
optional low‑ℓ TT and EE log‑normal likelihood that Planck normally
provides through *Commander* + *SimAll*.

The file is a stripped‑down but numerically faithful rewrite of Cobaya's
``Planck2020CamSpecPython`` **plus** the low‑ℓ logic lifted from
``planck_2018_highell_lowell``.  Everything lives in a *single* class so
UPanda can compile it once and vectorise over samples.

Key toggles (in ``additional_args``)
-----------------------------------
* ``add_lowell`` – include both low‑ℓ TT **and** EE (default: ``False``)
* ``with_planck`` – include only low‑ℓ TT (for ACT × Planck combos)
* ``apply_planck_temperature_rescaling`` / ``apply_planck_pol_rescaling`` –
  enable the *m_t_planck* and *m_e_planck* rescaling parameters.

Required extra data files (identical to Planck 2018 ones)
--------------------------------------------------------
Under ``camspec_root/low_ell`` we expect two ASCII tables:

* ``lognormal_fit_2bins_TT.txt`` (columns: ℓ_min ℓ_max μ σ)
* ``lognormal_fit_3bins_EE.txt`` (columns: ℓ_min ℓ_max μ σ loc)

These are exactly the same files the Planck 2018 likelihood ships.  You
can override their paths via ``additional_args`` if you keep them
elsewhere.
"""

is_jax = config.get_jax_enabled()

# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities (light copies from Cobaya)
# ─────────────────────────────────────────────────────────────────────────────


def _range_to_ells(use_range):
    if isinstance(use_range, str):
        rng = []
        for part in use_range.split():
            if "-" in part:
                lo, hi = map(int, part.split("-"))
                rng.append(np.arange(lo, hi + 1))
            else:
                rng.append(np.array([int(part)]))
        return concatenate(rng) if rng else asarray([], dtype=int32)
    return asarray(use_range)


def _read_normalized(fname, pivot=None):
    dat = np.loadtxt(fname)
    assert int(dat[0, 0]) == 2
    dat = np.hstack(([0, 0], dat[:, 1]))
    if pivot is not None:
        dat /= dat[pivot]
    return dat


class camspec_pr4(Likelihood):

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
        if cosmology_params_fixed is None:
            cosmology_params_fixed = []
        if additional_args is None:
            additional_args = {}
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        # remember toggles ------------------------------------------------
        self.apply_planck_temperature_rescaling = additional_args.get(
            "apply_planck_temperature_rescaling", False
        )
        self.apply_planck_pol_rescaling = additional_args.get(
            "apply_planck_pol_rescaling", False
        )
        self.add_lowell = additional_args.get("add_lowell", False)
        self.with_planck = additional_args.get("with_planck", False)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_directory, "data/camspec_pr4")

        cl_hat_file = self.data_dir + "/like_NPIPE_12.6_unified_spectra.txt"
        covmat_file = self.data_dir + "/like_NPIPE_12.6_unified_cov.bin"
        data_ranges_file = self.data_dir + "/like_NPIPE_12.6_unified_data_ranges.txt"

        # foreground templates (unchanged)
        sz143file = self.data_dir + "/tsz_143_eps0.50.dat"
        kszfile = self.data_dir + "/cl_ksz_148_trac.dat"
        tszxcibfile = self.data_dir + "/sz_x_cib_template.dat"
        cib217file = self.data_dir + "/cib217.txt"
        dust100file = self.data_dir + "/100x100_10.5_dust.dat"
        dust143file = self.data_dir + "/143x143_10.5_dust.dat"
        dust217file = self.data_dir + "/217x217_10.5_dust.dat"
        dust143x217file = self.data_dir + "/143x217_10.5_dust.dat"

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
        # -----------------------------------------------------------------
        # 2. primary spectra & bookkeeping --------------------------------
        # -----------------------------------------------------------------
        spectra = np.loadtxt(cl_hat_file)
        with open(data_ranges_file, "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.cl_names = []
        self.lmin = []
        self.lmax = []
        for ln in lines:
            tp, mn, mx = ln.split()
            self.cl_names.append(tp)
            self.lmin.append(int(mn))
            self.lmax.append(int(mx))
        self.lmin = np.asarray(self.lmin, int)
        self.lmax = np.asarray(self.lmax, int)

        n_tot = np.sum(self.lmax - self.lmin + 1)
        use_cl_default = "143x143 217x217 143x217 TE EE"
        self.use_cl = use_cl_default.split()

        used_ell = None

        data_vec_all, used_indices = [], []
        nX = 0
        for tp, mn, mx in zip(self.cl_names, self.lmin, self.lmax):
            if mx < mn:
                continue
            spec = spectra[mn : mx + 1, self.cl_names.index(tp)]
            data_vec_all.append(spec)
            if tp in self.use_cl:
                if used_ell is not None and (
                    not isinstance(used_ell, dict) or tp in used_ell
                ):
                    ells = used_ell[tp] if isinstance(used_ell, dict) else used_ell
                    ells = np.array([L for L in range(mn, mx + 1) if L in ells])
                else:
                    ells = np.arange(mn, mx + 1)
                used_indices.append(ells + (nX - mn))
            nX += mx - mn + 1

        self.ell_ranges = [None] * len(self.cl_names)
        self.used_sizes = [0] * len(self.cl_names)
        for idx, (tp, mn, mx) in enumerate(zip(self.cl_names, self.lmin, self.lmax)):
            if tp in self.use_cl:
                ells = np.arange(mn, mx + 1)
                self.ell_ranges[idx] = ells
                self.used_sizes[idx] = len(ells)

        self.data_vector = np.concatenate(data_vec_all)[np.concatenate(used_indices)]

        print("DATA VECTOR SHAPE", self.data_vector.shape)

        # covariance ------------------------------------------------------
        dtype_cov = np.float32
        cov_full = (
            np.fromfile(covmat_file, dtype=dtype_cov).reshape(nX, nX).astype(np.float64)
        )
        cov = cov_full[
            np.ix_(np.concatenate(used_indices), np.concatenate(used_indices))
        ]
        self.cov = cov
        # numpy LAPACK inversion (~30 s for 9915^2); the JAX/XLA `inv` of a matrix this size took 40 min to
        # COMPILE on CPU (2026-09-14), once per process. Numerically the same float64 result.
        self.covinv = array(np.linalg.inv(self.cov))

        # foreground templates -------------------------------------------
        pivot = 1500
        max_l = np.max(self.lmax)
        self.ls = np.arange(max_l + 1)
        self.llp1 = self.ls * (self.ls + 1)
        self.sz_143 = array(_read_normalized(sz143file, pivot))[: max_l + 1]
        self.ksz = array(_read_normalized(kszfile, pivot))[: max_l + 1]
        self.tszxcib = array(_read_normalized(tszxcibfile, pivot))[: max_l + 1]
        self.cib_217 = array(_read_normalized(cib217file, pivot))[: max_l + 1]
        self.dust = array(
            np.vstack(
                [
                    _read_normalized(dust100file)[: max_l + 1],
                    _read_normalized(dust143file)[: max_l + 1],
                    _read_normalized(dust217file)[: max_l + 1],
                    _read_normalized(dust143x217file)[: max_l + 1],
                ]
            )
        )
        self.lnrat = np.zeros_like(self.ls, float)
        lmin_use = np.min(self.lmin[np.isin(self.cl_names, self.use_cl)])
        self.lnrat[lmin_use:] = np.log(self.ls[lmin_use:] / float(pivot))

        # bookkeeping -----------------------------------------------------
        self.dtype = dtype
        self.emu_model_list = emu_model_list
        self.cosmology_params_fixed = cosmology_params_fixed
        self.additional_args = additional_args
        self.external_spectra = external_spectra

    # ------------------------------------------------------------------
    #  Calibrations & foregrounds (unchanged)
    # ------------------------------------------------------------------
    def _get_cals(self, p):
        # 1. grab A_planck as a length-batch array
        default_vals = ones_like(atleast_1d(p.get("A_planck", 1.0)))
        A_planck = atleast_1d(p.get("A_planck", 1.0))
        A2 = A_planck**2  # shape = (batch,)

        # 2. for each cal-parameter, make it a length-batch array
        cal0 = atleast_1d(p.get("cal0", default_vals))
        cal2 = atleast_1d(p.get("cal2", default_vals))
        calTE = atleast_1d(p.get("calTE", default_vals))
        calEE = atleast_1d(p.get("calEE", default_vals))

        # 3. stack them into shape (batch,6)
        #    order: [cal0, 1, cal2, sqrt(cal2), calTE, calEE]
        cals = stack(
            [
                cal0,
                ones_like(cal0),
                cal2,
                sqrt(cal2),
                calTE,
                calEE,
            ],
            axis=1,
        )  # (batch,6)

        # 4. apply the A_planck² rescaling per sample
        return cals * A2[:, None]  # still (batch,6)

    def _get_foregrounds(self, p):
        default_vals = ones_like(atleast_1d(p.get("A_planck", 1.0)))

        amp_100 = p.get("amp_100", default_vals)
        amp_143 = p.get("amp_143", default_vals)
        amp_217 = p.get("amp_217", default_vals)
        amp_143x217 = p.get("amp_143x217", default_vals)

        n_100 = p.get("n_100", default_vals)
        n_143 = p.get("n_143", default_vals)
        n_217 = p.get("n_217", default_vals)
        n_143x217 = p.get("n_143x217", default_vals)

        # 2. stack them into (batch,4)
        amps = stack([amp_100, amp_143, amp_217, amp_143x217], axis=1)
        tilts = stack([n_100, n_143, n_217, n_143x217], axis=1)

        # 3. build cluster of ℓ by (ℓmax+1)
        ls = self.ls  # shape = (ℓmax+1,)
        pivot = 1500

        # 4. for each of the 4 components, compute (batch,ℓmax+1)
        #    result will be (batch,4,ℓmax+1)
        C = zeros((amps.shape[0], 4, ls.size))
        for i in range(4):
            if is_jax:
                C = C.at[:, i, :].set(
                    amps[:, i : i + 1] * (ls[None, :] / pivot) ** (tilts[:, i : i + 1])
                )
            else:
                C[:, i, :] = amps[:, i : i + 1] * (ls[None, :] / pivot) ** (
                    tilts[:, i : i + 1]
                )
        return C

    # ------------------------------------------------------------------
    #  Prediction (returns band‑power vector) + stores low‑ℓ Cl's
    # ------------------------------------------------------------------
    def get_prediction(self, p):
        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        uK2 = 1e12
        if self.external_spectra is not None:
            ClTT = self.external_spectra["tt"] * uK2
            ClTE = self.external_spectra["te"] * uK2
            ClEE = self.external_spectra["ee"] * uK2
        else:
            tt_emu, te_emu, ee_emu = self.emu_model_list
            if self.additional_args.get("integrated_model"):
                ClTT = tt_emu.predict(params) * uK2
                ClTE = te_emu.predict(params) * uK2
                ClEE = ee_emu.predict(params) * uK2
            else:
                ClTT = exp(tt_emu.predictions_np(params)) * uK2
                ClTE = te_emu.predictions_np(params) * uK2
                ClEE = exp(ee_emu.predictions_np(params)) * uK2

        # optional Planck rescaling ------------------------------------
        batch = ClTT.shape[0]
        if self.apply_planck_temperature_rescaling:
            m_t = (
                atleast_1d(params.get("m_t_planck", 0.0))[:, None]
                if batch > 1
                else params.get("m_t_planck", 0.0)
            )
            ClTT = ClTT * ((1 + m_t) ** 2)
        if self.apply_planck_pol_rescaling:
            m_e = (
                atleast_1d(params.get("m_e_planck", 0.0))[:, None]
                if batch > 1
                else params.get("m_e_planck", 0.0)
            )
            ClEE = ClEE * ((1 + m_e) ** 2)
            ClTE = (
                ClTE * ((1 + m_t) * (1 + m_e))
                if self.apply_planck_temperature_rescaling
                else ClTE * ((1 + m_e))
            )

        # store raw low‑ℓ spectra (ℓ≥2 index 0→ℓ=2)
        self.Cltt = ClTT.copy()
        self.Clee = ClEE.copy()

        # construct D_ell with zeros at ℓ=0,1 --------------------------
        ell_tt = arange(2, ClTT.shape[1] + 2)
        fac_tt = ell_tt * (ell_tt + 1) / (2 * pi)
        ell_te = arange(2, ClTE.shape[1] + 2)
        fac_te = ell_te * (ell_te + 1) / (2 * pi)
        ell_ee = arange(2, ClEE.shape[1] + 2)
        fac_ee = ell_ee * (ell_ee + 1) / (2 * pi)
        DTT = hstack((zeros((batch, 2)), ClTT * fac_tt))
        DTE = hstack((zeros((batch, 2)), ClTE * fac_te))
        DEE = hstack((zeros((batch, 2)), ClEE * fac_ee))

        # calibration & foregrounds -----------------------------------
        cals = self._get_cals(params)
        fg = self._get_foregrounds(params) if np.any(self.used_sizes[:4]) else None
        pred = zeros((batch, self.data_vector.size))
        ix = 0
        for i, (tp, n) in enumerate(zip(self.cl_names, self.used_sizes)):
            if n == 0:
                continue
            ells = self.ell_ranges[i]
            spec = (DTT if i <= 3 else DTE if i == 4 else DEE)[:, ells]  # (batch, n)
            if i <= 3:
                spec += fg[:, i, ells]
            spec /= cals[:, i : i + 1]
            if is_jax:
                pred = pred.at[:, ix : ix + n].set(spec)
            else:
                pred[:, ix : ix + n] = spec
            ix += n
        return pred

    # ------------------------------------------------------------------
    #  Log‑likelihood (high‑ℓ CamSpec + optional low‑ℓ)
    # ------------------------------------------------------------------
    def compute_like(self, p):
        pred = self.get_prediction(p)
        diff = pred - self.data_vector
        # --- Binning/Compression logic ---
        binning_matrix = self.additional_args.get("binning_matrix", None)
        binned_covinv = self.additional_args.get("binned_covinv", None)
        if binning_matrix is not None:
            # Bin the residuals directly (consistent with hillipop approach)
            diff_binned = binning_matrix @ diff.T  # (n_bins, batch)
            if binned_covinv is not None:
                covinv_binned = binned_covinv
            else:
                # Correctly bin the covariance matrix first, then invert
                cov_binned = binning_matrix @ self.cov @ binning_matrix.T
                covinv_binned = linalg.inv(cov_binned)
            # Compute chi2 in binned space
            like = -0.5 * (
                einsum("bi,ij,jb->b", diff_binned.T, covinv_binned, diff_binned)
                if is_jax
                else np.einsum("bi,ij,jb->b", diff_binned.T, covinv_binned, diff_binned)
            )
        else:
            like = -0.5 * (
                einsum("bi,ij,bj->b", diff, self.covinv, diff)
                if is_jax
                else np.einsum("bi,ij,bj->b", diff, self.covinv, diff)
            )

        if self.add_lowell or self.with_planck:
            # rebuild params dict quickly (for extracting calibration param A_planck)
            params_dict = utils_likelihood.prepare_params_dict(
                p, self.dtype, self.cosmology_params_fixed, self.additional_args
            )
            A2 = (
                (array(params_dict["A_planck"])[:, None] ** 2)
                if like.ndim
                else params_dict["A_planck"] ** 2
            )
            Cls_tt_2_29 = (self.Cltt / A2)[:, 0:28]
            Cls_ee_2_29 = (self.Clee / A2)[:, 0:28]

            if self.add_lowell:
                print("lowell")
                ln_like_ee = utils_likelihood.planck_lowE_binned_loglike(
                    Cls_ee_2_29,
                    self.mu_LN_EE,
                    self.sig_LN_EE,
                    self.loc_LN_EE,
                    self.lmin_list_EE,
                    self.lmax_list_EE,
                )
                ln_like_tt = utils_likelihood.planck_lowT_binned_loglike(
                    Cls_tt_2_29,
                    self.mu_LN_TT,
                    self.sig_LN_TT,
                    self.lmin_list_TT,
                    self.lmax_list_TT,
                )
                like += ln_like_tt + ln_like_ee
            elif self.with_planck:
                print("with planck")
                ln_like_tt = utils_likelihood.planck_lowT_binned_loglike(
                    Cls_tt_2_29,
                    self.mu_LN_TT,
                    self.sig_LN_TT,
                    self.lmin_list_TT,
                    self.lmax_list_TT,
                )
                like += ln_like_tt

        return like

    def set_external_spectra(self, spectra):
        self.external_spectra = spectra
