"""Utility functions for likelihood computations.

This module provides helper functions for cosmological likelihood calculations,
including:
- Parameter conversions (omega_m to omega_cdm, S8 to sigma8, etc.)
- Data transformations and binning
- Covariance matrix handling
- Spectrum processing (binning, selection, standardization)
- Density space conversions

All functions are designed to work with both NumPy and JAX backends.
"""

import logging
import pickle

import h5py
import numpy as np
from scipy import integrate, interpolate, stats
from scipy.io import FortranFile

from UPanda.module import *

logger = logging.getLogger(__name__)

is_jax = config.get_jax_enabled()
if is_jax:
    pass  # JAX-specific imports handled elsewhere


from ._numerics import gaussian_loglike  # Backward-compatible utility export


def convert_om_to_ocdm(emu_dict):
    """Convert omega_m to omega_cdm in parameter dictionary.

    Computes omega_cdm = omega_m * h^2 - omega_nu - omega_b, where
    omega_nu is computed from m_nu assuming m_nu_tot = 3 * m_ncdm.

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters.
            Must include 'omega_m', 'omega_b', 'h', and 'm_nu'.

    Returns:
        dict: Updated dictionary with 'omega_cdm' added.
    """
    # Note omega_m is Capital Om!
    # Specific to AR set-up where m_nu_tot = 3*m_ncdm
    omega_m = array(emu_dict["omega_m"])
    omega_b = array(emu_dict["omega_b"])
    h = array(emu_dict["h"])
    m_nu = array(emu_dict["m_nu"])

    # Om = Onu + Oc + Ob
    o_nu = m_nu * 3 / 93.14

    omega_cdm = (omega_m * h * h) - o_nu - omega_b

    # remove omega_m
    # emu_dict.pop("omega_m", None)

    emu_dict["omega_cdm"] = omega_cdm

    return emu_dict


def get_sigma8(emu_dict, converter_model):
    """Compute sigma8 from cosmological parameters using a converter model.

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters.
        converter_model: Trained model that predicts sigma8 from parameters.

    Returns:
        dict: Updated dictionary with 'sigma8' added.
    """
    sigma8 = converter_model.predict(emu_dict)[:, 0]
    emu_dict["sigma8"] = sigma8
    return emu_dict


def get_theta_s_and_As(emu_dict, converter_model):
    """Convert (h, sigma8) to (theta_s_100, ln10^{10}A_s) using a trained converter.

    This is the REVERSE conversion: from MCMC sampling basis to CMB emulator basis.
    Use this when your MCMC samples in (h, sigma8) but your CMB emulators
    were trained with (theta_s_100, ln10^{10}A_s).

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters including
            'h' and 'sigma8'.
        converter_model: Trained model that predicts (theta_s_100, ln10^{10}A_s)
            from parameters. Output shape should be (n_samples, 2).

    Returns:
        dict: Updated dictionary with 'theta_s_100' and 'ln10^{10}A_s' added.
    """
    converted = converter_model.predict(emu_dict)
    emu_dict["theta_s_100"] = converted[:, 0]
    emu_dict["ln10^{10}A_s"] = converted[:, 1]
    return emu_dict


def get_As_from_sigma8(emu_dict, converter_model):
    """Convert sigma8 to ln10^{10}A_s using a trained converter.

    Use this when MCMC samples in sigma8 but CMB emulators expect ln10^{10}A_s.

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters including
            'sigma8'.
        converter_model: Trained model that predicts ln10^{10}A_s from
            parameters. Output shape should be (n_samples, 1).

    Returns:
        dict: Updated dictionary with 'ln10^{10}A_s' added.
    """
    converted = converter_model.predict(emu_dict)
    emu_dict["ln10^{10}A_s"] = converted[:, 0]
    return emu_dict


def get_theta_s_100(emu_dict, converter_model):
    """Compute theta_s_100 from (h, sigma8, ...) using a trained converter.

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters including 'h'.
        converter_model: Trained model that predicts theta_s_100 (and optionally A_s).

    Returns:
        dict: Updated dictionary with 'theta_s_100' added.
    """
    converted = converter_model.predict(emu_dict)
    emu_dict["theta_s_100"] = converted[:, 0]
    return emu_dict


def get_ln_As(emu_dict, converter_model):
    """Compute ln10^{10}A_s from (h, sigma8, ...) using a trained converter.

    Args:
        emu_dict (dict): Dictionary containing cosmological parameters including 'sigma8'.
        converter_model: Trained model that predicts ln10^{10}A_s (second output).

    Returns:
        dict: Updated dictionary with 'ln10^{10}A_s' added.
    """
    converted = converter_model.predict(emu_dict)
    emu_dict["ln10^{10}A_s"] = converted[:, 1]
    return emu_dict


def convert_S8_ocdm_to_sigma8_Om(emu_dict):
    """Convert S8 and omega_cdm to sigma8 and omega_m.

    Computes omega_m from omega_cdm, omega_b, and omega_nu, then converts
    S8 to sigma8 using the relation sigma8 = S8 * sqrt(0.3 / omega_m).

    Args:
        emu_dict (dict): Dictionary containing 'S8', 'omega_cdm', 'omega_b',
            'h', and 'm_nu'.

    Returns:
        dict: Updated dictionary with 'sigma8' and 'omega_m' added.
    """
    omega_b = array(emu_dict["omega_b"])
    h = array(emu_dict["h"])
    m_nu = array(emu_dict["m_nu"])
    o_nu = m_nu * 3 / 93.14

    S8 = array(emu_dict["S8"])
    omega_cdm = array(emu_dict["omega_cdm"])

    omega_m = (omega_cdm + omega_b + o_nu) / h / h
    sigma8 = S8 * sqrt(0.3 / omega_m)

    emu_dict["omega_m"] = omega_m
    emu_dict["sigma8"] = sigma8

    return emu_dict


def transform_to_dict(p, dtype, cosmology_params_fixed):
    """Transform MCMC sampler output to parameter dictionary format.

    Converts structured array from emcee sampler to dictionary format
    expected by emulator models, including fixed parameters.

    Args:
        p (ndarray): Parameter array from sampler with shape (n_samples, n_params)
            or structured array with shape (n_samples,).
        dtype (dtype): Structured dtype with parameter names.
        cosmology_params_fixed (list, optional): List of fixed parameter
            dictionaries with 'name' and 'value' keys.

    Returns:
        dict: Dictionary mapping parameter names to arrays of values.
    """
    # put the p in the emcee sampler into cosmopower dictionary form
    params_dict = {}

    # Handle structured arrays (shape (n_samples,)) vs regular arrays (shape (n_samples, n_params))
    if dtype.names is not None and p.dtype.names is not None:
        # Structured array - access fields by name
        for name in dtype.names:
            params_dict[name] = p[name]
        n_samples = len(p)
    else:
        # Regular array - access by column index
        for i, name in enumerate(dtype.names):
            params_dict[name] = p[:, i]
        n_samples = p.shape[0]

    if cosmology_params_fixed is not None:
        for item in cosmology_params_fixed:
            params_dict[item["name"]] = ones((n_samples,)) * item["value"]

    return params_dict


def prepare_params_dict(p, dtype, cosmology_params_fixed, additional_args=None):
    """Transform parameters and apply all necessary conversions.

    This is a convenience function that combines transform_to_dict with
    all standard parameter conversions:
    - omega_m -> omega_cdm (if omega_m is sampled but omega_cdm needed)
    - (h, sigma8) -> (theta_s_100, ln10^{10}A_s) for CMB emulators (need_theta_s)
    - (theta_s, A_s) -> sigma8 for LSS emulators (need_sigma8)

    Args:
        p (ndarray): Parameter array from sampler with shape (n_samples, n_params)
            or structured array with shape (n_samples,).
        dtype (dtype): Structured dtype with parameter names.
        cosmology_params_fixed (list, optional): List of fixed parameter
            dictionaries with 'name' and 'value' keys.
        additional_args (dict, optional): Dictionary with conversion flags:
            - 'need_theta_s': bool, if True converts (h, sigma8) -> (theta_s_100, A_s)
            - 'converter_h_to_theta_s': converter model for h/sigma8 -> theta_s/A_s
            - 'need_sigma8': bool, if True converts (theta_s, A_s) -> sigma8
            - 'converter_model': converter model for theta_s/A_s -> sigma8

    Returns:
        dict: Parameter dictionary with all conversions applied.
    """
    params_dict = transform_to_dict(p, dtype, cosmology_params_fixed)

    # Standard conversion: omega_m -> omega_cdm (if omega_m sampled but omega_cdm needed)
    if "omega_cdm" not in params_dict and "omega_m" in params_dict:
        params_dict = convert_om_to_ocdm(params_dict)

    if additional_args is not None:
        # Convert (h, sigma8) -> (theta_s_100, ln10^{10}A_s) for CMB emulators
        # Use this when MCMC samples in (h, sigma8) but CMB emulators expect (theta_s, A_s)
        if additional_args.get("need_theta_s", False):
            converter = additional_args.get("converter_h_to_theta_s")
            if converter is not None:
                params_dict = get_theta_s_and_As(params_dict, converter)

        # Convert sigma8 -> ln10^{10}A_s for CMB emulators
        # Use this when MCMC samples in sigma8 but CMB emulators expect ln10^{10}A_s
        if additional_args.get("need_sigma8_to_As", False):
            converter = additional_args.get("converter_sigma8_to_As")
            if converter is not None:
                params_dict = get_As_from_sigma8(params_dict, converter)

        # Convert (theta_s, A_s) -> sigma8 for LSS emulators (existing pattern)
        # Use this when MCMC samples in (theta_s, A_s) but LSS emulators expect sigma8
        if additional_args.get("need_sigma8", False):
            converter = additional_args.get("converter_model")
            if converter is not None:
                params_dict = get_sigma8(params_dict, converter)

    return params_dict


def load_covmat_fortran(cov_file, nbin):
    """Load a covariance matrix from a Fortran binary file.

    Reads a Fortran-order flat array and reshapes it to a symmetric matrix,
    mirroring the upper triangle onto the lower.

    Args:
        cov_file (str): Path to the Fortran binary file.
        nbin (int): Number of bins (matrix will be ``nbin x nbin``).

    Returns:
        ndarray: Symmetric covariance matrix of shape ``(nbin, nbin)``.
    """
    f = FortranFile(cov_file, "r")
    covmat = f.read_reals(dtype=float).reshape((nbin, nbin))
    for i in range(nbin):
        for j in range(i, nbin):
            covmat[i, j] = covmat[j, i]

    return covmat


def lognormal(x, mu, sig, loc=0):
    """Evaluate the log-normal probability density function.

    Args:
        x (ndarray): Values at which to evaluate the PDF.
        mu (float): Mean of the underlying normal distribution (log-space).
        sig (float): Standard deviation of the underlying normal distribution.
        loc (float, optional): Location parameter (shift). Defaults to 0.

    Returns:
        ndarray: Log-normal PDF evaluated at ``x``.
    """
    LN = (
        1
        / ((x - loc) * sig * sqrt(2 * pi))
        * exp(-((log(x - loc) - mu) ** 2) / (2 * sig**2))
    )

    return LN


def log_lognormal(x, mu, sig, loc=0):
    """Numerically stable natural log of the lognormal PDF (see :func:`lognormal`).

    Computes ``log LN`` directly instead of ``log(exp(...))``. The old
    ``log(lognormal(...))`` path underflows to ``log(0) = -inf`` for small
    ``x - loc`` and returns ``NaN`` for ``x <= loc`` (``log`` of a negative
    number). Across a wide sampling prior those out-of-range bandpowers make the
    Planck low-ℓ EE/TT term ``-inf``/``NaN`` almost everywhere, which turns the
    whole prior box into a flat ``-inf`` plateau and starves gradient-free nested
    samplers (nautilus collapses to the -1e9 floor). Here ``x - loc`` is clamped
    to a tiny positive value so the result is always finite; for the physical
    region (``x - loc > 0``) it is numerically identical to
    ``log(lognormal(x, mu, sig, loc))``.
    """
    d = where(x - loc > 1e-30, x - loc, 1e-30)
    return -log(d) - log(sig * sqrt(2 * pi)) - (log(d) - mu) ** 2 / (2 * sig**2)


def get_binned_D_from_theory_Cls(ell, Cl, lmin_list, lmax_list):
    """Bin D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) into bandpowers.

    Converts C_ℓ to D_ℓ, averages within each bin defined by ``lmin_list``
    and ``lmax_list``, and returns the binned D_ℓ values.

    Args:
        ell (ndarray): Array of multipole moments, shape ``(n_ell,)``.
        Cl (ndarray): Power spectrum array, shape ``(n_samples, n_ell)``.
        lmin_list (array-like): Lower multipole edges for each bin.
        lmax_list (array-like): Upper multipole edges for each bin.

    Returns:
        ndarray: Binned D_ℓ values, shape ``(n_samples, n_bins)``.

    Note:
        Assumes ``ell`` starts at 2 (i.e., ``Cl[:, 0]`` corresponds to ℓ=2).
    """
    ell = asarray(ell)
    lower = asarray(lmin_list)
    upper = asarray(lmax_list)
    mask = (ell[newaxis, :] >= lower[:, newaxis]) & (ell[newaxis, :] <= upper[:, newaxis])
    weights = mask / sum(mask, axis=1, keepdims=True)
    dl = asarray(Cl) * ell * (ell + 1) / (2 * pi)
    return matmul(dl, weights.T)


def get_binned_D_from_theory_Cls_tt(ell, Cl):
    """Bin TT D_ℓ into fixed Planck low-ℓ TT bandpowers (ℓ=2–29).

    Uses hardcoded bins [2–15] and [16–29], matching the Planck low-ℓ TT
    likelihood convention.

    Args:
        ell (ndarray): Array of multipole moments, shape ``(n_ell,)``.
        Cl (ndarray): TT power spectrum array, shape ``(n_samples, n_ell)``.

    Returns:
        ndarray: Binned D_ℓ values, shape ``(n_samples, 2)``.
    """
    lmin_list = [2, 16]
    lmax_list = [15, 29]

    D_fac = ell * (ell + 1) / (2 * pi)
    Dl = D_fac * Cl

    Dl_bin = zeros((Cl.shape[0], len(lmin_list)))

    for i, lmin in enumerate(lmin_list):
        lmax = lmax_list[i]

        if is_jax:
            Dl_bin = Dl_bin.at[:, i].set(mean(Dl[:, lmin - 2 : lmax - 2 + 1], axis=1))

        else:
            Dl_bin[:, i] = mean(Dl[:, int(lmin) - 2 : int(lmax) - 2 + 1], axis=1)

    return Dl_bin


def planck_lowE_binned_loglike(
    Cl_theory, mu_LN_EE, sig_LN_EE, loc_LN_EE, lmin_list_EE, lmax_list_EE
):
    """Compute Planck low-ℓ EE log-likelihood using lognormal bandpower distributions.

    Bins the theory C_ℓ^EE into bandpowers and evaluates the log-likelihood
    using pre-computed lognormal distribution parameters (from Planck's low-ℓ
    EE likelihood).

    Args:
        Cl_theory (ndarray): Theory EE power spectrum, shape ``(n_samples, n_ell)``,
            for ℓ=2–29.
        mu_LN_EE (array-like): Lognormal mean parameters, one per bin.
        sig_LN_EE (array-like): Lognormal sigma parameters, one per bin.
        loc_LN_EE (array-like): Lognormal location parameters, one per bin.
        lmin_list_EE (array-like): Lower ℓ edges for each bin.
        lmax_list_EE (array-like): Upper ℓ edges for each bin.

    Returns:
        ndarray: Log-likelihood values, shape ``(n_samples,)``.
    """
    ell = arange(2, 30)
    Dl_theory_bin = get_binned_D_from_theory_Cls(
        ell, Cl_theory, lmin_list_EE, lmax_list_EE
    )
    loglike = zeros((Cl_theory.shape[0],), dtype=float64)

    for i in range(Dl_theory_bin.shape[1]):
        D = Dl_theory_bin[:, i]
        loglike += log_lognormal(D, mu_LN_EE[i], sig_LN_EE[i], loc_LN_EE[i])
    return loglike


def planck_lowT_binned_loglike(
    Cl_theory, mu_LN_TT, sig_LN_TT, lmin_list_TT, lmax_list_TT
):
    """Compute Planck low-ℓ TT log-likelihood using lognormal bandpower distributions.

    Uses the fixed TT bins [2–15] and [16–29] and evaluates the log-likelihood
    against pre-computed lognormal distribution parameters.

    Args:
        Cl_theory (ndarray): Theory TT power spectrum, shape ``(n_samples, n_ell)``,
            for ℓ=2–29.
        mu_LN_TT (array-like): Lognormal mean parameters for each TT bin.
        sig_LN_TT (array-like): Lognormal sigma parameters for each TT bin.
        lmin_list_TT (array-like): Lower ℓ edges (currently unused; bins are fixed).
        lmax_list_TT (array-like): Upper ℓ edges (currently unused; bins are fixed).

    Returns:
        ndarray: Log-likelihood values, shape ``(n_samples,)``.
    """
    ell = arange(2, 30)
    Dl_theory_bin = get_binned_D_from_theory_Cls_tt(ell, Cl_theory)
    loglike = zeros((Cl_theory.shape[0],), dtype=float64)
    for i in range(Dl_theory_bin.shape[1]):
        D = Dl_theory_bin[:, i]
        loglike += log_lognormal(D, mu_LN_TT[i], sig_LN_TT[i])
    return loglike


def select_indices(arr, indices_list):
    """Select a subset of columns from an emulator output array.

    Args:
        arr (ndarray): Array of shape ``(n_samples, n_features)``.
        indices_list (array-like): Column indices to select.

    Returns:
        ndarray: Subset array of shape ``(n_samples, len(indices_list))``.
    """
    new_arr = arr[:, indices_list]

    return new_arr


def create_cov_lss(spectra, ufalcon_database, ufalcon_database_2, hartlap=True):
    """Compute an LSS covariance matrix from simulation realisations in HDF5 files.

    Reads C_ℓ realisations for each spectrum from one or two UFalcon simulation
    databases, concatenates them, removes any zero realisations, and computes
    the sample covariance matrix. Optionally applies the Hartlap correction
    to account for noise from a finite number of simulations.

    Args:
        spectra (list of str): List of spectrum keys to include (e.g.
            ``['kk', 'kg_1']``). The ``'kk'`` spectrum is rescaled by
            ``2/π`` before covariance estimation.
        ufalcon_database (str): Path to the primary HDF5 simulation database.
        ufalcon_database_2 (str or None): Path to a secondary HDF5 database,
            or ``None`` if only one database is used.
        hartlap (bool, optional): If True, apply the Hartlap et al. (2007)
            correction factor to the covariance matrix. Defaults to True.

    Returns:
        ndarray: Covariance matrix of shape ``(n_data, n_data)``, where
            ``n_data`` is the total number of data points across all spectra.
    """
    with h5py.File(ufalcon_database, mode="r") as f:

        # get the number of realisations from the database
        num_real = f[f"{spectra[0]}"].shape[0]

        cl_realisations = []
        for index in range(num_real):
            cl_realisation = []
            for spec in spectra:
                if spec == "kk":
                    cl_realisation.extend(f[spec][index] * 2 / pi)
                else:
                    cl_realisation.extend(f[spec][index])

            cl_realisations.append(cl_realisation)

    if ufalcon_database_2 is not None:
        with h5py.File(ufalcon_database_2, mode="r") as f:
            # get the number of realisations from the database
            num_real_2 = f[f"{spectra[0]}"].shape[0]
            num_real += num_real_2

            for index in range(num_real_2):
                cl_realisation = []
                for spec in spectra:
                    if spec == "kk":
                        cl_realisation.extend(f[spec][index] * 2 / pi)
                    else:
                        cl_realisation.extend(f[spec][index])

                cl_realisations.append(cl_realisation)

    cl_tot = array(cl_realisations)

    # idx = argwhere(all(abs(cl_tot[:, ...]) == 0, axis=1))
    # idx anywhere the realizations are exactly zero in each row this should never be the case
    idx = np.flatnonzero(np.all(np.asarray(cl_tot) == 0, axis=1))

    num_real -= idx.shape[0]

    logger.info("Number of realisations after removing zero realisations: %d", num_real)
    if num_real < 1000:
        logger.warning(
            "Only %d realisations — consider increasing for a well-converged covariance matrix",
            num_real,
        )
    cl_tot = delete(cl_tot, idx, axis=0)

    if num_real < 2:
        raise ValueError("At least two nonzero realizations are required")
    covmat = np.atleast_2d(np.cov(np.asarray(cl_tot).T))

    if hartlap:
        n_sims = num_real
        if n_sims <= covmat.shape[0] + 2:
            raise ValueError("Hartlap correction requires n_sims > n_data + 2")
        fac_hartlap = (n_sims - covmat.shape[0] - 2) / (n_sims - 1)
        covmat /= fac_hartlap  # divide the cov by this so that when the inverse is carried out this is accounted for

    return covmat


def create_datavector(spectra, data_vector_database, mc_norm_file=None):
    """Load an observed data vector from an HDF5 file with optional MC normalisation.

    Reads C_ℓ measurements for each spectrum from an HDF5 file. For
    lensing cross-spectra, applies an optional Monte-Carlo normalisation
    correction loaded from a pickle file.

    Args:
        spectra (list of str): List of spectrum keys to include (e.g.
            ``['kg_1', 'dd_11']``). The ``'kk'`` and ``'kaka'`` spectra
            are skipped (convergence power spectra are not added).
        data_vector_database (str): Path to the HDF5 file containing
            the observed C_ℓ values.
        mc_norm_file (str or None, optional): Path to a pickle file
            containing a dict of MC normalisation corrections keyed by
            spectrum name. Applied to lensing cross-spectra only.
            Defaults to None.

    Returns:
        ndarray: Concatenated observed data vector as a 1D NumPy array.
    """
    cl_obs = []
    lensing_cross_spectra = [
        "kt",
        "lk",
        "ck",
        "kg_1",
        "kg_2",
        "kg_3",
        "kg_4",
        "kg_5",
    ] + [
        "lka",
        "cka",
        "kag_1",
        "kag_2",
        "kag_3",
        "kag_4",
        "kag_5",
        "kad_1",
        "kad_2",
        "kad_3",
        "kad_4",
    ]

    # Open the data vector HDF5 file
    with h5py.File(data_vector_database, mode="r") as data_file:
        for spec in spectra:
            if spec not in ["kk", "kaka"]:

                if spec in lensing_cross_spectra and mc_norm_file is not None:
                    # Load pickle file with mcnorm correction
                    with open(mc_norm_file, "rb") as mc_file:
                        mcnorm = pickle.load(mc_file)

                    # Access data_file for the spectrum and apply mcnorm correction
                    cl_obs.extend(data_file[spec][:] * mcnorm[spec])
                    logger.debug("Applying MC normalisation correction to spectrum: %s", spec)

                else:
                    # No mcnorm correction, just access the data_file
                    cl_obs.extend(data_file[spec])
            else:
                logger.debug("Skipping spectrum '%s' (not added to data vector)", spec)

    cl_obs = np.array(cl_obs)

    return cl_obs


def bin_spectrum(ells, cells, bin_edges, ell_weighted=False):
    """Bin a power spectrum into bandpowers.

    Args:
        ells (ndarray): Array of multipole moments.
        cells (ndarray): Power spectrum values at each ℓ.
        bin_edges (array-like): Bin edge values defining the bandpowers.
        ell_weighted (bool, optional): If True, compute ℓ-weighted mean
            (sum(ℓ·C_ℓ)/sum(ℓ)). If False, compute unweighted mean.
            Defaults to False.

    Returns:
        ndarray: Binned power spectrum values, one per bin.
    """
    if ell_weighted:
        sums = stats.binned_statistic(ells, ells, statistic="sum", bins=bin_edges)
        cl = stats.binned_statistic(ells, ells * cells, statistic="sum", bins=bin_edges)
        cl = cl[0] / sums[0]
    else:
        cl = stats.binned_statistic(ells, cells, statistic="mean", bins=bin_edges)[0]

    return cl


def select_from_matrix(mat, selectionA, selectionB=None):
    """Select a sub-matrix using boolean selection arrays.

    Args:
        mat (ndarray): Square or rectangular matrix to select from.
        selectionA (ndarray): Boolean array for row selection.
        selectionB (ndarray or None, optional): Boolean array for column
            selection. If None, uses ``selectionA``. Defaults to None.

    Returns:
        ndarray: Selected sub-matrix.

    Raises:
        AssertionError: If ``mat.shape`` does not match
            ``(len(selectionA), len(selectionB))``.
    """
    if selectionB is None:
        selectionB = selectionA
    assert mat.shape == (len(selectionA), len(selectionB))
    selection_matrix = np.outer(selectionA, selectionB)

    return mat[np.where(selection_matrix)].reshape(
        np.sum(selectionA), np.sum(selectionB)
    )


def standardize_input_cl(ells, cells, lmax=None, lmin=None):
    """Standardize a power spectrum onto a contiguous ℓ grid.

    Places the input C_ℓ values at the correct ℓ positions in a zero-padded
    output array covering ``[lmin, lmax]``.

    Args:
        ells (ndarray): Integer multipole moments for the input spectrum.
        cells (ndarray): Power spectrum values corresponding to ``ells``.
        lmax (int or None, optional): Maximum ℓ for output array.
            Defaults to ``max(ells)``.
        lmin (int or None, optional): Minimum ℓ for output array.
            Defaults to 0.

    Returns:
        ndarray: Power spectrum on a contiguous grid of length
            ``lmax - lmin + 1``, zero-padded where no data is given.

    Raises:
        AssertionError: If ``ells`` contains non-integer values.
    """
    if lmax is None:
        lmax = int(np.max(ells))
    if lmin is None:
        lmin = 0

    assert np.all(ells % 1 == 0), "Input ells must be integers."
    ells = ells.astype(int)

    out_cells = np.zeros(lmax - lmin + 1)
    out_cells[ells[(lmin <= ells) & (ells <= lmax)] - lmin] = cells[
        (lmin <= ells) & (ells <= lmax)
    ]

    return out_cells


def density_space(xs, ps, n, endpoint=False, order=1, random=False):
    """Draw samples with spacing specified by a density function.

    Copyright Han-Kwang Nienhuys (2020).
    License: any of: CC-BY, CC-BY-SA, BSD, LGPL, GPL.
    Reference: https://stackoverflow.com/a/62740029/6228891

    Parameters:

    - xs: array, ordered by increasing values.
    - ps: array, corresponding densities (not normalized).
    - n: number of output values.
    - endpoint: whether to include x[-1] in the output.
    - order: interpolation order (1 or 2). Order 2 will
      require dense sampling and a smoothly varying density
      to work correctly.
    - random: whether to return random samples, ignoring endpoint).
      in this case, n can be a shape tuple.

    Return:

    - array, shape (n,), with values from xs[0] to xs[-1]
    """

    # Use cumulative_trapezoid for newer scipy, fallback to cumtrapz
    if hasattr(integrate, "cumulative_trapezoid"):
        cps = integrate.cumulative_trapezoid(ps, xs, initial=0)
    else:
        cps = integrate.cumtrapz(ps, xs, initial=0)
    cps *= 1 / cps[-1]
    intfunc = interpolate.interp1d(cps, xs, kind=order, fill_value="extrapolate")
    if random:
        return intfunc(np.random.uniform(size=n))
    else:
        return intfunc(np.linspace(0, 1, n, endpoint=endpoint))


def combine_inputs(*arrays, sorted=True):
    """Combine and deduplicate multiple arrays of values.

    Args:
        *arrays: Variable number of arrays to combine.
        sorted (bool, optional): If True, return sorted unique values.
            If False, return unique values in their first occurrence order.
            Defaults to True.

    Returns:
        ndarray: Combined unique values.
    """
    if sorted:
        return np.sort(np.unique(np.concatenate(arrays)))
    else:
        return np.unique(np.concatenate(arrays))


def evaluate_pk_kmax(pk_interp, z, k, kmax=None, log=False):
    """Evaluate a matter power spectrum interpolator, masking modes above k_max.

    Modes with ``k > kmax`` are set to NaN rather than extrapolated.

    Args:
        pk_interp: Power spectrum interpolator with ``.P(z, k, grid=False)``
            and ``.logP(z, k, grid=False)`` methods (e.g. from CAMB).
        z (ndarray): Redshifts, shape ``(n_k,)`` — one per k-mode.
        k (ndarray): Wavenumbers in h/Mpc, shape ``(n_k,)``.
        kmax (float or None, optional): Maximum wavenumber. Values above this
            are set to NaN. Defaults to ``pk_interp.kmax``.
        log (bool, optional): If True, use the log interpolator (``logP``).
            Defaults to False.

    Returns:
        ndarray: Power spectrum values, shape ``(n_k,)``.
            NaN where ``k > kmax``.
    """
    k = np.asarray(k)
    z = np.asarray(z)
    out = np.full(k.shape, np.nan, dtype=float)
    if kmax is None:
        kmax = pk_interp.kmax
    k_sel = k <= kmax
    if log:
        out[k_sel] = pk_interp.logP(z[np.where(k_sel)[0]], k[k_sel], grid=False)
    else:
        out[k_sel] = pk_interp.P(z[np.where(k_sel)[0]], k[k_sel], grid=False)

    return out


def compute_act_lensing_auto_corrections(self, cl_TEB, clkk):
    """
    Compute normalization and N1 corrections for the CMB lensing power spectrum.

    Parameters:
    -----------
    cl_TEB : array_like
        CMB power spectra [TT, EE, BB, TE] with shape (batch, 4, lmax+1)
    clkk : array_like
        Theoretical lensing power spectrum

    Returns:
    --------
    clkk_corrected : array_like
        Corrected lensing power spectrum
    """
    # Upstream convention: derivative rows index output L, columns input ell.
    # https://github.com/ACTCollaboration/act_dr6_lenslike/blob/main/act_dr6_lenslike/act_dr6_lenslike.py
    clkk = asarray(clkk)
    cl_TEB = asarray(cl_TEB)

    def fit_length(values, length):
        values = asarray(values)[..., :length]
        widths = [(0, 0)] * values.ndim
        widths[-1] = (0, max(0, length - values.shape[-1]))
        return pad(values, widths)

    norm_length = self.dNorm_dCl.shape[-1]
    diff_norm = (fit_length(cl_TEB, norm_length)
                 - fit_length(self.fiducial_cls, norm_length))
    norm_raw = -einsum("tLl,btl->bL", self.dNorm_dCl, diff_norm)
    norm_fid = fit_length(self.norm_fid, norm_raw.shape[-1])
    # Unobservable modes can have zero fiducial normalization. Avoid infinities
    # and undefined derivatives even under JIT, which evaluates both branches.
    norm = where(norm_fid != 0, norm_raw / where(norm_fid != 0, norm_fid, 1), 0)
    norm *= 2 * fit_length(self.clkk_fid, norm.shape[-1])

    cmb_length = self.dN1_dCl.shape[-1]
    diff_cmb = (fit_length(cl_TEB, cmb_length)
                - fit_length(self.fiducial_cls, cmb_length))
    n1_cmb = einsum("tLl,btl->bL", self.dN1_dCl, diff_cmb)
    kk_length = self.dN1_dClkk.shape[-1]
    diff_kk = fit_length(clkk, kk_length) - fit_length(self.clkk_fid, kk_length)
    n1_kk = einsum("ij,bj->bi", self.dN1_dClkk, diff_kk)

    output_length = max(clkk.shape[-1], self.dN1_dClkk.shape[0])
    return (fit_length(clkk, output_length) + fit_length(norm, output_length)
            + fit_length(n1_cmb, output_length) + fit_length(n1_kk, output_length))
