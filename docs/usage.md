# API and compatibility contracts

## Imports and parameters

`UPanda` (capital U/P) exports `config`, `utils`, and `Provider`.
`from UPanda.likelihoods import desi_bao_dr2` returns a module;
`from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2` returns its class.
Those import forms and existing lowercase class names remain supported.

Likelihood constructors retain `data_vector`, `covariance_matrix`,
`emu_model_list`, `dtype`, `compression_vecs`, `cosmology_params_fixed`, and
`additional_args`. Not all subclasses use the generic data/covariance arguments:
use the subclass's documented data-loading or `mock_data`/`mock_cov` options.

Emulator evaluation takes a numeric array of shape `(n_samples, n_parameters)`
and a structured NumPy `dtype` naming columns in that exact order. Structured
arrays are also accepted by `transform_to_dict`. Fixed parameters are a list of
`{'name': ..., 'value': ...}` mappings and override matching sampled values.
`prepare_params_dict` adds the configured cosmological conversions.

The legacy name `omega_m` means **Omega_m**, while `omega_b` and `omega_cdm` mean
physical densities, Omega times h². The conversion assumes `m_nu` is a
**per-species mass in eV**, with total mass `3*m_nu`, and uses
`omega_nu = 3*m_nu/93.14`. Do not silently substitute a total neutrino mass.

An emulator supplies `predict(parameter_dictionary)` and returns rows of
observables. Both `integrated_model.IntegratedModel` and
`integrated_model_jax.IntegratedModel` retain their constructor and
`restore(filename)` interfaces. Restore using the artifact stem expected by that
implementation, with the matching model/scaler metadata; random or mismatched
emulator files cannot validate a physical likelihood.

## External theory

These are *theory predictions*, despite historical dictionary keys ending in
`_data`. Set them before `compute_like(None)`:

| Likelihood | `set_external_spectra` value | Shape/meaning |
| --- | --- | --- |
| `desi_bao_dr2` | `{'desi_bao_data': prediction}` | `(batch, n_data)` BAO observables in `mean.txt` order |
| `boss_dr16_rsd` | `{'boss_dr16_rsd_data': prediction}` | `(batch, n_data)` in `quantities`/`z_eff` order |
| `boss_dr16_fsigma8` | `{'boss_dr16_fsigma8_data': prediction}` | Six fσ8 values in `z_eff` order |
| `pantheon_plus`, `des_dovekie`, `union3` | prediction array directly | Magnitude/distance-modulus prediction in data order |
| `general_lss` | existing list of emulator-output arrays | Preserve the emulator-list layout, per-spectrum ordering and batch axes |

The listed BAO/RSD/SN Gaussian implementations also accept a single theory vector
and return a scalar; batch inputs return one log likelihood per row.
`desi_bao_2024` retains its emulator-only interface.

Supernova emulator outputs are **angular-diameter distances in Mpc**. UPanda
forms `5*log10((1+zhel)*(1+zcmb)*D_A)`. External predictions already contain that
transformation. A constant magnitude offset is projected out analytically.
The `use_shoes` argument on Pantheon+ is historical: this implementation still
uses the uncalibrated, magnitude-marginalized likelihood. It is not a SH0ES
calibration likelihood.

Gaussian results omit parameter-independent normalization constants. They are
suitable for parameter inference with fixed covariance; absolute evidence
comparisons across different data vectors require consistent normalization.
A best-fitting residual of zero has log likelihood zero and is valid.

## Provider and JAX state

`Provider.set_Cl` stores CMB **C_ell in microkelvin squared**, starting at ell=0.
`get_Cl(units='K2')` divides by 1e12; `units='FIRASmuK2'` preserves units.
`ell_factor=True` multiplies by ell(ell+1)/(2π). One-dimensional and batched
spectra work, including polarization-only dictionaries. Set fresh spectra via
`set_Cl` for every new theory calculation to invalidate the conversion cache.
Do not mutate the stored arrays/dictionary in place and expect cache invalidation.

Select the NumPy/JAX backend before constructing models. Refreshing backend
symbols keeps imported likelihood class identities, but does not convert
previously stored arrays, refresh external aliases imported from `UPanda.module`,
or invalidate an existing JIT cache. Recreate models and compiled callables when
switching. Global backend selection is process-wide, not a thread-local context.

The external-spectra and provider APIs are stateful. When JIT-compiling a combined
likelihood, inject theory inside the traced wrapper and evaluate all dependent
components there. Changing a Python object's attributes outside an already
compiled function does not make those values dynamic JAX inputs. CMB producers
must run before likelihoods that read their spectra from the shared provider.

## Sampling configuration

Call `UPanda.run_panda.run_mcmc` from a Python driver, as the downstream projects
do. `python -m UPanda.run_panda config.yaml` is not a supported CLI.

`get_mcmc_start_and_fixed_params`, `get_param_lims`, and `get_gaussian_priors`
use the same sampled column order: variable cosmology then nuisance parameters,
with fixed names excluded. Missing `distribution` means a flat prior;
`uniform` is accepted as an alias. Grouped Gaussian prior `indices` address that
sampled column order. Repeated declarations of a group are counted once.

`point_check` evaluates specified points without running a chain. This is useful
for downstream smoke tests. The ensemble branch uses strict open prior bounds;
points exactly on a bound are rejected. Existing nonfinite-value penalties are
preserved. Precision, prior edges, covariance cuts, and normalization conventions
should be recorded with every scientific result.
