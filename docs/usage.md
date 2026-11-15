# Usage

## Likelihood classes

Each likelihood is a class in its own module, with a lower-case name:

```python
from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2
```

The constructors share the arguments

- `emu_model_list`: the emulators that provide the theory. An emulator is any object with a
  `predict(parameters)` method, where `parameters` is a dictionary of arrays; `IntegratedModel` (TensorFlow) and
  `integrated_model_jax.IntegratedModel` (JAX) are the classes shipped with UPanda, and both have
  `restore(filename)`.
- `dtype`: a structured NumPy dtype naming the sampled parameters, in the order of the columns of the array
  given to `compute_like`.
- `cosmology_params_fixed`: a list of `{"name": ..., "value": ...}` for parameters that are held fixed.
- `additional_args`: a dictionary of options specific to the likelihood.

`compute_like(points)` takes an array of shape `(n_points, n_parameters)` and returns one log-likelihood per
point. Constant normalisation terms of the Gaussian likelihoods are left out.

Parameter conventions: `omega_b` and `omega_cdm` are physical densities (Ω h²), `omega_m` is Ω_m, and `m_nu` is
the mass of one neutrino species in eV (three degenerate species are assumed).

## Supplying the theory directly

Instead of an emulator, the theory prediction can be set by hand and the likelihood evaluated with
`compute_like(None)`:

| Likelihood | `set_external_spectra(...)` | Content |
| --- | --- | --- |
| `desi_bao_dr2` | `{"desi_bao_data": vector}` | BAO distances in the order of `mean.txt` |
| `boss_dr16_rsd` | `{"boss_dr16_rsd_data": vector}` | in the order of the attributes `quantities` and `z_eff` |
| `boss_dr16_fsigma8` | `{"boss_dr16_fsigma8_data": vector}` | fσ8 at `z_eff` |
| `pantheon_plus`, `union3`, `des_dovekie` | `vector` | 5 log10[(1 + z_hel)(1 + z_cmb) D_A / Mpc] at the attribute `zcmb` |
| `camspec_pr4`, `hillipop` | `{"tt": ..., "te": ..., "ee": ...}` | C_ℓ in K², from ℓ = 0 |
| `general_lss` | list of arrays | one per emulator, same layout as the emulator outputs |

A single vector gives a scalar; a two-dimensional array gives one value per row. The supernova likelihoods
marginalise analytically over a constant magnitude offset, so the normalisation of the distances does not matter.
With an emulator, the supernova likelihoods expect angular-diameter distances in Mpc and apply the conversion
above themselves.

## CMB spectra shared between likelihoods

`Provider.set_Cl` stores CMB C_ℓ in μK² from ℓ = 0, and `Provider.get_Cl` returns them to the likelihoods that
need them (for example the lensing likelihoods, for their CMB-dependent corrections). The CMB likelihood has to
be evaluated before the likelihoods that read its spectra.

## NumPy or JAX

```python
import jax
jax.config.update("jax_enable_x64", True)
from UPanda import config
config.set_jax_enabled(True)
```

Choose the backend before building likelihoods and emulators. After switching, build them again: arrays already
stored in existing objects are not converted. The setting is global to the process.

## Sampling

`UPanda.run_panda.run_mcmc` runs a chain for a list of likelihoods and emulators, with `sampler` one of `emcee`,
`nautilus`, `ultranest`, `hmc`, or the optimisers `adam` and `anneal`. Priors and starting points come from the
YAML configuration (see `configs/` for examples); a parameter without a `distribution` entry has a flat prior.
`point_check` evaluates the posterior at given points without running a chain.
