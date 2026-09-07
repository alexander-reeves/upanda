# Compatibility review

The September 2026 review uses actual imports and calls in `multiprobe_framework`,
`mi_project`, and `multiprobe_desi_act_isw` as the compatibility boundary.
Those applications import lowercase likelihood classes directly, call
`get_mcmc_start_and_fixed_params`/`run_mcmc`, restore integrated models, inject
external theory, and sometimes access covariance/data attributes and paths.

The [likelihood accuracy audit](likelihood_audit.md) records the later numerical
fixes, upstream comparisons and the traced ACT emulator convention.

## Changes and regression coverage

| Finding | Change / compatibility consequence |
| --- | --- |
| DESI constructors dereferenced `additional_args=None`; synthetic inputs still read a survey file | Normalize optional options and bypass survey reads for synthetic data. Existing constructor keywords remain valid. |
| Repeated Gaussian products built a batch-by-batch matrix and failed for a single vector | Shared private `_numerics.gaussian_loglike`, exported through `utils_likelihood`; preserve batched values and add scalar support. |
| `log L == 0` was interpreted as outside the prior | Use the actual bound mask; an exact Gaussian fit remains valid. NumPy evaluation no longer references a conditionally imported `jax`. |
| Fixed parameter names, starting-point columns and prior indices could disagree | All helpers use the same sampled parameter order. This changes only the inconsistent fixed/variable overlap case. |
| Duplicate grouped prior declarations shifted later means/covariances | Deduplicate whole prior tuples, accept missing/`uniform` flat distributions. |
| Scalar sampling distributions returned a 1D array and crashed | Normalize sample shape to `(n_samples, n_parameters)`. |
| Backend switching refreshed only two modules | Refresh imported backend symbols without recreating likelihood classes. Existing compiled functions/stored arrays still require recreation after a switch. |
| Provider required TT even for polarization-only spectra | Use an available spectrum and compute multipole factors per spectrum. |
| Spectrum gridding used absolute ell as an index with nonzero lmin | Subtract lmin when assigning to the output grid. |
| Dovekie cut a precision sub-block, conditioning on removed observations | Use the Schur complement to marginalize dropped data; checked against inversion of the retained covariance. If no observations are cut, values are unchanged. |
| CamSpec string range parser failed on isolated multipoles | Treat single multipoles as one-element arrays; preserve inclusive ranges. |
| BOSS sample generators were consumed during validation | Materialize the iterable once and reject empty selections. |
| Test fixtures changed global path functions; sanity script loaded data during collection | Scope reader mocks to individual likelihoods; move manual calculation under `main`. |
| README advertised nonexistent downloads/CLI; metadata conflicted | Document real interfaces, use `pyproject.toml` as metadata authority, include explicit package data, simplify Make/CI commands. |

## Structure

Public modules stay in place. `likelihoods/likelihood.py` is the shared base;
`likelihoods/_numerics.py` contains experiment-independent Gaussian operations;
`utils_likelihood.py` keeps parameter and data helpers with compatibility exports.
Data ordering, calibration and cuts stay in the experiment modules. This permits
small independently reviewable extractions without imposing a new application API.

Further work should split sampler implementations behind `run_panda.run_mcmc`
and move data provisioning into a versioned manifest once the actual upstream
assets and derived products have verified provenance. Avoid a wholesale rewrite
or moving all heavy dependencies into optional extras in a compatibility patch.

## Known scientific and integration limits

The current Pantheon+ `use_shoes` keyword does not activate a calibrated likelihood.
The BOSS fσ8-only class deliberately uses a diagonal covariance approximation.
The existing low-ell lognormal floor and nonfinite sampler penalties are preserved;
these conventions should be reviewed separately before changing posterior support.
Large WMAP/HiLLiPoP/CamSpec/reference-data analyses and production posterior chains
are not validated by small synthetic tests. Coverage is reported for the full
package so those gaps remain visible.

`multiprobe_framework` also references `UPanda.integrated_model_cnn`, which is
absent from this checkout. This predates the review; inventing a replacement
would not establish compatibility with the expected trained CNN artifacts.
No changes to the three downstream applications are required for the covered
public calls. Full end-to-end application runs also require their external theory
engines, datasets, emulator files, and cluster execution environments.
