# History

## Unreleased

- eBOSS DR16 ELG likelihoods from the official SDSS files: the one-dimensional fσ8 table and the
  three-dimensional BAO + full-shape grid (`eboss_elg`), available through `boss_dr16_fsigma8` and
  `boss_dr16_rsd`.
- `boss_dr16_fsigma8` uses the published SDSS covariance by default; `additional_args={"diagonal": True}` gives the previous
  behaviour. `boss_dr16_rsd.drop_fsigma8()` marginalises fσ8 for distance-only analyses.
- `general_lss`: `act_lensing_range` selects the ACT DR6 lensing bandpowers (`all`, `baseline`, `extended`).
- `run_mcmc` (emcee): chains are written to an HDF5 file while running and can be resumed; walkers can be
  started from an existing chain.
- `camspec_pr4`: the covariance is inverted with NumPy, which removes a 40 minute JAX compilation on CPU.
- Tests against cobaya on the real data for DESI, SDSS, Pantheon+, Union3, DES-Dovekie, Planck 2018 plik-lite
  and lensing, and against pyactlike for ACT DR4 (see [Validation](validation.md)).

## 0.2.0 (September 2026)

- New likelihoods: Union3 and Union3.1, BOSS DR12 + eBOSS DR16 BAO and RSD.
- ACT DR6 lensing: corrected the contraction of the N1 correction and the multipole indexing of the fiducial
  spectrum.
- WMAP: corrected the off-diagonal Fisher matrices and the SZ template.
- HiLLiPoP, LoLLiPoP and CamSpec: fixes to covariance compression, binning and the JAX code paths.
- DES-Dovekie: redshift cuts now marginalise over the removed supernovae.
- Sampling: consistent parameter ordering with fixed parameters and grouped Gaussian priors.

Some of these fixes change likelihood values with respect to 0.1.0.

## 0.1.0

First version: likelihoods, emulator classes (TensorFlow and JAX), samplers.
