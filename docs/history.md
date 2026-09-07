# History

## 0.2.0 — September 2026

- Add the Union3.1/UNITY1.8 compressed-distance option and bundle its 22-bin data.
- Add Union3 binned supernova distances and BOSS/eBOSS DR16 joint BAO+RSD and
  approximate f-sigma8-only likelihoods, including data and source links.
- Preserve public calls and historical emulator metadata layouts used by
  `multiprobe_framework`.
- Fix ACT N1 correction contraction, fiducial multipole indexing, spectrum
  padding and batch handling. Preserve the trained emulator's normalization.
- Restore JAX conversion of assembled LSS covariance matrices before replacing
  official Planck or ACT lensing blocks; this fixes default ACT 5x2pt startup.
- Fix covariance marginalization/compression, zero-weight handling, JAX binning
  and foreground paths, immutable TensorFlow bias rescaling, and prior ordering.
- Correct WMAP off-diagonal variance products, SZ template precision and JAX
  diagonal updates; compare default TT/TE components directly with pyWMAP.
- Expand numerical, backend, external-theory, persistence and gradient tests;
  add reproducible comparisons with upstream likelihood code and a real emulator.
- Document data provenance, numerical conventions and remaining validation gaps.
- Consolidate packaging metadata, explicitly include bundled survey data, and
  add strict documentation/build checks and full-package coverage reporting.

Some fixes change posterior values. Read the [likelihood audit](likelihood_audit.md)
for the affected paths and scientific validation limits. This release does not
claim complete test coverage or complete agreement with every upstream survey mode.

## 0.1.0

Initial package: cosmological likelihoods, integrated emulators, multiple sampler
backends and NumPy/JAX support. Earlier work introduced the MkDocs documentation.
