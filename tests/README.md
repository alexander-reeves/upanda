# Tests

Install `python -m pip install -e '.[dev]'`, then run `make test` or
`python -m pytest`. Tests must run on CPU without downloading datasets.

- `test_regressions.py`: independent Gaussian calculations, external theory,
  SN offset marginalization/covariance cuts, backend changes, and spectrum grids.
- `test_sampling_regressions.py`: fixed-parameter ordering, grouped priors,
  one-dimensional sampling, and real posterior evaluation with `point_check`.
- Existing tests cover provider caching, parameter conversion, likelihood setup,
  sampler wrappers and TensorFlow/JAX emulator components.
- `test_lensing_corrections.py` and `test_general_lss_contracts.py`: ACT matrix
  contractions, ell support, batch independence and downstream spectrum conventions.
- `test_cmb_contracts.py`, `test_camspec_synthetic.py`, `test_hillipop_contracts.py`,
  `test_polarization_tools.py`, `test_foregrounds.py`, and `test_wmap_numerics.py`:
  bundled and synthetic CMB data, calibration, covariance projection, NumPy/JAX
  agreement, JIT, WMAP Fisher matrices and stable Blackwell–Rao evaluation.
- `test_model_compatibility.py`: saved metadata formats and independently checked
  inverse transforms/gradients. `test_bundled_distances.py`: real bundled distance
  data, covariance validation and BAO/RSD block ordering.
Use `make coverage` for full-package branch coverage and `htmlcov/index.html`.
Inspect skips with `python -m pytest -rs`; never equate skipped tests with validation.
For a focused check: `python -m pytest tests/test_regressions.py`.

Set `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true` for numerical comparisons.
On a shared cluster also set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1` to limit thread use.
The `integration` marker identifies checks that require external survey data.
The `upstream` marker identifies direct NumPy comparisons against pinned upstream
implementations; these run in their own GitHub Actions job.
Unit tests use small synthetic data; they do not establish published-survey
likelihood agreement or emulator accuracy across a cosmological training domain.

The automated NumPy upstream comparisons live in `test_upstream_numpy.py`.
Additional real-data checks are reproducible scripts in `scripts/`:
`check_act_lensing_reference.py` accepts a real saved emulator and pinned ACT
source, and `check_wmap_reference.py` accepts a trusted pyWMAP checkout. See the
[audit](../docs/likelihood_audit.md) for exact scope and reference revisions.
CI enforces a 60% floor across the complete package, including branch coverage;
the floor is a regression guard, not the final coverage target.
