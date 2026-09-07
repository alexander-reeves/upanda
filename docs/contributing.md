# Development workflow

Create a dedicated Python 3.10–3.12 environment and install `python -m pip install
-e '.[dev]'`. The package's authoritative metadata and tool settings live in
`pyproject.toml`. `uv.lock` records the release dependency resolution and is
checked with `uv lock --check`.
Make targets use the active interpreter; they do not run `uv sync` or update
an active analysis environment.

Before submitting a change, run:

```bash
make style-check
make test
make coverage
make docs
make build
```

Use `PYTHON=/path/to/python` to select an interpreter. CI runs CPU tests on Python
3.10, 3.11, and 3.12. Documentation uses MkDocs, is built with `--strict`, and
is published to the `gh-pages` branch from `master`.

## Compatible changes

Preserve existing module paths, lowercase likelihood class names, constructor
keywords, `predict`, `restore`, `set_external_spectra`, and `compute_like`.
Add shared implementation details in private modules and retain compatibility
exports. Avoid moving dataset directories that downstream scripts access directly.
For bug fixes that change likelihood values, document the trigger and test against
an independently computed result, with a named scientific convention.

For new likelihoods, document data release/version, upstream code and data links,
observable ordering, redshifts, units, normalization, nuisance parameters,
calibration, cuts and covariance treatment. Never infer a data checksum or DOI.
Use synthetic positive-definite covariance fixtures for routine tests; separate
comparisons requiring large data/reference software with an `integration` marker.

## Test design

Use fixed random seeds, small deterministic emulators, analytic quadratic forms,
and NumPy linear solves as references. Check scalar/batch shapes and both
NumPy and JAX/JIT/gradient paths for modified numerical code. Missing a required
package import should fail CI; only genuinely optional dependencies/data justify
skips. Do not catch arbitrary exceptions and call them passes or expected bugs.

Patch module-local readers or an individual likelihood's `os` reference when
isolating data access. Patching `os.path.dirname` globally affects pytest and
other libraries and can cause recursion during failure reporting.

Coverage reports must include the full package, including large likelihoods that
are not fully exercised by synthetic tests. Report focused module coverage
separately rather than excluding difficult modules to inflate the total.
The `upstream` test job compares NumPy likelihood kernels directly with pinned
Cobaya and LoLLiPoP implementations. Full survey-reference validation remains a
separate scientific task from code coverage.
