# Contributing

```bash
pip install -e '.[dev]'
make style-check      # ruff
make test             # unit tests
make test-reference   # comparisons with reference likelihoods, see Validation
make docs             # documentation, built with MkDocs
```

The continuous integration runs the unit tests on Python 3.10 to 3.12 and the reference comparisons on 3.10.
The documentation is published from `master`.

Several analysis codes import the likelihood classes directly, so please keep module paths, class names and
constructor arguments as they are.

When adding a likelihood:

- say in the module docstring where the data come from, and add a row to [Data sources](data_sources.md);
- if cobaya or the collaboration provides a likelihood for the same data, add a comparison to
  `tests/test_reference.py` and a row to [Validation](validation.md);
- check that it runs in both NumPy and JAX.
