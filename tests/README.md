# Tests

```bash
pip install -e '.[dev]'
make test              # unit tests: small synthetic data, CPU, no downloads
make test-reference    # comparisons with cobaya and other reference codes (see docs/validation.md)
make coverage
```

`test_reference.py` is the file to look at if you want to know whether a likelihood agrees with its reference
implementation. The other files are unit tests of the individual pieces (Gaussian likelihoods, binning,
foregrounds, emulator classes, samplers, NumPy/JAX agreement).

Set `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true` when comparing numbers.
