# UPanda

[![CI](https://github.com/alexander-reeves/upanda/actions/workflows/ci.yml/badge.svg)](https://github.com/alexander-reeves/upanda/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-4051b5.svg)](https://alexander-reeves.github.io/upanda/)
[![arXiv: Multiprobe dark energy](https://img.shields.io/badge/arXiv-2510.06114-b31b1b.svg)](https://arxiv.org/abs/2510.06114)
[![arXiv: Tuning the cosmic instrument](https://img.shields.io/badge/arXiv-2502.01722-b31b1b.svg)](https://arxiv.org/abs/2502.01722)
[![arXiv: 12x2pt combined probes](https://img.shields.io/badge/arXiv-2309.03258-b31b1b.svg)](https://arxiv.org/abs/2309.03258)

**U**ltrafast **P**arameter **A**nalysis using **N**eural-accelerated **D**riven **A**lgorithms

UPanda evaluates cosmological likelihoods with emulator predictions or external theory
spectra. It includes CMB, BAO, redshift-space distortion, supernova, and general LSS
likelihoods, with NumPy evaluation and JAX differentiation, plus sampling utilities.

## Installation

Python 3.10–3.12 are the supported CI targets. From this checkout:

```bash
python -m pip install -e '.[dev]'
# Optional CUDA 12 dependencies:
python -m pip install -e '.[gpu]'
```

Use a dedicated environment for development. Runtime dependencies are retained for
compatibility; the default installation still includes TensorFlow and sampler packages.
[Installation details](docs/installation.md).

## Evaluate a likelihood

This self-contained example uses synthetic observations and a tiny emulator. No
trained model, downloaded dataset, or sampler run is required.

```python
import numpy as np
from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

class ExampleTheory:
    def predict(self, params):
        return np.column_stack([params['x'], 2 * params['x']])

like = desi_bao_dr2(
    dtype=np.dtype([('x', 'f8')]),
    emu_model_list=[ExampleTheory()],
    additional_args={'mock_data': np.array([1., 2.]), 'mock_cov': np.eye(2)},
)
print(like.compute_like(np.array([[1.], [2.]])))  # [0., -2.5]

# External theory interface used by downstream analysis projects:
like.set_external_spectra({'desi_bao_data': np.array([[1., 2.]])})
print(like.compute_like(None))  # [0.]
```

For real DESI observations, omit `mock_data` and `mock_cov` and supply a theory
prediction in the data file's observable/redshift order. The example above is a
software demonstration, not a cosmological theory model.

## JAX

Choose the backend before constructing likelihoods and compiling functions:

```python
import jax
jax.config.update('jax_enable_x64', True)
from UPanda import config
config.set_jax_enabled(True)
from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2
```

UPanda refreshes backend symbols in already imported UPanda modules while preserving
likelihood class identities. Recreate existing models and compiled functions after a
switch; switching does not convert stored data or invalidate JAX compilation caches.
JAX theory functions must use JAX operations. See [API contracts](docs/usage.md)
for shapes, units, external spectra, and JIT caveats.

## Likelihood data and citations

Data are read from `src/UPanda/likelihoods/data/<dataset>/` in editable installations.
The package includes an explicit selection of small/medium data assets. CamSpec,
HiLLiPoP, ACT DR6 lensing, and WMAP require separately provisioned data.
**Automatic downloading is not implemented.** There are no UPanda Zenodo archive
URLs or checksums to use for those large datasets.

[The likelihood accuracy audit](docs/likelihood_audit.md) records upstream comparisons,
corrections, emulator conventions and unverified modes.
[The data-source catalogue](docs/data_sources.md) links upstream likelihood code,
data releases, and papers, and records implementation-specific approximations.
Do not assume that a link to an upstream release proves a byte-for-byte match to
this checkout. For scientific analyses, record the exact data files, emulator
artifacts, cuts, nuisance priors, and software revision.

Please cite the relevant survey papers and the UPanda research using the code:
[Reeves et al. 2023](https://arxiv.org/abs/2309.03258),
[Reeves et al. 2025a](https://arxiv.org/abs/2502.01722), and
[Reeves et al. 2025b](https://arxiv.org/abs/2510.06114).

## Development

```bash
make test          # deterministic unit and synthetic likelihood tests
make coverage      # full package coverage, HTML and XML reports
make style-check   # syntax and undefined-name checks
make docs          # strict MkDocs build
make build         # wheel and source distribution
```

Commands use `python` from your active environment; override with
`make test PYTHON=/path/to/python`. No implicit environment synchronization occurs.
See [contributing](docs/contributing.md), [tests](tests/README.md), and the
[compatibility review](docs/review.md). Existing import paths, class names,
constructor keywords, and downstream external-theory interfaces are retained.

## License

[MIT](LICENSE). Survey data retain their upstream attribution and usage requirements.
