# UPanda

[![CI](https://github.com/alexander-reeves/upanda/actions/workflows/ci.yml/badge.svg)](https://github.com/alexander-reeves/upanda/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-4051b5.svg)](https://alexander-reeves.github.io/upanda/)
[![arXiv](https://img.shields.io/badge/arXiv-2309.03258-b31b1b.svg)](https://arxiv.org/abs/2309.03258)
[![arXiv](https://img.shields.io/badge/arXiv-2502.01722-b31b1b.svg)](https://arxiv.org/abs/2502.01722)
[![arXiv](https://img.shields.io/badge/arXiv-2510.06114-b31b1b.svg)](https://arxiv.org/abs/2510.06114)

**U**ltrafast **P**arameter **A**nalysis using **N**eural-accelerated **D**riven **A**lgorithms

UPanda is a collection of cosmological likelihoods written for fast inference. Each likelihood takes its
theory prediction either from a neural-network emulator or directly from the user, and runs in NumPy or in JAX
(differentiable, and compatible with `jit` and `vmap`). The package also wraps a few samplers (emcee, nautilus,
UltraNest, HMC) behind one interface.

| | Likelihoods |
|---|---|
| CMB | Planck 2018 (plik-lite, low-ℓ, SRoll2 low-ℓ EE), Planck PR4 (CamSpec, HiLLiPoP, LoLLiPoP), ACT DR4, ACT DR6, WMAP |
| CMB lensing | Planck 2018, ACT DR6 |
| BAO and RSD | DESI DR1 and DR2, BOSS DR12 and eBOSS DR16 |
| Supernovae | Pantheon+, Union3 and Union3.1, DES-Dovekie |
| Large-scale structure | angular power spectra of galaxy clustering, weak lensing and their cross-correlations with CMB lensing (`general_lss`) |

## Installation

```bash
git clone https://github.com/alexander-reeves/upanda.git
cd upanda
pip install -e .            # add '.[gpu]' for CUDA 12, '.[dev]' for the tests and the documentation
```

Python 3.10 to 3.12.

## Example

A likelihood is built once and then evaluated on a theory vector, here the DESI DR2 BAO distances:

```python
import numpy as np
from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

like = desi_bao_dr2()                                    # DESI DR2 data and covariance
theory = np.loadtxt(f"{like.data_dir}/mean.txt", usecols=1) * 1.01
like.set_external_spectra({"desi_bao_data": theory})     # D_V/r_d, D_M/r_d, D_H/r_d in the order of mean.txt
print(like.compute_like(None))                           # log-likelihood
```

With an emulator, pass it at construction together with the parameter names, and evaluate batches of points:

```python
like = desi_bao_dr2(dtype=np.dtype([("omega_cdm", "f8"), ("h", "f8")]), emu_model_list=[emulator])
like.compute_like(points)                                # points: array of shape (n_points, n_parameters)
```

For JAX, switch the backend before building the likelihoods:

```python
import jax
jax.config.update("jax_enable_x64", True)
from UPanda import config
config.set_jax_enabled(True)
```

The [documentation](https://alexander-reeves.github.io/upanda/) describes the interface of each likelihood, and
`notebooks/` has worked examples with trained emulators.

## Data

The smaller data sets are part of the repository (`src/UPanda/likelihoods/data/`). The Planck PR4 CamSpec and
HiLLiPoP data, the ACT DR6 lensing likelihood and WMAP are too large and have to be downloaded;
[docs/data_sources.md](docs/data_sources.md) lists where every data set comes from.

## Validation

Where a public reference exists (cobaya, or the likelihood code released by the collaboration), the UPanda
likelihood is tested against it on the real data. [docs/validation.md](docs/validation.md) lists what is
compared with what, and how to run the comparisons.

## Citing

If you use UPanda, please cite [Reeves et al. (2024)](https://arxiv.org/abs/2309.03258),
[Reeves et al. (2025)](https://arxiv.org/abs/2502.01722) and
[Reeves et al. (2026)](https://arxiv.org/abs/2510.06114), together with the papers of the data sets you use.

## Development

```bash
make test             # unit tests
make test-reference   # comparisons with cobaya and other reference codes
make docs             # build the documentation
```

## License

[MIT](LICENSE). The survey data keep the terms of their original releases.
