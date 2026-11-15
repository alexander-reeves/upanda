# UPanda

UPanda stands for:
**U**ltrafast 
**P**arameter
**A**nalysis using
**N**eural-accelerated
**D**riven
**A**lgorithms

[![arXiv](https://img.shields.io/badge/arXiv-2309.03258-b31b1b.svg)](https://arxiv.org/abs/2309.03258)
[![arXiv](https://img.shields.io/badge/arXiv-2502.01722-b31b1b.svg)](https://arxiv.org/abs/2502.01722)
[![arXiv](https://img.shields.io/badge/arXiv-2510.06114-b31b1b.svg)](https://arxiv.org/abs/2510.06114)

A Python library for rapid cosmological parameter inference using neural network-accelerated emulators. UPanda provides MCMC sampling capabilities with support for various cosmological likelihoods including CMB, LSS, and BAO data. The framework is designed to work with JAX for differentiable computations and supports both traditional TensorFlow/Keras emulators and JAX-based implementations.

## Citing 

When using this code please cite the following papers: 

- Reeves et al. (2023): [arXiv:2309.03258](https://arxiv.org/abs/2309.03258)
- Reeves et al. (2025a): [arXiv:2502.01722](https://arxiv.org/abs/2502.01722)
- Reeves et al. (2025b): [arXiv:2510.06114](https://arxiv.org/abs/2510.06114)

## Features

- **Fast MCMC sampling** with multiple sampler options (emcee, nautilus, dynesty, ultranest)
- **Support for various cosmological likelihoods** (Planck PR4/PR3, ACT DR4/DR6, Pantheon+, and a general likelihood class for LSS cross-correlations)
- **Neural network emulator integration** for rapid theory predictions
- **JAX-compatible implementation** for differentiable computations (`vmap`/`jit` compatible)
- **Flexible configuration** via YAML files

## Quick Start

```bash
git clone https://github.com/alexander-reeves/upanda.git
cd upanda
pip install -e .
```

## Examples

The best way to learn UPanda is through the working examples in the `notebooks/` directory:

- **[Planck PR4 with JAX](https://github.com/alexander-reeves/upanda/blob/master/notebooks/planck_pr4_jax.ipynb)** - Full CMB analysis with neural emulators
- **[DESI BAO DR2](https://github.com/alexander-reeves/upanda/blob/master/notebooks/tutorial_desi_bao_dr2.ipynb)** - BAO parameter inference
- **[Planck PR3 Tutorial](https://github.com/alexander-reeves/upanda/blob/master/notebooks/tutorial_planck_pr3.ipynb)** - Complete MCMC walkthrough

See the [Examples](examples.md) page for the full list of notebooks.

## Feedback

If you have any suggestions or questions about **UPanda**, feel free to email me at [areeves@phys.ethz.ch](mailto:areeves@phys.ethz.ch).

If you encounter any errors or problems with **UPanda**, please let me know!

## Validation

Where a public reference exists (cobaya, or the collaboration's own code), each likelihood is tested against it on the real data. See [Validation](validation.md).

## Acknowledgments

Developed at ETH Zurich (IPA). Thanks to John Hennig and the Euler cluster support team at ETH Zürich for software support.

## License

[MIT](https://choosealicense.com/licenses/mit/)
