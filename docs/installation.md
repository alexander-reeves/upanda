# Installation

## Prerequisites

UPanda requires Python 3.10 or higher.

## Installation from Source

To install UPanda, first clone the git repository:

```bash
git clone https://github.com/alexander-reeves/upanda.git
```

Then, navigate to the cloned directory:

```bash
cd upanda
```

Finally, install the package:

```bash
pip install -e .
```

This will install UPanda in editable mode, allowing you to modify the source code and see changes immediately.

## Development Installation

For development dependencies (including testing and linting tools):

```bash
pip install -e ".[dev]"
```

This includes:

- Testing tools (pytest, pytest-cov)
- Code formatting (black, isort)
- Linting (ruff)
- Pre-commit hooks

## Documentation Dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

## Verifying Installation

To verify your installation, run the tests:

```bash
make test
```

Or directly with pytest:

```bash
pytest tests/
```

## Dependencies

UPanda depends on several key packages:

- **JAX**: For differentiable computing and GPU acceleration
- **NumPy/SciPy**: For numerical operations
- **BlackJAX/Numpyro**: For MCMC sampling
- **TensorFlow**: For neural network emulators
- **Astropy**: For cosmological utilities
- **SACC**: For data handling

See `pyproject.toml` for the complete list of dependencies.
