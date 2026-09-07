# Examples

The best way to learn how to use UPanda is through the Jupyter notebooks in the `notebooks/` directory. These are complete, working examples that you can run directly.

## Available Notebooks

### Planck CMB Analysis

- **[`planck_pr4_jax.ipynb`](https://github.com/alexander-reeves/upanda/blob/master/notebooks/planck_pr4_jax.ipynb)** - Planck PR4 analysis using JAX-accelerated emulators
  - Demonstrates JAX integration
  - Shows how to use neural network emulators
  - Full MCMC sampling example

- **[`tutorial_planck_pr3.ipynb`](https://github.com/alexander-reeves/upanda/blob/master/notebooks/tutorial_planck_pr3.ipynb)** - Complete Planck PR3 analysis tutorial
  - Step-by-step parameter inference
  - Chain analysis and visualization
  - Convergence diagnostics

### BAO Analysis

- **[`tutorial_desi_bao_dr2.ipynb`](https://github.com/alexander-reeves/upanda/blob/master/notebooks/tutorial_desi_bao_dr2.ipynb)** - DESI BAO DR2 analysis
  - BAO likelihood setup
  - Parameter inference with BAO data
  - Comparison with CMB constraints

- **[`test_desi_bao.ipynb`](https://github.com/alexander-reeves/upanda/blob/master/notebooks/test_desi_bao.ipynb)** - DESI BAO testing and validation
  - Testing BAO likelihood implementation
  - Validation against official results

## Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/alexander-reeves/upanda.git
cd upanda
```

2. Install UPanda with dependencies:
```bash
pip install -e ".[dev]"
```

3. Launch Jupyter:
```bash
jupyter notebook notebooks/
```

4. Open any of the notebooks and run the cells!

## What You'll Learn

From these notebooks, you'll learn how to:

- Set up and run cosmological parameter inference
- Use JAX-accelerated neural network emulators
- Configure different likelihoods (CMB, BAO, LSS)
- Analyze MCMC chains and assess convergence
- Combine multiple datasets
- Use different samplers (emcee, nautilus, etc.)
- Optimize parameters and find best-fit values
- Create corner plots and visualizations

## Need Help?

If you encounter issues running the notebooks or have questions:

- Check the [API Reference](api/run_panda.md) for detailed function documentation
- Open an issue on [GitHub](https://github.com/alexander-reeves/upanda/issues)
- Contact: [areeves@phys.ethz.ch](mailto:areeves@phys.ethz.ch)

