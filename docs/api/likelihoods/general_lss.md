# General LSS Likelihood

The general large-scale structure likelihood module provides a flexible framework for handling various LSS observables and cross-correlations including galaxy clustering, weak lensing, and CMB lensing cross-correlations.

::: UPanda.likelihoods.general_lss.general_lss

## Quick Example

```python
from UPanda.likelihoods.general_lss import general_lss
import jax.numpy as jnp

# Initialize likelihood
config = {
    'data_file': 'lss_data.sacc',
    'data_types': ['galaxy_shear_cl_ee', 'galaxy_density_cl'],
    'ell_range': [30, 3000],
    'nuisance_params': {
        'm1': {'prior': 'gaussian', 'mean': 0.0, 'std': 0.01},
        'm2': {'prior': 'gaussian', 'mean': 0.0, 'std': 0.01}
    }
}

likelihood = general_lss(config)

# Evaluate likelihood
params = {
    'omega_m': 0.3,
    'sigma_8': 0.8,
    'm1': 0.0,
    'm2': 0.0
}

loglike = likelihood.loglike(params)
```


