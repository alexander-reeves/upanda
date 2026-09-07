# Utils

Utility functions for UPanda.

## Overview

The utils module provides helper functions for:

- MCMC diagnostics
- Statistical calculations
- Data manipulation
- Likelihood utilities

## API Reference

::: UPanda.utils

## Common Functions

### Gelman-Rubin Diagnostic

```python
from UPanda.utils import compute_gelman_rubin

# Check convergence
R_hat = compute_gelman_rubin(chain)  # Want R_hat < 1.1
```

### Effective Sample Size

```python
from UPanda.utils import compute_ess

ess = compute_ess(chain)
print(f"Effective samples: {ess}")
```

### Autocorrelation Time

```python
from UPanda.utils import compute_autocorr_time

tau = compute_autocorr_time(chain)
print(f"Autocorrelation time: {tau}")
```

