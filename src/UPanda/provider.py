"""Provider class for sharing CMB power spectra between likelihood components.

This module provides the Provider class which enables efficient sharing of
CMB power spectra computed by one likelihood with other likelihoods that
need them (e.g., lensing likelihoods that require CMB spectra for forward
modeling). The Provider includes intelligent caching to avoid redundant
computations when processing batches of parameters.
"""

# Use JAX operations when available for better JIT performance
try:
    from UPanda.module import config

    is_jax = config.get_jax_enabled()
    if is_jax:
        import jax.numpy as jnp
    else:
        import numpy as jnp
except (ImportError, AttributeError):
    import numpy as jnp


class Provider:
    """Facilitates data sharing between different likelihood components.

    The Provider class handles CMB power spectra sharing for forward modeling
    in lensing likelihoods. It stores CMB spectra computed by one likelihood
    and makes them available to other likelihoods that need them, with
    intelligent caching to avoid redundant computations.

    Attributes:
        cmb_cls (dict): Dictionary containing CMB power spectra ('tt', 'ee', etc.)
        cosmology_params (dict): Cosmological parameters used to compute spectra
        has_cmb_data (bool): Flag indicating whether CMB data is available
        ell_max_cmb (int): Maximum multipole moment ℓ in stored spectra
        stored_units (str): Units of stored data ('FIRASmuK2' by default)
    """

    def __init__(self):
        self.cmb_cls = None
        self.cosmology_params = None
        self.has_cmb_data = False
        self.ell_max_cmb = None  # Track the ell range of stored CMB data
        self.stored_units = "FIRASmuK2"  # Track units of stored data

        # Batch processing cache to avoid repeated expensive operations
        self._batch_cache = {}
        self._current_batch_id = None

    def set_Cl(self, cls_dict, params=None):
        """Store CMB power spectra computed by a likelihood.

        Args:
            cls_dict (dict): Dictionary with CMB power spectra. Keys should be
                spectrum types ('tt', 'ee', 'bb', 'te', etc.) and values are
                arrays of power spectrum values. Arrays can be 1D (single spectrum)
                or 2D (batch of spectra with shape (n_samples, n_ell)).
            params (dict, optional): Cosmological parameters used to compute
                these spectra. Defaults to None.
        """
        if not cls_dict:
            raise ValueError("At least one CMB spectrum is required")
        self.cmb_cls = cls_dict
        self.cosmology_params = params
        self.has_cmb_data = True

        # Track ell range (assume all spectra have same shape)
        reference = cls_dict.get("tt", next(iter(cls_dict.values())))
        self.ell_max_cmb = reference.shape[-1] - 1

        # Generate new batch ID when CMB data changes
        self._current_batch_id = id(reference)
        self._batch_cache.clear()  # Clear cache when new data arrives

    def get_Cl(self, ell_factor=False, units="K2"):
        """Retrieve stored CMB power spectra with intelligent batch caching.

        Args:
            ell_factor (bool, optional): Whether to apply the ℓ(ℓ+1)/(2π) factor
                to convert C_ℓ to D_ℓ. Defaults to False.
            units (str, optional): Units for the output spectra. Options are:
                - 'K2': Convert from µK² to K² (divide by 1e12)
                - 'FIRASmuK2': Return in µK² units (no conversion)
                Defaults to 'K2'.

        Returns:
            dict: Dictionary with CMB power spectra. Keys are spectrum types
                ('tt', 'ee', etc.) and values are arrays of power spectrum
                values in the requested units and format.

        Raises:
            ValueError: If no CMB data has been set via set_Cl().
        """
        if not self.has_cmb_data:
            raise ValueError(
                "No CMB data available. Please run a CMB likelihood first."
            )

        # Create cache key for this specific request
        cache_key = (ell_factor, units)

        # Check if we already processed this exact request for this batch
        if self._current_batch_id is not None and cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        # Optimize: Skip unnecessary copy and operations when no processing needed
        if not ell_factor and units == "FIRASmuK2":
            result = (
                self.cmb_cls
            )  # Return directly, no processing needed (already in µK²)
        else:
            # Only copy if we need to modify
            cls_copy = {}
            for key in self.cmb_cls:
                if units == "FIRASmuK2":
                    # CMB likelihoods already send µK² data, so no conversion needed
                    cls_copy[key] = self.cmb_cls[key]
                else:
                    # Convert µK² to K² if K2 units requested
                    cls_copy[key] = self.cmb_cls[key] / 1e12

            # Apply ell factor if requested (this is rarely used for lensing corrections)
            if ell_factor:
                # Handle batch dimensions properly for JAX compatibility
                for key in cls_copy:
                    ells = jnp.arange(cls_copy[key].shape[-1])
                    factor = ells * (ells + 1) / (2 * jnp.pi)
                    cls_copy[key] = cls_copy[key] * factor

            result = cls_copy

        # Cache the result for this batch
        if self._current_batch_id is not None:
            self._batch_cache[cache_key] = result

        return result
