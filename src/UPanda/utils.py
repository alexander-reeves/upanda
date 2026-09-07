"""Utility functions for configuration loading and data manipulation.

This module provides helper functions for:
- Loading YAML configuration files
- Importing likelihood modules dynamically
- Inverting scalars and matrices (with JAX/NumPy compatibility)
"""

import importlib

import yaml

from UPanda.module import *

is_jax = config.get_jax_enabled()
if is_jax:
    from jax import numpy as jnp


def load_config(filename_config):
    """
    Load configuration from a YAML file.

    Args:
        filename_config (str): Path to the YAML configuration file

    Returns:
        dict: Loaded configuration dictionary
    """
    with open(filename_config) as f:
        conf = yaml.safe_load(f)
    return conf


def get_likelihood_module(name):
    """
    Import and return a likelihood module by name.

    Args:
        name (str): Name of the likelihood module

    Returns:
        module: Imported likelihood module
    """
    module_name = f"UPanda.likelihoods.{name}"
    module = importlib.import_module(module_name)
    return module


def invert_input(value):
    """
    Invert a scalar or matrix input.

    Args:
        value (ndarray, int, float): Input to invert

    Returns:
        ndarray, float: Inverted value

    Raises:
        ValueError: If input is an invalid shape
        TypeError: If input type is unsupported
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.shape == ():  # Scalar input
            if is_jax:
                return jnp.reciprocal(value)
            else:
                return 1.0 / value.item()
        elif len(value.shape) == 2:  # Matrix input
            return linalg.inv(value)
        else:
            raise ValueError("Invalid input: Expecting a scalar or a 2D matrix.")
    elif isinstance(value, (int, float)):
        return 1.0 / value
    else:
        raise TypeError("Unsupported type for inversion.")
