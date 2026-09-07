"""Configuration module for managing JAX/NumPy backend switching.

This module provides a simple interface to enable or disable JAX backend
usage across the UPanda package. When JAX is enabled, operations use
JAX arrays and functions for improved performance with JIT compilation
and automatic differentiation.

Attributes:
    is_jax_enabled (bool): Global flag indicating whether JAX backend
        is currently enabled. Defaults to False.
"""

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

is_jax_enabled = False  # Default setting


def set_jax_enabled(value: bool):
    """Enable or disable JAX backend globally.

    This function sets the JAX backend flag and automatically reloads
    all UPanda modules that depend on the backend selection to ensure
    they pick up the correct numpy/jax functions.

    Args:
        value (bool): True to enable JAX backend, False to use NumPy.
    """
    global is_jax_enabled
    old_value = is_jax_enabled
    is_jax_enabled = value

    # If the value changed, reload modules to pick up the new backend
    if old_value != value:
        try:
            _reload_modules()
        except Exception:
            is_jax_enabled = old_value
            _reload_modules()
            raise


def get_jax_enabled():
    """Get the current JAX backend status.

    Returns:
        bool: True if JAX is enabled, False if using NumPy.
    """
    return is_jax_enabled


def _reload_modules():
    """Reload UPanda modules after JAX backend change.

    This ensures that all modules pick up the correct numpy/jax functions
    after the backend is switched. Modules are reloaded in dependency order.
    """
    backend = sys.modules.get("UPanda.module")
    if backend is None:
        return
    previous = {name: getattr(backend, name) for name in backend.__all__}
    importlib.reload(backend)

    # Refresh imported backend symbols without recreating likelihood classes.
    # Existing instances and ``from ... import function`` references retain
    # their identities and see the updated module globals.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("UPanda.") or module is None or module is backend:
            continue
        namespace = vars(module)
        for symbol, old in previous.items():
            if symbol in namespace and namespace[symbol] is old:
                namespace[symbol] = getattr(backend, symbol)

    # This module also has conditional JAX sampler imports.
    sampler = sys.modules.get("UPanda.run_panda")
    if sampler is not None:
        importlib.reload(sampler)
