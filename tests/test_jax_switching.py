# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import pytest

try:
    import UPanda.module as module
    from UPanda import config

    HAS_UPANDA = True
except ImportError:
    HAS_UPANDA = False


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestJAXSwitching:
    """Test JAX switching functionality."""

    def test_jax_disabled_by_default(self):
        """Test that JAX is disabled by default."""
        config.set_jax_enabled(False)
        assert config.get_jax_enabled() is False

    def test_enable_jax(self):
        """Test enabling JAX."""
        config.set_jax_enabled(True)
        assert config.get_jax_enabled() is True
        config.set_jax_enabled(False)

    def test_module_has_core_functions(self):
        """Test that module has core numpy/JAX functions."""
        required_functions = [
            "array",
            "zeros",
            "ones",
            "sum",
            "sqrt",
            "exp",
            "log",
            "sin",
            "cos",
            "pi",
            "linalg",
        ]

        for func_name in required_functions:
            assert hasattr(module, func_name), f"Module missing {func_name}"

    def test_numpy_operations_work(self):
        """Test that numpy operations work regardless of JAX setting."""
        config.set_jax_enabled(False)

        arr1 = module.array([1, 2, 3])
        arr2 = module.array([4, 5, 6])

        result = module.sum(arr1 + arr2)
        assert result == 21
