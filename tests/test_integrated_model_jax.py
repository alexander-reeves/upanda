# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    import UPanda.integrated_model_jax as integrated_model_jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestIntegratedModelJAX:
    """Test JAX-based integrated model functionality."""

    def test_custom_activation_jax(self):
        """Test CustomActivation_jax module."""
        activation = integrated_model_jax.CustomActivation_jax(a=1.0, b=0.5)
        x = jnp.array([[1.0, 2.0, 3.0]])

        rng = jax.random.PRNGKey(42)
        params = activation.init(rng, x)
        output = activation.apply(params, x)

        assert output.shape == x.shape

    def test_insert_zero_columns(self):
        """Test insert_zero_columns function."""
        # Make sure JAX is enabled for this test
        from UPanda import config

        original_jax_state = config.get_jax_enabled()

        try:
            import jax.numpy as jnp

            config.set_jax_enabled(True)

            # Use JAX arrays directly - ensure it's a JAX array
            prediction = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # Verify it's a JAX array
            assert hasattr(prediction, "at")

            # Convert list to JAX array for compatibility with JAX operations
            zero_columns_indices = jnp.array([1, 3])

            result = integrated_model_jax.insert_zero_columns(
                prediction, zero_columns_indices
            )

            assert result.shape == (2, 5)
            # Convert to numpy for comparison
            result_np = np.array(result)
            assert np.allclose(result_np[:, 1], 0.0)
            assert np.allclose(result_np[:, 3], 0.0)
            # Check non-zero columns are correct
            assert np.allclose(result_np[:, 0], [1.0, 4.0])
            assert np.allclose(result_np[:, 2], [2.0, 5.0])
            assert np.allclose(result_np[:, 4], [3.0, 6.0])
        finally:
            config.set_jax_enabled(original_jax_state)
