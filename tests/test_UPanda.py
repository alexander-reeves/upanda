# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import os
import tempfile

import numpy as np
import pytest
import yaml

try:
    from UPanda import config, utils
    from UPanda.likelihoods import Likelihood
except ImportError as e:
    pytest.skip(f"UPanda not installed: {e}", allow_module_level=True)


class TestConfig:
    """Test configuration module functionality."""

    def test_get_jax_enabled_default(self):
        """Test that JAX is disabled by default."""
        # Reset to default
        config.set_jax_enabled(False)
        assert config.get_jax_enabled() is False

    def test_set_jax_enabled(self):
        """Test setting JAX enabled state."""
        config.set_jax_enabled(True)
        assert config.get_jax_enabled() is True

        config.set_jax_enabled(False)
        assert config.get_jax_enabled() is False


class TestLikelihood:
    """Test base Likelihood class."""

    def test_likelihood_init(self):
        """Test Likelihood initialization."""
        data_vector = np.array([1.0, 2.0, 3.0])
        cov_matrix = np.eye(3)

        likelihood = Likelihood(data_vector=data_vector, covariance_matrix=cov_matrix)

        assert likelihood.data_vector is not None
        assert likelihood.covariance_matrix is not None
        np.testing.assert_array_equal(likelihood.data_vector, data_vector)

    def test_likelihood_init_minimal(self):
        """Test Likelihood initialization with minimal arguments."""
        likelihood = Likelihood()
        assert likelihood.data_vector is None
        assert likelihood.covariance_matrix is None


class TestUtils:
    """Test utility functions."""

    def test_load_config_basic(self):
        """Test loading YAML configuration."""
        config_dict = {
            "sampler": "emcee",
            "n_walkers": 100,
            "n_steps": 1000,
            "parameters": {"omega_m": {"min": 0.1, "max": 0.5}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            loaded_config = utils.load_config(temp_path)
            assert loaded_config["sampler"] == "emcee"
            assert loaded_config["n_walkers"] == 100
            assert "parameters" in loaded_config
        finally:
            os.unlink(temp_path)

    def test_invert_input_scalar(self):
        """Test scalar inversion."""
        result = utils.invert_input(2.0)
        assert result == 0.5

        result = utils.invert_input(4)
        assert result == 0.25

    def test_invert_input_matrix(self):
        """Test matrix inversion."""
        # Create a simple 2x2 matrix
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = utils.invert_input(matrix)

        # Expected inverse
        expected = np.linalg.inv(matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_invert_input_invalid_shape(self):
        """Test inversion with invalid array shape."""
        # 1D array should raise ValueError
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid input"):
            utils.invert_input(arr)

    def test_invert_input_unsupported_type(self):
        """Test inversion with unsupported type."""
        with pytest.raises(TypeError, match="Unsupported type"):
            utils.invert_input("not a number")

    def test_invert_input_zero_scalar(self):
        """Test inversion of zero scalar."""
        with pytest.raises((ZeroDivisionError, FloatingPointError)):
            utils.invert_input(0.0)

    def test_invert_input_negative(self):
        """Test inversion of negative number."""
        result = utils.invert_input(-2.0)
        assert result == -0.5

    def test_invert_input_singular_matrix(self):
        """Test inversion of singular matrix."""
        singular_matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(np.linalg.LinAlgError):
            utils.invert_input(singular_matrix)

    def test_load_config_complex(self):
        """Test load_config with complex YAML structure."""
        import os
        import tempfile

        # Create a temporary config file with more complex structure
        config_content = """
variable_cosmology_parameters:
  - name: omega_m
    prior_min: 0.1
    prior_max: 0.5
    distribution: flat
  - name: omega_b
    prior_min: 0.01
    prior_max: 0.1
nuisance_parameters:
  - name: A_sz
    prior_min: 0.0
    prior_max: 10.0
fixed_cosmology_parameters:
  - name: h
    value: 0.7
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            conf = utils.load_config(temp_path)
            assert "variable_cosmology_parameters" in conf
            assert "nuisance_parameters" in conf
            assert "fixed_cosmology_parameters" in conf
            assert len(conf["variable_cosmology_parameters"]) == 2
            assert len(conf["nuisance_parameters"]) == 1
            assert len(conf["fixed_cosmology_parameters"]) == 1
        finally:
            os.unlink(temp_path)

    def test_get_likelihood_module_valid(self):
        """Test get_likelihood_module with valid likelihood name."""
        # Test with a known likelihood
        module = utils.get_likelihood_module("desi_bao_dr2")
        assert module is not None
        assert hasattr(module, "desi_bao_dr2")

    def test_get_likelihood_module_invalid(self):
        """Test get_likelihood_module with invalid likelihood name."""
        with pytest.raises(ModuleNotFoundError):
            utils.get_likelihood_module("nonexistent_likelihood")

    def test_invert_input_jax_scalar(self):
        """Test invert_input with JAX scalar array."""
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not available")

        from UPanda.module import config as module_config

        original_jax = module_config.get_jax_enabled()

        try:
            module_config.set_jax_enabled(True)
            # The public switch refreshes imported symbols. Manually reloading
            # the backend afterward leaves other modules holding stale wrappers.
            x = jnp.array(2.0)
            result = utils.invert_input(x)
            result_np = np.array(result)
            assert abs(result_np - 0.5) < 1e-10
        finally:
            module_config.set_jax_enabled(original_jax)

    def test_invert_input_1d_array(self):
        """Test invert_input with 1D array (should raise ValueError)."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid input"):
            utils.invert_input(x)

    def test_invert_input_3d_array(self):
        """Test invert_input with 3D array (should raise ValueError)."""
        x = np.random.rand(2, 3, 4)
        with pytest.raises(ValueError, match="Invalid input"):
            utils.invert_input(x)


class TestModuleImports:
    """Test that core modules can be imported."""

    def test_import_config(self):
        """Test config module import."""
        from UPanda import config

        assert hasattr(config, "get_jax_enabled")
        assert hasattr(config, "set_jax_enabled")

    def test_import_provider(self):
        """Test Provider import."""
        from UPanda import Provider

        assert Provider is not None

    def test_import_utils(self):
        """Test utils module import."""
        from UPanda import utils

        assert hasattr(utils, "load_config")
        assert hasattr(utils, "invert_input")

    def test_import_likelihoods(self):
        """Test likelihoods module import."""
        from UPanda.likelihoods import Likelihood

        assert Likelihood is not None

    def test_import_run_panda(self):
        """Test run_panda module import."""
        try:
            import UPanda.run_panda as run_panda

            assert hasattr(run_panda, "run_mcmc")
            assert hasattr(run_panda, "get_param_lims")
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

    def test_import_module(self):
        """Test module import."""
        import UPanda.module as module

        assert module is not None

    def test_import_integrated_model(self):
        """Test integrated_model module import."""
        try:
            import UPanda.integrated_model as integrated_model

            assert integrated_model is not None
        except ImportError as e:
            pytest.skip(f"integrated_model module not available: {e}")


class TestRunPanda:
    """Test run_panda utility functions."""

    def test_get_param_lims(self):
        """Test parameter limits extraction."""
        try:
            import UPanda.run_panda as run_panda
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

        conf = {
            "variable_cosmology_parameters": [
                {"name": "omega_m", "prior_min": 0.1, "prior_max": 0.5},
                {"name": "omega_b", "prior_min": 0.01, "prior_max": 0.1},
            ],
            "fixed_cosmology_parameters": None,
        }

        params_list, lims = run_panda.get_param_lims(conf)

        assert len(params_list) == 2
        assert params_list[0] == "omega_m"
        assert params_list[1] == "omega_b"
        assert lims.shape == (2, 2)
        assert lims[0, 0] == 0.1
        assert lims[0, 1] == 0.5

    def test_get_param_lims_with_nuisance(self):
        """Test parameter limits with nuisance parameters."""
        try:
            import UPanda.run_panda as run_panda
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

        conf = {
            "variable_cosmology_parameters": [
                {"name": "omega_m", "prior_min": 0.1, "prior_max": 0.5}
            ],
            "nuisance_parameters": [
                {"name": "A_sz", "prior_min": 0.0, "prior_max": 10.0}
            ],
            "fixed_cosmology_parameters": None,
        }

        params_list, lims = run_panda.get_param_lims(conf)

        assert len(params_list) == 2
        assert params_list[0] == "omega_m"
        assert params_list[1] == "A_sz"

    def test_get_param_lims_with_fixed_params(self):
        """Test parameter limits with fixed parameters."""
        try:
            import UPanda.run_panda as run_panda
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

        conf = {
            "variable_cosmology_parameters": [
                {"name": "omega_m", "prior_min": 0.1, "prior_max": 0.5},
                {"name": "omega_b", "prior_min": 0.01, "prior_max": 0.1},
            ],
            "fixed_cosmology_parameters": [{"name": "omega_m", "value": 0.3}],
        }

        params_list, lims = run_panda.get_param_lims(conf)

        # omega_m should be excluded, only omega_b should be in params_list
        assert params_list == ["omega_b"]
        assert lims.shape == (1, 2)

    def test_log_gauss_scalar(self):
        """Test log Gaussian with scalar variance."""
        try:
            import UPanda.run_panda as run_panda
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

        x = np.array([1.0, 2.0, 3.0])
        mu = 2.0
        cov = 1.0

        result = run_panda.log_gauss(x, mu, cov)

        # Should return log probability for each point
        assert len(result) == 3
        # Check that result is reasonable (should be negative for log probability)
        assert np.all(result <= 0)

    def test_log_gauss_matrix(self):
        """Test log Gaussian with covariance matrix."""
        try:
            import UPanda.run_panda as run_panda
        except ImportError as e:
            pytest.skip(f"run_panda module not available: {e}")

        x = np.array([[1.0], [2.0], [3.0]])
        mu = np.array([2.0])
        cov = np.eye(1) * 1.0

        result = run_panda.log_gauss(x, mu, cov)

        # Should return log probability for each point
        assert len(result) == 3


class TestUtilsExtended:
    """Additional tests for utility functions."""

    def test_get_likelihood_module(self):
        """Test getting likelihood module by name."""
        # Test that it can import a known likelihood module
        # We'll skip if the module doesn't exist to avoid import errors
        try:
            module = utils.get_likelihood_module("likelihood")
            assert module is not None
        except ImportError:
            pytest.skip("Likelihood module not available")

    def test_load_config_invalid_file(self):
        """Test loading config from non-existent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            utils.load_config("/nonexistent/path/config.yaml")

    def test_invert_input_zero(self):
        """Test inversion of zero (should raise error or handle gracefully)."""
        # This should raise an error or return inf
        with pytest.raises((ZeroDivisionError, FloatingPointError, ValueError)):
            utils.invert_input(0.0)

    def test_invert_input_negative(self):
        """Test inversion of negative number."""
        result = utils.invert_input(-2.0)
        assert result == -0.5

    def test_invert_input_singular_matrix(self):
        """Test inversion of singular matrix."""
        # Create a singular matrix
        singular_matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
        # This should raise LinAlgError
        with pytest.raises(np.linalg.LinAlgError):
            utils.invert_input(singular_matrix)
