# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

"""
Pytest configuration and shared fixtures for testing.
"""

import os

import numpy as np
import pytest
import yaml

try:
    from UPanda.likelihoods import Likelihood
    from UPanda.module import config as module_config

    HAS_UPANDA = True
except ImportError:
    HAS_UPANDA = False


@pytest.fixture
def mock_likelihood_data_dir(tmpdir, monkeypatch):
    """
    Create a mock data directory structure and patch os.path operations.

    Returns a dict with helper functions to create mock data files.
    """
    data_dir = tmpdir.mkdir("data")

    def create_mock_data_file(subdir, filename, content):
        """Helper to create mock data files."""
        subdir_path = data_dir.mkdir(subdir, exist_ok=True)
        file_path = subdir_path.join(filename)
        if isinstance(content, np.ndarray):
            np.savetxt(str(file_path), content)
        elif isinstance(content, (list, tuple)):
            np.savetxt(str(file_path), np.array(content))
        else:
            file_path.write(content)
        return str(file_path)

    return {
        "data_dir": str(data_dir),
        "create_file": create_mock_data_file,
        "tmpdir": tmpdir,
    }


@pytest.fixture
def mock_emulator_model():
    """Create a mock emulator model that can be used in likelihood tests."""

    class MockEmulator:
        def __init__(self, output_shape=(10,)):
            self.output_shape = output_shape

        def predict(self, params_dict):
            """Mock predict that returns array of correct shape."""
            # Handle both dict and array inputs
            if isinstance(params_dict, dict):
                n_samples = len(next(iter(params_dict.values())))
            else:
                n_samples = 1
            return np.random.randn(n_samples, *self.output_shape)

        def predictions_np(self, params_dict):
            """Alternative predict method for some likelihoods."""
            return self.predict(params_dict)

    return MockEmulator


@pytest.fixture
def standard_cosmology_dtype():
    """Standard cosmology parameter dtype for testing."""
    return np.dtype(
        [
            ("omega_m", "f8"),
            ("omega_b", "f8"),
            ("h", "f8"),
            ("n_s", "f8"),
            ("ln10^{10}A_s", "f8"),
        ]
    )


@pytest.fixture
def simple_cosmology_dtype():
    """Simple cosmology parameter dtype for testing."""
    return np.dtype([("omega_m", "f8"), ("omega_b", "f8")])


@pytest.fixture
def enable_jax():
    """Fixture to enable JAX for tests that need it."""
    if not HAS_UPANDA:
        pytest.skip("UPanda not installed")
    original_jax = module_config.get_jax_enabled()
    try:
        module_config.set_jax_enabled(True)
        yield
    finally:
        module_config.set_jax_enabled(original_jax)


@pytest.fixture
def create_test_config(tmpdir):
    """Fixture to create test config files."""

    def _create_config(add_distribution=False, config_dict=None):
        """Helper to create a test configuration file."""
        if config_dict is None:
            params = [
                {"name": "omega_m", "prior_min": 0.1, "prior_max": 0.5},
                {"name": "omega_b", "prior_min": 0.01, "prior_max": 0.1},
            ]
            if add_distribution:
                for param in params:
                    param["distribution"] = "flat"

            config_content = {
                "variable_cosmology_parameters": params,
                "fixed_cosmology_parameters": None,
            }
        else:
            config_content = config_dict

        config_path = os.path.join(tmpdir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)
        return config_path

    return _create_config


@pytest.fixture
def mock_likelihood_class():
    """Create a mock likelihood class factory."""
    if not HAS_UPANDA:
        pytest.skip("UPanda not installed")

    def _create_mock_likelihood(has_provider=False, return_zeros=False):
        """Create a mock likelihood class."""

        class MockLikelihood(Likelihood):
            def __init__(self, *args, **kwargs):
                likelihood_kwargs = {
                    "data_vector": None,
                    "covariance_matrix": None,
                    "emu_model_list": None,
                    "compression_vecs": None,
                }
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k in likelihood_kwargs
                }
                super().__init__(**filtered_kwargs)

                self.dtype = kwargs.get("dtype", None)
                self.cosmology_params_fixed = kwargs.get("cosmology_params_fixed", None)
                self.additional_args = kwargs.get("additional_args", {})
                self.emu_model_list = kwargs.get("emu_model_list", [])
                self.data_vector = np.array([1.0, 2.0])
                self.covariance_matrix = np.eye(2)
                self.provider = None

            def set_provider(self, provider):
                if has_provider:
                    self.provider = provider

            def compute_like(self, params):
                if hasattr(params, "block_until_ready"):
                    params = np.array(params)
                else:
                    params = np.asarray(params)

                if params.ndim == 1:
                    params = params.reshape(1, -1)

                if return_zeros:
                    return np.zeros(params.shape[0])

                diff = params[:, 0] - 0.3
                result = -0.5 * diff**2
                return np.array(result) if isinstance(result, (list, tuple)) else result

        return MockLikelihood

    return _create_mock_likelihood


@pytest.fixture
def mock_emcee_sampler():
    """Fixture to create mock emcee sampler."""

    class MockEmceeSampler:
        def __init__(self, *args, **kwargs):
            self.nwalkers = kwargs.get("nwalkers", 10)
            self.ndim = kwargs.get("ndim", 2)
            self.pool = kwargs.get("pool", None)

        def run_mcmc(self, initial_state, nsteps, **kwargs):
            return np.random.rand(self.nwalkers, self.ndim)

        def reset(self):
            pass

        def get_chain(self, flat=True, discard=0, thin=1):
            return np.random.rand(50, self.ndim)

        def get_log_prob(self, flat=True, discard=0, thin=1):
            return np.random.randn(50)

        def get_autocorr_time(self):
            return np.ones(self.ndim) * 10.0

    return MockEmceeSampler


@pytest.fixture
def mock_module_factory():
    """Factory fixture to create mock modules with likelihoods."""

    def _create_mock_module(likelihood_class, likelihood_name="desi_bao_dr2"):
        """Create a mock module with a likelihood attribute."""

        class MockModule:
            pass

        setattr(MockModule, likelihood_name, likelihood_class)
        return MockModule

    return _create_mock_module


class LikelihoodTestHelper:
    """Helper class for testing likelihoods with consistent mocking."""

    @staticmethod
    def create_mock_covariance(n_points, seed=42):
        """Create a positive definite covariance matrix."""
        np.random.seed(seed)
        A = np.random.randn(n_points, n_points)
        cov = np.dot(A, A.T)
        # Add small diagonal to ensure positive definite
        cov += np.eye(n_points) * 0.1
        return cov

    @staticmethod
    def create_mock_data_vector(n_points, seed=42):
        """Create a mock data vector."""
        np.random.seed(seed)
        return np.random.randn(n_points)

    @staticmethod
    def create_mock_likelihood_init_kwargs(
        dtype, emu_model_list=None, additional_args=None
    ):
        """Create standard kwargs for likelihood initialization."""
        if emu_model_list is None:
            emu_model_list = []
        if additional_args is None:
            additional_args = {}

        return {
            "dtype": dtype,
            "emu_model_list": emu_model_list,
            "cosmology_params_fixed": [],
            "additional_args": additional_args,
            "compression_vecs": None,
        }

    @staticmethod
    def setup_path_mocks(monkeypatch, data_dir, likelihood_name):
        """
        Setup mock functions for os.path.abspath and os.path.dirname.

        Parameters:
        -----------
        monkeypatch : pytest monkeypatch fixture
        data_dir : str
            Base data directory path
        likelihood_name : str
            Name of the likelihood module (e.g., 'desi_bao_dr2')

        Returns:
        --------
        None (modifies monkeypatch directly)
        """
        import importlib
        from types import SimpleNamespace

        module = importlib.import_module(f"UPanda.likelihoods.{likelihood_name}")
        # Patch only this likelihood's os reference: changing os.path globally
        # corrupts pytest's own path handling and causes recursive failures.
        path = SimpleNamespace(
            abspath=lambda _: os.path.join(data_dir, "likelihoods", f"{likelihood_name}.py"),
            dirname=lambda _: os.path.join(data_dir, "likelihoods"),
            join=os.path.join,
            exists=os.path.exists,
        )
        monkeypatch.setattr(module, "os", SimpleNamespace(path=path))

    @staticmethod
    def create_predictable_emulator(output_shape):
        """Create an emulator that returns predictable (ones) output."""

        class PredictableEmulator:
            def __init__(self, output_shape):
                self.output_shape = output_shape

            def predictions_np(self, params_dict):
                n_samples = len(next(iter(params_dict.values())))
                return np.ones((n_samples, self.output_shape[0]))

            def predict(self, params_dict):
                # Some likelihoods use predict instead of predictions_np
                return self.predictions_np(params_dict)

        return PredictableEmulator(output_shape)
