# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import numpy as np
import pytest

try:
    from UPanda.provider import Provider

    HAS_UPANDA = True
except ImportError:
    HAS_UPANDA = False


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestProvider:
    """Test Provider class for CMB data sharing."""

    def test_provider_init(self):
        """Test Provider initialization."""
        provider = Provider()

        assert provider.cmb_cls is None
        assert provider.cosmology_params is None
        assert provider.has_cmb_data is False
        assert provider.ell_max_cmb is None
        assert provider.stored_units == "FIRASmuK2"
        assert provider._batch_cache == {}
        assert provider._current_batch_id is None

    def test_set_Cl_basic(self):
        """Test set_Cl with basic CMB spectra."""
        provider = Provider()

        cls_dict = {
            "tt": np.ones(10) * 1000.0,
            "ee": np.ones(10) * 100.0,
            "te": np.ones(10) * 50.0,
        }
        params = {"omega_m": 0.3, "omega_b": 0.05}

        provider.set_Cl(cls_dict, params)

        assert provider.has_cmb_data is True
        assert provider.cmb_cls == cls_dict
        assert provider.cosmology_params == params
        assert provider.ell_max_cmb == 9  # 10 elements, ell 0-9

    def test_set_Cl_without_params(self):
        """Test set_Cl without optional params."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1000.0}

        provider.set_Cl(cls_dict)

        assert provider.has_cmb_data is True
        assert provider.cosmology_params is None

    def test_get_Cl_no_data(self):
        """Test get_Cl raises error when no CMB data."""
        provider = Provider()

        with pytest.raises(ValueError, match="No CMB data available"):
            provider.get_Cl()

    def test_get_Cl_default_units(self):
        """Test get_Cl with default units (K2 conversion)."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12, "ee": np.ones(10) * 1e11}  # In µK²
        provider.set_Cl(cls_dict)

        result = provider.get_Cl()

        # Should convert from µK² to K² (divide by 1e12)
        assert "tt" in result
        assert "ee" in result
        np.testing.assert_array_almost_equal(result["tt"], np.ones(10))
        np.testing.assert_array_almost_equal(result["ee"], np.ones(10) * 0.1)

    def test_get_Cl_FIRASmuK2_units(self):
        """Test get_Cl with FIRASmuK2 units (no conversion)."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12, "ee": np.ones(10) * 1e11}
        provider.set_Cl(cls_dict)

        result = provider.get_Cl(units="FIRASmuK2")

        # Should return original values
        np.testing.assert_array_almost_equal(result["tt"], cls_dict["tt"])
        np.testing.assert_array_almost_equal(result["ee"], cls_dict["ee"])

    def test_get_Cl_with_ell_factor(self):
        """Test get_Cl with ell_factor=True."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict)

        result = provider.get_Cl(ell_factor=True, units="K2")

        # Should apply D_ell = ell(ell+1) / (2π) factor
        assert "tt" in result
        # First element (ell=0) should be 0
        assert result["tt"][0] == 0.0
        # Other elements should be multiplied by factor
        assert result["tt"][1] > 0

    def test_get_Cl_caching(self):
        """Test that get_Cl caches results."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict)

        # First call
        result1 = provider.get_Cl(units="K2")

        # Second call with same params should use cache
        result2 = provider.get_Cl(units="K2")

        # Results should be identical (same object from cache)
        assert result1 is result2

    def test_get_Cl_cache_clearing_on_new_data(self):
        """Test that cache is cleared when new data is set."""
        provider = Provider()

        cls_dict1 = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict1)
        result1 = provider.get_Cl(units="K2")

        # Set new data
        cls_dict2 = {"tt": np.ones(10) * 2e12}
        provider.set_Cl(cls_dict2)

        # Cache should be cleared, new result should be different
        result2 = provider.get_Cl(units="K2")

        assert not np.array_equal(result1["tt"], result2["tt"])

    def test_get_Cl_different_cache_keys(self):
        """Test that different ell_factor/units combinations use different cache entries."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict)

        result1 = provider.get_Cl(units="K2", ell_factor=False)
        result2 = provider.get_Cl(units="K2", ell_factor=True)
        result3 = provider.get_Cl(units="FIRASmuK2", ell_factor=False)

        # All should be different
        assert result1 is not result2
        assert result1 is not result3
        assert result2 is not result3

    def test_get_Cl_batch_dimensions(self):
        """Test get_Cl with batch dimensions (2D arrays)."""
        provider = Provider()

        # Create batch of spectra (n_samples, n_ell)
        cls_dict = {"tt": np.ones((5, 10)) * 1e12}
        provider.set_Cl(cls_dict)

        result = provider.get_Cl(units="K2", ell_factor=True)

        assert result["tt"].shape == (5, 10)
        # Should handle batch dimension correctly
        assert result["tt"][0, 0] == 0.0  # ell=0 is zero

    def test_get_Cl_optimization_no_copy(self):
        """Test that get_Cl returns original dict when no processing needed."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict)

        # Requesting FIRASmuK2 with no ell_factor should return original
        result = provider.get_Cl(units="FIRASmuK2", ell_factor=False)

        # Should be same object (optimization)
        assert result is provider.cmb_cls

    def test_provider_set_provider_method(self):
        """Test that likelihoods can call set_provider."""
        provider = Provider()

        # This is used by likelihoods to set the provider
        # Test that the method exists and can be called
        assert hasattr(provider, "set_Cl")
        assert hasattr(provider, "get_Cl")

    def test_get_Cl_K2_with_ell_factor_optimization(self):
        """Test get_Cl K2 units path when ell_factor=False but units=K2."""
        provider = Provider()

        cls_dict = {"tt": np.ones(10) * 1e12}
        provider.set_Cl(cls_dict)

        # This should go through the else branch at line 82-90
        result = provider.get_Cl(units="K2", ell_factor=False)

        # Should convert to K2
        np.testing.assert_array_almost_equal(result["tt"], np.ones(10))
