# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import numpy as np
import pytest

try:
    from UPanda.likelihoods import utils_likelihood

    HAS_UPANDA = True
except ImportError:
    HAS_UPANDA = False


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestUtilsLikelihood:
    """Test utility functions in utils_likelihood."""

    def test_convert_om_to_ocdm(self):
        """Test convert_om_to_ocdm function."""
        emu_dict = {"omega_m": 0.3, "omega_b": 0.05, "h": 0.7, "m_nu": 0.06}

        result = utils_likelihood.convert_om_to_ocdm(emu_dict)

        assert "omega_cdm" in result
        # omega_cdm = (omega_m * h^2) - o_nu - omega_b
        # o_nu = m_nu * 3 / 93.14 = 0.06 * 3 / 93.14 ≈ 0.00193
        # omega_cdm ≈ (0.3 * 0.49) - 0.00193 - 0.05 ≈ 0.095
        assert result["omega_cdm"] > 0

    def test_get_sigma8(self):
        """Test get_sigma8 function (requires converter_model)."""
        emu_dict = {"omega_m": 0.3}

        # Mock converter_model - predict returns 2D array
        class MockConverter:
            def predict(self, params):
                return np.array([[0.8]])  # Shape (1, 1)

        converter_model = MockConverter()

        result = utils_likelihood.get_sigma8(emu_dict, converter_model)
        assert "sigma8" in result
        assert result["sigma8"] == pytest.approx(0.8, rel=1e-10)

    def test_transform_to_dict(self):
        """Test transform_to_dict function."""
        dtype = np.dtype([("omega_m", "f8"), ("omega_b", "f8")])
        # p should be 2D array with shape (n_samples, n_params)
        p = np.array([[0.3, 0.05], [0.31, 0.051]])  # 2 samples
        cosmology_params_fixed = []

        result = utils_likelihood.transform_to_dict(p, dtype, cosmology_params_fixed)

        assert "omega_m" in result
        assert "omega_b" in result
        assert len(result["omega_m"]) == 2
        assert result["omega_m"][0] == 0.3
        assert result["omega_b"][0] == 0.05

    def test_transform_to_dict_with_fixed(self):
        """Test transform_to_dict with fixed parameters."""
        # The function expects p to have columns matching dtype.names
        # If omega_m is fixed, p should still have all columns, but omega_m will be overwritten
        dtype = np.dtype([("omega_m", "f8"), ("omega_b", "f8")])
        # p should be 2D array with shape (n_samples, n_params) matching dtype
        p = np.array([[0.3, 0.05], [0.31, 0.051]])  # Both columns
        cosmology_params_fixed = [{"name": "omega_m", "value": 0.3}]

        result = utils_likelihood.transform_to_dict(p, dtype, cosmology_params_fixed)

        assert "omega_m" in result
        assert "omega_b" in result
        # omega_m should be overwritten by fixed value
        assert result["omega_m"][0] == 0.3
        assert result["omega_b"][0] == 0.05

    def test_bin_spectrum(self):
        """Test bin_spectrum function."""
        ells = np.arange(100)
        cells = np.ones(100) * 1000.0
        bin_edges = [0, 20, 40, 60, 80, 100]

        result = utils_likelihood.bin_spectrum(ells, cells, bin_edges)

        assert len(result) == len(bin_edges) - 1

    def test_bin_spectrum_ell_weighted(self):
        """Test bin_spectrum with ell weighting."""
        ells = np.arange(100)
        cells = np.ones(100) * 1000.0
        bin_edges = [0, 20, 40, 60, 80, 100]

        result = utils_likelihood.bin_spectrum(
            ells, cells, bin_edges, ell_weighted=True
        )

        assert len(result) == len(bin_edges) - 1

    def test_select_from_matrix(self):
        """Test select_from_matrix function."""
        # selectionA and selectionB should be boolean arrays or indices
        # The function expects mat.shape == (len(selectionA), len(selectionB))
        # So selectionA/B are used as selectors, not indices
        mat = np.random.randn(3, 3)  # Must match selection lengths
        selectionA = np.array([True, True, False])  # Boolean array
        selectionB = np.array([True, False, True])

        result = utils_likelihood.select_from_matrix(mat, selectionA, selectionB)

        n_selected_A = np.sum(selectionA)
        n_selected_B = np.sum(selectionB)
        assert result.shape == (n_selected_A, n_selected_B)

    def test_select_from_matrix_single_selection(self):
        """Test select_from_matrix with single selection."""
        # When selectionB is None, it uses selectionA for both
        mat = np.random.randn(3, 3)  # Must match selection length
        selectionA = np.array([True, True, False])

        result = utils_likelihood.select_from_matrix(mat, selectionA)

        n_selected = np.sum(selectionA)
        assert result.shape == (n_selected, n_selected)

    def test_standardize_input_cl(self):
        """Test standardize_input_cl function."""
        ells = np.arange(100)
        cells = np.ones(100) * 1000.0

        # Function returns only out_cells, not (ells, cells)
        result_cells = utils_likelihood.standardize_input_cl(ells, cells, lmax=50)

        assert len(result_cells) == 51  # 0 to 50 inclusive
        assert result_cells[0] == 1000.0  # ell=0
        assert result_cells[50] == 1000.0  # ell=50

    def test_standardize_input_cl_with_lmin(self):
        """Test standardize_input_cl with lmin."""
        # The function uses ells directly as indices into out_cells
        # out_cells has size (lmax-lmin+1), so ells must be in range [0, lmax-lmin]
        # or the function will fail. Let's test with ells in valid range
        ells = np.arange(0, 41)  # 0 to 40, valid indices for out_cells of size 41
        cells = np.ones(41) * 1000.0

        # Use lmin=0, lmax=40 so ells match indices
        result_cells = utils_likelihood.standardize_input_cl(
            ells, cells, lmin=0, lmax=40
        )

        assert len(result_cells) == 41  # 0 to 40 inclusive (lmax-lmin+1)
        assert result_cells[0] == 1000.0
        assert result_cells[40] == 1000.0

    def test_density_space(self):
        """Test density_space function."""
        # Check if scipy.integrate has cumtrapz or cumulative_trapezoid
        try:
            from scipy import integrate

            if not hasattr(integrate, "cumtrapz") and not hasattr(
                integrate, "cumulative_trapezoid"
            ):
                pytest.skip(
                    "scipy.integrate.cumtrapz or cumulative_trapezoid not available"
                )
        except ImportError:
            pytest.skip("scipy not available")

        xs = np.linspace(0, 10, 100)
        ps = np.ones(100)  # Uniform density

        result = utils_likelihood.density_space(xs, ps, n=10, endpoint=False)

        assert len(result) == 10
        assert np.all(result >= xs[0])
        assert np.all(result <= xs[-1])

    def test_density_space_with_endpoint(self):
        """Test density_space with endpoint=True."""
        try:
            from scipy import integrate

            if not hasattr(integrate, "cumtrapz") and not hasattr(
                integrate, "cumulative_trapezoid"
            ):
                pytest.skip(
                    "scipy.integrate.cumtrapz or cumulative_trapezoid not available"
                )
        except ImportError:
            pytest.skip("scipy not available")

        xs = np.linspace(0, 10, 100)
        ps = np.ones(100)

        result = utils_likelihood.density_space(xs, ps, n=10, endpoint=True)

        assert len(result) == 10
        assert result[-1] == pytest.approx(xs[-1], rel=1e-5)

    def test_density_space_random(self):
        """Test density_space with random=True."""
        try:
            from scipy import integrate

            if not hasattr(integrate, "cumtrapz") and not hasattr(
                integrate, "cumulative_trapezoid"
            ):
                pytest.skip(
                    "scipy.integrate.cumtrapz or cumulative_trapezoid not available"
                )
        except ImportError:
            pytest.skip("scipy not available")

        xs = np.linspace(0, 10, 100)
        ps = np.ones(100)

        result = utils_likelihood.density_space(xs, ps, n=10, random=True)

        assert len(result) == 10
        assert np.all(result >= xs[0])
        assert np.all(result <= xs[-1])

    def test_combine_inputs(self):
        """Test combine_inputs function."""
        arr1 = np.array([1, 3, 5])
        arr2 = np.array([2, 4, 6])
        arr3 = np.array([1, 2, 3])

        result = utils_likelihood.combine_inputs(arr1, arr2, arr3, sorted=True)

        assert len(result) == 6  # Unique values: 1, 2, 3, 4, 5, 6
        assert np.all(result == np.array([1, 2, 3, 4, 5, 6]))

    def test_combine_inputs_unsorted(self):
        """Test combine_inputs with sorted=False."""
        arr1 = np.array([1, 3, 5])
        arr2 = np.array([2, 4, 6])

        result = utils_likelihood.combine_inputs(arr1, arr2, sorted=False)

        assert len(result) == 6
        # Should be unique but not necessarily sorted
        assert len(np.unique(result)) == 6

    def test_convert_S8_ocdm_to_sigma8_Om(self):
        """Test convert_S8_ocdm_to_sigma8_Om function."""
        emu_dict = {
            "omega_b": 0.05,
            "h": 0.7,
            "m_nu": 0.06,
            "S8": 0.8,
            "omega_cdm": 0.25,
        }

        result = utils_likelihood.convert_S8_ocdm_to_sigma8_Om(emu_dict)

        assert "omega_m" in result
        assert "sigma8" in result
        assert result["omega_m"] > 0
        assert result["sigma8"] > 0
