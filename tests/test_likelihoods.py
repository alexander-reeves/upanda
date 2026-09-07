# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import os
from unittest.mock import patch

import numpy as np
import pytest

# Import test helpers from conftest
try:
    from .conftest import LikelihoodTestHelper
except ImportError:
    # If relative import fails, try absolute
    from tests.conftest import LikelihoodTestHelper

try:
    from UPanda.likelihoods import Likelihood

    HAS_UPANDA = True
except ImportError:
    HAS_UPANDA = False


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestLikelihoodBase:
    """Test base Likelihood class."""

    def test_likelihood_init(self):
        """Test Likelihood initialization."""
        data_vector = np.array([1.0, 2.0, 3.0])
        cov_matrix = np.eye(3)

        likelihood = Likelihood(data_vector=data_vector, covariance_matrix=cov_matrix)

        assert likelihood.data_vector is not None
        assert likelihood.covariance_matrix is not None

    def test_likelihood_get_prediction(self):
        """Test Likelihood get_prediction placeholder."""
        likelihood = Likelihood()
        params = np.array([[0.3, 0.05]])
        # Base class implementation tries to iterate over emu_model_list
        # If None, it will fail, so we test with an empty list
        likelihood.emu_model_list = []
        # This should work but return empty list
        result = likelihood.get_prediction(params)
        assert result == []  # Returns empty list for empty emu_model_list

    def test_likelihood_compute_like(self):
        """Test Likelihood compute_like placeholder."""
        likelihood = Likelihood()
        params = np.array([[0.3, 0.05]])
        # Base class needs data_vector and covariance_matrix
        likelihood.data_vector = np.array([1.0, 2.0])
        likelihood.covariance_matrix = np.eye(2)

        # Create a mock model with predict method that returns array
        class MockModel:
            def predict(self, p):
                return np.array([1.0, 2.0])  # Returns 1D array

        likelihood.emu_model_list = [MockModel()]

        # The code has a bug: return -0.5 * chi2_values where chi2_values is a list
        # This will raise TypeError. Let's expect this failure.
        with pytest.raises(NotImplementedError, match="must implement compute_like"):
            result = likelihood.compute_like(params)


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestDESIBAODR2:
    """Test desi_bao_dr2 likelihood with mocked data."""

    def test_import(self):
        """Test that desi_bao_dr2 can be imported."""
        from UPanda.likelihoods import desi_bao_dr2

        assert hasattr(desi_bao_dr2, "desi_bao_dr2")

    def test_init_with_mocked_data(
        self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch
    ):
        """Test initialization with properly mocked data files."""
        from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

        # Create mock data files
        n_data = 5
        mock_mean = LikelihoodTestHelper.create_mock_data_vector(n_data, seed=42)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_data, seed=42)

        data_dir = mock_likelihood_data_dir["data_dir"]
        # Create the correct directory structure: {data_dir}/likelihoods/data/desi_bao_dr2/
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        target_data_dir = os.path.join(likelihood_dir, "data", "desi_bao_dr2")
        os.makedirs(target_data_dir, exist_ok=True)

        # Create mean.txt with 2 columns (index, value) because usecols=(1) takes second column
        mean_with_index = np.column_stack([np.arange(len(mock_mean)), mock_mean])
        np.savetxt(os.path.join(target_data_dir, "mean.txt"), mean_with_index)
        np.savetxt(os.path.join(target_data_dir, "cov.txt"), mock_cov)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "desi_bao_dr2")

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            simple_cosmology_dtype, additional_args={}
        )
        likelihood = desi_bao_dr2(**kwargs)

        assert hasattr(likelihood, "desi_bao_data")
        assert hasattr(likelihood, "desi_bao_covariance")
        assert hasattr(likelihood, "inv_c_bao")
        assert likelihood.desi_bao_data.shape[0] == n_data
        assert likelihood.desi_bao_covariance.shape == (n_data, n_data)

    def test_get_prediction_with_mock_emulator(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test get_prediction with a mock emulator."""
        from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

        # Setup mock data
        n_data = 5
        mock_mean = LikelihoodTestHelper.create_mock_data_vector(n_data, seed=42)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_data, seed=42)

        data_dir = mock_likelihood_data_dir["data_dir"]
        # Create the correct directory structure: {data_dir}/likelihoods/data/desi_bao_dr2/
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        target_data_dir = os.path.join(likelihood_dir, "data", "desi_bao_dr2")
        os.makedirs(target_data_dir, exist_ok=True)

        # Create mean.txt with 2 columns (index, value) because usecols=(1) takes second column
        mean_with_index = np.column_stack([np.arange(len(mock_mean)), mock_mean])
        np.savetxt(os.path.join(target_data_dir, "mean.txt"), mean_with_index)
        np.savetxt(os.path.join(target_data_dir, "cov.txt"), mock_cov)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "desi_bao_dr2")

        # Create mock emulator
        mock_emu = mock_emulator_model(output_shape=(n_data,))

        # desi_bao_dr2 needs h and m_nu for convert_om_to_ocdm if omega_cdm not present
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h, emu_model_list=[mock_emu], additional_args={}
        )
        likelihood = desi_bao_dr2(**kwargs)

        # Test get_prediction - use (1,) shape for structured array, not (1,1)
        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        prediction = likelihood.get_prediction(params)

        assert prediction is not None
        assert len(prediction.shape) >= 1

    def test_compute_like_with_mock_data(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test compute_like with mocked data and emulator."""
        from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

        # Setup mock data
        n_data = 5
        mock_mean = LikelihoodTestHelper.create_mock_data_vector(n_data, seed=42)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_data, seed=42)

        data_dir = mock_likelihood_data_dir["data_dir"]
        # Create the correct directory structure: {data_dir}/likelihoods/data/desi_bao_dr2/
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        target_data_dir = os.path.join(likelihood_dir, "data", "desi_bao_dr2")
        os.makedirs(target_data_dir, exist_ok=True)

        # Create mean.txt with 2 columns (index, value) because usecols=(1) takes second column
        mean_with_index = np.column_stack([np.arange(len(mock_mean)), mock_mean])
        np.savetxt(os.path.join(target_data_dir, "mean.txt"), mean_with_index)
        np.savetxt(os.path.join(target_data_dir, "cov.txt"), mock_cov)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "desi_bao_dr2")

        # Create mock emulator that returns predictable output
        class PredictableEmulator:
            def predict(self, params_dict):
                n_samples = len(next(iter(params_dict.values())))
                # Return mock_mean for each sample (predictable)
                return np.tile(mock_mean, (n_samples, 1))

        mock_emu = PredictableEmulator()

        # desi_bao_dr2 needs h and m_nu for convert_om_to_ocdm if omega_cdm not present
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h, emu_model_list=[mock_emu], additional_args={}
        )
        likelihood = desi_bao_dr2(**kwargs)

        # Test compute_like
        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        loglike = likelihood.compute_like(params)

        assert loglike is not None
        # Should return log likelihood (negative or zero)
        assert np.all(loglike <= 0)  # Log likelihood should be <= 0

    @patch("os.path.dirname")
    @patch("os.path.join")
    @patch("numpy.loadtxt")
    def test_init_with_mock_data(self, mock_loadtxt, mock_join, mock_dirname):
        """Test initialization with mocked data files."""
        from UPanda.likelihoods import desi_bao_dr2

        # Mock file paths
        mock_dirname.return_value = "/fake/path"
        mock_join.side_effect = lambda *args: "/".join(args)

        # Mock data files
        mock_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_cov = np.eye(5)
        mock_loadtxt.side_effect = [mock_mean, mock_cov]

        dtype = np.dtype([("omega_m", "f8"), ("h", "f8")])

        likelihood = desi_bao_dr2.desi_bao_dr2(
            dtype=dtype, additional_args={"mock_data": mock_mean}
        )

        assert likelihood.desi_bao_data is not None
        assert hasattr(likelihood, "inv_c_bao")

    @patch("os.path.dirname")
    @patch("os.path.join")
    @patch("numpy.loadtxt")
    def test_set_external_spectra(self, mock_loadtxt, mock_join, mock_dirname):
        """Test set_external_spectra method."""
        from UPanda.likelihoods import desi_bao_dr2

        mock_dirname.return_value = "/fake/path"
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_loadtxt.side_effect = [np.array([1.0]), np.eye(1)]

        dtype = np.dtype([("omega_m", "f8"), ("h", "f8")])
        likelihood = desi_bao_dr2.desi_bao_dr2(
            dtype=dtype, additional_args={"mock_data": np.array([1.0])}
        )

        external_data = {"desi_bao_data": np.array([2.0, 3.0, 4.0])}
        likelihood.set_external_spectra(external_data)
        assert likelihood.external_spectra is not None


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestDESIBAO2024:
    """Test desi_bao_2024 likelihood with mocked data."""

    def test_import(self):
        """Test that desi_bao_2024 can be imported."""
        from UPanda.likelihoods import desi_bao_2024

        assert hasattr(desi_bao_2024, "desi_bao_2024")

    @patch("os.path.dirname")
    @patch("os.path.join")
    @patch("numpy.loadtxt")
    def test_init(self, mock_loadtxt, mock_join, mock_dirname):
        """Test initialization with mocked data."""
        from UPanda.likelihoods import desi_bao_2024

        mock_dirname.return_value = "/fake/path"
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_loadtxt.side_effect = [np.array([1.0, 2.0]), np.eye(2)]

        dtype = np.dtype([("omega_m", "f8"), ("h", "f8")])

        likelihood = desi_bao_2024.desi_bao_2024(dtype=dtype, additional_args={})

        assert hasattr(likelihood, "dtype")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPantheonPlus:
    """Test pantheon_plus likelihood with mocked data."""

    def test_import(self):
        """Test that pantheon_plus can be imported."""
        from UPanda.likelihoods import pantheon_plus

        assert hasattr(pantheon_plus, "pantheon_plus")

    def test_init_with_mocked_data(
        self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch
    ):
        """Test initialization with properly mocked data files."""
        from UPanda.likelihoods.pantheon_plus import pantheon_plus

        # Create mock Pantheon+ data
        n_sne = 100  # Smaller number for testing
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        pantheon_dir = os.path.join(likelihood_dir, "data", "pantheon_plus")
        os.makedirs(pantheon_dir, exist_ok=True)

        # Create mock data file with header
        data_file = os.path.join(pantheon_dir, "Pantheon+SH0ES.dat")
        with open(data_file, "w") as f:
            f.write("m_b_corr zHD zHEL\n")
            for i in range(n_sne):
                z_cmb = 0.01 + i * 0.01
                z_hel = z_cmb * 1.001  # Small difference
                mag = 19.0 + z_cmb * 5.0
                f.write(f"{mag} {z_cmb} {z_hel}\n")

        # Create mock covariance matrix
        cov_file = os.path.join(pantheon_dir, "Pantheon+SH0ES_STAT+SYS.cov")
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_sne, seed=42)
        np.savetxt(cov_file, mock_cov)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "pantheon_plus")

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            simple_cosmology_dtype, additional_args={}
        )
        likelihood = pantheon_plus(use_shoes=False, **kwargs)

        assert hasattr(likelihood, "cov")
        assert hasattr(likelihood, "inv_cov")
        assert hasattr(likelihood, "mag")
        assert hasattr(likelihood, "zcmb")
        assert hasattr(likelihood, "zhel")
        # After masking z > 0.01, should have fewer SNe
        assert hasattr(likelihood, "nsn")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestACDR4:
    """Test act_dr4 likelihood with mocked data."""

    def test_import(self):
        """Test that act_dr4 can be imported."""
        from UPanda.likelihoods import act_dr4

        assert hasattr(act_dr4, "act_dr4")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestACDR6:
    """Test act_dr6 likelihood with mocked data."""

    def test_import(self):
        """Test that act_dr6 can be imported."""
        from UPanda.likelihoods import act_dr6

        assert hasattr(act_dr6, "act_dr6")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestACDR6Lensing:
    """Test act_dr6_lensing likelihood with mocked data."""

    def test_import(self):
        """Test that act_dr6_lensing can be imported."""
        from UPanda.likelihoods import act_dr6_lensing

        assert hasattr(act_dr6_lensing, "act_dr6_lensing")

    def test_init(self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch):
        """Test initialization with mocked data."""
        from UPanda.likelihoods import act_dr6_lensing

        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        act_lens_dir = os.path.join(likelihood_dir, "data", "act_dr6_lensing")
        os.makedirs(act_lens_dir, exist_ok=True)

        # Create mock data files
        n_bins = 9
        mock_data = LikelihoodTestHelper.create_mock_data_vector(n_bins, seed=42)
        mock_binning = np.eye(n_bins)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42)

        np.savetxt(os.path.join(act_lens_dir, "clkk_bandpowers_act.txt"), mock_data)
        np.savetxt(
            os.path.join(act_lens_dir, "binning_matrix_ACT-DR6_Clkk.txt"), mock_binning
        )
        np.savetxt(
            os.path.join(act_lens_dir, "covmat_ACT-DR6_Clkk_baseline.txt"), mock_cov
        )
        np.savetxt(
            os.path.join(act_lens_dir, "covmat_ACT-DR6_Clkk_cmbmarg.txt"), mock_cov
        )

        # Create mock fiducial data files
        fiducial_data = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000) * 1e-10,  # TT
                np.ones(3000) * 1e-12,  # EE
                np.ones(3000) * 1e-14,  # BB
                np.ones(3000) * 1e-11,  # TE
            ]
        )
        np.savetxt(
            os.path.join(act_lens_dir, "cosmo2017_10K_acc3_lensedCls.dat"),
            fiducial_data,
        )

        fiducial_phi = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000) * 1e-10,  # TT
                np.ones(3000) * 1e-12,  # EE
                np.ones(3000) * 1e-14,  # BB
                np.ones(3000) * 1e-11,  # TE
                np.ones(3000) * 1e-13,  # PhiPhi
            ]
        )
        np.savetxt(
            os.path.join(act_lens_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"),
            fiducial_phi,
        )

        # Create N0 and N1der files
        np.savetxt(
            os.path.join(
                act_lens_dir, "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"
            ),
            np.ones(2401),
        )
        np.savetxt(
            os.path.join(act_lens_dir, "N1der_KK_lmin600_lmax3000_full.txt"),
            np.ones((n_bins, 2401)),
        )
        for spec in ["TT", "EE", "BB", "TE"]:
            np.savetxt(
                os.path.join(act_lens_dir, f"N1der_{spec}_lmin600_lmax3000_full.txt"),
                np.ones((n_bins, 2401)),
            )
        np.save(
            os.path.join(act_lens_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"),
            np.ones((n_bins, 2401)),
        )

        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "act_dr6_lensing")

        dtype = np.dtype([("omega_m", "f8")])

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype, additional_args={"want_lensing_lklh_correction": False}
        )
        likelihood = act_dr6_lensing.act_dr6_lensing(**kwargs)

        assert hasattr(likelihood, "dtype")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestCamSpecPR4:
    """Test camspec_pr4 likelihood with mocked data."""

    def test_import(self):
        """Test that camspec_pr4 can be imported."""
        from UPanda.likelihoods import camspec_pr4

        assert hasattr(camspec_pr4, "camspec_pr4")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestGeneralLSS:
    """Test general_lss likelihood with mocked data."""

    def test_import(self):
        """Test that general_lss can be imported."""
        from UPanda.likelihoods import general_lss

        assert hasattr(general_lss, "general_lss")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPlanck2018Lensing:
    """Test planck_2018_lensing likelihood with mocked data."""

    def test_import(self):
        """Test that planck_2018_lensing can be imported."""
        from UPanda.likelihoods import planck_2018_lensing

        assert hasattr(planck_2018_lensing, "planck_2018_lensing")

    def test_init_with_mocked_data(
        self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch
    ):
        """Test initialization with properly mocked data files."""
        from UPanda.likelihoods.planck_2018_lensing import planck_2018_lensing

        # Create mock Planck 2018 Lensing data
        n_bins = 9
        n_ell = 2500
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        lensing_dir = os.path.join(likelihood_dir, "data", "planck_2018_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        # Create mock data files
        binning_matrix = np.random.randn(n_bins, n_ell)
        cl_hat = np.ones(n_bins)
        covariance = LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42)
        theory_offset = np.zeros(n_bins)

        np.savetxt(
            os.path.join(lensing_dir, "binning_plus_correction_matrix.txt"),
            binning_matrix,
        )
        np.savetxt(os.path.join(lensing_dir, "cl_hat.txt"), cl_hat)
        np.savetxt(os.path.join(lensing_dir, "covariance.txt"), covariance)
        np.savetxt(os.path.join(lensing_dir, "theory_offset.txt"), theory_offset)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(
            monkeypatch, data_dir, "planck_2018_lensing"
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            simple_cosmology_dtype, additional_args={}
        )
        likelihood = planck_2018_lensing(**kwargs)

        assert hasattr(likelihood, "bin_mat_lens")
        assert hasattr(likelihood, "cl_hat")
        assert hasattr(likelihood, "covariance")
        assert hasattr(likelihood, "theory_offset")
        assert hasattr(likelihood, "inv_c_lens")
        assert likelihood.bin_mat_lens.shape == (n_bins, n_ell)
        assert likelihood.cl_hat.shape == (n_bins,)
        assert likelihood.covariance.shape == (n_bins, n_bins)

    def test_get_prediction_with_mock_emulator(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test get_prediction with a mock emulator."""
        from UPanda.likelihoods.planck_2018_lensing import planck_2018_lensing

        # Setup mock data
        n_bins = 9
        n_ell = 2499  # ell from 2 to 2500 = 2499 elements
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        lensing_dir = os.path.join(likelihood_dir, "data", "planck_2018_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        binning_matrix = np.random.randn(n_bins, n_ell)
        cl_hat = np.ones(n_bins)
        covariance = LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42)
        theory_offset = np.zeros(n_bins)

        np.savetxt(
            os.path.join(lensing_dir, "binning_plus_correction_matrix.txt"),
            binning_matrix,
        )
        np.savetxt(os.path.join(lensing_dir, "cl_hat.txt"), cl_hat)
        np.savetxt(os.path.join(lensing_dir, "covariance.txt"), covariance)
        np.savetxt(os.path.join(lensing_dir, "theory_offset.txt"), theory_offset)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(
            monkeypatch, data_dir, "planck_2018_lensing"
        )

        # Create mock emulator - output should be Cl_kk for ell=2 to 2500
        n_ell_range = n_ell  # ell from 2 to 2500 = 2499 elements
        mock_emu = mock_emulator_model(output_shape=(n_ell_range,))

        # planck_2018_lensing needs 'h' and 'm_nu' parameters
        dtype_with_h_mnu = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h_mnu,
            emu_model_list=[mock_emu],
            additional_args={"integrated_model": False, "lens_exp": False},
        )
        likelihood = planck_2018_lensing(**kwargs)

        # Test get_prediction - use (1,) shape for structured array, not (1,1)
        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h_mnu)
        prediction = likelihood.get_prediction(params)

        assert prediction is not None
        assert prediction.shape[0] == 1  # 1 sample
        assert prediction.shape[1] == n_bins  # After binning

    def test_compute_like_with_mock_data(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test compute_like with mocked data and emulator."""
        from UPanda.likelihoods.planck_2018_lensing import planck_2018_lensing

        # Setup mock data
        n_bins = 9
        n_ell = 2499  # ell from 2 to 2500 = 2499 elements
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        lensing_dir = os.path.join(likelihood_dir, "data", "planck_2018_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        binning_matrix = np.random.randn(n_bins, n_ell)
        cl_hat = np.ones(n_bins)
        covariance = LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42)
        theory_offset = np.zeros(n_bins)

        np.savetxt(
            os.path.join(lensing_dir, "binning_plus_correction_matrix.txt"),
            binning_matrix,
        )
        np.savetxt(os.path.join(lensing_dir, "cl_hat.txt"), cl_hat)
        np.savetxt(os.path.join(lensing_dir, "covariance.txt"), covariance)
        np.savetxt(os.path.join(lensing_dir, "theory_offset.txt"), theory_offset)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(
            monkeypatch, data_dir, "planck_2018_lensing"
        )

        # Create mock emulator
        n_ell_range = n_ell  # ell from 2 to 2500 = 2499 elements
        mock_emu = mock_emulator_model(output_shape=(n_ell_range,))

        # planck_2018_lensing needs 'h' and 'm_nu' parameters
        dtype_with_h_mnu = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h_mnu,
            emu_model_list=[mock_emu],
            additional_args={"integrated_model": False, "lens_exp": False},
        )
        likelihood = planck_2018_lensing(**kwargs)

        # Test compute_like
        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h_mnu)
        loglike = likelihood.compute_like(params)

        assert loglike is not None
        assert np.all(loglike <= 0)  # Log likelihood should be <= 0


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPlanck2018HighellLowell:
    """Test planck_2018_highell_lowell likelihood with mocked data."""

    def test_import(self):
        """Test that planck_2018_highell_lowell can be imported."""
        from UPanda.likelihoods import planck_2018_highell_lowell

        assert hasattr(planck_2018_highell_lowell, "planck_2018_highell_lowell")

    def test_init(self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch):
        """Test initialization with mocked data."""
        from UPanda.likelihoods import planck_2018_highell_lowell

        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        highell_dir = os.path.join(likelihood_dir, "data", "planck_2018_highell")
        os.makedirs(highell_dir, exist_ok=True)
        lowell_dir = os.path.join(likelihood_dir, "data", "planck_2018_lowl")
        os.makedirs(lowell_dir, exist_ok=True)

        # Create mock binning files
        n_bins = 613
        np.savetxt(os.path.join(highell_dir, "blmin.dat"), np.arange(30, 30 + n_bins))
        np.savetxt(os.path.join(highell_dir, "blmax.dat"), np.arange(31, 31 + n_bins))
        np.savetxt(os.path.join(highell_dir, "bweight.dat"), np.ones(n_bins))

        # Create mock data vector file (bval, data_vector, X_sig)
        mock_plik_data = np.column_stack(
            [np.arange(n_bins), np.random.randn(n_bins) * 1e-6, np.ones(n_bins) * 1e-7]
        )
        np.savetxt(os.path.join(highell_dir, "cl_cmb_plik_v22.dat"), mock_plik_data)

        # Create mock covariance (Fortran format)
        from scipy.io import FortranFile

        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42)
        cov_file = os.path.join(highell_dir, "c_matrix_plik_v22.dat")
        with FortranFile(cov_file, "w") as f:
            f.write_record(mock_cov.astype(np.float64).T)

        # Create mock low-ell lognormal files
        ee_data = np.array(
            [
                [2, 29, 0.1, 0.2, 0.0],
                [30, 99, 0.15, 0.25, 0.0],
                [100, 199, 0.2, 0.3, 0.0],
            ]
        )
        np.savetxt(os.path.join(lowell_dir, "lognormal_fit_3bins_EE.txt"), ee_data)

        tt_data = np.array([[2, 29, 0.1, 0.2], [30, 99, 0.15, 0.25]])
        np.savetxt(os.path.join(lowell_dir, "lognormal_fit_2bins_TT.txt"), tt_data)

        LikelihoodTestHelper.setup_path_mocks(
            monkeypatch, data_dir, "planck_2018_highell_lowell"
        )

        dtype = np.dtype([("omega_m", "f8"), ("omega_b", "f8")])

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype, additional_args={}
        )
        likelihood = planck_2018_highell_lowell.planck_2018_highell_lowell(**kwargs)

        assert hasattr(likelihood, "dtype")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPlanck2018ForACT:
    """Test planck_2018_for_act likelihood with mocked data."""

    def test_import(self):
        """Test that planck_2018_for_act can be imported."""
        from UPanda.likelihoods import planck_2018_for_act

        assert hasattr(planck_2018_for_act, "planck_2018_for_act")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPlanck2018ForACTDR6:
    """Test planck_2018_for_act_dr6 likelihood with mocked data."""

    def test_import(self):
        """Test that planck_2018_for_act_dr6 can be imported."""
        from UPanda.likelihoods import planck_2018_for_act_dr6

        assert hasattr(planck_2018_for_act_dr6, "planck_2018_for_act_dr6")


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestPlanckSroll2LowellEE:
    """Test planck_sroll2_lowell_ee likelihood with mocked data."""

    def test_import(self):
        """Test that planck_sroll2_lowell_ee can be imported."""
        from UPanda.likelihoods import planck_sroll2_lowell_ee

        assert hasattr(planck_sroll2_lowell_ee, "planck_sroll2_lowell_ee")

    @patch("os.path.dirname")
    @patch("os.path.join")
    @patch("numpy.loadtxt")
    def test_init(self, mock_loadtxt, mock_join, mock_dirname):
        """Test initialization with mocked data."""
        from UPanda.likelihoods import planck_sroll2_lowell_ee

        mock_dirname.return_value = "/fake/path"
        mock_join.side_effect = lambda *args: "/".join(args)

        # Mock probability table
        mock_loadtxt.return_value = np.random.randn(100, 29)

        dtype = np.dtype([("omega_m", "f8"), ("omega_b", "f8")])

        likelihood = planck_sroll2_lowell_ee.planck_sroll2_lowell_ee(
            dtype=dtype, additional_args={}, cosmology_params_fixed=None
        )

        assert hasattr(likelihood, "dtype")


# ============================================================================
# Comprehensive tests for all likelihoods: get_prediction and compute_like
# ============================================================================


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestDESIBAO2024Comprehensive:
    """Comprehensive tests for desi_bao_2024 likelihood."""

    def test_desi_bao_2024_get_prediction(
        self, mock_likelihood_data_dir, mock_emulator_model, monkeypatch
    ):
        """Test desi_bao_2024 get_prediction method."""
        from UPanda.likelihoods.desi_bao_2024 import desi_bao_2024

        n_data = 5
        mock_mean = LikelihoodTestHelper.create_mock_data_vector(n_data, seed=42)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_data, seed=42)

        data_dir = mock_likelihood_data_dir["data_dir"]
        desi_dir = os.path.join(data_dir, "data", "desi_bao_2024")
        os.makedirs(desi_dir, exist_ok=True)

        mean_with_index = np.column_stack([np.arange(len(mock_mean)), mock_mean])
        np.savetxt(os.path.join(desi_dir, "mean.txt"), mean_with_index)
        np.savetxt(os.path.join(desi_dir, "cov.txt"), mock_cov)

        def mock_abspath(path):
            return os.path.join(data_dir, "likelihoods", "desi_bao_2024.py")

        def mock_dirname(path):
            if "likelihoods" in str(path) and "desi_bao_2024.py" in str(path):
                return data_dir
            import os.path as orig_path

            return orig_path.dirname(path)

        monkeypatch.setattr(os.path, "abspath", mock_abspath)
        monkeypatch.setattr(os.path, "dirname", mock_dirname)

        mock_emu = mock_emulator_model(output_shape=(n_data,))

        # DESI BAO needs h and m_nu parameters
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h, emu_model_list=[mock_emu], additional_args={}
        )
        likelihood = desi_bao_2024(**kwargs)

        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        prediction = likelihood.get_prediction(params)

        assert prediction is not None
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == n_data

    def test_desi_bao_2024_compute_like(
        self, mock_likelihood_data_dir, mock_emulator_model, monkeypatch
    ):
        """Test desi_bao_2024 compute_like method."""
        from UPanda.likelihoods.desi_bao_2024 import desi_bao_2024

        n_data = 5
        mock_mean = LikelihoodTestHelper.create_mock_data_vector(n_data, seed=42)
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_data, seed=42)

        data_dir = mock_likelihood_data_dir["data_dir"]
        desi_dir = os.path.join(data_dir, "data", "desi_bao_2024")
        os.makedirs(desi_dir, exist_ok=True)

        mean_with_index = np.column_stack([np.arange(len(mock_mean)), mock_mean])
        np.savetxt(os.path.join(desi_dir, "mean.txt"), mean_with_index)
        np.savetxt(os.path.join(desi_dir, "cov.txt"), mock_cov)

        def mock_abspath(path):
            return os.path.join(data_dir, "likelihoods", "desi_bao_2024.py")

        def mock_dirname(path):
            if "likelihoods" in str(path) and "desi_bao_2024.py" in str(path):
                return data_dir
            import os.path as orig_path

            return orig_path.dirname(path)

        monkeypatch.setattr(os.path, "abspath", mock_abspath)
        monkeypatch.setattr(os.path, "dirname", mock_dirname)

        class PredictableEmulator:
            def predict(self, params_dict):
                n_samples = len(next(iter(params_dict.values())))
                return np.tile(mock_mean, (n_samples, 1))

        mock_emu = PredictableEmulator()

        # DESI BAO needs h and m_nu parameters
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h, emu_model_list=[mock_emu], additional_args={}
        )
        likelihood = desi_bao_2024(**kwargs)

        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        loglike = likelihood.compute_like(params)

        assert loglike is not None
        assert np.all(loglike <= 0)

    def test_pantheon_plus_set_external_spectra(
        self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch
    ):
        """Test pantheon_plus set_external_spectra method."""
        from UPanda.likelihoods.pantheon_plus import pantheon_plus

        n_sne = 50
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        pantheon_dir = os.path.join(likelihood_dir, "data", "pantheon_plus")
        os.makedirs(pantheon_dir, exist_ok=True)

        data_file = os.path.join(pantheon_dir, "Pantheon+SH0ES.dat")
        with open(data_file, "w") as f:
            f.write("m_b_corr zHD zHEL\n")
            for i in range(n_sne):
                z_cmb = 0.01 + i * 0.01
                z_hel = z_cmb * 1.001
                mag = 19.0 + z_cmb * 5.0
                f.write(f"{mag} {z_cmb} {z_hel}\n")

        cov_file = os.path.join(pantheon_dir, "Pantheon+SH0ES_STAT+SYS.cov")
        mock_cov = LikelihoodTestHelper.create_mock_covariance(n_sne, seed=42)
        np.savetxt(cov_file, mock_cov)

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "pantheon_plus")

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            simple_cosmology_dtype, additional_args={}
        )
        likelihood = pantheon_plus(use_shoes=False, **kwargs)

        external_data = {"distance_modulus": np.ones(likelihood.nsn) * 19.0}
        likelihood.set_external_spectra(external_data)
        assert likelihood.external_spectra is not None

        # Test get_prediction uses external spectra
        params = np.array([(0.3, 0.05)], dtype=simple_cosmology_dtype)
        prediction = likelihood.get_prediction(params)
        assert prediction is not None


@pytest.mark.skipif(not HAS_UPANDA, reason="UPanda not installed")
class TestACDR6LensingComprehensive:
    """Comprehensive tests for act_dr6_lensing likelihood."""

    def test_act_dr6_lensing_get_prediction(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test act_dr6_lensing get_prediction method."""
        from UPanda.likelihoods import act_dr6_lensing

        n_bins = 10
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        lensing_dir = os.path.join(likelihood_dir, "data", "act_dr6_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        # Create mock data files
        np.savetxt(
            os.path.join(lensing_dir, "clkk_bandpowers_act.txt"), np.ones(n_bins)
        )
        np.savetxt(
            os.path.join(lensing_dir, "binning_matrix_ACT-DR6_Clkk.txt"),
            np.random.randn(n_bins, 3000),
        )
        np.savetxt(
            os.path.join(lensing_dir, "covmat_ACT-DR6_Clkk_baseline.txt"),
            LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42),
        )
        np.savetxt(
            os.path.join(lensing_dir, "covmat_ACT-DR6_Clkk_cmbmarg.txt"),
            LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42),
        )

        # Mock fiducial files
        fiducial_data = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lensedCls.dat"), fiducial_data
        )
        fiducial_phi = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"),
            fiducial_phi,
        )
        np.savetxt(
            os.path.join(
                lensing_dir, "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"
            ),
            np.ones(2401),
        )
        np.savetxt(
            os.path.join(lensing_dir, "N1der_KK_lmin600_lmax3000_full.txt"),
            np.ones((n_bins, 2401)),
        )
        for spec in ["TT", "EE", "BB", "TE"]:
            np.savetxt(
                os.path.join(lensing_dir, f"N1der_{spec}_lmin600_lmax3000_full.txt"),
                np.ones((n_bins, 2401)),
            )
        np.save(
            os.path.join(lensing_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"),
            np.ones((n_bins, 2401)),
        )

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "act_dr6_lensing")

        # ACT DR6 Lensing needs Cl_kk emulator and h parameter
        mock_emu = mock_emulator_model(output_shape=(3000,))
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h,
            emu_model_list=[mock_emu],
            additional_args={
                "want_lensing_lklh_correction": False,
                "cmb_marginalized": False,
            },
        )
        likelihood = act_dr6_lensing.act_dr6_lensing(**kwargs)

        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        prediction = likelihood.get_prediction(params)

        assert prediction is not None
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == n_bins

    def test_act_dr6_lensing_compute_like(
        self,
        mock_likelihood_data_dir,
        simple_cosmology_dtype,
        mock_emulator_model,
        monkeypatch,
    ):
        """Test act_dr6_lensing compute_like method."""
        from UPanda.likelihoods import act_dr6_lensing

        n_bins = 10
        data_dir = mock_likelihood_data_dir["data_dir"]
        lensing_dir = os.path.join(data_dir, "data", "act_dr6_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        np.savetxt(
            os.path.join(lensing_dir, "clkk_bandpowers_act.txt"), np.ones(n_bins)
        )
        np.savetxt(
            os.path.join(lensing_dir, "binning_matrix_ACT-DR6_Clkk.txt"),
            np.random.randn(n_bins, 3000),
        )
        np.savetxt(
            os.path.join(lensing_dir, "covmat_ACT-DR6_Clkk_baseline.txt"),
            LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42),
        )

        fiducial_data = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lensedCls.dat"), fiducial_data
        )
        fiducial_phi = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"),
            fiducial_phi,
        )
        np.savetxt(
            os.path.join(
                lensing_dir, "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"
            ),
            np.ones(2401),
        )
        np.savetxt(
            os.path.join(lensing_dir, "N1der_KK_lmin600_lmax3000_full.txt"),
            np.ones((n_bins, 2401)),
        )
        for spec in ["TT", "EE", "BB", "TE"]:
            np.savetxt(
                os.path.join(lensing_dir, f"N1der_{spec}_lmin600_lmax3000_full.txt"),
                np.ones((n_bins, 2401)),
            )
        np.save(
            os.path.join(lensing_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"),
            np.ones((n_bins, 2401)),
        )

        def mock_abspath(path):
            return os.path.join(data_dir, "likelihoods", "act_dr6_lensing.py")

        def mock_dirname(path):
            if "likelihoods" in str(path) and "act_dr6_lensing.py" in str(path):
                return data_dir
            import os.path as orig_path

            return orig_path.dirname(path)

        monkeypatch.setattr(os.path, "abspath", mock_abspath)
        monkeypatch.setattr(os.path, "dirname", mock_dirname)

        mock_emu = LikelihoodTestHelper.create_predictable_emulator((3000,))
        dtype_with_h = np.dtype(
            [("omega_m", "f8"), ("omega_b", "f8"), ("h", "f8"), ("m_nu", "f8")]
        )

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            dtype_with_h,
            emu_model_list=[mock_emu],
            additional_args={
                "want_lensing_lklh_correction": False,
                "cmb_marginalized": False,
            },
        )
        likelihood = act_dr6_lensing.act_dr6_lensing(**kwargs)

        params = np.array([(0.3, 0.05, 0.7, 0.06)], dtype=dtype_with_h)
        loglike = likelihood.compute_like(params)

        assert loglike is not None
        assert np.all(loglike <= 0)

    def test_act_dr6_lensing_set_provider(
        self, mock_likelihood_data_dir, simple_cosmology_dtype, monkeypatch
    ):
        """Test act_dr6_lensing set_provider method."""
        from UPanda.likelihoods import act_dr6_lensing
        from UPanda.provider import Provider

        n_bins = 10
        data_dir = mock_likelihood_data_dir["data_dir"]
        likelihood_dir = os.path.join(data_dir, "likelihoods")
        os.makedirs(likelihood_dir, exist_ok=True)
        lensing_dir = os.path.join(likelihood_dir, "data", "act_dr6_lensing")
        os.makedirs(lensing_dir, exist_ok=True)

        np.savetxt(
            os.path.join(lensing_dir, "clkk_bandpowers_act.txt"), np.ones(n_bins)
        )
        np.savetxt(
            os.path.join(lensing_dir, "binning_matrix_ACT-DR6_Clkk.txt"),
            np.random.randn(n_bins, 3000),
        )
        np.savetxt(
            os.path.join(lensing_dir, "covmat_ACT-DR6_Clkk_baseline.txt"),
            LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42),
        )
        np.savetxt(
            os.path.join(lensing_dir, "covmat_ACT-DR6_Clkk_cmbmarg.txt"),
            LikelihoodTestHelper.create_mock_covariance(n_bins, seed=42),
        )

        fiducial_data = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lensedCls.dat"), fiducial_data
        )
        fiducial_phi = np.column_stack(
            [
                np.arange(2, 3002),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
                np.ones(3000),
            ]
        )
        np.savetxt(
            os.path.join(lensing_dir, "cosmo2017_10K_acc3_lenspotentialCls.dat"),
            fiducial_phi,
        )
        np.savetxt(
            os.path.join(
                lensing_dir, "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt"
            ),
            np.ones(2401),
        )
        np.savetxt(
            os.path.join(lensing_dir, "N1der_KK_lmin600_lmax3000_full.txt"),
            np.ones((n_bins, 2401)),
        )
        for spec in ["TT", "EE", "BB", "TE"]:
            np.savetxt(
                os.path.join(lensing_dir, f"N1der_{spec}_lmin600_lmax3000_full.txt"),
                np.ones((n_bins, 2401)),
            )
        np.save(
            os.path.join(lensing_dir, "norm_correction_matrix_Lmin0_Lmax4000.npy"),
            np.ones((n_bins, 2401)),
        )

        # Use setup_path_mocks helper to avoid recursion
        LikelihoodTestHelper.setup_path_mocks(monkeypatch, data_dir, "act_dr6_lensing")

        kwargs = LikelihoodTestHelper.create_mock_likelihood_init_kwargs(
            simple_cosmology_dtype,
            additional_args={
                "want_lensing_lklh_correction": False,
                "cmb_marginalized": False,
            },
        )
        likelihood = act_dr6_lensing.act_dr6_lensing(**kwargs)

        provider = Provider()
        likelihood.set_provider(provider)
        assert hasattr(likelihood, "provider")
        assert likelihood.provider is not None
