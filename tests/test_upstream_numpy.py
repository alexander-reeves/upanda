"""Direct NumPy comparisons with pinned upstream likelihood implementations."""

import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from UPanda import config

pytestmark = pytest.mark.upstream


@pytest.fixture(autouse=True)
def numpy_backend():
    config.set_jax_enabled(False)
    yield
    config.set_jax_enabled(False)


def test_lollipop_matches_upstream_real_data():
    tools = pytest.importorskip("planck_2020_lollipop.tools")
    upstream = pytest.importorskip("planck_2020_lollipop.lollipop")
    from UPanda.likelihoods.lollipop import lollipop

    like = lollipop(dtype=np.dtype([("A_planck", "f8")]))
    reference = SimpleNamespace(
        bins=tools.get_binning(2, 30),
        mode="lowlE",
        cldata=np.asarray(like.cldata),
        cloff=np.asarray(like.cloff),
        clfid=np.asarray(like.clfid),
        invclcov=np.asarray(like.invclcov),
        marginalised_over_covariance=False,
        Nsim=400,
        log=logging.getLogger("upstream-reference"),
    )
    ee = np.pad(np.asarray(like.clfid[0]), (2, 0))
    expected, actual = [], []
    for scale, calibration in [(1.0, 1.0), (0.9, 1.02), (1.1, 0.98)]:
        theory = ee * scale
        expected.append(
            -0.5
            * upstream._LollipopLikelihood._compute_chi2_1field(
                reference, {"ee": theory}, A_planck=calibration
            )
        )
        like.set_external_spectra({"ee": theory[None, 2:] / 1e12})
        actual.append(float(like.compute_like(np.array([[calibration]]))[0]))
    np.testing.assert_allclose(actual, expected, rtol=1e-11)


def test_camspec_powerlaw_and_calibration_match_cobaya():
    module = pytest.importorskip(
        "cobaya.likelihoods.planck_2018_highl_CamSpec2021."
        "planck_2018_CamSpec2021_python"
    )
    from UPanda.likelihoods.camspec_pr4 import camspec_pr4

    like = camspec_pr4.__new__(camspec_pr4)
    like.ls = np.arange(3001.0)
    parameters = {
        key: np.array([value])
        for key, value in {
            "A_planck": 1.02,
            "cal0": 0.98,
            "cal2": 1.01,
            "calTE": 0.99,
            "calEE": 1.03,
            "amp_100": 10.0,
            "amp_143": 12.0,
            "amp_217": 14.0,
            "amp_143x217": 16.0,
            "n_100": 0.5,
            "n_143": 0.7,
            "n_217": 0.9,
            "n_143x217": 1.1,
        }.items()
    }
    scalar = {key: float(value[0]) for key, value in parameters.items()}
    upstream = module.Planck2018CamSpec2021Python
    np.testing.assert_allclose(
        like._get_foregrounds(parameters)[0],
        upstream.get_powerlaw_residuals(like, scalar),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        like._get_cals(parameters)[0], upstream.get_cals(like, scalar), rtol=1e-13
    )


@pytest.mark.parametrize("name", ["union3", "pantheon_plus", "des_dovekie"])
def test_supernova_marginalization_matches_cobaya(name):
    module = pytest.importorskip("cobaya.likelihoods.sn.pantheonplus")
    covariance = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, 0.3], [0.1, 0.3, 1.0]])
    likelihood_class = getattr(
        importlib.import_module(f"UPanda.likelihoods.{name}"), name
    )
    like = likelihood_class(
        additional_args={
            "mock_data": np.array([22.0, 23.0, 24.0]),
            "mock_cov": covariance,
            "mock_zcmb": np.array([0.1, 0.2, 0.3]),
            "mock_zhel": np.array([0.1, 0.2, 0.3]),
        }
    )
    reference = SimpleNamespace(
        mag=np.array([22.0, 23.0, 24.0]),
        invcov=np.linalg.inv(covariance),
        use_abs_mag=False,
    )
    module.PantheonPlus._marginalize_abs_mag(reference)
    predictions = np.array([[21.5, 22.9, 24.3], [22.0, 23.0, 24.0]])
    like.set_external_spectra(predictions)
    expected = [
        module.PantheonPlus.alpha_beta_logp(reference, row) for row in predictions
    ]
    np.testing.assert_allclose(like.compute_like(None), expected, atol=1e-12)
