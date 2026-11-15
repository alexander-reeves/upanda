"""Comparisons with reference likelihood codes: cobaya and the collaborations' own packages.

Each test evaluates the reference likelihood, gives UPanda the same theory and compares the two
log-likelihoods on the real data. The cobaya comparisons need CAMB and cobaya's data:

    pip install cobaya camb
    cobaya-install bao.desi_dr2.desi_bao_all bao.desi_2024_bao_all bao.sdss_dr16_baoplus_lrg \
        bao.sdss_dr16_baoplus_qso bao.sdss_dr16_baoplus_elg sn.pantheonplus sn.union3 sn.desdovekie \
        planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lensing.CMBMarged -p <packages>
    COBAYA_PACKAGES_PATH=<packages> python -m pytest -m reference

A test is skipped when its reference code or data is not installed.
"""

import importlib
import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

from UPanda import config

pytestmark = pytest.mark.reference

C_KMS = 299792.458
COBAYA_LIKELIHOODS = [
    "bao.desi_dr2.desi_bao_all",
    "bao.desi_2024_bao_all",
    "bao.sdss_dr16_baoplus_lrg",
    "bao.sdss_dr16_baoplus_qso",
    "bao.sdss_dr16_baoplus_elg",
    "sn.pantheonplus",
    "sn.union3",
    "sn.desdovekie",
]
# Two cosmologies, so that an agreement does not rest on a single (near best-fit) point.
POINTS = [{"H0": 67.5, "omch2": 0.119}, {"H0": 71.0, "omch2": 0.128}]


@pytest.fixture(autouse=True)
def numpy_backend():
    config.set_jax_enabled(False)
    yield
    config.set_jax_enabled(False)


class Background:
    """Distances and growth of the cosmology cobaya has just evaluated."""

    def __init__(self, provider):
        self.provider = provider
        self.rd = provider.get_param("rdrag")

    def DA(self, z):
        return self.provider.get_angular_diameter_distance(np.atleast_1d(z))

    def H(self, z):
        return self.provider.get_Hubble(np.atleast_1d(z), units="km/s/Mpc")

    def DM_over_rs(self, z):
        return (1 + z) * self.DA(z) / self.rd

    def DH_over_rs(self, z):
        return C_KMS / self.H(z) / self.rd

    def DV_over_rs(self, z):
        return ((1 + z) ** 2 * self.DA(z) ** 2 * C_KMS * z / self.H(z)) ** (1 / 3) / self.rd

    def f_sigma8(self, z):
        return self.provider.get_fsigma8(np.atleast_1d(z))

    def vector(self, redshifts, quantities):
        return np.array([float(getattr(self, q)(float(z))[0]) for z, q in zip(redshifts, quantities)])


@pytest.fixture(scope="module")
def cobaya():
    """Returns evaluate(point) -> (log-likelihoods by name, Background)."""
    pytest.importorskip("camb")
    model_module = pytest.importorskip("cobaya.model")
    packages = os.environ.get("COBAYA_PACKAGES_PATH")
    if not packages or not os.path.isdir(os.path.join(packages, "data", "bao_data")):
        pytest.skip("COBAYA_PACKAGES_PATH does not point to cobaya's BAO and SN data")
    model = model_module.get_model(
        {
            "theory": {"camb": None},
            "likelihood": {name: None for name in COBAYA_LIKELIHOODS},
            "packages_path": packages,
            "params": {
                "H0": {"prior": {"min": 50, "max": 90}},
                "omch2": {"prior": {"min": 0.05, "max": 0.2}},
                "ombh2": 0.0222,
                "ns": 0.965,
                "As": 2.1e-9,
                "tau": 0.055,
                "mnu": 0.06,
            },
        }
    )

    def evaluate(point):
        loglikes = model.loglikes(point)[0]
        return dict(zip(model.likelihood, loglikes)), Background(model.provider)

    return evaluate


def _rows(filename):
    with open(filename) as f:
        return [line.split() for line in f if line.strip() and not line.startswith("#")]


def test_desi_bao_dr2_matches_cobaya(cobaya):
    from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2

    like = desi_bao_dr2()
    rows = _rows(os.path.join(like.data_dir, "mean.txt"))
    for point in POINTS:
        reference, background = cobaya(point)
        theory = background.vector([r[0] for r in rows], [r[2] for r in rows])
        like.set_external_spectra({"desi_bao_data": theory})
        np.testing.assert_allclose(
            like.compute_like(None), reference["bao.desi_dr2.desi_bao_all"], rtol=1e-9
        )


def test_desi_bao_2024_matches_cobaya(cobaya):
    from UPanda.likelihoods.desi_bao_2024 import desi_bao_2024

    emulator = SimpleNamespace()
    like = desi_bao_2024(dtype=np.dtype([("x", "f8")]), emu_model_list=[emulator])
    rows = _rows(os.path.join(like.data_dir, "mean.txt"))
    for point in POINTS:
        reference, background = cobaya(point)
        theory = background.vector([r[0] for r in rows], [r[2] for r in rows])
        emulator.predict = lambda params, theory=theory: theory[None, :]
        np.testing.assert_allclose(
            like.compute_like(np.zeros((1, 1)))[0], reference["bao.desi_2024_bao_all"], rtol=1e-9
        )


@pytest.mark.parametrize(
    "samples, cobaya_name, atol",
    [
        # cobaya's LRG likelihood holds the two BOSS DR12 bins and the eBOSS LRG bin
        (("boss_dr12", "eboss_lrg"), "bao.sdss_dr16_baoplus_lrg", 1e-8),
        (("eboss_qso",), "bao.sdss_dr16_baoplus_qso", 1e-8),
        # ELG: both read the official grid; cobaya interpolates it linearly in ln L, UPanda smooths it
        # by 0.7 cells and uses a cubic interpolation, hence the looser tolerance
        (("eboss_elg",), "bao.sdss_dr16_baoplus_elg", 0.05),
    ],
)
def test_sdss_dr16_bao_rsd_matches_cobaya(cobaya, samples, cobaya_name, atol):
    from UPanda.likelihoods.boss_dr16_rsd import boss_dr16_rsd

    try:
        like = boss_dr16_rsd(samples=samples)
    except FileNotFoundError:
        pytest.skip("the eBOSS ELG likelihood files are not in data/eboss_dr16_elg")
    for point in POINTS:
        reference, background = cobaya(point)
        like.set_external_spectra(
            {"boss_dr16_rsd_data": background.vector(like.z_eff, like.quantities)}
        )
        np.testing.assert_allclose(
            np.ravel(like.compute_like(None))[0], reference[cobaya_name], rtol=0, atol=atol
        )


@pytest.mark.parametrize(
    "name, cobaya_name",
    [
        ("pantheon_plus", "sn.pantheonplus"),
        ("union3", "sn.union3"),
        ("des_dovekie", "sn.desdovekie"),
    ],
)
def test_supernovae_match_cobaya(cobaya, name, cobaya_name):
    like = getattr(importlib.import_module(f"UPanda.likelihoods.{name}"), name)()
    for point in POINTS:
        reference, background = cobaya(point)
        like.set_external_spectra(
            5 * np.log10((1 + like.zhel) * (1 + like.zcmb) * background.DA(like.zcmb))
        )
        np.testing.assert_allclose(like.compute_like(None), reference[cobaya_name], rtol=1e-8)


@pytest.fixture(scope="module")
def cobaya_cmb():
    """Returns evaluate(point) -> (log-likelihoods by name, CAMB spectra from cobaya's provider)."""
    pytest.importorskip("camb")
    model_module = pytest.importorskip("cobaya.model")
    packages = os.environ.get("COBAYA_PACKAGES_PATH")
    if not packages or not os.path.isdir(os.path.join(packages, "data", "planck_2018_pliklite_native")):
        pytest.skip("COBAYA_PACKAGES_PATH does not point to cobaya's Planck 2018 data")
    model = model_module.get_model(
        {
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "likelihood": {
                "planck_2018_highl_plik.TTTEEE_lite_native": None,
                "planck_2018_lensing.CMBMarged": None,
            },
            "packages_path": packages,
            "params": {
                "H0": {"prior": {"min": 50, "max": 90}},
                "omch2": {"prior": {"min": 0.05, "max": 0.2}},
                "ombh2": 0.0222,
                "ns": 0.965,
                "As": 2.1e-9,
                "tau": 0.055,
                "mnu": 0.06,
                "A_planck": 1.0,
            },
        }
    )

    def evaluate(point):
        loglikes = model.loglikes(point)[0]
        spectra = model.provider.get_Cl(ell_factor=False, units="FIRASmuK2")
        return dict(zip(model.likelihood, loglikes)), spectra

    return evaluate


def _emulator(spectrum):
    """Stand-in for an emulator: returns a fixed spectrum starting at ell = 2."""
    return SimpleNamespace(predict=lambda params: spectrum[None, 2:])


CMB_POINTS = [{"H0": 67.5, "omch2": 0.119}, {"H0": 69.0, "omch2": 0.122}]


def test_planck_2018_plik_lite_matches_cobaya(cobaya_cmb):
    from UPanda.likelihoods.planck_2018_highell_lowell import planck_2018_highell_lowell

    for point in CMB_POINTS:
        reference, cl = cobaya_cmb(point)
        like = planck_2018_highell_lowell(
            dtype=np.dtype([("A_planck", "f8")]),
            emu_model_list=[_emulator(cl[x] / 1e12) for x in ("tt", "te", "ee")],   # C_ell in K^2
            additional_args={"integrated_model": True},
        )
        np.testing.assert_allclose(
            np.ravel(like.compute_like(np.ones((1, 1))))[0],
            reference["planck_2018_highl_plik.TTTEEE_lite_native"],
            rtol=1e-7,
        )


def test_planck_2018_lensing_matches_cobaya(cobaya_cmb):
    from UPanda.likelihoods.planck_2018_lensing import planck_2018_lensing

    for point in CMB_POINTS:
        reference, cl = cobaya_cmb(point)
        ell = cl["ell"]
        kappa = (ell * (ell + 1.0)) ** 2 / 4.0 * cl["pp"]                            # C_L^kappa-kappa
        like = planck_2018_lensing(
            dtype=np.dtype([("A_planck", "f8")]),
            emu_model_list=[_emulator(kappa)],
            additional_args={"integrated_model": True},
        )
        np.testing.assert_allclose(
            np.ravel(like.compute_like(np.ones((1, 1))))[0],
            reference["planck_2018_lensing.CMBMarged"],
            rtol=1e-6,
        )


def test_act_dr4_matches_pyactlike():
    pyactlike = pytest.importorskip("pyactlike")
    from UPanda.likelihoods.act_dr4 import act_dr4

    emulators = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]
    like = act_dr4(
        dtype=np.dtype([("yp2", "f8")]),
        emu_model_list=emulators,
        additional_args={"integrated_model": True},
    )
    reference = pyactlike.ACTPowerSpectrumData()
    # the two best-fit spectra distributed with pyactlike: ell, D_ell^TT, D_ell^TE, D_ell^EE in muK^2 from ell = 2
    for name in ("bf_ACTPol_lcdm.minimum.theory_cl", "bf_ACTPol_WMAP_lcdm.minimum.theory_cl"):
        ell, tt, te, ee = np.loadtxt(os.path.join(like.data_dir, name), usecols=(0, 1, 2, 3), unpack=True)
        for emulator, dell in zip(emulators, (tt, te, ee)):
            # the emulators give C_ell in K^2
            emulator.predict = lambda params, dell=dell: (dell * 2 * np.pi / (ell * (ell + 1)))[None, :] / 1e12
        for yp2 in (1.0, 1.003):
            np.testing.assert_allclose(
                like.compute_like(np.array([[yp2]]))[0], reference.loglike(tt, te, ee, yp2), rtol=0, atol=1e-3
            )


def test_lollipop_matches_planck_npipe_code():
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
        log=logging.getLogger("lollipop-reference"),
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


def test_camspec_foregrounds_and_calibration_match_cobaya():
    """The power-law foreground residuals and the calibration factors, not the full likelihood."""
    module = pytest.importorskip(
        "cobaya.likelihoods.planck_2018_highl_CamSpec2021.planck_2018_CamSpec2021_python"
    )
    from UPanda.likelihoods.camspec_pr4 import camspec_pr4

    like = camspec_pr4.__new__(camspec_pr4)
    like.ls = np.arange(3001.0)
    values = {
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
    }
    parameters = {key: np.array([value]) for key, value in values.items()}
    upstream = module.Planck2018CamSpec2021Python
    np.testing.assert_allclose(
        like._get_foregrounds(parameters)[0],
        upstream.get_powerlaw_residuals(like, values),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        like._get_cals(parameters)[0], upstream.get_cals(like, values), rtol=1e-13
    )
