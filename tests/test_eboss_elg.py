"""eBOSS DR16 ELG likelihoods read from the official SDSS files, and their use in the BOSS/eBOSS likelihoods."""

from types import SimpleNamespace

import numpy as np
import pytest

from UPanda import config
from UPanda.likelihoods import eboss_elg
from UPanda.likelihoods.boss_dr16_fsigma8 import boss_dr16_fsigma8
from UPanda.likelihoods.boss_dr16_rsd import boss_dr16_rsd


@pytest.fixture(autouse=True)
def numpy_backend():
    config.set_jax_enabled(False)
    yield
    config.set_jax_enabled(False)


def test_fsigma8_table_gives_the_published_measurement():
    table = eboss_elg.ELGfs8Table()
    # Alam et al. 2021, Table III: f sigma8 = 0.315 +- 0.095
    assert table.mean == pytest.approx(0.315, abs=1e-3)
    assert table.sd == pytest.approx(0.095, abs=1e-3)
    centre, low, high, far_low, far_high = table.lnL(
        np.array([table.mean, table.mean - table.sd, table.mean + table.sd, -1.0, 3.0])
    )
    assert centre > low > far_low and centre > high > far_high
    assert np.isfinite([far_low, far_high]).all()


def test_grid_likelihood_in_three_and_two_dimensions():
    grid, marginal = eboss_elg.ELGGrid(), eboss_elg.ELGGrid(mode="2d")
    dm, dh, fs8 = grid.mean
    assert -1.0 < grid.lnL(dm, dh, fs8) <= 1e-6
    assert -1.0 < marginal.lnL(dm, dh) <= 1e-6
    assert grid.lnL(dm + 2.0, dh, fs8) < grid.lnL(dm, dh, fs8)
    assert grid.lnL(np.full(4, dm), np.full(4, dh), np.full(4, fs8)).shape == (4,)
    outside = grid.lnL(1e3, dh, fs8)       # outside the tabulated range: finite, strongly disfavoured
    assert np.isfinite(outside) and outside < -50
    with pytest.raises(ValueError):
        eboss_elg.ELGGrid(mode="1d")


def test_rsd_likelihood_with_the_elg_grid():
    like = boss_dr16_rsd(samples=("eboss_qso", "eboss_elg"))
    assert like.quantities[-3:] == ["DM_over_rs", "DH_over_rs", "f_sigma8"]
    theory = np.asarray(like.data_vector, dtype=float).copy()
    theory[-3:] = eboss_elg.ELGGrid().mean
    like.set_external_spectra({"boss_dr16_rsd_data": theory})
    at_mean = float(np.ravel(like.compute_like(None))[0])
    assert -1.0 < at_mean <= 1e-6
    like.set_external_spectra({"boss_dr16_rsd_data": theory + np.r_[np.zeros(3), 1.0, 0.0, 0.0]})
    assert float(np.ravel(like.compute_like(None))[0]) < at_mean

    kept = like.drop_fsigma8()             # distances only
    assert "f_sigma8" not in like.quantities and len(kept) == 4
    like.set_external_spectra({"boss_dr16_rsd_data": theory[kept]})
    assert -1.0 < float(np.ravel(like.compute_like(None))[0]) <= 1e-6


def test_fsigma8_covariance_and_elg_options():
    def build(**options):
        return boss_dr16_fsigma8(
            dtype=np.dtype([("x", "f8")]),
            emu_model_list=[SimpleNamespace(predict=lambda p: prediction[None, :])],
            additional_args=options,
        )

    published, diagonal = build(), build(diagonal=True)
    prediction = np.asarray(published.data_vector, dtype=float) + 0.02
    i, j = (int(np.argmin(np.abs(np.asarray(published.z_eff) - z))) for z in (0.38, 0.51))
    assert published.cov[i, j] != 0.0      # the two BOSS DR12 bins are correlated
    assert np.count_nonzero(diagonal.cov - np.diag(np.diag(diagonal.cov))) == 0
    point = np.zeros((1, 1))
    assert published.compute_like(point)[0] != pytest.approx(diagonal.compute_like(point)[0])

    table = build(elg="table")             # official ELG likelihood instead of a Gaussian for that point
    assert np.isfinite(table.compute_like(point)[0])
    assert table.compute_like(point)[0] != pytest.approx(published.compute_like(point)[0])
    with pytest.raises(ValueError):
        build(elg="unknown")
