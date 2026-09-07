"""Independent references for spectrum binning and simulation covariances."""
import numpy as np
import pytest
from scipy import stats
from UPanda import config
from UPanda.likelihoods import utils_likelihood as util


@pytest.mark.parametrize('jax_enabled', [False, True])
def test_low_ell_binning_with_lists_and_jit(jax_enabled):
    old = config.get_jax_enabled()
    config.set_jax_enabled(jax_enabled)
    ell = np.arange(2, 30)
    cl = np.stack([1 / (ell * (ell + 1)), 2 / (ell * (ell + 1))])
    fn = lambda c: util.get_binned_D_from_theory_Cls(ell, c, [2, 10, 20], [9, 19, 29])
    try:
        if jax_enabled:
            import jax
            fn = jax.jit(fn)
        np.testing.assert_allclose(fn(cl), np.array([[1.] * 3, [2.] * 3]) / (2 * np.pi), rtol=2e-6)
        mu, sig, loc = np.zeros(3), np.ones(3), np.zeros(3)
        actual = util.planck_lowE_binned_loglike(cl, mu, sig, loc, [2, 10, 20], [9, 19, 29])
        expected = 3 * stats.lognorm.logpdf(np.array([1., 2.]) / (2 * np.pi), s=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6)
        actual = util.planck_lowT_binned_loglike(cl, mu[:2], sig[:2], [2, 16], [15, 29])
        np.testing.assert_allclose(actual, expected * 2 / 3, rtol=2e-6)
    finally:
        config.set_jax_enabled(old)


@pytest.mark.parametrize('hartlap', [False, True])
@pytest.mark.parametrize('split', [False, True])
def test_simulation_covariance_matches_numpy(tmp_path, hartlap, split):
    import h5py
    values = np.random.default_rng(8).normal(size=(16, 3))
    values[0, 0] = 0  # Legitimate zero in a nonempty realization must be retained.
    all_values = np.vstack([np.zeros(3), values])
    paths = [tmp_path / 'a.h5', tmp_path / 'b.h5']
    blocks = np.array_split(all_values, 2) if split else [all_values]
    for path, block in zip(paths, blocks):
        with h5py.File(path, 'w') as f:
            f['kk'] = block[:, :1]
            f['gg'] = block[:, 1:]
    transformed = values.copy()
    transformed[:, 0] *= 2 / np.pi
    expected = np.cov(transformed.T)
    if hartlap:
        expected /= (16 - 3 - 2) / (16 - 1)
    actual = util.create_cov_lss(['kk', 'gg'], str(paths[0]), str(paths[1]) if split else None, hartlap)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_hartlap_invalid_sample_count(tmp_path):
    import h5py
    path = tmp_path / 'short.h5'
    with h5py.File(path, 'w') as f:
        f['gg'] = np.ones((3, 2))
    with pytest.raises(ValueError, match='n_sims'):
        util.create_cov_lss(['gg'], path, None)


@pytest.mark.parametrize('spec,expected', [('2-4 9', [2, 3, 4, 9]), ('7', [7]), ('', []), ([3, 7], [3, 7])])
def test_camspec_ell_range_parser(spec, expected):
    from UPanda.likelihoods.camspec_pr4 import _range_to_ells
    np.testing.assert_array_equal(_range_to_ells(spec), expected)


def test_fixed_densities_are_available_to_conversions():
    dtype = np.dtype([('x', 'f8')])
    fixed = [{'name': k, 'value': v} for k, v in dict(omega_m=.3, h=.7, m_nu=.02, omega_b=.022).items()]
    result = util.prepare_params_dict(np.ones((2, 1)), dtype, fixed)
    np.testing.assert_allclose(result['omega_cdm'], .3 * .7 ** 2 - .06 / 93.14 - .022)
    fixed.append({'name': 'omega_cdm', 'value': .12})
    result = util.prepare_params_dict(np.ones((2, 1)), dtype, fixed)
    np.testing.assert_allclose(result['omega_cdm'], .12)


def test_hillipop_binning_matches_explicit_operator():
    from UPanda.likelihoods.tools_hillipop import Bins
    bins = Bins.fromdeltal(2, 9, 2)
    p = np.zeros((4, 10))
    for i, lower in enumerate([2, 4, 6, 8]):
        p[i, lower:lower + 2] = .5
    spectrum = np.arange(20.).reshape(2, 10)
    np.testing.assert_allclose(bins.bin_spectra(spectrum), spectrum @ p.T)
    cov = np.diag(np.arange(1., 11.))
    np.testing.assert_allclose(bins.bin_covariance(cov), p @ cov @ p.T)
    bins.cut_binning(4, 7)
    assert bins.nbins == 2
    np.testing.assert_array_equal(bins.bins(), [[4, 6], [5, 7]])
