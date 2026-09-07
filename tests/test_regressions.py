"""Numerical regression tests for public interfaces used by downstream apps."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from UPanda import Provider, config
from UPanda.likelihoods import utils_likelihood as util
from UPanda.likelihoods._numerics import gaussian_loglike, marginal_precision


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def backend(request):
    if request.param:
        pytest.importorskip('jax')
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    yield request.param
    config.set_jax_enabled(old)


@pytest.mark.parametrize('shape', [(3,), (1, 3), (5, 3), (2, 4, 3)])
def test_gaussian_matches_independent_solve(backend, shape):
    rng = np.random.default_rng(84)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3)
    residual = rng.normal(size=shape)
    expected = [-0.5 * r @ np.linalg.solve(cov, r) for r in residual.reshape(-1, 3)]
    actual = gaussian_loglike(residual, np.linalg.inv(cov))
    np.testing.assert_allclose(actual, np.array(expected).reshape(shape[:-1]), rtol=2e-6)


@pytest.mark.parametrize('residual,precision', [(0., np.eye(2)), ([1., 2.], [1., 2.]), ([1., 2.], np.eye(3))])
def test_gaussian_rejects_incompatible_shapes(residual, precision):
    with pytest.raises(ValueError):
        gaussian_loglike(residual, precision)


@pytest.mark.parametrize('name', ['desi_bao_dr2', 'desi_bao_2024', 'boss_dr16_rsd', 'pantheon_plus', 'des_dovekie', 'union3'])
def test_likelihood_predictions_and_external_paths(backend, name, monkeypatch):
    module = importlib.import_module('UPanda.likelihoods.' + name)
    data = np.array([2., 4., 8.])
    covariance = np.array([[2., .2, .1], [.2, 1., .3], [.1, .3, 3.]])
    # Synthetic mode must work without any survey files.
    monkeypatch.setattr(module.np, 'loadtxt', lambda *a, **kw: pytest.fail('unexpected survey file access'))
    kwargs = dict(dtype=np.dtype([('x', 'f8')]), additional_args={'mock_data': data, 'mock_cov': covariance})
    like = getattr(module, name)(**kwargs)
    prediction = np.array([[1., 4., 9.], [2., 5., 7.]])
    if name.startswith('desi') or name.startswith('boss'):
        precision = np.linalg.inv(covariance)
    else:
        raw = np.linalg.inv(covariance)
        one = np.ones(3)
        precision = raw - np.outer(raw @ one, raw @ one) / (one @ raw @ one)
    expected = [-.5 * r @ precision @ r for r in data - prediction]
    like.emu_model_list = [SimpleNamespace(predict=lambda _: prediction)]
    if name in ('pantheon_plus', 'des_dovekie', 'union3'):
        distance = 10 ** (prediction / 5) / ((1 + like.zhel) * (1 + like.zcmb))
        like.emu_model_list = [SimpleNamespace(predict=lambda _: distance)]
    np.testing.assert_allclose(like.compute_like(np.ones((2, 1))), expected, rtol=2e-6)
    if hasattr(like, 'set_external_spectra'):
        def evaluate(values):
            key = 'desi_bao_data' if name == 'desi_bao_dr2' else name + '_data'
            spectra = {key: values} if name.startswith(('desi', 'boss')) else values
            like.set_external_spectra(spectra)
            return like.compute_like(None)
        np.testing.assert_allclose(evaluate(prediction), expected, rtol=2e-6)
        np.testing.assert_allclose(evaluate(prediction[0]), expected[0], rtol=2e-6)
        if backend:
            import jax
            import jax.numpy as jnp
            actual = jax.jit(evaluate)(jnp.array(prediction))
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
            gradient = jax.jit(jax.grad(lambda x: evaluate(x)))(jnp.array(prediction[0]))
            np.testing.assert_allclose(gradient, precision @ (data - prediction[0]), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize('name', ['desi_bao_dr2', 'desi_bao_2024'])
def test_bao_default_options(name):
    module = importlib.import_module('UPanda.likelihoods.' + name)
    like = getattr(module, name)()
    assert like.desi_bao_covariance.shape == (len(like.desi_bao_data),) * 2


@pytest.mark.parametrize('name', ['pantheon_plus', 'des_dovekie', 'union3'])
def test_supernova_offset_is_marginalized(name):
    cls = getattr(importlib.import_module('UPanda.likelihoods.' + name), name)
    like = cls(additional_args={'mock_data': np.array([1., 2., 4.])})
    like.set_external_spectra(np.array([2., 3., 6.]))
    reference = like.compute_like(None)
    like.set_external_spectra(np.array([2., 3., 6.]) + 25)
    np.testing.assert_allclose(like.compute_like(None), reference, atol=1e-10)


def test_precision_cut_marginalizes_discarded_data():
    covariance = np.array([[3., .8, .2], [.8, 2., .6], [.2, .6, 1.]])
    mask = np.array([True, False, True])
    precision = np.linalg.inv(covariance)
    np.testing.assert_allclose(marginal_precision(precision, mask), np.linalg.inv(covariance[np.ix_(mask, mask)]))
    np.testing.assert_allclose(marginal_precision(precision, np.ones(3, dtype=bool)), precision)
    from UPanda.likelihoods.des_dovekie import des_dovekie
    like = des_dovekie.__new__(des_dovekie)
    like.mag = like.zcmb = like.zhel = np.arange(3.)
    like.inv_cov = precision
    like._apply_mask(mask)
    np.testing.assert_allclose(like.inv_cov, np.linalg.inv(covariance[np.ix_(mask, mask)]))
    assert like.nsn == 2


@pytest.mark.parametrize('lmin,lmax,expected', [(2, 5, [4, 0, 8, 10]), (3, 4, [0, 8]), (0, 2, [0, 2, 4])])
def test_spectrum_grid_uses_offset(lmin, lmax, expected):
    np.testing.assert_array_equal(util.standardize_input_cl(np.array([1, 2, 4, 5]), np.array([2, 4, 8, 10]), lmin=lmin, lmax=lmax), expected)


def test_switch_updates_existing_imports_and_classes(backend):
    from UPanda import utils
    from UPanda.likelihoods.desi_bao_dr2 import desi_bao_dr2
    from UPanda.provider import Provider as original_provider
    original_function = util.convert_om_to_ocdm
    config.set_jax_enabled(not backend)
    config.set_jax_enabled(backend)
    import UPanda.module as module
    assert utils.array is module.array
    assert util.array is module.array
    assert original_function is util.convert_om_to_ocdm
    assert Provider is original_provider
    assert desi_bao_dr2 is importlib.import_module('UPanda.likelihoods.desi_bao_dr2').desi_bao_dr2
    if backend:
        import jax
        import jax.numpy as jnp
        np.testing.assert_allclose(jax.jit(utils.invert_input)(jnp.eye(2)), np.eye(2))


def test_provider_polarization_without_tt(backend):
    provider = Provider()
    values = np.ones((2, 5))
    provider.set_Cl({'ee': values})
    assert provider.ell_max_cmb == 4
    expected = np.arange(5) * (np.arange(5) + 1) / (2 * np.pi)
    np.testing.assert_allclose(provider.get_Cl(True, 'FIRASmuK2')['ee'], np.broadcast_to(expected, (2, 5)), rtol=2e-6)
    provider.set_Cl({'ee': values * 2})
    np.testing.assert_allclose(provider.get_Cl(False, 'K2')['ee'], values * 2e-12)
    with pytest.raises(ValueError, match='spectrum'):
        Provider().set_Cl({})


def test_boss_sample_generator_and_single_prediction():
    from UPanda.likelihoods.boss_dr16_rsd import boss_dr16_rsd
    from UPanda.likelihoods.boss_dr16_fsigma8 import boss_dr16_fsigma8
    like = boss_dr16_rsd(samples=(s for s in ['mgs']))
    assert like.samples == ('mgs',)
    like.set_external_spectra({'boss_dr16_rsd_data': like.data_vector})
    assert like.compute_like(None) == 0
    with pytest.raises(ValueError, match='at least one'):
        boss_dr16_rsd(samples=[])
    with pytest.raises(ValueError, match='Unknown'):
        boss_dr16_rsd(samples=['missing'])
    like = boss_dr16_fsigma8()
    like.set_external_spectra({'boss_dr16_fsigma8_data': like.data_vector + .1})
    expected = -.5 * .01 * np.diag(like.inv_cov).sum()
    np.testing.assert_allclose(like.compute_like(None), expected)
