"""Parameter ordering, grouped priors, and real posterior evaluation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from UPanda import config, run_panda
from UPanda.likelihoods.utils_likelihood import transform_to_dict


def test_fixed_parameter_order_is_shared_by_all_helpers():
    conf = {
        'variable_cosmology_parameters': [
            {'name': 'fixed', 'prior_min': 0, 'prior_max': 1},
            {'name': 'x', 'prior_min': 1, 'prior_max': 3},
        ],
        'nuisance_parameters': [{'name': 'n', 'prior_min': -1, 'prior_max': 1,
                                'distribution': 'gaussian', 'distribution_kwargs': {'mu': 0., 'cov': 1.}}],
        'fixed_cosmology_parameters': [{'name': 'fixed', 'value': .4}],
    }
    original = deepcopy(conf)
    names, lims = run_panda.get_param_lims(conf)
    dtype, start, fixed = run_panda.get_mcmc_start_and_fixed_params(conf, 4)
    assert names == list(dtype.names) == ['x', 'n']
    assert start.shape == (4, 2)
    assert run_panda.get_gaussian_priors(conf)[0][0] == 1
    assert np.all(start > lims[:, 0]) and np.all(start < lims[:, 1])
    np.testing.assert_array_equal(transform_to_dict(start, dtype, fixed)['fixed'], [.4] * 4)
    assert conf == original


def test_grouped_priors_do_not_shift_later_means():
    group = {'mu': [0., 0.], 'cov': [[1., .2], [.2, 1.]], 'indices': [0, 1]}
    conf = {'variable_cosmology_parameters': [
        {'name': 'a', 'distribution': 'gaussian', 'distribution_kwargs': group},
        {'name': 'b', 'distribution': 'gaussian', 'distribution_kwargs': group},
        {'name': 'c', 'distribution': 'gaussian', 'distribution_kwargs': {'mu': 42., 'cov': 2.}},
        {'name': 'd', 'distribution': 'uniform'},
        {'name': 'e'},
    ]}
    priors = run_panda.get_gaussian_priors(conf)
    assert len(priors) == 2
    assert priors[1] == (2, 42., 2.)


def test_one_dimensional_sampling():
    import chaospy
    values = run_panda.sample_sequence(chaospy.Uniform(-1, 1), 8)
    assert values.shape == (8, 1)
    assert np.all(np.abs(values) <= 1)
    with pytest.raises(ValueError, match='n_gridpoints'):
        run_panda.sample_sequence(chaospy.Uniform(-1, 1), 8, n_gridpoints=3)


@pytest.mark.parametrize('jax_enabled', [False, True])
def test_zero_loglike_is_valid_and_outside_prior_is_rejected(jax_enabled, tmp_path, monkeypatch):
    if jax_enabled:
        pytest.importorskip('jax')
    old = config.get_jax_enabled()
    config.set_jax_enabled(jax_enabled)
    path = tmp_path / 'config.yaml'
    path.write_text('variable_cosmology_parameters:\n  - {name: x, prior_min: -1, prior_max: 1}\nfixed_cosmology_parameters: null\n')

    class Gaussian:
        def __init__(self, **kwargs):
            pass

        def compute_like(self, p):
            return -.5 * p[:, 0] ** 2

    monkeypatch.setattr(run_panda.utils, 'get_likelihood_module', lambda _: SimpleNamespace(gaussian=Gaussian))
    try:
        points = np.array([[0.], [.5], [2.], [-1.]])
        _, logp = run_panda.run_mcmc(total_emu_model_list=[[]], experiments_list=['gaussian'], compression_list=[None], filename_config=str(path), n_walkers=4, point_check=points)
        np.testing.assert_allclose(np.ravel(logp)[:2], [0., -.125])
        assert np.isneginf(np.ravel(logp)[2:]).all()
    finally:
        config.set_jax_enabled(old)


def test_numpy_emcee_executes_real_callback(tmp_path, monkeypatch):
    old = config.get_jax_enabled()
    config.set_jax_enabled(False)
    path = tmp_path / 'config.yaml'
    path.write_text('variable_cosmology_parameters:\n  - {name: x, prior_min: -5, prior_max: 5}\nfixed_cosmology_parameters: null\n')
    class Gaussian:
        def __init__(self, **kwargs):
            pass
        def compute_like(self, p):
            return -.5 * p[:, 0] ** 2
    monkeypatch.setattr(run_panda.utils, 'get_likelihood_module', lambda _: SimpleNamespace(gaussian=Gaussian))
    np.random.seed(921)
    try:
        chain, logp = run_panda.run_mcmc(total_emu_model_list=[[]], experiments_list=['gaussian'], compression_list=[None], filename_config=str(path), n_samples=64, n_walkers=8, burn_in_frac=0)
        assert chain.dtype.names == ("x",) and len(chain) > 0
        np.testing.assert_allclose(logp, -.5 * chain["x"] ** 2, rtol=2e-6)
    finally:
        config.set_jax_enabled(old)


def test_gaussian_prior_accepts_yaml_matrix():
    covariance = [[2., .2], [.2, 1.]]
    point = np.array([[1., 2.]])
    expected = -.5 * point[0] @ np.linalg.solve(covariance, point[0])
    np.testing.assert_allclose(run_panda.log_gauss(point, [0., 0.], covariance), [expected])
