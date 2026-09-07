"""Bundled CMB assets and synthetic spectra: units, calibration and batch values.

These check UPanda's implementation, not published best-fit likelihood numbers.
"""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from UPanda import Provider, config


class Spectra:
    def __init__(self, values, logarithmic=False):
        self.values = values
        self.logarithmic = logarithmic
    def predict(self, params):
        return np.broadcast_to(self.values, (len(next(iter(params.values()))), len(self.values))).copy()
    def predictions_np(self, params):
        result = self.predict(params)
        return np.log(result) if self.logarithmic else result


@pytest.mark.parametrize('name', ['planck_2018_highell_lowell', 'planck_2018_for_act', 'planck_2018_for_act_dr6', 'act_dr4', 'act_dr6'])
@pytest.mark.parametrize('integrated', [False, True])
def test_bundled_cmb_gaussian_values(name, integrated):
    config.set_jax_enabled(False)
    cls = getattr(importlib.import_module('UPanda.likelihoods.' + name), name)
    names = ['x', 'A_planck', 'yp2', 'A_act', 'P_act', 'm_t_planck', 'm_e_planck', 'm_t_act', 'm_e_act']
    params = np.array([[1., 1., 1., 1., 1., 0., 0., 0., 0.], [1., 1.1, 1.1, 1.1, 1.1, .01, .02, .01, .02]])
    options = dict(integrated_model=integrated, apply_planck_temperature_rescaling=True,
                   apply_planck_pol_rescaling=True, apply_act_temperature_rescaling=True)
    like = cls(dtype=np.dtype([(n, 'f8') for n in names]), additional_args=options)
    length = like.lmax_win - 1 if name == 'act_dr4' else 9000
    ell = np.arange(2, length + 2)
    values = 1e-10 / (ell * (ell + 1))
    like.emu_model_list = [Spectra(values, not integrated), Spectra(values * .1), Spectra(values * .2, not integrated)]
    prediction = like.get_prediction(params)
    if name == 'act_dr4':
        prediction = prediction.T
    residual = np.asarray(prediction) - like.data_vector
    if name.startswith('act'):
        precision = like.fisher
    elif name == 'planck_2018_for_act_dr6':
        precision = like.inv_covariance
    elif name == 'planck_2018_for_act':
        residual = residual[:, :78]
        precision = np.linalg.inv(like.covariance_matrix[:78, :78])
    else:
        precision = np.linalg.inv(like.covariance_matrix)
    expected = [-.5 * row @ precision @ row for row in residual]
    if name == 'planck_2018_for_act':
        from UPanda.likelihoods import utils_likelihood
        low_tt = utils_likelihood.planck_lowT_binned_loglike(
            like.Cltt[:, :28], like.mu_LN_TT, like.sig_LN_TT,
            like.lmin_list_TT, like.lmax_list_TT)
        np.testing.assert_allclose(like.compute_like(params), expected + low_tt, rtol=2e-8)
        like.additional_args['add_lowell_ee'] = True
        low_ee = utils_likelihood.planck_lowE_binned_loglike(
            like.Clee[:, :28], like.mu_LN_EE, like.sig_LN_EE,
            like.loc_LN_EE, like.lmin_list_EE, like.lmax_list_EE)
        np.testing.assert_allclose(like.compute_like(params), expected + low_tt + low_ee, rtol=2e-8)
        return
    np.testing.assert_allclose(like.compute_like(params), expected, rtol=2e-8)
    if name.startswith('planck'):
        # Low-ell options must add finite contributions to the same high-ell result.
        like.additional_args['with_planck'] = True
        with_low_t = like.compute_like(params)
        np.testing.assert_allclose(with_low_t - expected, like.lognorm_tt, rtol=1e-6)
        like.additional_args['with_planck'] = False
        like.additional_args['add_lowell'] = True
        with_low = like.compute_like(params)
        np.testing.assert_allclose(with_low - expected, like.lognorm_tt + like.lognorm_ee, rtol=1e-6)


@pytest.mark.parametrize('jax_enabled', [False, True])
def test_act_dr6_window_calibration_and_provider(jax_enabled):
    from UPanda.likelihoods.act_dr6 import act_dr6
    old = config.get_jax_enabled()
    config.set_jax_enabled(jax_enabled)
    try:
        like = act_dr6.__new__(act_dr6)
        like.additional_args = {}
        like.dtype = np.dtype([('A_act', 'f8'), ('P_act', 'f8')])
        like.cosmology_params_fixed = []
        like.A_act = like.P_act = 1.
        like.emu_model_list = [None] * 3
        like.external_cls = {p: np.ones((2, 3)) * v for p, v in [('tt', 1.), ('te', 2.), ('ee', 3.)]}
        like.apply_act_temperature_rescaling = False
        like.data_vector = np.zeros(3)
        like.spec_meta = [dict(idx=np.array([i]), pol=p, window=SimpleNamespace(weight=np.array([[1.]], dtype='>f8'), values=np.array([2]))) for i, p in enumerate(['tt', 'te', 'ee'])]
        provider = Provider()
        like.set_provider(provider)
        params = np.array([[1., 1.], [2., 3.]])
        expected = np.array([[1., 2., 3.], [1/4, 2/12, 3/36]]) * 3 / np.pi
        np.testing.assert_allclose(like.get_prediction(params), expected, rtol=2e-6)
        np.testing.assert_allclose(provider.get_Cl(units='FIRASmuK2')['tt'][:, 2:], np.ones((2, 3)))
    finally:
        config.set_jax_enabled(old)


def test_sroll2_lookup_and_calibration():
    config.set_jax_enabled(True)
    import jax
    import jax.numpy as jnp
    from UPanda.likelihoods.planck_sroll2_lowell_ee import planck_sroll2_lowell_ee
    table = np.arange(5.)[:, None] + np.arange(28.)[None, :]
    like = planck_sroll2_lowell_ee(additional_args={'mock_prob_table': table})
    prediction = jnp.array([[.0001] * 28, [.0002] * 28])
    like.get_prediction = lambda _: prediction
    np.testing.assert_allclose(jax.jit(like.compute_like)(None), table[[1, 2]].sum(axis=1))
    config.set_jax_enabled(False)
