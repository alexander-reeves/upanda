"""Small WMAP covariance and emulator contracts; no survey downloads required."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from UPanda import config
from UPanda.likelihoods.wmap import wmap


@pytest.fixture(params=[False, True])
def backend(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    from UPanda import module
    yield module
    config.set_jax_enabled(old)


def small_wmap(xp):
    obj = wmap.__new__(wmap)
    obj.ttmin = obj.temin = obj.tt_highl_start = obj.te_highl_start = 2
    obj.ttmax = obj.temax = obj.tt_lmax = obj.te_lmax = 5
    obj.TE_logdet_offset = 1.2
    obj.r_off_tttt = xp.array((np.ones((6, 6)) - np.eye(6)) * .01)
    obj.r_off_tete = xp.array((np.ones((6, 6)) - np.eye(6)) * .02)
    obj.epsilon = xp.array(np.ones((6, 6)) * .03)
    obj.ntt = xp.array(np.linspace(.3, .8, 6))
    obj.cltt_dat = xp.array(np.linspace(3., 4., 6))
    obj.clte_dat = xp.array(np.linspace(.3, .4, 6))
    return obj


@pytest.mark.parametrize('cross', [0., .2])
def test_fisher_and_high_ell_against_dense_reference(backend, cross):
    obj = small_wmap(backend)
    tt = np.array([np.arange(2., 8.), np.arange(3., 9.)])
    te = tt / 3
    covtt, covte = tt * 2, tt * 3
    covcross = np.full_like(tt, cross)
    args = [backend.array(v) for v in (covtt, covcross, covte)]
    fisher_tt = np.asarray(obj.get_fisher_TTTT(backend.array(tt), *args))
    fisher_te = np.asarray(obj.get_fisher_TETE(backend.array(te), *args))
    reference_tt, reference_te = [], []
    for b in range(2):
        ft = -np.asarray(obj.r_off_tttt) / np.sqrt(np.outer(covtt[b], covtt[b]))
        ft += np.asarray(obj.epsilon) / np.outer(covtt[b], covtt[b])
        fe = -np.asarray(obj.r_off_tete) / np.sqrt(np.outer(covte[b], covte[b]))
        dt = covte[b] / (covtt[b] * covte[b] - cross**2)
        de = covtt[b] / (covtt[b] * covte[b] - cross**2)
        dt[:2] = de[:2] = 0
        np.fill_diagonal(ft, dt)
        np.fill_diagonal(fe, de)
        np.testing.assert_allclose(fisher_tt[b], ft)
        np.testing.assert_allclose(fisher_te[b], fe)
        np.testing.assert_allclose(ft, ft.T)
        residual = tt[b, 2:] - np.asarray(obj.cltt_dat)[2:]
        shifted = tt[b, 2:] + np.asarray(obj.ntt)[2:]
        dz = np.log(np.asarray(obj.cltt_dat)[2:] + np.asarray(obj.ntt)[2:]) - np.log(shifted)
        reference_tt.append(-.5 * (residual @ ft[2:, 2:] @ residual / 3
                                  + 2 / 3 * (dz * shifted) @ ft[2:, 2:] @ (dz * shifted)))
        dte = te[b, 2:] - np.asarray(obj.clte_dat)[2:]
        reference_te.append((-.5 * dte @ fe[2:, 2:] @ dte,
                             .5 * (np.linalg.slogdet(fe[2:, 2:])[1] + obj.TE_logdet_offset)))
    np.testing.assert_allclose(obj.loglike_highl_TT(backend.array(tt), *args), reference_tt)
    np.testing.assert_allclose(np.array(obj.loglike_highl_TE(backend.array(te), *args)).T, reference_te)
    if config.get_jax_enabled():
        import jax
        np.testing.assert_allclose(jax.jit(obj.loglike_highl_TT)(backend.array(tt), *args), reference_tt)


def test_sz_template_keeps_fractional_values(backend, tmp_path):
    obj = wmap.__new__(wmap)
    obj.sz_file = tmp_path / 'sz.txt'
    np.savetxt(obj.sz_file, [[2, .25], [3, 1.75], [5, .125]])
    obj.setup_for_sz()
    np.testing.assert_allclose(obj.sz_spec, [0, 0, .25, 1.75, 0, .125])


def test_blackwell_rao_batch_and_extreme_tail(backend):
    obj = wmap.__new__(wmap)
    obj.lmin, obj.lmax = 2, 4
    obj.sigmas = backend.array(np.arange(1., 31.).reshape(5, 2, 3))
    obj.first_eval = True
    fid = backend.array(np.arange(2., 7.))
    obj.compute_br_estimator(fid)
    # An extreme tail must remain a finite log probability in both backends.
    spectra = np.array([fid, np.asarray(fid) * .0001])
    sigma = np.asarray(obj.sigmas)
    expected = []
    for cl in spectra:
        terms = []
        for chain in range(2):
            for sample in range(3):
                s = 0.
                for i, ell in enumerate(range(2, 5)):
                    x = sigma[i, chain, sample] / cl[ell]
                    s += .5 * (2 * ell + 1) * (np.log(x) - x) - np.log(sigma[i, chain, sample])
                terms.append(s - float(obj.offset))
        expected.append(logsumexp(terms))
    np.testing.assert_allclose(obj.compute_br_estimator(backend.array(spectra)), expected)


@pytest.mark.parametrize('integrated', [False, True])
def test_emulator_units_and_calibration(backend, integrated):
    obj = small_wmap(backend)
    obj.dtype = np.dtype([('A_sz', float), ('m_t_wmap', float), ('m_e_wmap', float)])
    obj.cosmology_params_fixed = []
    obj.additional_args = {'integrated_model': integrated}
    obj.apply_wmap_temperature_rescaling = True
    obj.apply_wmap_polarization_rescaling = True
    spectra = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]) * 1e-12
    def model(logarithmic):
        return SimpleNamespace(predict=lambda _: backend.array(spectra),
                               predictions_np=lambda _: backend.array(np.log(spectra) if logarithmic else spectra))
    obj.emu_model_list = [model(True), model(False), model(True)]
    captured = {}
    def loglike(cls, params):
        captured.update(cls)
        return backend.array([0., 0.]), {}
    obj.loglike = loglike
    obj.compute_like(backend.array([[0., .1, .2], [0., .3, .4]]))
    factor = np.arange(2, 6) * np.arange(3, 7) / (2 * np.pi)
    for name, calibration in [('tt', np.array([1.1, 1.3])**2),
                              ('te', np.array([1.1 * 1.2, 1.3 * 1.4])),
                              ('ee', np.array([1.2, 1.4])**2)]:
        np.testing.assert_allclose(np.asarray(captured[name])[:, 2:], spectra * 1e12 * calibration[:, None] * factor)
        np.testing.assert_array_equal(np.asarray(captured[name])[:, :2], 0)


def test_provider_calculate_forwards_parameter_dictionary(backend):
    obj = wmap.__new__(wmap)
    obj.provider = SimpleNamespace(get_Cl=lambda **_: {'tt': backend.array([[1.]])})
    obj.loglike = lambda cls, params: (params['A_sz'], {'test': params['A_sz']})
    state = {'derived': None}
    assert obj.calculate(state, A_sz=-2.)
    assert state == {'logp': -2., 'derived': {'chi2_wmap_test': 4., 'logp_test': -2.}}
