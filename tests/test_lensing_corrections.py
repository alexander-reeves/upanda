"""ACT corrections compared to independent matrix products, including gradients."""
from types import SimpleNamespace
import numpy as np
import pytest
from UPanda import config, Provider
from UPanda.likelihoods import utils_likelihood as u


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def backend(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    yield request.param
    config.set_jax_enabled(old)


def correction_data():
    rng = np.random.default_rng(91)
    return SimpleNamespace(dNorm_dCl=rng.normal(size=(4, 5, 6)),
        dN1_dCl=rng.normal(size=(4, 5, 6)), dN1_dClkk=rng.normal(size=(5, 5)),
        norm_fid=np.arange(1., 6.), fiducial_cls=rng.normal(size=(4, 6)),
        clkk_fid=rng.normal(size=5))


def test_correction_matrix_orientation_and_batches(backend):
    data = correction_data()
    rng = np.random.default_rng(4)
    cmb = rng.normal(size=(2, 4, 6))
    kk = rng.normal(size=(2, 5))
    expected = []
    for c, k in zip(cmb, kk):
        delta = c - data.fiducial_cls
        norm = sum(data.dNorm_dCl[t] @ delta[t] for t in range(4))
        n1 = sum(data.dN1_dCl[t] @ delta[t] for t in range(4))
        expected.append(k - 2*norm/data.norm_fid*data.clkk_fid + n1 + data.dN1_dClkk @ (k-data.clkk_fid))
    np.testing.assert_allclose(u.compute_act_lensing_auto_corrections(data, cmb, kk), expected, atol=1e-13)
    if backend:
        import jax
        fn = lambda k: u.compute_act_lensing_auto_corrections(data, cmb, k)
        np.testing.assert_allclose(jax.jit(fn)(kk), expected, atol=1e-13)
        jacobian = jax.jacrev(fn)(kk)
        np.testing.assert_allclose(jacobian[0, :, 0, :], np.eye(5)+data.dN1_dClkk)
        np.testing.assert_allclose(jacobian[0, :, 1, :], 0.)


def test_padding_and_zero_normalizations(backend):
    data = correction_data()
    data.dNorm_dCl *= 0
    data.dN1_dCl *= 0
    data.norm_fid *= 0
    data.dN1_dClkk = np.eye(5)
    data.clkk_fid *= 0
    value = u.compute_act_lensing_auto_corrections(data, np.ones((1, 4, 3)), np.ones((1, 3)))
    np.testing.assert_allclose(value, [[2, 2, 2, 0, 0]])


@pytest.mark.parametrize('provider', [False, True])
def test_act_likelihood_corrections_reach_binning(backend, provider):
    from UPanda.likelihoods.act_dr6_lensing import act_dr6_lensing
    like = act_dr6_lensing.__new__(act_dr6_lensing)
    like.__dict__.update(vars(correction_data()))
    like.dtype = np.dtype([('x', 'f8')])
    like.cosmology_params_fixed = []
    like.additional_args = {'integrated_model': True}
    like.want_lensing_lklh_correction = True
    like.cmb_marginalized = False
    like.binning_matrix = np.eye(5)[:2]
    like.emu_model_list = [SimpleNamespace(predict=lambda p: np.ones((len(p['x']), 5)))]
    if provider:
        like.provider = Provider()
        like.provider.set_Cl({name: np.broadcast_to(like.fiducial_cls[i], (2, 6)) for i, name in enumerate(['tt', 'ee', 'bb', 'te'])})
    expected = np.ones(5) + like.dN1_dClkk @ (np.ones(5)-like.clkk_fid)
    np.testing.assert_allclose(like.get_prediction(np.array([[1.], [2.]])), np.tile(expected[:2], (2, 1)))


def test_pk_mask_accepts_integer_wavenumbers():
    interpolator = SimpleNamespace(kmax=2, P=lambda z, k, grid: k*.5,
                                   logP=lambda z, k, grid: np.log(k*.5))
    np.testing.assert_allclose(u.evaluate_pk_kmax(interpolator, [0, 0, 0], [1, 2, 3]), [.5, 1., np.nan])
    np.testing.assert_allclose(u.evaluate_pk_kmax(interpolator, [0, 0], [1, 2], log=True), [np.log(.5), 0])


@pytest.mark.parametrize('length', [2, 3, 7])
def test_general_lss_act_auto_accepts_different_theory_support(backend, length):
    from UPanda.likelihoods.general_lss import general_lss
    options = dict(spectra=['gg'], input_cov=np.eye(2), input_dv=np.zeros(2),
                   integrated_model=True, want_lensing_lklh_correction=False)
    like = general_lss(dtype=np.dtype([('x', 'f8')]), emu_model_list=[None], additional_args=options)
    like.spectra = ['kaka']
    like.bin_mat_act = np.eye(5)[2:4]
    like.want_lensing_lklh_correction = True
    like.__dict__.update(vars(correction_data()))
    like.dNorm_dCl *= 0
    like.dN1_dCl *= 0
    like.dN1_dClkk *= 0
    like.set_external_spectra([np.ones((2, length))])
    params = np.array([[1.], [2.]])
    np.testing.assert_allclose(like.compute_like(params), [-1., -1.])
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(like.compute_like)(params), [-1., -1.])
