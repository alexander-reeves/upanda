"""External theory contracts used in the three active multiprobe projects."""
import numpy as np
import pytest
from UPanda import config, Provider
from UPanda.likelihoods.general_lss import general_lss


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def backend(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    yield request.param
    config.set_jax_enabled(old)


def make_lss(options=None, size=6):
    opts = dict(spectra=['gg'], input_cov=np.eye(size), input_dv=np.zeros(size), integrated_model=True, want_lensing_lklh_correction=False)
    opts.update(options or {})
    names = ['x', 'm1', 'm_k', 'm_k_act', 'm_t_planck', 'sn_1', 'sigma8', 'omega_cdm', 'omega_b', 'h', 'm_nu']
    like = general_lss(dtype=np.dtype([(n, 'f8') for n in names]), emu_model_list=[None], additional_args=opts)
    p = np.array([[1., .1, .2, .3, .4, 2., .8, .12, .022, .7, .02], [1., .2, .3, .4, .5, 3., .9, .13, .023, .72, .02]])
    return like, p


@pytest.mark.parametrize('lensing,official_key', [('kk', 'add_kk_official'),
                                                   ('kaka', 'add_ka_official')])
def test_official_lensing_covariance_accepts_numpy_source(backend, lensing,
                                                          official_key,
                                                          monkeypatch):
    """Production covariance readers return NumPy even with JAX enabled."""
    import UPanda.likelihoods.general_lss as module

    bins = 9 if lensing == 'kk' else 2
    official = np.eye(bins) * 3

    def fake_loadtxt(path, *args, **kwargs):
        name = str(path)
        if 'covariance.txt' in name or 'covmat_ACT' in name:
            return official
        if 'cl_hat.txt' in name or 'clkk_bandpowers_act.txt' in name:
            return np.arange(1., bins + 1)
        if 'theory_offset.txt' in name:
            return np.zeros(bins)
        if 'binning_plus_correction_matrix.txt' in name:
            return np.zeros((bins, 3))
        if 'binning_matrix_ACT' in name:
            return np.zeros((bins, 3))
        raise AssertionError(f'unexpected file read: {name}')

    monkeypatch.setattr(module.np, 'loadtxt', fake_loadtxt)
    opts = dict(spectra=[lensing], input_cov=np.eye(bins), input_dv=np.array([]),
                integrated_model=True, want_lensing_lklh_correction=False,
                **{official_key: True})
    if lensing == 'kk':
        opts['mock_kk'] = np.arange(1., bins + 1)
    else:
        opts['mock_kaka'] = np.arange(1., bins + 1)
    like = general_lss(dtype=np.dtype([('x', 'f8')]), emu_model_list=[None],
                       additional_args=opts)
    np.testing.assert_allclose(np.asarray(like.inv_C), np.linalg.inv(official))
    if backend:
        assert hasattr(like.inv_C, 'at')


def test_external_nuisance_calibrations_do_not_mutate_theory(backend):
    opts = dict(indices_list_mg=[[[[0], [], [], [], []], [[1], [], [], [], []]]], indices_list_k_cross=[[2]], indices_list_ka_cross=[[3]], indices_list_t_cross=[[4]], apply_t_indices=True, indices_shot=[[[5]]])
    like, p = make_lss(opts)
    original = np.ones((2, 6))
    values = original.copy()
    if backend:
        import jax.numpy as jnp
        values = jnp.asarray(values)
    like.set_external_spectra([values])
    expected = np.column_stack([1+p[:, 1], (1+p[:, 1])**2, 1+p[:, 2], 1+p[:, 3], 1+p[:, 4], 1+p[:, 5]*1e-6])
    for _ in range(2):
        np.testing.assert_allclose(like.get_prediction(p), expected, rtol=2e-6)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(like.compute_like(p), -.5 * (expected**2).sum(axis=1), rtol=2e-6)


@pytest.mark.parametrize('lensing', ['kk', 'kaka'])
def test_auto_calibration_only_scales_lensing_block(backend, lensing):
    like, p = make_lss(dict(apply_kappa_rescaling=True), size=2)
    like.spectra = ['gg', lensing]
    like.emu_model_list = [None, None]
    galaxy = np.ones((2, 2))
    lens = np.ones((2, 3))
    if backend:
        import jax.numpy as jnp
        galaxy, lens = jnp.asarray(galaxy), jnp.asarray(lens)
    like.set_external_spectra([galaxy, lens])
    if lensing == 'kk':
        like.ell_min_kk, like.ell_max_kk = 2, 4
        like.bin_mat_lens = np.ones((9, 3)) / 3
        like.theory_offset_kk = np.zeros(9)
        expected = np.ones((2, 9)) * (2 / np.pi) * (1+p[:, 2:3])**2
    else:
        like.bin_mat_act = np.zeros((3, 7))
        like.bin_mat_act[:, 2:5] = np.eye(3)
        expected = np.ones((2, 3)) * (1+p[:, 3:4])**2
    actual = like.get_prediction(p)
    np.testing.assert_array_equal(actual[:, :2], np.ones((2, 2)))
    np.testing.assert_allclose(actual[:, 2:], expected, rtol=2e-6)


def test_cross_correction_is_sample_specific(backend):
    like, p = make_lss(dict(indices_list_ka_cross=[[0, 1]]), size=2)
    like.want_lensing_lklh_correction = True
    like.fiducial_cls = np.zeros((4, 3))
    like.dNorm_dCl = np.ones((4, 2, 3)) * .1
    like.norm_fid = np.ones(2)
    like.set_external_spectra([np.ones((2, 2))])
    provider = Provider()
    provider.set_Cl({s: np.array([[1.] * 3, [2.] * 3]) for s in ['tt', 'ee', 'bb', 'te']})
    like.set_provider(provider)
    expected = np.ones((2, 2)) * (1+p[:, 3:4]) * np.array([[2.2], [3.4]])
    if backend:
        import jax.numpy as jnp
        like.set_external_spectra([jnp.ones((2, 2))])
    np.testing.assert_allclose(like.get_prediction(p), expected, rtol=2e-6)


@pytest.mark.parametrize('prior,mean,sigma', [('kl_prior', .815, .019), ('kog_prior', .759, .022)])
def test_lss_s8_priors(backend, prior, mean, sigma):
    like, p = make_lss({prior: True}, size=2)
    like.set_external_spectra([np.zeros((2, 2))])
    s8 = p[:, 6] * np.sqrt((p[:, 7]+p[:, 8]+3*p[:, 10]/93.14)/p[:, 9]**2/.3)
    np.testing.assert_allclose(like.compute_like(p), -.5*((s8-mean)/sigma)**2, rtol=2e-6)
