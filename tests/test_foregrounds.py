"""Foreground normalization, frequency scaling, template I/O and JAX contracts."""
import numpy as np
import pytest
from UPanda import config
from UPanda.likelihoods import foregrounds_hillipop as fg


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def backend(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    yield request.param
    config.set_jax_enabled(old)


@pytest.fixture
def template(tmp_path):
    path = tmp_path / 'template.txt'
    ell = np.arange(2, 3002)
    np.savetxt(path, np.column_stack([ell, ell*(ell+1)]))
    return str(path)


def test_powerlaw_and_frequency_normalization(backend, template):
    model = fg.fgmodel(20, [100, 143, 217])
    assert model.ll2pi.shape == (21,)
    for name in ['_tszRatio', '_cibRatio', '_dustRatio', '_radioRatio', '_syncRatio']:
        np.testing.assert_allclose(getattr(model, name)(143., 143.), 1.)
    power = model._gen_dl_powerlaw(-2., lnorm=10)
    ell = np.arange(2, 21.)
    np.testing.assert_allclose(power[2:], ((ell+1)/ell)/(11/10))
    raw = model._gen_dl_powerlaw(-2., lnorm=None)
    np.testing.assert_allclose(raw[2:], (ell+1)/ell/(2*np.pi))
    loaded = model._read_dl_template(template)
    assert loaded.shape == (21,)
    np.testing.assert_allclose(loaded[2:], ell*(ell+1)/(3000*3001))


@pytest.mark.parametrize('name,amplitude', [('ps','Aps_143x143'), ('ps_radio','Aradio'), ('ps_dusty','Adusty'), ('cib_model','Acib'), ('sync_model','AsyncT')])
def test_foreground_batch_amplitude_and_disabled_modes(backend, name, amplitude):
    model = getattr(fg, name)(20, [143, 143])
    pars = {amplitude: np.array([1., 3.]), 'beta_radio': np.array([-.7, -.7]), 'beta_cib': np.array([1.75, 1.75])}
    result = np.asarray(model.compute_dl(pars))
    assert result.shape == (1, 2, 21)
    np.testing.assert_allclose(result[0, 1], 3*result[0, 0])
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(model.compute_dl)(pars), result, atol=1e-14)
    model.mode = 'TE'
    assert model.compute_dl(pars) == 0.
    if name == 'sync_model':
        model.mode = 'EE'
        pars['AsyncP'] = pars['AsyncT']
        np.testing.assert_allclose(model.compute_dl(pars), result)


def test_sz_models_and_cross_template(backend, template):
    for cls, amp in [(fg.tsz_model, 'Atsz'), (fg.ksz_model, 'Aksz')]:
        model = cls(20, [143, 143], filename=template)
        result = np.asarray(model.compute_dl({amp: np.array([1., 2.])}))
        assert result.shape == (1, 2, 21)
        np.testing.assert_allclose(result[0, 1], 2*result[0, 0])
    act = fg.tsz_model(20, [150, 150], filename=template, template='battaglia')
    value = np.asarray(act.compute_dl({'Atsz': np.array([1.]), 'alpha_tsz': np.array([-.5])}))
    assert np.isfinite(value).all() and value[0, 0, 0] == 0
    assert 150 not in fg.fgmodel.fsz  # instance configuration must not leak
    pars = {k: np.array([v]) for k, v in dict(xi=.2, Acib=4., Atsz=9., beta_cib=1.75).items()}
    cross = fg.szxcib_model(20, [143, 143], filename=template)
    expected = -2*.2*np.sqrt(4*9)*cross._cibRatio(fg.fgmodel.fcib[143], 143, beta=1.75)*np.asarray(cross.x_tmpl)
    np.testing.assert_allclose(cross.compute_dl(pars), expected[None, None, :])
    cross.mode = 'EE'
    assert cross.compute_dl(pars) == 0.
    combined = fg.szxcib_model(20, [143, 143], filenames=[template, template])
    np.testing.assert_allclose(combined.x_tmpl, cross.x_tmpl**2)


@pytest.mark.parametrize('name', ['sync_model','cib_model','tsz_model','szxcib_model'])
def test_unknown_frequencies(name):
    with pytest.raises(ValueError, match='Missing'):
        getattr(fg, name)(20, [999, 143])


def test_missing_templates():
    with pytest.raises(ValueError, match='Missing template'):
        fg.szxcib_model(20, [100, 143])
    with pytest.raises(ValueError, match='data_folder'):
        fg.tsz_model(20, [100, 143], template='battaglia')
    with pytest.raises(ValueError, match='template must'):
        fg.tsz_model(20, [100, 143], template='typo')


@pytest.mark.parametrize('mode', ['TT', 'EE', 'TE', 'ET'])
def test_dust_template_ordering_and_amplitude(backend, tmp_path, mode):
    base = tmp_path / 'dust'
    ell = np.arange(2, 22)
    data = np.column_stack([ell] + [ell*0.+i for i in range(1, 7)])
    np.savetxt(str(base)+'_'+mode+'.txt', data)
    model = fg.dust_model(20, [100, 143], filename=str(base), mode=mode)
    pars = {k: np.array([v, v]) for k, v in dict(beta_dustT=1.5, beta_dustP=1.6, AdustT=2., AdustP=3.).items()}
    result = model.compute_dl(pars)
    assert result.shape == (1, 2, 21)
    np.testing.assert_allclose(result[:, :, :2], 0.)
    first, second = mode
    amplitudes = {'T': 2., 'E': 3.}
    betas = {'T': 1.5, 'E': 1.6}
    expected = 2*amplitudes[first]*amplitudes[second]*model._dustRatio(model.fdust[100], model.fdust[353], beta=betas[first])*model._dustRatio(model.fdust[143], model.fdust[353], beta=betas[second])
    np.testing.assert_allclose(result[:, :, 2:], expected)
    legacy = fg.dust(20, [100, 143], filename=str(base), mode=mode)
    np.testing.assert_allclose(legacy.compute_dl({f'Ad{f}{p}': 2. for f in [100, 143, 217] for p in ['T', 'P']})[:, 2:], 8.)
