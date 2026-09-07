"""CamSpec file loading, external theory and compression on a tiny data product."""
import importlib
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
from UPanda import config


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def camspec(request, tmp_path, monkeypatch):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    mod = importlib.import_module('UPanda.likelihoods.camspec_pr4')
    root = tmp_path / 'data'
    data = root / 'camspec_pr4'
    data.mkdir(parents=True)
    shutil.copytree(Path(mod.__file__).parent/'data'/'planck_2018_lowl', root/'planck_2018_lowl')
    names = ['100x100', '143x143', '217x217', '143x217', 'TE', 'EE']
    (data/'like_NPIPE_12.6_unified_data_ranges.txt').write_text(''.join(f'{n} 2 4\n' for n in names))
    np.savetxt(data/'like_NPIPE_12.6_unified_spectra.txt', np.zeros((5, 6)))
    np.eye(18, dtype='f4').tofile(data/'like_NPIPE_12.6_unified_cov.bin')
    for file in ['tsz_143_eps0.50.dat', 'cl_ksz_148_trac.dat', 'sz_x_cib_template.dat', 'cib217.txt', '100x100_10.5_dust.dat', '143x143_10.5_dust.dat', '217x217_10.5_dust.dat', '143x217_10.5_dust.dat']:
        np.savetxt(data/file, np.column_stack([np.arange(2, 1502), np.ones(1500)]))
    monkeypatch.setattr(mod, '__file__', str(tmp_path/'camspec_pr4.py'))
    like = mod.camspec_pr4(dtype=np.dtype([('A_planck', 'f8')]))
    yield like, request.param
    config.set_jax_enabled(old)


@pytest.mark.parametrize('source', ['external', 'integrated', 'legacy'])
def test_predictions_units_cuts_and_compression(camspec, source):
    like, backend = camspec
    spectra = {k: np.ones((2, 28))*v*1e-12 for k,v in [('tt', 1), ('te', 2), ('ee', 3)]}
    if source == 'external':
        like.set_external_spectra(spectra)
    else:
        like.additional_args['integrated_model'] = source == 'integrated'
        like.emu_model_list = [SimpleNamespace(predict=lambda p, k=k: spectra[k], predictions_np=lambda p, k=k: np.log(spectra[k]) if k != 'te' else spectra[k]) for k in ['tt','te','ee']]
    ell = np.arange(2., 5.)
    fac = ell*(ell+1)/(2*np.pi)
    foreground = ell/1500
    expected = np.concatenate([fac+foreground]*3+[2*fac, 3*fac])
    params = np.array([[1.], [2.]])
    prediction = like.get_prediction(params)
    np.testing.assert_allclose(prediction, np.stack([expected, expected/4]))
    np.testing.assert_allclose(like.compute_like(params), [-.5*expected@expected, -.5*expected@expected/16])
    projection = np.zeros((2, 15))
    projection[0, :3] = 1/3
    projection[1, 3:6] = 1/3
    like.additional_args['binning_matrix'] = projection
    cov = projection @ like.cov @ projection.T
    compressed = np.asarray(prediction) @ projection.T
    expected_like = [-.5*r@np.linalg.solve(cov,r) for r in compressed]
    np.testing.assert_allclose(like.compute_like(params), expected_like)
    like.additional_args['binned_covinv'] = np.linalg.inv(cov)
    np.testing.assert_allclose(like.compute_like(params), expected_like)
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(like.compute_like)(params), expected_like)
    like.apply_planck_temperature_rescaling = True
    like.apply_planck_pol_rescaling = True
    np.testing.assert_allclose(like.get_prediction(params), prediction)
    like.with_planck = True
    assert np.isfinite(like.compute_like(params)).all()
    like.add_lowell = True
    assert np.isfinite(like.compute_like(params)).all()


def test_calibration_defaults(camspec):
    like, _ = camspec
    np.testing.assert_allclose(like._get_cals({}), np.ones((1, 6)))
    assert like._get_foregrounds({}).shape == (1, 4, 5)
