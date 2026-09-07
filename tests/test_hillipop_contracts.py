"""Small multi-frequency data check residuals, weighting and compression."""
from types import SimpleNamespace
import numpy as np
import pytest
from UPanda import config
from UPanda.likelihoods.hillipop import hillipop


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def like(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    obj = hillipop.__new__(hillipop)
    obj.frequencies = [100,100,143,143,217,217]
    obj._mapnames = ['100A','100B','143A','143B','217A','217B']
    obj._nmap, obj._nfreq, obj._nxfreq, obj._nxspec = 6,3,6,15
    obj._xspec2xfreq = obj._xspec2xfreq()
    obj.lmax = 4
    obj._is_mode = dict(TT=True, EE=True, TE=True, ET=True, BB=False)
    obj._lmins = {k: np.full(15, 2) for k in ['TT','EE','TE','ET']}
    obj._lmaxs = {k: np.full(15, 4) for k in obj._lmins}
    obj._dlweight = {k: np.ones((15, 5)) for k in obj._lmins}
    obj._dldata = {k: np.ones((15, 5))*2 for k in obj._lmins}
    obj.fgs = {k: [] for k in obj._lmins}
    obj.apply_hillipop_temperature_rescaling = False
    obj.apply_hillipop_pol_rescaling = False
    obj.additional_args = {}
    covariance = np.eye(54)*2 + .1
    obj._invkll = np.linalg.inv(covariance)
    yield obj, request.param, covariance
    config.set_jax_enabled(old)


def nuisance(like):
    return {**{f'cal{n}': np.ones(2) for n in like._mapnames},
            **{f'pe{n}': np.ones(2) for n in like._mapnames},
            'A_planck': np.ones(2), 'm_t_hillipop': np.zeros(2),
            'm_e_hillipop': np.zeros(2)}


def test_frequency_weighted_means_and_empty_bins(like):
    obj, backend, _ = like
    cl = np.arange(15*2*5.).reshape(15,2,5)
    weights = np.ones((15,5))
    weights[:,0] = 0
    result = obj._xspectra_to_xfreq(cl, weights)
    for i in range(6):
        select = np.asarray(obj._xspec2xfreq)==i
        np.testing.assert_allclose(result[i,:,1:], cl[select,:,1:].mean(axis=0))
    np.testing.assert_allclose(result[:,:,0], 0.)
    assert obj._get_matrix_size() == 54
    assert obj.dof() == 54
    assert obj.reduction_matrix('TT').shape == (18,5)
    assert obj.get_requirements() == {'Cl': {'tt':4,'ee':4,'te':4}}


@pytest.mark.parametrize('compression', ['none','bin','precomputed','projection'])
def test_residual_gaussian_and_compression(like, compression):
    obj, backend, covariance = like
    theory = {k: np.stack([np.ones(5), np.ones(5)*.5]) for k in obj._lmins}
    residual = np.stack([np.ones(54), np.ones(54)*1.5])
    p = nuisance(obj)
    projection = np.zeros((2,54))
    projection[0,:27] = 1/27
    projection[1,27:] = 1/27
    if compression == 'none':
        expected = [r@np.linalg.solve(covariance,r) for r in residual]
    else:
        cov = projection@covariance@projection.T
        expected = [(projection@r)@np.linalg.solve(cov,projection@r) for r in residual]
        obj.additional_args['hillipop_compression' if compression=='projection' else 'binning_matrix'] = projection
        if compression=='precomputed':
            obj.additional_args['binned_invkll'] = np.linalg.inv(cov)
    np.testing.assert_allclose(obj.compute_chi2(theory,p), expected, rtol=1e-12)
    np.testing.assert_allclose(obj.delta_cl,residual)
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(obj.compute_chi2)(theory,p), expected, rtol=1e-12)


@pytest.mark.parametrize('mode', ['TT','EE','TE','ET'])
def test_foreground_residual_and_rescaling(like,mode):
    obj, _, _ = like
    obj.apply_hillipop_temperature_rescaling = True
    obj.apply_hillipop_pol_rescaling = True
    obj.cib_systematic_factor = 2.
    obj.fgs[mode] = [SimpleNamespace(compute_dl=lambda p: np.ones((15,2,5))*.25)]
    theory = {mode: np.ones((2,5))}
    np.testing.assert_allclose(obj._compute_residuals(nuisance(obj),theory,mode), .5)


def test_readers_reject_missing_files(like,tmp_path):
    obj, _, _ = like
    for fn in [obj._set_multipole_ranges,obj._read_invcovmatrix]:
        with pytest.raises(ValueError,match='File missing'):
            fn(str(tmp_path/'missing.fits'))
