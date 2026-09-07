"""Independent binning and polarization calculations on both array backends."""
import numpy as np
import pytest

from UPanda import config
from UPanda.likelihoods import tools_hillipop as hill, tools_lollipop as lol


@pytest.fixture(params=[False, True], ids=['numpy', 'jax'])
def backend(request):
    old = config.get_jax_enabled()
    config.set_jax_enabled(request.param)
    yield request.param
    config.set_jax_enabled(old)


def test_bin_means_and_covariance(backend):
    bins = lol.Bins([2, 4], [3, 6])
    spectrum = np.arange(7.) ** 2
    p = np.zeros((2, 7))
    p[0, 2:4] = .5
    p[1, 4:7] = 1/3
    covariance = np.diag(np.arange(1., 8.)) + .2
    np.testing.assert_allclose(bins.bin_spectra(spectrum, *bins.bins()), [6.5, 77/3])
    np.testing.assert_allclose(bins.bin_covariance(covariance, *bins.bins()), p @ covariance @ p.T)
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(lambda x: bins.bin_spectra(x, *bins.bins()))(spectrum), p @ spectrum)
        np.testing.assert_allclose(jax.jacrev(lambda x: bins.bin_spectra(x, *bins.bins()))(spectrum), p)
    weighted, unbin = bins._bin_operators(*bins.bins(), Dl=True)
    np.testing.assert_allclose(weighted @ unbin, np.eye(2), atol=1e-14)
    bins.cut_binning(4, 6)
    assert bins.nbins == 1 and bins.lmin == 4 and bins.lmax == 6
    np.testing.assert_allclose(bins.bin_spectra(spectrum, *bins.bins()), [77/3])


@pytest.mark.parametrize('limits', [(2, 30), (2, 55), (36, 65)])
def test_binning_covers_requested_complete_bins(backend, limits):
    bins, lo, hi = lol.get_binning(*limits)
    assert bins.lmin == limits[0] and bins.lmax == limits[1]
    assert len(lo) == bins.nbins == len(hi)
    np.testing.assert_allclose(bins.bin_spectra(np.ones(limits[1]+1), lo, hi), 1.)
    assert lol.Bins.fromdeltal(2, 9, 2).nbins == 4


@pytest.mark.parametrize('limits', [(1, 30), (2, 201), (30, 2), (36, 40)])
def test_invalid_bin_ranges(limits):
    with pytest.raises(ValueError):
        lol.get_binning(*limits)


@pytest.mark.parametrize('lo,hi', [([2], [3, 4]), ([4], [3]), ([], [])])
def test_invalid_bins(lo, hi):
    with pytest.raises(ValueError):
        lol.Bins(lo, hi)


def test_polarization_covariance_blocks(backend):
    bins = lol.Bins([2, 4], [3, 5])
    rng = np.random.default_rng(3)
    a = rng.normal(size=(12, 12))
    covariance = a @ a.T
    p = np.kron(np.eye(3), np.array([[.5, .5, 0, 0], [0, 0, .5, .5]]))
    expected = p @ covariance @ p.T
    np.testing.assert_allclose(lol.bin_covEB(covariance, bins, *bins.bins()), expected, atol=1e-13)
    np.testing.assert_allclose(lol.bin_covEE(covariance, bins, *bins.bins()), expected[:2, :2])
    np.testing.assert_allclose(lol.bin_covBB(covariance, bins, *bins.bins()), expected[2:4, 2:4])
    with pytest.raises(ValueError, match='cover'):
        lol.bin_covEE(np.eye(3), bins, *bins.bins())


def test_matrix_conversion_and_hl(backend):
    np.testing.assert_allclose(lol.vec2mat([2., 3.]), np.diag([2., 3.]))
    np.testing.assert_allclose(lol.vec2mat([2., 3., .5]), [[2., .5], [.5, 3.]])
    np.testing.assert_allclose(lol.mat2vec(lol.vec2mat([2., 3., .5])), [2., 3., .5])
    x = np.array([.5, 1., 2.])
    np.testing.assert_allclose(lol.ghl(x), np.sign(x-1)*np.sqrt(2*(x-np.log(x)-1)))
    ell = np.array([2., 5.])
    # With zero fiducial signal the offset is the analytic noise variance.
    np.testing.assert_allclose(lol.compute_offsets(ell, np.array([4., 9.]), np.zeros(2)), np.sqrt([10., 49.5]))


def test_read_polarization_grid(backend, tmp_path):
    path = tmp_path / 'spectra.txt'
    np.savetxt(path, [[2, 1, 2, 3], [350, 4, 5, 6]])
    result = np.asarray(lol.read_dl(path))
    assert result.shape == (3, 351)
    np.testing.assert_array_equal(result[:, [2, 350]], [[1, 4], [2, 5], [3, 6]])
    np.savetxt(path, [[2, 1, 2, 3]])
    assert lol.read_dl(path).shape == (3, 301)
    np.savetxt(path, [[-1, 1, 2, 3]])
    with pytest.raises(ValueError, match='Multipoles'):
        lol.read_dl(path)


def test_contour_threshold_selects_highest_density_pixel():
    grid = np.array([[1, 2], [3, 4]])
    np.testing.assert_allclose(hill.convert_to_stdev(grid), [[1., .9], [.7, .4]])
    np.testing.assert_allclose(hill.ctr_level(grid, [0., .4, .7, 1.]), [4, 4, 3, 1])


def test_fits_binning_roundtrip_and_smoothing(tmp_path):
    from astropy.io import fits
    path = tmp_path / 'bins.fits'
    bins = [(2, 3), (4, 7)]
    hill.create_bin_file(path, bins, bins, bins, bins, bins)
    with fits.open(path) as hdus:
        assert len(hdus) == 6
        np.testing.assert_array_equal(hdus[1].data['LMIN'], [2, 4])
    for cut in [0, 20]:
        np.testing.assert_allclose(hill.SG(np.arange(50), np.ones(50), lcut=cut), 1.)


def test_lollipop_external_theory_without_emulators(backend):
    from UPanda.likelihoods.lollipop import lollipop
    like = lollipop(dtype=np.dtype([('A_planck', 'f8')]))
    spectra = np.asarray(like.cldata[0])[None, :] / 1e12
    like.set_external_spectra({'ee': spectra})
    params = np.array([[1.]])
    np.testing.assert_allclose(like.compute_like(params), 0., atol=1e-12)
    assert like.get_requirements() == {'Cl': {'ee': 30, 'bb': 30}}
    like.marginalised_over_covariance = True
    np.testing.assert_allclose(like.compute_like(params), 0., atol=1e-12)
    if backend:
        import jax
        np.testing.assert_allclose(jax.jit(like.compute_like)(params), 0., atol=1e-12)
