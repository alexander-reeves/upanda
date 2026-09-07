"""Bundled distance/RSD products: actual readers, ordering and marginalization."""
import numpy as np
import pytest
from UPanda import config
from UPanda.likelihoods.union3 import union3
from UPanda.likelihoods.boss_dr16_fsigma8 import boss_dr16_fsigma8
from UPanda.likelihoods.boss_dr16_rsd import boss_dr16_rsd


def test_union3_bundled_data_constant_offset_invariance():
    config.set_jax_enabled(False)
    like = union3()
    assert like.nsn == 22 and np.all(np.diff(like.zcmb)>0)
    for offset in [0., 5., -12.]:
        like.set_external_spectra(like.mag + offset)
        np.testing.assert_allclose(like.compute_like(None), 0., atol=1e-9)
    np.testing.assert_allclose(like.inv_cov@np.ones(22),0.,atol=1e-10)


def test_union31_bundled_data_and_version_validation():
    like = union3(additional_args={"union_version": "31"})
    assert like.union_version == "31"
    assert like.nsn == 22
    assert np.all(np.diff(like.zcmb) > 0)
    np.testing.assert_allclose(like.inv_cov @ np.ones(22), 0.0, atol=1e-10)
    with pytest.raises(ValueError, match="union_version"):
        union3(additional_args={"union_version": "4"})


def test_union3_reader_preserves_header_among_comments(tmp_path):
    path = tmp_path/'sn.txt'
    path.write_text('# explanatory comment\n# name zcmb zhel mb\n\na .1 .11 23\n# trailing comment\nb .2 .21 24\n')
    like=union3.__new__(union3)
    like._read_data(path)
    np.testing.assert_allclose(like.mag,[23,24])
    path.write_text('a .1 .11 23\n')
    with pytest.raises(ValueError,match='header'):
        like._read_data(path)


@pytest.mark.parametrize('entries', [[1.5, 1], [0], [2, 1, 0, 0], [1, 1, 2], [1, np.nan], [2, 1, 2, 0, 1]])
def test_union3_covariance_rejects_corrupt_files(tmp_path,entries):
    path=tmp_path/'cov.txt'
    np.savetxt(path,entries)
    with pytest.raises(ValueError):
        union3._read_covmat(path)


@pytest.mark.parametrize('samples', [('mgs',),('boss_dr12',),('eboss_lrg',),('eboss_qso',)])
def test_rsd_bundled_metadata_and_gaussian(samples):
    like=boss_dr16_rsd(samples=samples)
    assert len(like.z_eff)==len(like.quantities)==len(like.data_vector)
    theory=np.asarray(like.data_vector)+.01
    like.set_external_spectra({'boss_dr16_rsd_data':theory})
    expected=-.5*np.ones(len(theory))@np.asarray(like.inv_cov)@np.ones(len(theory))*.01**2
    np.testing.assert_allclose(like.compute_like(None),expected,rtol=1e-12)


def test_fsigma8_emulator_and_mock_data():
    from types import SimpleNamespace
    like=boss_dr16_fsigma8(dtype=np.dtype([('x','f8')]),
        emu_model_list=[SimpleNamespace(predict=lambda p: np.ones((len(p['x']),6))*.5)],
        additional_args={'mock_data':np.ones(6)*.5})
    np.testing.assert_allclose(like.compute_like(np.array([[1.],[2.]])),[0,0])
