# Validation

Where a public reference implementation exists, the UPanda likelihood is compared with it on the real data.
The reference is evaluated for a given cosmology, UPanda receives the same theory prediction, and the two
log-likelihoods are compared. For the cobaya comparisons the theory comes from CAMB through cobaya, at two
ΛCDM cosmologies. ACT DR4 is evaluated on the two best-fit spectra distributed with pyactlike.

| UPanda likelihood | Reference | Difference in ln L |
| --- | --- | --- |
| `desi_bao_dr2` | cobaya `bao.desi_dr2.desi_bao_all` | < 1e-9 (relative) |
| `desi_bao_2024` | cobaya `bao.desi_2024_bao_all` | < 1e-9 (relative) |
| `boss_dr16_rsd`, BOSS DR12 + eBOSS LRG | cobaya `bao.sdss_dr16_baoplus_lrg` | < 1e-8 |
| `boss_dr16_rsd`, eBOSS QSO | cobaya `bao.sdss_dr16_baoplus_qso` | < 1e-8 |
| `boss_dr16_rsd`, eBOSS ELG | cobaya `bao.sdss_dr16_baoplus_elg` | 0.03 to 0.04 |
| `pantheon_plus` | cobaya `sn.pantheonplus` | < 1e-8 (relative) |
| `union3` (Union3) | cobaya `sn.union3` | < 1e-8 (relative) |
| `des_dovekie` | cobaya `sn.desdovekie` | < 1e-8 (relative) |
| `planck_2018_highell_lowell`, high-ℓ part | cobaya `planck_2018_highl_plik.TTTEEE_lite_native` | < 1e-7 (relative) |
| `planck_2018_lensing` | cobaya `planck_2018_lensing.CMBMarged` | < 1e-6 (relative) |
| `act_dr4` | [pyactlike](https://github.com/ACTCollaboration/pyactlike) | 1e-4 |
| `lollipop` | [planck-npipe/lollipop](https://github.com/planck-npipe/lollipop) | < 1e-11 (relative) |
| `camspec_pr4` | cobaya CamSpec: foreground and calibration model only | < 1e-13 (relative) |

The eBOSS ELG likelihood is a tabulated, non-Gaussian likelihood on a grid in (D_M/r_d, D_H/r_d, fσ8). Both
codes read the official grid; cobaya interpolates it linearly in ln L, while UPanda first smooths it by 0.7 grid
cells to remove its sampling noise and then uses a cubic interpolation. The difference quoted above comes from
that choice.

These tests are in `tests/test_reference.py`. To run them:

```bash
pip install cobaya camb
pip install 'planck_2020_lollipop @ git+https://github.com/planck-npipe/lollipop.git' \
    'pyactlike @ git+https://github.com/ACTCollaboration/pyactlike.git'
cobaya-install bao.desi_dr2.desi_bao_all bao.desi_2024_bao_all bao.sdss_dr16_baoplus_lrg \
    bao.sdss_dr16_baoplus_qso bao.sdss_dr16_baoplus_elg sn.pantheonplus sn.union3 sn.desdovekie \
    planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lensing.CMBMarged -p <packages>
COBAYA_PACKAGES_PATH=<packages> make test-reference
```

A test is skipped if its reference code or data is missing. They also run in the continuous integration.

Two more comparisons need large data sets and are run by hand:

- `scripts/check_act_lensing_reference.py` compares the ACT DR6 lensing bandpowers and likelihood with
  [act_dr6_lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike);
- `scripts/check_wmap_reference.py` compares the WMAP TT and TE likelihood terms with
  [pyWMAP](https://github.com/HTJense/pyWMAP).

## Not compared yet

The following have unit tests but no end-to-end comparison with a reference code: `hillipop` and the full
`camspec_pr4` likelihood, `act_dr6`, `planck_sroll2_lowell_ee`, the compressed low-ℓ TT and EE terms of the
Planck 2018 likelihoods, the ACT variants `planck_2018_for_act` and `planck_2018_for_act_dr6`, and the MGS block
of `boss_dr16_rsd`. Union3.1 and `boss_dr16_fsigma8` have no cobaya counterpart. `general_lss` takes its data
vector and covariance from the user, so there is nothing to compare it with.
