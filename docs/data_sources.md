# Data sources

The data are read from `src/UPanda/likelihoods/data/<directory>/`. The small data sets are part of the
repository; the others ("download" below) have to be placed in that directory by hand.

| Likelihood | Data | Source | In the repository |
| --- | --- | --- | --- |
| `desi_bao_dr2` | DESI DR2 BAO, [arXiv:2503.14738](https://arxiv.org/abs/2503.14738) | [cobaya bao_data](https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2) | yes |
| `desi_bao_2024` | DESI DR1 BAO, [arXiv:2404.03002](https://arxiv.org/abs/2404.03002) | [cobaya bao_data](https://github.com/CobayaSampler/bao_data) | yes |
| `boss_dr16_rsd`, `boss_dr16_fsigma8` | SDSS MGS, BOSS DR12, eBOSS DR16 LRG and QSO consensus BAO + RSD, [arXiv:2007.08991](https://arxiv.org/abs/2007.08991) | [SDSS DR16 likelihoods](https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-plus/) | yes |
| `eboss_elg` | eBOSS DR16 ELG likelihood grid and fσ8 table, [arXiv:2007.09008](https://arxiv.org/abs/2007.09008) | same SDSS directory; see `data/eboss_dr16_elg/README.txt` | yes (processed grid; the raw 60 MB grid is not) |
| `pantheon_plus` | Pantheon+, [arXiv:2202.04077](https://arxiv.org/abs/2202.04077) | [Pantheon+ data release](https://github.com/PantheonPlusSH0ES/DataRelease) | yes |
| `union3` | Union3 ([arXiv:2311.12098](https://arxiv.org/abs/2311.12098)) and Union3.1 ([arXiv:2601.19854](https://arxiv.org/abs/2601.19854)), 22 binned distances | [cobaya sn_data](https://github.com/CobayaSampler/sn_data), [Union3 release](https://github.com/rubind/union3_release) | yes |
| `des_dovekie` | DES-SN5YR Dovekie reanalysis, [arXiv:2511.07517](https://arxiv.org/abs/2511.07517) | [cobaya sn_data](https://github.com/CobayaSampler/sn_data) | yes |
| `planck_2018_highell_lowell`, `planck_2018_for_act`, `planck_2018_for_act_dr6` | Planck 2018 plik-lite and compressed low-ℓ likelihoods, [arXiv:2104.05715](https://arxiv.org/abs/2104.05715) | [planck-lite-py](https://github.com/heatherprince/planck-lite-py), [planck-low-py](https://github.com/heatherprince/planck-low-py) | yes |
| `planck_2018_lensing` | Planck 2018 lensing | [Planck Legacy Archive](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code); the correction matrices were extracted from the clik files | yes |
| `planck_sroll2_lowell_ee` | SRoll2 low-ℓ EE | probability table of the low-ℓ EE spectrum | yes |
| `camspec_pr4` | Planck PR4 CamSpec, [arXiv:2205.10869](https://arxiv.org/abs/2205.10869) | PR4_12.6cl release, as used by [cobaya](https://github.com/CobayaSampler/cobaya) | download (500 MB) |
| `hillipop` | Planck PR4 HiLLiPoP, [arXiv:2309.10034](https://arxiv.org/abs/2309.10034) | [planck-npipe/hillipop](https://github.com/planck-npipe/hillipop) | download (6.7 GB) |
| `lollipop` | Planck PR4 LoLLiPoP | [planck-npipe/lollipop](https://github.com/planck-npipe/lollipop) | yes |
| `act_dr4` | ACT DR4 foreground-marginalised bandpowers, [arXiv:2007.07288](https://arxiv.org/abs/2007.07288), [arXiv:2007.07289](https://arxiv.org/abs/2007.07289) | [pyactlike](https://github.com/ACTCollaboration/pyactlike) | yes |
| `act_dr6` | ACT DR6 CMB-only bandpowers | [ACT DR6 data products](https://act.princeton.edu/act-dr6-data-products), [DR6-ACT-lite](https://github.com/ACTCollaboration/DR6-ACT-lite) | yes |
| `act_dr6_lensing` | ACT DR6 lensing, [arXiv:2304.05203](https://arxiv.org/abs/2304.05203) | [LAMBDA](https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_lh_info.html), [act_dr6_lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike) | download (1.5 GB) |
| `wmap` | WMAP nine-year likelihood | [LAMBDA](https://lambda.gsfc.nasa.gov/product/wmap/dr5/likelihood_info.html), [pyWMAP](https://github.com/HTJense/pyWMAP) | download (1.4 GB) |
| `general_lss` | measured angular power spectra and simulations for the covariance | supplied by the user | no |

Notes:

- `pantheon_plus` applies the z > 0.01 cut and marginalises over the absolute magnitude; the SH0ES calibration
  is not implemented, although the data file carries SH0ES in its name.
- The origin of the SRoll2 low-ℓ EE probability table bundled here has not been traced to a public release.
