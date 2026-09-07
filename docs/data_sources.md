# Likelihood data sources

Links below were checked during the September 2026 review. They identify upstream
releases and implementations; they are not claims that every local file has been
verified against a particular upstream commit. Preserve a copy of your actual
inputs and cite the survey paper for the likelihood used.

| UPanda module(s) | Upstream data/code and reference | Local directory / conventions |
| --- | --- | --- |
| `desi_bao_dr2` | [DESI DR2 results](https://arxiv.org/abs/2503.14738), [DESI release guide](https://www.desi.lbl.gov/2025/03/19/desi-dr2-results-march-19-guide/), [Cobaya BAO data](https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2) | `desi_bao_dr2`: `mean.txt` and `cov.txt`. Prediction order follows rows of the mean file; values are BAO distance/drag-scale observables, not CMB spectra. |
| `desi_bao_2024` | [Cobaya BAO data](https://github.com/CobayaSampler/bao_data) (`desi_2024_gaussian_bao_ALL_GCcomb_*`) | `desi_bao_2024`: older release; do not substitute the DR2 vector into the same emulator. |
| `boss_dr16_rsd` | [Alam et al. consensus analysis](https://arxiv.org/abs/2007.08991), [SDSS/Cobaya BAO+RSD products](https://github.com/CobayaSampler/bao_data) | `boss_dr16_rsd`: MGS, BOSS DR12, eBOSS LRG/QSO blocks in canonical sample order. `quantities` and `z_eff` expose the exact ordering. |
| `boss_dr16_fsigma8` | [Alam et al., Table III](https://arxiv.org/abs/2007.08991) | `boss_dr16_fsigma8`: six fσ8 points. UPanda uses a diagonal approximation; this is not the full joint consensus covariance. Do not combine with overlapping joint BAO+RSD points. |
| `pantheon_plus` | [Pantheon+ / SH0ES data release](https://github.com/PantheonPlusSH0ES/DataRelease) | `pantheon_plus`: `Pantheon+SH0ES.dat`, `Pantheon+SH0ES_STAT+SYS.cov`; zHD > 0.01 cut and absolute-magnitude marginalization. The current UPanda implementation does not implement SH0ES calibration even though the input filename includes SH0ES. |
| `des_dovekie` | [Cobaya SN data](https://github.com/CobayaSampler/sn_data), [Cobaya likelihood source](https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods/sn) | `des_dovekie`: `DES-Dovekie_HD.csv` and `STAT+SYS.npz`; pre-inverted covariance, zHD > 0.01, magnitude marginalization. Check the upstream release citation with these files; the local module records arXiv:2511.07517. |
| `union3` | [Rubin et al., Union3/UNITY1.5](https://arxiv.org/abs/2311.12098), [Union3.1 release](https://github.com/rubind/union3_release), [Rubin et al., Union3.1/UNITY1.8](https://arxiv.org/abs/2601.19854), [Cobaya SN data](https://github.com/CobayaSampler/sn_data) | 22 binned distances (`lcparam_full.txt`, `mag_covmat.txt`) with magnitude marginalization. `union_version="3"` (default) selects Union3; `union_version="31"` selects Union3.1. |
| `planck_2018_highell_lowell`, `planck_2018_for_act`, `planck_2018_for_act_dr6` | [Planck-lite-py](https://github.com/heatherprince/planck-lite-py), [Planck-low-py](https://github.com/heatherprince/planck-low-py), [compressed low-ell paper](https://arxiv.org/abs/2104.05715) | `planck_2018_highell`, `planck_2018_lowl`: compressed/binned implementations. ACT combinations apply additional spectrum/multipole selections. |
| `planck_2018_lensing` | [Planck Legacy Archive likelihood products](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code) | `planck_2018_lensing`: local README states correction files were generated using `extract_lensing_corr.py`. These are derived products, not just the untouched archive. |
| `camspec_pr4` | [Rosenberg, Gratton & Efstathiou](https://arxiv.org/abs/2205.10869), [Cobaya](https://github.com/CobayaSampler/cobaya) | `camspec_pr4`: local README identifies PR4_12.6cl. Large files separately provisioned. Low-ell terms use the Planck compressed fits when enabled. |
| `hillipop` | [Planck NPIPE HiLLiPoP](https://github.com/planck-npipe/hillipop) | `planck_pr4_hillipop`: separate large data; locally compressed binning/inverse matrices may be analysis-specific derived products. |
| `lollipop` | [Planck NPIPE LoLLiPoP](https://github.com/planck-npipe/lollipop) | `planck_pr4_lollipop`: low-ell polarization; preserve the selected bins and covariance variant. |
| `planck_sroll2_lowell_ee` | [Planck/Cobaya likelihood implementations](https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods) | `planck_sroll2_lowell_ee`: local low-ell EE table. Exact upstream table provenance still needs confirmation; do not identify it as NPIPE LoLLiPoP. |
| `act_dr4` | [ACT DR4 pyactlike](https://github.com/ACTCollaboration/pyactlike) | `act_dr4`: foreground-marginalized CMB bandpowers/windows, with calibration nuisance parameters. Upstream requests Aiola et al. and Choi et al. citations. |
| `act_dr6` | [ACT DR6 data products](https://act.princeton.edu/act-dr6-data-products) | `act_dr6`: local SACC product. Record its filename/version, cuts, and calibration convention; do not assume equivalence to every upstream DR6 likelihood. |
| `act_dr6_lensing` | [ACT lensing likelihood](https://github.com/ACTCollaboration/act_dr6_lenslike), [NASA LAMBDA likelihood data](https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_lh_info.html), [lensing cosmology paper](https://arxiv.org/abs/2304.05203) | `act_dr6_lensing`: separately provisioned reconstruction/correction files. Distinguish CMB-marginalized and full CMB-dependent corrections. |
| `wmap` | [NASA WMAP nine-year likelihood](https://lambda.gsfc.nasa.gov/product/wmap/dr5/likelihood_info.html), [pyWMAP implementation](https://github.com/HTJense/pyWMAP) | `wmap`: separate large nine-year likelihood assets; the module references pyWMAP's setup. |
| `general_lss` | User-supplied UFalcon HDF5 realizations/data vectors, with optional Planck/ACT lensing inputs linked above | No single survey likelihood defines this module. Record simulation count, spectra order, shot noise, masks, Hartlap correction, bandpowers, and normalization files. The small `KiDS_1000` asset alone is not a complete KiDS likelihood. |

## Provisioning and reproducibility

Editable installations use the existing `src/UPanda/likelihoods/data` layout.
The explicit package-data entries in `pyproject.toml` include the smaller assets
in wheels and source distributions. CamSpec, HiLLiPoP, ACT DR6 lensing, and WMAP
are excluded from built distributions because of their size. Provision those
separately in the layout expected by their constructors. Some likelihoods have
additional path overrides; consult their constructor before relocating inputs.

UPanda has no working automatic data downloader. The older README's proposed
Zenodo downloads/cache were not implemented. Upstream archives may have different
layouts and require conversion; copying an arbitrary tarball into the directory
is not sufficient. The upstream links establish a starting point for provenance,
not a universal installation recipe.

The review keeps numerical data values unchanged. A release should record source
URL, upstream revision, download date, derivation commands and SHA256 values for
all externally provisioned/derived assets. Keep survey and emulator provenance
together: matching a covariance to an incompatible prediction ordering can
produce finite but incorrect likelihoods.
