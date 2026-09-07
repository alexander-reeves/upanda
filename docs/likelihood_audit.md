# Likelihood accuracy audit — September 2026

This review preserves public import paths, constructor keywords and trained-model
formats. It does **not** certify every survey likelihood or every optional mode.
The table distinguishes direct upstream comparisons from independent numerical
checks and unverified paths. Regression coverage is not a substitute for agreement
with a reference likelihood at a specified cosmology.

## ACT lensing: emulator convention comes first

The active `multiprobe_framework` path uses `general_lss`, `spectra=['kaka']`,
`integrated_model=True` and `lens_exp=False`. Its training targets are CCL
convergence spectra **C_ell^kappa-kappa**, not D_ell or lensing-potential spectra.
`combined_probes_data/emulator_training_indices/indices_cmb_lens.txt` selects
exactly 2,998 entries, ell=2 through 2999. The integrated model reverses the log
preprocessing; the likelihood must **not exponentiate that output again**.
It prepends two zeros and applies the ACT binning matrix with **no pi factor**.

The evidence chain in the downstream checkout is:

- `src/multiprobe_framework/theory_code.py::get_theory_cls` calls
  `theory_utils.get_lss_spectra`, which uses CCL CMBLensingTracer/angular_cl.
- `theory_utils.apply_binning` returns `kk`/`kaka` without bandpower binning.
- `scripts/generate_emulator_training_data.py` writes those spectra to training data.
- `submission_scripts/submit_train_kk_modrec_v3.yaml` selects the indices above and
  log preprocessing. `scripts/train_emulator.py` stores the inverse-transform metadata.
- `emulator.prepare_inference` passes the integrated-model flags, and
  `scripts/run_minimizer_no_emu.py` passes 2,998 external ACT entries to the same API.

The legacy standalone `act_dr6_lensing` branch retains its existing `2/pi`
conversion for `predictions_np`; the integrated branch retains factor one.
Planck's existing `2/pi` conversion and its derived bandpower files are also
preserved. These interfaces must not be assigned a universal conversion merely
because their methods have similar names.

`python scripts/check_act_lensing_reference.py --emulator /path/to/kk_emu
--reference /path/to/act_dr6_lenslike.py` reproduces a direct comparison using
real ACT matrices and a trusted saved emulator. The September run used
`multiprobe_framework/combined_probes_data/saved_models/modrec/kk_emu` and
[ACT revision 5d54308](https://github.com/ACTCollaboration/act_dr6_lenslike/blob/5d54308ec60acbf545fd4e46a10091cc6f8a2237/act_dr6_lenslike/act_dr6_lenslike.py).
The two NumPy log-likelihoods were -12.83904002499 and -11.76128971493.
The maximum absolute JAX-versus-reference bandpower difference was 7.24e-15.
The check also compares nonzero CMB corrections on common multipole support.
This verifies likelihood normalization and restoring this artifact; it does not
measure the emulator's cosmological interpolation error.

## Findings that change results or fix broken calls

| Finding | Resolution and evidence |
| --- | --- |
| ACT N1 lensing correction contracted `ij,bj->bj`, summing the output axis instead of the input axis. | Use `ij,bj->bi`, as in upstream `dN1_kk @ (clkk-clkk_fid)`. Nonsymmetric matrix regression, batch independence, gradients, and the real-data comparison above. Results change away from the fiducial spectrum. |
| The ACT fiducial lensing table starts at ell=2 but was subtracted as if its first element were ell=0. | Place it on the explicitly indexed multipole grid. Keep its existing conversion from the CAMB table; no emulator pi factors are changed. |
| Standalone ACT corrections called a nonexistent method; provider spectra and repeated fiducial batches had inconsistent axes. | Call the shared correction helper with `(batch,4,ell)` in TT, EE, BB, TE order. Test distinct samples and provider/fiducial paths. |
| ACT padding tested batch count instead of spectrum length; long spectra could fail at binning. | Pad/trim along the multipole axis. Test short/exact/long support and the actual 2998-element model. |
| General LSS reused sample zero's CMB correction for all samples and rescaled unrelated galaxy/shear blocks with the lensing auto calibration. | Compute per-sample corrections; apply the auto calibration only to the final lensing emulator. Existing cross-correlation normalization convention is retained; see limitations below. |
| HiLLiPoP compression without a supplied compressed precision discarded the full precision, then tried to use it. It also projected precision instead of covariance. | Retain the full precision when needed; compute `(B C B.T)^-1` via a solve against `C^-1`. Correlated, nontrivial bin-average tests cover both compression keywords. Supplied `binned_invkll` behavior is preserved. |
| HiLLiPoP JAX zero-weight bins could divide by zero; zero-error replacement was discarded. | Keep functional updates and a safe denominator. Frequency-weighted means and empty bins agree across backends. |
| LoLLiPoP tools silently forced 29 bins/ellmax=30, `cut_binning` called a missing method, and JAX `vec2mat` discarded updates. | Share the bin metadata implementation, derive operator sizes, and build matrices functionally. Preserve the default ell=2..30 calculation; test arbitrary valid bins and polarization covariance blocks. |
| CamSpec JAX initialization passed a Python list to `jnp.hstack`; legacy EE prediction used NumPy inside JIT; external-only prediction still unpacked missing emulators. | Keep file preprocessing in NumPy, use backend exponentiation, and require emulators only on their prediction path. Test real-format synthetic files, calibration, foregrounds, compression and JIT. |
| TensorFlow bias rescaling attempted indexed assignment into an immutable tensor. | Use concatenation; finite differences verify gradients including bias, amplitude, PCA, log and Chebyshev inverse transforms. |
| Several foreground constructors forced ellmax=2500; JAX power-law/dust assignments failed; synchrotron indexed scalar ratios; explicit ACT 150 GHz was rejected before registration. | Honor requested support and use functional operations/scalar broadcasting. Test normalization, frequency ratios, file alignment, batching and JIT. Negative tSZ tilts avoid an undefined monopole product. |
| Dovekie cuts took a precision sub-block, conditioning on dropped supernovae. | Marginalize discarded data with the Schur complement before removing the absolute-magnitude mode; compare with inversion of the retained covariance. |
| Exact-fit likelihood zero was mistaken for an out-of-prior point; fixed/grouped-prior parameter ordering could disagree. | Use the actual bound mask and a common sampled-parameter order; exercise real emcee callbacks and grouped Gaussian priors. |
| WMAP's off-diagonal Fisher matrices used a single multipole's variance instead of the product at both multipoles, producing asymmetric matrices. Zero TT–TE covariance was also replaced by one. | Restore the upstream outer products and retain zero cross-covariance. Default TT/TE components now agree with pinned pyWMAP on real local data at two input spectra. |
| WMAP's SZ template was allocated with integer dtype; its JAX diagonal update referenced an unavailable `vmap`; provider parameter forwarding was invalid. | Preserve fractional SZ values, update diagonals functionally, and forward the parameter dictionary. SZ integer truncation is also present upstream, so this correction deliberately differs from that loader. |
| WMAP's NumPy Blackwell–Rao calculation underflowed in the tails. | Use log-sum-exp, as the JAX path already did. Preserve UPanda's unbounded log tail; pyWMAP instead applies a far-tail floor. Synthetic tests cover this deliberate difference. |
| `general_lss` covariance assembly returned a NumPy array under the JAX backend, while the official Planck/ACT block replacements called JAX `.at`. Default ACT 5x2pt setup therefore failed during likelihood construction. | Convert the assembled covariance at the backend boundary before applying indices or official blocks. Constructor regressions cover both `add_kk_official` and default `add_ka_official`; the real `mi_project` 5x2pt reproduction now initializes and JIT-compiles all likelihoods. |

## Upstream mapping and extent of verification

| UPanda implementation | Upstream GitHub reference | Verification / remaining difference |
| --- | --- | --- |
| `act_dr6_lensing`, ACT auto part of `general_lss` | [ACT DR6 lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike) | Direct real-matrix/reference comparison with a downstream emulator, NumPy and JAX; detailed above. |
| `lollipop` | [LoLLiPoP](https://github.com/planck-npipe/lollipop/tree/280a9c93d33bc6a058d6bf769ec82d9f7fdbd2b6) | Bundled EE data and the upstream one-field transformation; direct reference script. UPanda's public constructor selects EE; its dormant two-field branch is not certified. |
| `hillipop`, foregrounds | [HiLLiPoP](https://github.com/planck-npipe/hillipop/tree/47659c9930f1b56005d874b77a338ec8b80cbc79) | Reviewed frequency/calibration/foreground equations; independent residual and covariance tests. Full high-dimensional survey likelihood and analysis-specific compression products still need a cosmology reference run. |
| `camspec_pr4` | [Cobaya CamSpec2021 residual model](https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/planck_2018_highl_CamSpec2021/planck_2018_CamSpec2021_python.py), [NPIPE subclass](https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/planck_NPIPE_highl_CamSpec/planck_NPIPE_CamSpec_python.py) | Power-law foregrounds at pivot 1500 and calibration compared directly. UPanda implements the power-law residual mode (`use_fg_residual_model=0`), not all upstream foreground modes. Synthetic file/cut/compression tests; no full PR4 cosmology comparison. |
| `act_dr4` | [pyactlike](https://github.com/ACTCollaboration/pyactlike/tree/1cac8c5d047bc2cad991890f2ebf1d8e3fb483b3) | Bundled bandpowers/windows, calibration, Gaussian calculation and integrated/legacy outputs tested. Full upstream numerical comparison remains outstanding. |
| `act_dr6` | [DR6 ACT-lite](https://github.com/ACTCollaboration/DR6-ACT-lite) | Local SACC Gaussian/window/calibration/provider contracts tested. The local data filename/version must be matched before claiming agreement with a specific release. |
| Planck high/low and ACT combinations | [Planck-lite-py](https://github.com/heatherprince/planck-lite-py), [Planck-low-py](https://github.com/heatherprince/planck-low-py) | Bundled Gaussian data and low-ell additions tested. `planck_2018_for_act` uses only 78 TT bins through approximately ell=645 and always adds low TT; this is not full Plik with an exact ell=650 cut. |
| `desi_bao_2024`, `desi_bao_dr2`, BOSS/eBOSS | [Cobaya BAO data](https://github.com/CobayaSampler/bao_data), [Cobaya BAO implementation](https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/base_classes/bao.py) | Gaussian contractions and data ordering tested. The f-sigma8-only class deliberately discards cross-measurement covariance; it is an approximation, not the full joint RSD likelihood. |
| `pantheon_plus`, `union3`, `des_dovekie` | [Cobaya SN](https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods/sn), [SN data](https://github.com/CobayaSampler/sn_data) | Direct comparison of constant-magnitude marginalization; synthetic correlated covariance/cut tests. `use_shoes` still does not implement a calibrated SH0ES likelihood. Dovekie covariance cuts deliberately use marginalization even where an upstream inherited mask only handles covariance dictionaries. |
| `planck_2018_lensing` | [Cobaya Planck lensing](https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods/planck_2018_lensing) | Local derived binning/offset conventions retained and tested; no claim of exact agreement with every PLA product. |
| `planck_sroll2_lowell_ee` | [Cobaya likelihoods](https://github.com/CobayaSampler/cobaya/tree/master/cobaya/likelihoods) | Lookup/index/calibration tests; exact provenance of the local probability table is unresolved. |
| `wmap` | [pyWMAP](https://github.com/HTJense/pyWMAP/tree/c1adc0eca8551b87085c3eb1754c3d7d8d04faa0) | Real default TT/TE covariance, low-TT and high-TT/TE components agree at the supplied test spectrum and 1.02 times that spectrum (tolerance 1e-10). NumPy/JAX synthetic tests cover Fisher matrices, log tails, units and calibrations. SZ truncation and far-tail differences are described above. Disabled beam/TB/pixel modes and production chains remain unvalidated. |

`python -m pytest -m upstream tests/test_upstream_numpy.py` runs the automated
NumPy comparisons. CI pins Cobaya at `b76b6fe` and LoLLiPoP at `280a9c9` so an
upstream change cannot silently alter the reference result.

`python scripts/check_wmap_reference.py --reference /path/to/pyWMAP/wmaplike/wmap.py`
runs the real-data WMAP comparison. At unit spectrum amplitude the component
log probabilities are `6.807435606988155`, `-596.830768186799`,
`-659.5720443436185`, and `-1765.5023879835906` (low TT, high TT,
TE quadratic term, TE determinant term). This comparison excludes SZ so that
the upstream template truncation does not conceal the covariance comparison.

## Open scientific questions

General LSS cross-correlation corrections use analysis-specific already-binned
vectors and a historical normalization factor. Their batch bug is fixed, but
matching their bandpower response to the reconstruction normalization requires
the originating analysis products; the ACT *auto* reference does not validate
those cross-correlation conventions. Likewise, cut selections, priors, covariance
estimation, and emulator training-domain accuracy require analysis-level checks.

Optional HiLLiPoP subpixel foregrounds still assume an ell=2500 pivot in the
supplied spectrum and scalar amplitudes. They are not validated for shorter
multipole grids or batched nuisance parameters; the foreground tests above do
not certify that optional component.

The review has not replaced survey data values, retrained emulators, or changed
the established Planck/ACT emulator conversion factors. Consult
[data provenance](data_sources.md) and [compatibility review](review.md) together
with this note before comparing chains made with earlier revisions.
