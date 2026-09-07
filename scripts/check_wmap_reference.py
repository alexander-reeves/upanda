"""Compare WMAP's default TT/TE kernels using real local survey data.

Pass --reference /path/to/pyWMAP/wmaplike/wmap.py from a trusted upstream
checkout (reviewed revision c1adc0eca8551b87085c3eb1754c3d7d8d04faa0).
Requires Cobaya and the separately provisioned UPanda WMAP data directory.
The SZ template is tested separately: upstream also truncates it to integers.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('wmap_reference', args.reference)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    from UPanda import config
    config.set_jax_enabled(False)
    from UPanda.likelihoods.wmap import wmap
    like = wmap()
    reference = upstream.WMAPLike.__new__(upstream.WMAPLike)
    reference.__dict__.update(like.__dict__)
    data = np.loadtxt(Path(like.data_folder) / 'test_cls_v5.dat')
    ell = data[:, 0].astype(int)
    spectra = {}
    for name, column in [('tt', 1), ('te', 2), ('ee', 3), ('bb', 4)]:
        cl = np.zeros(max(ell) + 1)
        cl[ell] = data[:, column]
        spectra[name] = cl
    results = []
    for scale in [1., 1.02]:
        cls = {k: v * scale for k, v in spectra.items()}
        sizes = [like.tt_lmax + 1] + [like.te_lmax + 1] * 4
        ref_cov = [np.zeros(n) for n in sizes]
        reference.loglike_init_covterms(cls.copy(), *ref_cov)
        actual_cov = like.loglike_init_covterms(
            {k: v[None, :] for k, v in cls.items()},
            *(np.zeros((1, n)) for n in sizes))
        for expected, actual in zip(ref_cov, actual_cov):
            np.testing.assert_allclose(actual[0], expected, atol=1e-12)
        expected = [reference.loglike_lowl_TT(cls['tt']),
                    reference.loglike_highl_TT(cls['tt'], *ref_cov[:3]),
                    *reference.loglike_highl_TE(cls['te'], *ref_cov[:3])]
        actual = [like.loglike_lowl_TT(cls['tt'][None, :])[0],
                  like.loglike_highl_TT(cls['tt'][None, :], *actual_cov[:3])[0],
                  *(v[0] for v in like.loglike_highl_TE(cls['te'][None, :], *actual_cov[:3]))]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        results.append({'scale': scale, 'lowl_tt_highl_tt_te_chi_te_det': actual})
    print(json.dumps({'comparisons': results, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
