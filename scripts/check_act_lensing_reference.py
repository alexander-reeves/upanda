"""Compare ACT lensing using real data and a downstream integrated emulator.

Run from an environment with UPanda, TensorFlow and JAX installed::

    python scripts/check_act_lensing_reference.py --emulator /path/to/kk_emu \
        --reference /path/to/act_dr6_lenslike.py

The reference file must be a trusted upstream ACT source file. Only its
get_corrected_clkk function is evaluated. No datasets are downloaded or changed.
The emulator is a trusted local UPanda model stem, without .pkl/_model.h5.
"""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def reference_function(path):
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'get_corrected_clkk')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_corrected_clkk']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emulator', required=True)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args()
    from UPanda import config, integrated_model
    from UPanda.likelihoods.general_lss import general_lss
    from UPanda.likelihoods.utils_likelihood import compute_act_lensing_auto_corrections

    config.set_jax_enabled(False)
    model = integrated_model.IntegratedModel(None, None, None, None)
    model.restore(args.emulator)
    fiducial = dict(omega_cdm=.12, omega_b=.0224, sigma8=.81, n_s=.965, h=.674,
                    logT_AGN=7.8, m_nu=.02, w0=-1., wa=0., tau=.054,
                    **{'ln10^{10}A_s': 3.044})
    fiducial.update({f'q_{i}': 0. for i in range(1, 8)})
    missing = set(model.parameter_names)-fiducial.keys()
    if missing:
        raise ValueError(f'Provide fiducial values for emulator parameters: {missing}')
    params = {name: np.array([fiducial[name], fiducial[name]]) for name in model.parameter_names}
    if 'sigma8' in params:
        params['sigma8'][1] *= 1.02
    prediction = np.asarray(model.predict(params))
    if prediction.shape != (2, 2998):
        raise ValueError(f'Expected ell=2..2999 emulator outputs, got {prediction.shape}')
    like = general_lss(dtype=np.dtype([(n, 'f8') for n in model.parameter_names]),
        emu_model_list=[model], additional_args=dict(spectra=['kaka'],
            input_cov=np.eye(18), input_dv=np.empty(0), integrated_model=True,
            cmb_marginalized=True, want_lensing_lklh_correction=True))
    points = np.column_stack([params[n] for n in model.parameter_names])
    theory = np.pad(prediction, ((0, 0), (2, 0)))  # no pi factor: CCL C_ell^kk
    reference = reference_function(args.reference)
    n = theory.shape[1]
    data = {'fiducial_cl_kk': np.asarray(like.clkk_fid[:n]),
            'dN1_kk': np.asarray(like.dN1_dClkk[:n, :n]),
            'dAL_dC': np.asarray(like.dNorm_dCl[:, :n, :n]),
            'fAL': np.asarray(like.norm_fid[:n])}
    for i, name in enumerate(['tt', 'ee', 'bb', 'te']):
        data['fiducial_cl_'+name] = np.asarray(like.fiducial_cls[i, :n])
        data['dN1_'+name] = np.asarray(like.dN1_dCl[i, :n, :n])
    cmb = {k: data['fiducial_cl_'+k] for k in ['tt','ee','bb','te']}
    expected_spectra = np.stack([reference(data, k, cmb['tt'], cmb['te'], cmb['ee'], cmb['bb']) for k in theory])
    expected_bins = expected_spectra @ like.bin_mat_act.T
    expected_logp = np.array([-.5*r@like.inv_C@r for r in expected_bins-like.y_obs])
    np.testing.assert_allclose(like.get_prediction(points), expected_bins, rtol=2e-10, atol=1e-18)
    np.testing.assert_allclose(like.compute_like(points), expected_logp, rtol=2e-10)

    # Also exercise nonzero, sample-dependent CMB corrections at common support.
    view = SimpleNamespace(dNorm_dCl=data['dAL_dC'], norm_fid=data['fAL'],
        clkk_fid=data['fiducial_cl_kk'], dN1_dClkk=data['dN1_kk'],
        fiducial_cls=np.stack([cmb[k] for k in ['tt','ee','bb','te']]),
        dN1_dCl=np.stack([data['dN1_'+k] for k in ['tt','ee','bb','te']]))
    varied_cmb = np.stack([view.fiducial_cls*.99, view.fiducial_cls*1.01])
    expected_varied = np.stack([reference(data, k, c[0], c[3], c[1], c[2]) for k,c in zip(theory,varied_cmb)])
    actual_varied = compute_act_lensing_auto_corrections(view, varied_cmb, theory)
    np.testing.assert_allclose(actual_varied[:, 2:], expected_varied[:, 2:], rtol=2e-10, atol=1e-18)

    # Restore the same trained artifact through the JAX API and compile inference.
    import jax
    jax.config.update('jax_enable_x64', True)
    config.set_jax_enabled(True)
    from UPanda import integrated_model_jax
    jax_model = integrated_model_jax.IntegratedModel(None, None, None, None)
    jax_model.restore(args.emulator)
    np.testing.assert_allclose(jax_model.predict(params), prediction, rtol=2e-5, atol=1e-14)
    like.emu_model_list = [jax_model]
    # The reference for this backend uses its own emulator output so emulator
    # float32 roundoff is not conflated with a likelihood normalization error.
    jax_theory = np.pad(np.asarray(jax_model.predict(params)), ((0,0),(2,0)))
    jax_reference = np.stack([reference(data, k, cmb['tt'], cmb['te'], cmb['ee'], cmb['bb']) for k in jax_theory]) @ like.bin_mat_act.T
    actual_bins = np.asarray(jax.jit(like.get_prediction)(points))
    np.testing.assert_allclose(actual_bins, jax_reference, rtol=2e-5, atol=1e-14)
    print(json.dumps(dict(emulator=args.emulator, reference=args.reference,
        output_shape=list(prediction.shape), numpy_loglike=expected_logp.tolist(),
        jax_max_bandpower_error=float(np.max(np.abs(actual_bins-jax_reference))),
        status='passed'), indent=2))


if __name__ == '__main__':
    main()
