"""Small persisted artifacts exercise legacy metadata and both restore APIs."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from UPanda import config
from UPanda._model_io import decode_attributes


def metadata(n):
    scaler = SimpleNamespace(mean_=np.array([0., 0.]), scale_=np.array([1., 1.]))
    base = [scaler, scaler, None, False, None]
    if n >= 8:
        base += [None]
    base += [['a', 'b'], None]
    if n >= 9:
        base += [False]
    if n == 10:
        base += [None]
    if n >= 11:
        base += [False, None]
    if n == 12:
        base += [None]
    return base


@pytest.mark.parametrize('n', range(7, 13))
def test_metadata_layout_defaults(n):
    actual = decode_attributes(metadata(n))
    assert actual['parameter_names'] == ['a', 'b']
    assert actual['pca_scaler'] is None
    assert actual['galaxy_bias_config'] is None
    assert actual['cheby_preprocess'] is False


def test_metadata_unknown_layout_fails():
    with pytest.raises(ValueError, match='Unknown model format'):
        decode_attributes([None] * 6)


@pytest.mark.parametrize('n', range(7, 13))
@pytest.mark.parametrize('jax_backend', [False, True])
def test_restore_predict_and_resave(n, jax_backend, tmp_path):
    import tensorflow as tf
    from UPanda import integrated_model, integrated_model_jax
    old = config.get_jax_enabled()
    config.set_jax_enabled(jax_backend)
    try:
        inputs = tf.keras.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(2)(inputs)
        model = tf.keras.Model(inputs, outputs)
        weight = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        bias = np.array([.5, -.5], dtype=np.float32)
        model.layers[-1].set_weights([weight, bias])
        stem = str(tmp_path / 'legacy')
        model.save(stem + '_model.h5')
        with open(stem + '.pkl', 'wb') as f:
            pickle.dump(metadata(n), f)
        module = integrated_model_jax if jax_backend else integrated_model
        restored = module.IntegratedModel(None, None, None, None)
        restored.restore(stem)
        params = {'b': np.array([2., 4.]), 'a': np.array([1., 3.])}
        expected = np.array([[1., 2.], [3., 4.]]) @ weight + bias
        np.testing.assert_allclose(restored.predict(params), expected, rtol=1e-6)
        if jax_backend:
            import jax
            np.testing.assert_allclose(jax.jit(restored.predict)(params), expected, rtol=1e-6)
        restored.save(str(tmp_path / 'again'))
        with open(tmp_path / 'again.pkl', 'rb') as f:
            assert len(pickle.load(f)) == 12
    finally:
        config.set_jax_enabled(old)


@pytest.mark.parametrize('transform', ['plain', 'pca', 'cheby', 'log', 'amplitude', 'bias'])
def test_tensorflow_gradient_matches_finite_difference(transform):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from UPanda.integrated_model import IntegratedModel
    from sklearn.decomposition import PCA
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    keras_model = tf.keras.Model(inputs, outputs)
    keras_model.layers[-1].set_weights([np.array([[.1, .2], [.3, .4]], dtype='f4'), np.array([.2, -.1], dtype='f4')])
    ins = StandardScaler().fit(np.array([[0., 0.], [2., 4.]]))
    outs = StandardScaler().fit(np.array([[0., 0.], [4., 6.]]))
    names = ['ln10^{10}A_s', 'b']
    options = {}
    if transform == 'pca':
        options = dict(pca=PCA(2).fit(np.array([[1., 0., 1.], [0., 2., 0.], [2., 1., 3.]])),
                       pca_scaler=StandardScaler().fit(np.array([[0., 0., 0.], [2., 4., 6.]])))
    elif transform == 'cheby':
        options = dict(cheby_preprocess=True, cheby_meta=dict(c_denom=np.array([1., 2.]), c_min=np.array([.1, .2]), V=np.array([[1., 0.], [.5, .5], [0., 1.]]), D_std=np.ones(3)*2, D_mean=np.ones(3), ell_factor=np.array([2., 3., 4.])))
    elif transform == 'log':
        options = dict(log_preprocess=True, offset=.1)
    elif transform == 'amplitude':
        options = dict(rescale_by_as=True)
    elif transform == 'bias':
        options = dict(galaxy_bias_config=dict(enabled=True, quadratic=[(0, 1, 'b')], linear=[(1, 2, 'b')]))
    model = IntegratedModel(keras_model, ins, outs, names, **options)
    model.scaler_mean_in, model.scaler_scale_in = ins.mean_, ins.scale_
    model.scaler_mean_out, model.scaler_scale_out = outs.mean_, outs.scale_
    data = {names[0]: np.array([.5, 1.5]), names[1]: np.array([1., 2.])}
    actual = model.compute_gradient(data)
    reference = []
    for name in names:
        plus = {k: v.copy() for k, v in data.items()}
        minus = {k: v.copy() for k, v in data.items()}
        plus[name] += .002
        minus[name] -= .002
        reference.append((model.predict(plus) - model.predict(minus)) / .004)
    np.testing.assert_allclose(actual, np.stack(reference, axis=-1), rtol=2e-3, atol=1e-12)
