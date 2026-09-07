# Copyright (C) 2024 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves

import numpy as np
import pytest

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    import UPanda.integrated_model as integrated_model

    HAS_TF = True
except ImportError:
    HAS_TF = False


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
class TestIntegratedModel:
    """Test IntegratedModel class for TensorFlow/Keras emulators."""

    def test_integrated_model_init(self):
        """Test IntegratedModel initialization."""
        input_dim = 3
        output_dim = 5
        inputs = Input(shape=(input_dim,))
        x = Dense(10, activation="relu")(inputs)
        outputs = Dense(output_dim)(x)
        model = Model(inputs=inputs, outputs=outputs)

        from sklearn.preprocessing import StandardScaler

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        input_scaler.fit(np.random.randn(10, input_dim))
        output_scaler.fit(np.random.randn(10, output_dim))

        parameter_names = ["omega_m", "omega_b", "h"]

        integrated = integrated_model.IntegratedModel(
            keras_model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            parameter_names=parameter_names,
        )

        assert integrated.model is not None
        assert integrated.input_scaler is not None
        assert integrated.output_scaler is not None
        assert integrated.parameter_names == parameter_names

    def test_integrated_model_predict(self):
        """Test IntegratedModel prediction."""
        input_dim = 2
        output_dim = 3
        inputs = Input(shape=(input_dim,))
        x = Dense(5, activation="relu")(inputs)
        outputs = Dense(output_dim)(x)
        model = Model(inputs=inputs, outputs=outputs)

        from sklearn.preprocessing import StandardScaler

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        X_train = np.random.randn(20, input_dim)
        y_train = np.random.randn(20, output_dim)
        input_scaler.fit(X_train)
        output_scaler.fit(y_train)

        parameter_names = ["omega_m", "omega_b"]

        integrated = integrated_model.IntegratedModel(
            keras_model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            parameter_names=parameter_names,
        )

        # Set scaler attributes that predict() method expects
        integrated.scaler_mean_in = input_scaler.mean_
        integrated.scaler_scale_in = input_scaler.scale_
        integrated.scaler_mean_out = output_scaler.mean_
        integrated.scaler_scale_out = output_scaler.scale_

        data_dict = {
            "omega_m": np.array([0.3, 0.31, 0.32]),
            "omega_b": np.array([0.05, 0.051, 0.052]),
        }

        prediction = integrated.predict(data_dict)
        assert prediction.shape[0] == 3
        assert prediction.shape[1] == output_dim

    def test_custom_activation(self):
        """Test CustomActivation layer."""
        layer = integrated_model.CustomActivation()
        layer.build((None, 10))

        # Need to provide input with correct shape
        test_input = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        output = layer(test_input)

        assert output.shape == test_input.shape
        assert layer.alpha is not None
        assert layer.beta is not None

    def test_custom_loss(self):
        """Test CustomLoss layer."""
        element_weights = np.array([1.0, 2.0, 3.0])
        loss = integrated_model.CustomLoss(element_weights=element_weights)

        y_true = tf.constant([[1.0, 2.0, 3.0]])
        y_pred = tf.constant([[1.1, 2.1, 3.1]])

        loss_value = loss(y_true, y_pred)
        assert loss_value is not None
        assert loss_value.numpy() > 0

    def test_create_model(self):
        """Test create_model function."""
        model = integrated_model.create_model(
            input_dim=5, hidden_layers=[10, 5], output_dim=3
        )

        assert isinstance(model, Model)
        assert model.input_shape == (None, 5)
        assert model.output_shape == (None, 3)

    def test_integrated_model_predict_with_pca(self):
        """Test IntegratedModel prediction with PCA."""
        input_dim = 2
        output_dim = 5  # Original output dimension
        pca_components = 3  # PCA reduces to this
        inputs = Input(shape=(input_dim,))
        # Model outputs PCA dimension
        outputs = Dense(pca_components)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
        pca = PCA(n_components=pca_components)
        pca_scaler = StandardScaler()

        X_train = np.random.randn(20, input_dim)
        y_train = np.random.randn(20, output_dim)
        input_scaler.fit(X_train)
        output_scaler.fit(y_train)
        pca.fit(y_train)
        # pca_scaler should be fit on original output dimension, not PCA dimension
        # because after PCA inverse_transform, we get original dimension
        pca_scaler.fit(y_train)  # Fit on original output dimension

        parameter_names = ["omega_m", "omega_b"]

        integrated = integrated_model.IntegratedModel(
            keras_model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            parameter_names=parameter_names,
            pca=pca,
            pca_scaler=pca_scaler,
        )

        # Set scaler attributes - need to match the dimensions
        integrated.scaler_mean_in = input_scaler.mean_
        integrated.scaler_scale_in = input_scaler.scale_
        # The flow is:
        # 1. Model outputs (1, 3) - PCA dimension
        # 2. Scale: prediction * scaler_scale_out + scaler_mean_out - applied to PCA output, so needs (3,)
        # 3. Inverse PCA transform: (1, 3) -> (1, 5) - original output dimension
        # 4. pca_scaler.inverse_transform: expects (1, 5) since we fit it on original output
        integrated.scaler_mean_out = output_scaler.mean_[
            :pca_components
        ]  # PCA dimension (3,)
        integrated.scaler_scale_out = output_scaler.scale_[
            :pca_components
        ]  # PCA dimension (3,)

        data_dict = {"omega_m": np.array([0.3]), "omega_b": np.array([0.05])}

        prediction = integrated.predict(data_dict)
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == output_dim  # After PCA inverse transform

    def test_integrated_model_predict_with_log_preprocess(self):
        """Test IntegratedModel prediction with log preprocessing."""
        input_dim = 2
        output_dim = 3
        inputs = Input(shape=(input_dim,))
        outputs = Dense(output_dim)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        from sklearn.preprocessing import StandardScaler

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        X_train = np.random.randn(20, input_dim)
        y_train = np.random.randn(20, output_dim)
        input_scaler.fit(X_train)
        output_scaler.fit(y_train)

        parameter_names = ["omega_m", "omega_b"]

        integrated = integrated_model.IntegratedModel(
            keras_model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            parameter_names=parameter_names,
            log_preprocess=True,
            offset=1.0,
        )

        integrated.scaler_mean_in = input_scaler.mean_
        integrated.scaler_scale_in = input_scaler.scale_
        integrated.scaler_mean_out = output_scaler.mean_
        integrated.scaler_scale_out = output_scaler.scale_

        data_dict = {"omega_m": np.array([0.3]), "omega_b": np.array([0.05])}

        prediction = integrated.predict(data_dict)
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == output_dim

    def test_integrated_model_predict_with_rescale_as(self):
        """Test IntegratedModel prediction with A_s rescaling."""
        parameter_names = ["omega_m", "omega_b", "ln10^{10}A_s", "tau"]
        input_dim = len(parameter_names)  # Must match number of parameters
        output_dim = 3
        inputs = Input(shape=(input_dim,))
        outputs = Dense(output_dim)(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        from sklearn.preprocessing import StandardScaler

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        X_train = np.random.randn(20, input_dim)
        y_train = np.random.randn(20, output_dim)
        input_scaler.fit(X_train)
        output_scaler.fit(y_train)

        parameter_names = ["omega_m", "omega_b", "ln10^{10}A_s", "tau"]
        # Note: input_dim must match number of parameters
        input_dim = len(parameter_names)

        integrated = integrated_model.IntegratedModel(
            keras_model=model,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            parameter_names=parameter_names,
            rescale_by_as=True,
        )

        integrated.scaler_mean_in = input_scaler.mean_
        integrated.scaler_scale_in = input_scaler.scale_
        integrated.scaler_mean_out = output_scaler.mean_
        integrated.scaler_scale_out = output_scaler.scale_

        data_dict = {
            "omega_m": np.array([0.3]),
            "omega_b": np.array([0.05]),
            "ln10^{10}A_s": np.array([3.0]),
            "tau": np.array([0.05]),
        }

        prediction = integrated.predict(data_dict)
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == output_dim
