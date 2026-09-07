"""JAX-based integrated emulator models for cosmological predictions.

This module provides JAX/Flax-based implementations of integrated emulator
models, enabling JIT compilation and automatic differentiation. It includes
custom activation functions compatible with JAX transformations (vmap, jit).

The module provides utilities for converting TensorFlow models to JAX/Flax
and for inserting zero columns in predictions (useful for masked spectra).
"""

import logging

from UPanda.module import *  # noqa

is_jax = config.get_jax_enabled()
import pickle

logger = logging.getLogger(__name__)

import jax.random as random
import tensorflow as tf
from flax import linen as nn
from flax import traverse_util
from jax.nn import sigmoid
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model


class MyNetwork(nn.Module):
    """Flax neural network module with custom activation functions.

    This module implements a multi-layer neural network with custom activation
    functions in each hidden layer, matching the TensorFlow CustomActivation
    behavior but implemented in Flax for JAX compatibility.

    Attributes:
        weights (list): List of (weight, bias) tuples for each layer.
        hyper_params (list): List of (alpha, beta) tuples for custom activations.
    """

    weights: list  # List of (weight, bias) tuples for each layer
    hyper_params: list  # List of (alpha, beta) for each custom activation layer

    @nn.compact
    def __call__(self, x):
        # Loop over layers except the last one
        for (w, b), (a, b_hyper) in zip(self.weights[:-1], self.hyper_params):
            x = CustomActivation_jax(a=a, b=b_hyper)(dot(x, w) + b)

        # Final layer (no activation)
        final_w, final_b = self.weights[-1]
        x = dot(x, final_w) + final_b
        return x


class CustomActivation_jax(nn.Module):
    """JAX-compatible custom activation function matching arXiv:2106.03846.

    This activation function uses fixed hyperparameters alpha and beta to
    create a learnable activation: (β + sigmoid(αx) * (1 - β)) * x

    This is the JAX/Flax version of CustomActivation, designed to work with
    JAX transformations like vmap and jit.

    Attributes:
        a (float): Alpha hyperparameter.
        b (float): Beta hyperparameter.
    """

    a: float  # alpha hyperparameter
    b: float  # beta hyperparameter

    @nn.compact
    def __call__(self, x):
        """Apply the custom activation function.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Transformed input tensor.
        """
        return multiply(
            add(self.b, multiply(sigmoid(multiply(self.a, x)), subtract(1.0, self.b))),
            x,
        )


class CustomActivation(Layer):
    """TensorFlow custom activation function matching arXiv:2106.03846.

    This is kept for compatibility when loading TensorFlow models.
    The JAX version (CustomActivation_jax) should be used for new JAX models.
    """

    def build(self, input_shape):
        # Trainable weight variables for alpha and beta initialized with random normal distribution
        self.alpha = self.add_weight(
            name="alpha",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        super(CustomActivation, self).build(input_shape)

    def call(self, inputs):
        return (self.beta + tf.sigmoid(self.alpha * inputs) * (1 - self.beta)) * inputs


# Custom loss function for original tf model
class CustomLoss(Layer):
    def __init__(self, element_weights=None, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.element_weights = (
            tf.convert_to_tensor(element_weights, dtype=tf.float32)
            if element_weights is not None
            else tf.constant(1.0, dtype=tf.float32)
        )

    def call(self, y_true, y_pred):
        # Compute element-wise squared errors
        squared_errors = tf.square(y_true - y_pred)

        if self.element_weights is not None:
            # Apply element-wise weights
            squared_errors = squared_errors / self.element_weights

        # Compute the mean of squared errors
        return tf.reduce_mean(squared_errors)

    def get_config(self):
        config = super(CustomLoss, self).get_config()
        config.update({"element_weights": self.element_weights.numpy()})
        return config


def insert_zero_columns(prediction, zero_columns_indices):
    """Insert zero columns at specified indices in predictions.

    This function is useful for inserting masked or invalid multipole bins
    back into power spectrum predictions, maintaining compatibility with
    data vectors that include these bins.

    Args:
        prediction (ndarray): Prediction array with shape (n_samples, n_features).
        zero_columns_indices (array-like): Indices where zero columns should
            be inserted.

    Returns:
        ndarray: Prediction array with zero columns inserted. Shape is
            (n_samples, n_features + len(zero_columns_indices)).
    """
    # Ensure prediction is a JAX array if JAX is enabled
    if is_jax:
        import jax.numpy as jnp

        if not hasattr(prediction, "at"):
            prediction = jnp.array(prediction)

    # Total number of columns in the final output
    num_columns = prediction.shape[1] + len(zero_columns_indices)

    # Initialize the final_prediction array
    final_prediction = zeros((prediction.shape[0], num_columns))

    # Ensure final_prediction is a JAX array if JAX is enabled
    if is_jax:
        import jax.numpy as jnp

        if not hasattr(final_prediction, "at"):
            final_prediction = jnp.array(final_prediction)

    # Indices in final_prediction where values from prediction will be inserted
    non_zero_indices = delete(arange(num_columns), zero_columns_indices)

    # Ensure non_zero_indices is a JAX array if JAX is enabled
    if is_jax:
        import jax.numpy as jnp

        if not hasattr(non_zero_indices, "at"):
            non_zero_indices = jnp.array(non_zero_indices)

    # Insert values from prediction into the correct positions in final_prediction
    # Use .at syntax for JAX arrays, or regular indexing for NumPy
    if is_jax and hasattr(final_prediction, "at"):
        final_prediction = final_prediction.at[:, non_zero_indices].set(prediction)
    else:
        # Fallback for NumPy arrays
        final_prediction[:, non_zero_indices] = prediction

    return final_prediction


class IntegratedModel:
    """JAX-compatible wrapper for TensorFlow-trained IntegratedModel.
    
    This class mirrors integrated_model.IntegratedModel but uses JAX for predictions,
    enabling JIT compilation and automatic differentiation.
    """
    
    def __init__(
        self,
        keras_model,
        input_scaler,
        output_scaler,
        parameter_names,
        temp_file=None,
        offset=None,
        log_preprocess=False,
        cheby_preprocess=False,
        cheby_meta=None,
        rescaling_factor=None,
        pca=None,
        pca_scaler=None,
        rescale_by_as=False,
        galaxy_bias_config=None,
        verbose=False,
    ):
        self.model = keras_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.offset = offset
        self.rescaling_factor = rescaling_factor
        self.temp_file = temp_file
        self.train_losses = []
        self.val_losses = []
        self.log_preprocess = log_preprocess
        self.cheby_preprocess = cheby_preprocess
        self.cheby_meta = cheby_meta
        self.parameter_names = parameter_names
        self.pca = pca
        self.pca_scaler = pca_scaler
        self.rescale_by_as = rescale_by_as
        
        # Galaxy bias rescaling configuration (metadata-driven)
        # Format: {
        #   'enabled': bool,
        #   'linear': [(start_idx, end_idx, bias_param_name), ...],  # cross-spectra scale as b_i
        #   'quadratic': [(start_idx, end_idx, bias_param_name), ...],  # auto-spectra scale as b_i^2
        # }
        # Set to None to disable, or provide dict with 'enabled': True and index ranges
        self.galaxy_bias_config = galaxy_bias_config
        self.verbose = verbose

    def predict(self, data_dict):
        if self.parameter_names is not None:
            required_keys = set(sorted(data_dict.keys()))
            expected_keys = set(self.parameter_names)
            if not expected_keys.issubset(required_keys):
                raise ValueError(
                    f"Input dictionary keys {required_keys} do not match expected keys {expected_keys}"
                )

        # Map parameter values to the correct positions in the input matrix
        data = stack([asarray(data_dict[key]) for key in self.parameter_names], axis=1)

            # print(data)
            # print(self.parameter_names)
            # print(self.scaler_mean_in)

        # Ensure scaler arrays have the right shape for broadcasting
        scaler_mean_in = self.scaler_mean_in.reshape(1, -1)
        scaler_scale_in = self.scaler_scale_in.reshape(1, -1)
        scaled_data = (data - scaler_mean_in) / scaler_scale_in

        # Store A_s values if we need to rescale (matches integrated_model.py)
        amplitudes = None
        if self.rescale_by_as and "ln10^{10}A_s" in self.parameter_names:
            # Find the index of the A_s parameter
            as_index = list(self.parameter_names).index("ln10^{10}A_s")
            ln_10_10_as = data[:, as_index]
            amplitudes = exp(ln_10_10_as) * 1e-10

        # Convert to TensorFlow tensor
        scaled_data = array(scaled_data, dtype=float32)
        prediction = self.jax_model.apply(self.jax_params, scaled_data)
        
        # Inverse transforms applied in reverse order of training (matches integrated_model.py)
        
        # Apply inverse PCA transform if PCA is used
        if self.pca is not None:
            scaler_scale_out = self.scaler_scale_out.reshape(1, -1)
            scaler_mean_out = self.scaler_mean_out.reshape(1, -1)
            prediction = prediction * scaler_scale_out + scaler_mean_out

            # Invert the PCA using the mean and components
            prediction = dot(prediction, self.pca_components) + self.pca_mean
            prediction = prediction * self.pca_scaler_scale + self.pca_scaler_mean
        elif self.cheby_preprocess and self.cheby_meta is not None:
            # Inverse Chebyshev transform
            scaler_scale_out = self.scaler_scale_out.reshape(1, -1)
            scaler_mean_out = self.scaler_mean_out.reshape(1, -1)
            prediction = prediction * scaler_scale_out + scaler_mean_out
            
            # Undo min-max scaling
            coeffs = prediction * self.cheby_meta['c_denom'] + self.cheby_meta['c_min']
            
            # Reconstruct: coeffs -> standardized D_ell -> D_ell -> C_ell
            y_std = dot(coeffs, self.cheby_meta['V'].T)
            y_D = y_std * self.cheby_meta['D_std'] + self.cheby_meta['D_mean']
            prediction = y_D / self.cheby_meta['ell_factor']
        else:
            scaler_scale_out = self.scaler_scale_out.reshape(1, -1)
            scaler_mean_out = self.scaler_mean_out.reshape(1, -1)
            prediction = prediction * scaler_scale_out + scaler_mean_out

        # Inverse offset and log
        if self.log_preprocess:
            prediction = exp(prediction) + 2 * self.offset

        # Rescale by A_s if needed
        if self.rescale_by_as and amplitudes is not None:
            prediction = prediction * amplitudes[:, newaxis]

        # Apply galaxy bias rescaling if configured
        prediction = self._apply_galaxy_bias_rescaling(prediction, data_dict)

        return prediction

    def _apply_galaxy_bias_rescaling(self, prediction, data_dict):
        """Apply galaxy bias rescaling to prediction based on configuration.
        
        Compatible with JAX vmap and jit.
        
        Auto-spectra (dd_ii, gg_ii) scale as b_i^2 (quadratic).
        Cross-spectra (kad_i, td_i, kg_i, tg_i) scale as b_i (linear).
        
        Args:
            prediction: JAX array of shape (n_samples, n_output).
            data_dict: Dictionary with bias parameter values.
        
        Returns:
            Rescaled prediction array.
        """
        if self.galaxy_bias_config is None:
            return prediction
        if not self.galaxy_bias_config.get('enabled', False):
            return prediction
        
        # Quadratic rescaling for auto-spectra
        for start_idx, end_idx, bias_name in self.galaxy_bias_config.get('quadratic', []):
            if bias_name in data_dict:
                bias_vals = data_dict[bias_name]
                if not hasattr(bias_vals, 'at'):
                    bias_vals = array(bias_vals)
                prediction = prediction.at[:, start_idx:end_idx].multiply(
                    (bias_vals ** 2)[:, newaxis]
                )
        
        # Linear rescaling for cross-spectra
        for start_idx, end_idx, bias_name in self.galaxy_bias_config.get('linear', []):
            if bias_name in data_dict:
                bias_vals = data_dict[bias_name]
                if not hasattr(bias_vals, 'at'):
                    bias_vals = array(bias_vals)
                prediction = prediction.at[:, start_idx:end_idx].multiply(
                    bias_vals[:, newaxis]
                )
        
        return prediction

    def save(self, filename):
        """Save IntegratedModel to disk.
        
        Saves the model in a format compatible with integrated_model.IntegratedModel.
        Note: This saves the original TensorFlow model, not the JAX conversion.
        
        Args:
            filename (str): Base filename (without suffix) where model will
                be saved. Files will be saved as:
                - {filename}_model.h5 (Keras model)
                - {filename}.pkl (other attributes)
        """
        # Save the Keras model in the .h5 format
        self.model.save(filename + "_model.h5")
        
        # attributes list must match integrated_model.py exactly (12 attributes)
        attributes = [
            self.input_scaler,
            self.output_scaler,
            self.offset,
            self.log_preprocess,
            self.pca,
            self.pca_scaler,
            self.parameter_names,
            self.rescaling_factor,
            self.rescale_by_as,
            self.cheby_preprocess,
            self.cheby_meta,
            self.galaxy_bias_config,
        ]
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(attributes, f)
        
        logger.debug("Model saved to %s_model.h5", filename)

    def restore(self, filename):
        """Load pre-saved IntegratedModel from disk.

        Transform tf weights to jax weights dynamically and initialize the jax model

        Parameters:
            filename (str): filename tag (without suffix) where model was saved
        """
        # Load the Keras model from the .h5 format
        tf_model = load_model(
            filename + "_model.h5",
            custom_objects={
                "CustomActivation": CustomActivation,
                "CustomLoss": CustomLoss,
            },
            compile=False,
        )

        self.model = tf_model
        tf_params = []
        for layer in tf_model.layers:
            weights = layer.get_weights()
            if weights:  # Check if the layer has parameters
                tf_params.append(weights)

        # load in the weights and hyperparameters
        tf_weights = []
        tf_hyperparams = []
        for i, layer in enumerate(tf_params):
            if i % 2 == 0:
                tf_weights.append(tf_params[i])
            else:
                tf_hyperparams.append(tf_params[i])

        # Load the remaining attributes
        with open(filename + ".pkl", "rb") as f:
            attributes = pickle.load(f)

            from UPanda._model_io import decode_attributes
            for key, value in decode_attributes(attributes).items():
                setattr(self, key, value)

        self.scaler_mean_in = self.input_scaler.mean_
        self.scaler_scale_in = self.input_scaler.scale_

        logger.debug("input scaler mean shape: %s", self.scaler_mean_in.shape)

        self.scaler_scale_out = self.output_scaler.scale_
        self.scaler_mean_out = self.output_scaler.mean_

        logger.debug("output scaler mean shape: %s", self.scaler_mean_out.shape)

        try:
            self.pca_scaler_mean = self.pca_scaler.mean_
            self.pca_scaler_scale = self.pca_scaler.scale_
        except AttributeError:
            logger.debug("No PCA scaler found in model")

        try:
            self.pca_mean = self.pca.mean_
            self.pca_components = self.pca.components_
        except AttributeError:
            logger.debug("No PCA found in model")

        self.jax_model = MyNetwork(hyper_params=tf_hyperparams, weights=tf_weights)

        # Initialize the model with dummy data
        input_shape = tf_weights[0][0].shape[0]  # the input shape of the first layer
        rng = random.PRNGKey(0)
        dummy_input = ones((1, input_shape))
        params = self.jax_model.init(rng, dummy_input)

        # Flatten the parameters dictionary
        flattened_params = traverse_util.flatten_dict(params)

        # Replace the initialized parameters with your pre-trained weights and hyperparameters
        for i, layer_path in enumerate(flattened_params.keys()):
            if (
                "CustomActivation" in layer_path
            ):  # Assuming this is the name of your custom activation layers
                # Replace hyperparameters
                flattened_params[layer_path] = tf_hyperparams[i]
            else:
                # Replace weights and biases
                flattened_params[layer_path] = tf_weights[i]

        # Unflatten the parameters dictionary
        self.jax_params = traverse_util.unflatten_dict(flattened_params)

        logger.debug("IntegratedModel (JAX) restore successful")
