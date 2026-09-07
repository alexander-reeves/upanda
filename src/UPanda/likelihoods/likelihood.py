"""Base likelihood class for cosmological data analysis.

This module provides the base Likelihood class that serves as a parent
class for all specific likelihood implementations (CMB, LSS, BAO, etc.).
Subclasses must override ``get_prediction()`` and ``compute_like()`` to
implement the specific calculation for their experiment.
"""


class Likelihood:
    """Base class for cosmological likelihood calculations.

    This class provides a framework for computing likelihoods from
    observational data using emulator models. Subclasses should implement
    the get_prediction() method to compute theoretical predictions from
    cosmological parameters.

    Attributes:
        data_vector (ndarray): Observed data vector
        covariance_matrix (ndarray): Covariance matrix for the data
        emu_model_list (list): List of emulator models for predictions
        compression_vecs (ndarray, optional): Compression vectors if data
            is compressed. Defaults to None.
    """

    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        compression_vecs=None,
    ):
        """Initialize the likelihood with data and models.

        Args:
            data_vector (ndarray, optional): Observed data vector.
                Defaults to None.
            covariance_matrix (ndarray, optional): Covariance matrix for
                the data. Defaults to None.
            emu_model_list (list, optional): List of emulator models that
                can predict data from parameters. Defaults to None.
            compression_vecs (ndarray, optional): Compression vectors if
                data compression is used. Defaults to None.
        """
        self.data_vector = data_vector
        self.covariance_matrix = covariance_matrix
        self.emu_model_list = emu_model_list
        self.compression_vecs = compression_vecs

    def get_prediction(self, params):
        """Get theoretical predictions from emulator models.

        This method should be overridden by subclasses to implement
        specific prediction logic. The base implementation calls predict()
        on each emulator model in emu_model_list.

        Args:
            params (ndarray): Cosmological parameters. Shape should be
                (n_samples, n_params) or compatible with emulator input.

        Returns:
            list: List of predictions from each emulator model in
                emu_model_list. Returns empty list if emu_model_list is None
                or empty.
        """
        if self.emu_model_list is None:
            return []
        return [model.predict(params) for model in self.emu_model_list]

    def compute_like(self, params):
        """Compute the log-likelihood for given parameters.

        Subclasses must override this method to implement the specific
        likelihood calculation for their experiment.

        Args:
            params (ndarray): Cosmological parameters. Shape should be
                ``(n_samples, n_params)`` or compatible with emulator input.

        Returns:
            ndarray: Log-likelihood values, shape ``(n_samples,)``.

        Raises:
            NotImplementedError: Always — subclasses must override this method.

        Example:
            A minimal Gaussian likelihood implementation::

                def compute_like(self, params):
                    pred = self.get_prediction(params)
                    diff = self.data_vector - pred
                    chi2 = np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff)
                    return -0.5 * chi2
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_like(). "
            "See any likelihood subclass (e.g. desi_bao_dr2) for a reference."
        )
