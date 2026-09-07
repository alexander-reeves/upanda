"""DES Dovekie supernova likelihood module.

Implements the DES-SN5YR Dovekie Type Ia supernova likelihood, mathematically
identical to the Cobaya sn.desdovekie implementation (arXiv:2511.07517).

Data: 1820 SNe from DES 5YR + low-z surveys, pre-standardized distance moduli.
Covariance: STAT+SYS inverse covariance stored in NPZ upper-triangular format.
Marginalization: analytic marginalization over absolute magnitude M.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood
from ._numerics import marginal_precision


class des_dovekie(Likelihood):
    """Likelihood class for the DES Dovekie supernova dataset.

    Mathematically identical to Cobaya's sn.desdovekie. Distance moduli are
    pre-standardized (no alpha/beta nuisance parameters). Absolute magnitude
    is marginalized analytically.
    """

    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        dtype=None,
        compression_vecs=None,
        cosmology_params_fixed=None,
        additional_args=None,
        external_spectra=None,
    ):
        super().__init__(data_vector, covariance_matrix, emu_model_list)

        self.emu_model_list = emu_model_list
        self.external_spectra = external_spectra

        additional_args = additional_args or {}

        if additional_args.get("mock_data", None) is not None:
            self.mag = additional_args["mock_data"]
            self.zcmb = additional_args.get(
                "mock_zcmb", np.linspace(0.02, 1.0, len(self.mag))
            )
            self.zhel = additional_args.get("mock_zhel", self.zcmb)
            self.inv_cov = np.linalg.inv(
                additional_args.get("mock_cov", np.eye(len(self.mag)))
            )
            self.nsn = len(self.mag)
        else:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_directory, "data/des_dovekie")

            data_file = os.path.join(self.data_dir, "DES-Dovekie_HD.csv")
            self._read_data(data_file)

            npz_file = os.path.join(self.data_dir, "STAT+SYS.npz")
            self.inv_cov = self._read_inv_covmat(npz_file)

            # Apply zcmb > 0.01 mask (same as Cobaya PantheonPlus.configure)
            mask = self.zcmb > 0.01
            self._apply_mask(mask)

        self._marginalize_abs_mag()

        self.dtype = dtype
        self.cosmology_params_fixed = (
            cosmology_params_fixed if cosmology_params_fixed is not None else []
        )
        self.additional_args = additional_args

    def _read_data(self, data_file):
        """Parse DES-Dovekie_HD.csv.

        The file has comment lines starting with '#', then a VARNAMES header row
        ('VARNAMES: CID IDSURVEY zHD zHEL MU ...'), then data rows ('SN: ...').
        """
        zhd_list, zhel_list, mu_list = [], [], []

        with open(data_file) as f:
            header_cols = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("VARNAMES:"):
                    header_cols = line.split()[1:]  # drop "VARNAMES:"
                    zhd_idx = header_cols.index("zHD")
                    zhel_idx = header_cols.index("zHEL")
                    mu_idx = header_cols.index("MU")
                    continue
                if line.startswith("SN:"):
                    vals = line.split()[1:]  # drop "SN:"
                    zhd_list.append(float(vals[zhd_idx]))
                    zhel_list.append(float(vals[zhel_idx]))
                    mu_list.append(float(vals[mu_idx]))

        self.zcmb = np.array(zhd_list)
        self.zhel = np.array(zhel_list)
        self.mag = np.array(mu_list)
        self.nsn = len(self.mag)

    def _read_inv_covmat(self, filename):
        """Load pre-inverted covariance from NPZ upper-triangular format.

        Identical to Cobaya's SN._read_inv_covmat.
        """
        d = np.load(filename, allow_pickle=True)
        n = int(d[d.files[0]][0])
        inv_cov = np.zeros((n, n))
        inv_cov[np.triu_indices(n)] = d[d.files[1]]
        i_lower = np.tril_indices(n, -1)
        inv_cov[i_lower] = inv_cov.T[i_lower]
        return inv_cov

    def _apply_mask(self, mask):
        self.mag = self.mag[mask]
        self.zcmb = self.zcmb[mask]
        self.zhel = self.zhel[mask]
        self.inv_cov = marginal_precision(self.inv_cov, mask)
        self.nsn = int(mask.sum())

    def _marginalize_abs_mag(self):
        """Analytic marginalization over absolute magnitude M.

        Identical to Cobaya's PantheonPlus._marginalize_abs_mag.
        """
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.inv_cov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.inv_cov = self.inv_cov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def set_external_spectra(self, spectra):
        """Set distance moduli directly, bypassing the emulator."""
        self.external_spectra = spectra

    def get_prediction(self, p):
        """Return predicted distance moduli mu = 5*log10((1+zhel)*(1+zcmb)*D_A)."""
        if self.external_spectra is not None:
            return self.external_spectra
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )
        emu_model = self.emu_model_list[0]
        distances = emu_model.predict(params_dict)
        return 5 * log10((1 + self.zhel) * (1 + self.zcmb) * distances)

    def compute_like(self, params):
        """Compute log-likelihood: -0.5 * (mu_obs - mu_th)^T C^{-1} (mu_obs - mu_th).

        Works for both a single 1-D diff (returns scalar) and a batched 2-D diff
        with shape (n_samples, n_sn) (returns array of shape (n_samples,)).
        """
        mu_pred = self.get_prediction(params)
        diff = self.mag - mu_pred
        return utils_likelihood.gaussian_loglike(diff, self.inv_cov)
