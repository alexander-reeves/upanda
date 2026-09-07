"""Union3 and Union3.1 supernova likelihood module.

Implements the Union3 binned-distance likelihood (Rubin et al. 2023,
arXiv:2311.12098), mathematically identical to Cobaya's ``sn.union3``.

Data: 22 binned distance moduli ``mb`` at redshifts 0.05-2.26 (``lcparam_full.txt``,
columns zcmb zhel mb; zhel == zcmb by construction), with the full 22x22 magnitude
covariance (``mag_covmat.txt``: a leading line with n, then n*n entries).
The absolute-magnitude offset is marginalised analytically by projecting the
constant mode out of the inverse covariance, exactly as in Cobaya's
``PantheonPlus._marginalize_abs_mag`` (``use_abs_mag: False``), so any constant
offset between the binned ``mb`` and the predicted 5 log10 D_L drops out.

Added 2026-09-04 as a third SN sample alongside pantheon_plus and des_dovekie.
"""

import os

import numpy as np

from UPanda.likelihoods import utils_likelihood
from UPanda.module import *

from .likelihood import Likelihood


class union3(Likelihood):
    """Likelihood class for the Union3 binned supernova distances.

    Same interface as :class:`des_dovekie`: ``get_prediction`` returns
    5 log10[(1+zhel)(1+zcmb) D_A] and ``compute_like`` the chi-squared with the
    M-marginalised inverse covariance. No alpha/beta nuisance parameters.
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
                "mock_zcmb", np.linspace(0.05, 2.26, len(self.mag))
            )
            self.zhel = additional_args.get("mock_zhel", self.zcmb)
            self.inv_cov = np.linalg.inv(
                additional_args.get("mock_cov", np.eye(len(self.mag)))
            )
            self.nsn = len(self.mag)
        else:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.union_version = str(additional_args.get("union_version", "3"))
            data_directories = {"3": "union3", "31": "union31"}
            if self.union_version not in data_directories:
                raise ValueError(
                    "union_version must be '3' (UNITY1.5) or '31' (UNITY1.8)"
                )
            self.data_dir = os.path.join(
                current_directory, "data", data_directories[self.union_version]
            )
            self._read_data(os.path.join(self.data_dir, "lcparam_full.txt"))
            self.inv_cov = np.linalg.inv(
                self._read_covmat(os.path.join(self.data_dir, "mag_covmat.txt"))
            )

        self._marginalize_abs_mag()

        self.dtype = dtype
        self.cosmology_params_fixed = (
            cosmology_params_fixed if cosmology_params_fixed is not None else []
        )
        self.additional_args = additional_args

    def _read_data(self, data_file):
        """Parse lcparam_full.txt (Cobaya/JLA-style: '#name zcmb zhel dz mb ...')."""
        with open(data_file) as f:
            header = None
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    columns = line.lstrip("#").split()
                    if {"zcmb", "zhel", "mb"}.issubset(columns):
                        header = columns
                    continue
                rows.append(line.split())
        if header is None:
            raise ValueError(f"no header line in {data_file}")
        iz, izh, imb = header.index("zcmb"), header.index("zhel"), header.index("mb")
        self.zcmb = np.array([float(r[iz]) for r in rows])
        self.zhel = np.array([float(r[izh]) for r in rows])
        self.mag = np.array([float(r[imb]) for r in rows])
        self.nsn = len(self.mag)

    @staticmethod
    def _read_covmat(filename):
        """Cobaya SN covmat format: first entry n, then n*n values (row-major)."""
        vals = np.loadtxt(filename).ravel()
        if vals.size == 0 or not np.isfinite(vals[0]) or vals[0] < 1 or vals[0] != int(vals[0]):
            raise ValueError(f"{filename}: invalid covariance dimension")
        n = int(vals[0])
        if vals.size != 1 + n * n or not np.isfinite(vals).all():
            raise ValueError(f"{filename}: expected {n*n} finite covariance entries")
        cov = vals[1:].reshape(n, n)
        if not np.allclose(cov, cov.T, rtol=0, atol=1e-12 * np.abs(cov).max()):
            raise ValueError(f"{filename}: covariance is not symmetric")
        return cov

    def _marginalize_abs_mag(self):
        """Analytic marginalisation over the absolute magnitude (constant mode)."""
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.inv_cov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.inv_cov = self.inv_cov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def set_external_spectra(self, spectra):
        """Set distance moduli directly, bypassing the emulator."""
        self.external_spectra = spectra

    def get_prediction(self, p):
        """Predicted distance moduli 5 log10[(1+zhel)(1+zcmb) D_A] (offset irrelevant)."""
        if self.external_spectra is not None:
            return self.external_spectra
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )
        emu_model = self.emu_model_list[0]
        distances = emu_model.predict(params_dict)
        return 5 * log10((1 + self.zhel) * (1 + self.zcmb) * distances)

    def compute_like(self, params):
        """-0.5 * (mb - mu_th)^T Cinv_marg (mb - mu_th); batched over rows if 2-D."""
        mu_pred = self.get_prediction(params)
        diff = self.mag - mu_pred
        return utils_likelihood.gaussian_loglike(diff, self.inv_cov)
