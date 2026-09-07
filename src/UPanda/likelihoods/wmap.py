"""
:Synopsis: A python implementation of the WMAP likelihood. See Dunkley et al.
           (2008) for a full description.
:Authors: Hidde Jense, Umberto Natale.

Refactored by Alexander Reeves February 2024
"""

from UPanda.module import *

is_jax = config.get_jax_enabled()

import os
import time

import numpy as np
from jax.scipy.special import logsumexp
from scipy.special import logsumexp as numpy_logsumexp

# from scipy import linalg
from astropy.io import fits

from UPanda.likelihoods import utils_likelihood

from .likelihood import Likelihood


class wmap(Likelihood):
    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        dtype=None,
        compression_vecs=None,
        cosmology_params_fixed=None,
        additional_args=None,
    ):
        self.emu_model_list = emu_model_list
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed if cosmology_params_fixed is not None else []
        self.additional_args = additional_args if additional_args is not None else {}

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_folder = os.path.join(current_directory, "data/wmap")

        ## set-up taken from .yaml file at https://github.com/HTJense/pyWMAP/blob/main/wmaplike/WMAPLike.yaml

        # Some preset filenames
        self.gibbs_sigma_filename = "lowlT/gibbs/sigmaEllsHkeChu_test16_ilc_9yr_5deg_r5_2uK_corrected_kq85y9_June_r5_all.fits"
        self.gibbs_cl_filename = "lowlT/gibbs/test_cls.dat"

        self.tt_filename = "highl/wmap_likelihood_inputs_tt.p4v6.wmap9.kq85.cinv_v3.dat"
        self.te_filename = "highl/wmap_likelihood_inputs_te.p5_final.dat"
        self.tb_filename = "highl/wmap_likelihood_inputs_tb.p5_final.dat"
        self.ttoff_filename = (
            "highl/wmap_likelihood_inputs_tt_offdiag.p4v4.wmap9.kq85.cinv_v3.dat"
        )
        self.teoff_filename = "highl/wmap_likelihood_inputs_te_offdiag.p5_final.dat"

        self.sz_filename = "sz_spectra/wmap_sz_spectrum_61GHz_v5.txt"

        self.ifn_ptsrc_mode = "highl/wmap_likelihood_inputs_ptsrc.p5_final.dat"
        self.ifn_beam_modes = "highl/top_ten_modes.beam_covariance_VW_combined.dat"
        self.ifn_fiducial_cltt = "test_cls_v4.dat"

        # Likelihood settings
        self.ttmin = 2
        self.ttmax = 1200
        self.temin = 2
        self.temax = 800

        self.lowl_max = 32

        # Whether to use the TT gibbs/pixel likelihood.
        self.use_lowl_TT = True
        # Whether to use the TE EE BB pixel likelihood.
        self.use_lowl_pol = False
        # Whether to use the TB EB low l likelihood.
        self.use_lowl_TBEB = False
        # Whether to use the high-l MASTER TT likelihood.
        self.use_highl_TT = True
        # Whether to use the high-l MASTER TE likelihood.
        self.use_highl_TE = True
        # Whether to use the high-l MASTER TB likelihood.
        self.use_highl_TB = False
        # Whether to include beam/point source errors.
        self.use_highl_TT_beam_ptsrc = False

        # Whether or not to use the gibbs sampling method for lowl_TT component.
        self.use_gibbs = True

        # Settings for the gibbs sampling component.
        self.gibbs_first_iteration = 10
        self.gibbs_last_iteration = 120000
        self.gibbs_skip = 2
        self.gibbs_ell_max = 32

        # Settings for the beam/ptsrc correction component.
        self.beam_diagonal_sigma = True
        self.beam_gaussian_likelihood = True
        self.beam_fixed_fiducial_spectrum = False
        self.beam_include_beam_modes = True
        self.beam_include_ptsrc_mode = True
        self.n_beam_modes = 9
        self.ptsrc_err = 0.1

        # Settings for the TE/EE/BB pixel likelihood
        self.TEEEBB_filename_0 = "masked_ee_ninvplninv_qu_r3_corrected_9yr.KaQV.fits"
        self.TEEEBB_filename_1 = "masked_bb_ninvplninv_qu_r3_corrected_9yr.KaQV.fits"
        self.TEEEBB_filename_2 = "masked_ninv_qu_r3_corrected_9yr.KaQV.fits"
        self.TEEEBB_filename_3 = "wt_r3_9yr.KaQV.map_q"
        self.TEEEBB_filename_4 = "wt_r3_9yr.KaQV.map_u"
        self.TEEEBB_filename_5 = "lowlP/alm_tt_fs_r9_ilc_nopixwin_9yr.dat"
        self.TEEEBB_filename_6 = "masked_ninvy_e_qu_r3_corrected_9yr.KaQV.fits"
        self.TEEEBB_filename_9 = "healpix_data/pixel_window_n0008.txt"
        self.TEEEBB_maskfile = "lowlP/mask_r3_p06_jarosik.fits"

        # Whether or not to use the SZ template
        self.use_sz = True
        self.sz_file = os.path.join(self.data_folder, self.sz_filename)

        if self.use_sz:
            self.setup_for_sz()

        self.tt_lmax = 1200
        self.te_lmax = 800

        self.TEEEBB_pixlike_logdet_offset = 16078.083180
        self.TE_logdet_offset = 3584.277805
        self.TB_logdet_offset = 3598.152208

        self.use_cl = set()

        self.temin = 24  # use TE data from 24 to match ACT set-up

        if self.use_lowl_TT:
            """Initialize the low l TT likelihood."""
            self.use_cl.add("tt")

            if self.use_gibbs:
                self.setup_for_tt_gibbs()
            else:
                self.setup_for_tt_exact()

            self.tt_highl_start = self.lowl_max + 1
        else:
            self.tt_highl_start = self.ttmin

        if self.use_lowl_pol:
            self.use_cl.add("te")
            self.use_cl.add("ee")
            self.use_cl.add("bb")

            self.te_highl_start = 24

            if self.use_lowl_TBEB:
                self.use_cl.add("tb")
                self.use_cl.add("eb")

                self.setup_for_lowl_te_tb_ee_bb_eb()
            else:
                self.setup_for_lowl_te_ee_bb()
        else:
            self.te_highl_start = self.temin

        if self.use_highl_TT or self.use_highl_TE:
            self.use_cl.add("tt")
            self.use_cl.add("te")

            self.setup_for_tt_te_covariance()

        if self.use_highl_TT_beam_ptsrc:
            self.use_cl.add("tt")

            self.setup_for_tt_beam_ptsrc(2, self.tt_lmax)

        if self.use_highl_TB:
            self.use_cl.add("tt")
            self.use_cl.add("tb")
            self.use_cl.add("bb")

            self.setup_for_tt_te_covariance()

        self.additional_args = additional_args if additional_args is not None else {}
        self.apply_wmap_temperature_rescaling = self.additional_args.get(
            "apply_wmap_temperature_rescaling", False
        )
        self.apply_wmap_polarization_rescaling = self.additional_args.get(
            "apply_wmap_polarization_rescaling", False
        )

    def setup_for_sz(self) -> None:

        # if not os.path.exists(os.path.dirname(self.sz_file)):
        #     os.makedirs(os.path.dirname(self.sz_file))

        # import requests

        # url = "https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/sz_spectra/wmap_sz_spectrum_61GHz_v5.txt"  # noqa E501

        # req = requests.get(url, stream=True)
        # if req.ok:
        #     print("Downloading...")

        #     with open(self.sz_file, "wb") as fp:
        #         for chunk in req.iter_content(chunk_size=1024 * 8):
        #             if chunk:
        #                 fp.write(chunk)
        #                 fp.flush()
        #                 os.fsync(fp.fileno())
        # else:
        #     raise IOError(f"Failed to download SZ spectrum file: Error \
        #                     code {req.status_code}: {req.text}")

        sz_ell, sz_spec = np.loadtxt(self.sz_file, unpack=True, dtype=float)
        sz_ell = sz_ell.astype(int)

        # Making sure it's zero-indexed.
        self.sz_ell = arange(sz_ell.max() + 1)
        self.sz_spec = zeros_like(self.sz_ell, dtype=float)

        if is_jax:
            self.sz_spec = self.sz_spec.at[sz_ell].set(sz_spec)

        else:
            self.sz_spec[sz_ell] = sz_spec

    def setup_for_tt_gibbs(self) -> None:
        gibbs_filename = os.path.normpath(
            os.path.join(self.data_folder, self.gibbs_sigma_filename)
        )

        gibbs_file = fits.open(gibbs_filename)
        gibbs_chain = gibbs_file[0]

        lmax = gibbs_chain.header["LMAX"]
        numsamples = gibbs_chain.header["NUMSAMP"]
        numchains = gibbs_chain.header["NUMCHAIN"]
        numspec = gibbs_chain.header["NUMSPEC"]  # noqa F841

        data = gibbs_chain.data.T
        sigmas = array(data[:lmax, 0, :numchains, :numsamples])

        gibbs_file.close()

        cl_filename = os.path.normpath(
            os.path.join(self.data_folder, self.gibbs_cl_filename)
        )

        ls, cltt_fiducial = np.loadtxt(cl_filename, usecols=(0, 1), unpack=True)
        ls = ls.astype(int)
        self.cltt_fiducial = zeros((ls.max() + 1,))

        if is_jax:
            self.cltt_fiducial = self.cltt_fiducial.at[ls].set(cltt_fiducial)

        else:
            self.cltt_fiducial[ls] = cltt_fiducial

        self.lmin = self.ttmin
        self.lmax = self.gibbs_ell_max

        self.sigmas = sigmas[
            :,
            :numchains,
            self.gibbs_first_iteration
            - 1 : self.gibbs_last_iteration : self.gibbs_skip,
        ]
        self.numchain = self.sigmas.shape[1]
        self.numsamples = self.sigmas.shape[2]

        self.first_eval = True
        self.offset = -1.6375e30

        self.compute_br_estimator(self.cltt_fiducial)

        print("Initialized TT gibbs.")

    def setup_for_tt_beam_ptsrc(self, lmin: int, lmax: int) -> None:
        self.bptsrc_lmin = lmin
        self.bptsrc_lmax = lmax

        self.nmodes = 0
        self.ptsrc_mode_index = 0

        if self.beam_include_beam_modes:
            self.nmodes += self.n_beam_modes

        if self.beam_include_ptsrc_mode:
            self.ptsrc_mode_index = self.nmodes
            self.nmodes += 1

        if self.nmodes == 0:
            return

        self.beam_mode = zeros((lmax + 1, self.n_beam_modes))
        self.fiducial_cltt = zeros((lmax + 1,))
        self.ptsrc_mode = zeros((lmax + 1,))

        if self.beam_include_beam_modes:
            ifn = os.path.normpath(os.path.join(self.data_folder, self.ifn_beam_modes))
            i, ls, x = np.loadtxt(ifn, usecols=(0, 1, 2), unpack=True)
            i = i.astype(int)
            ls = ls.astype(int)

            ls = ls[i <= self.n_beam_modes]
            x = x[i <= self.n_beam_modes]
            i = i[i <= self.n_beam_modes] - 1

            self.beam_mode[ls, i] = x

        if self.beam_fixed_fiducial_spectrum:
            ifn = os.path.normpath(
                os.path.join(self.data_folder, self.ifn_fiducial_cltt)
            )
            ls, x = np.loadtxt(ifn, usecols=(0, 1), unpack=True)
            ls = ls.astype(int)

            self.fiducial_cltt[ls] = x

        if self.beam_include_ptsrc_mode:
            ifn = os.path.normpath(os.path.join(self.data_folder, self.ifn_ptsrc_mode))
            ls, x = np.loadtxt(ifn, usecols=(0, 1), unpack=True)
            ls = ls.astype(int)

            self.ptsrc_mode[ls] = x

        print("Initialized TT beam/ptsrc correction.")

    def setup_for_tt_exact(self) -> None:
        raise NotImplementedError(
            "TT Pixel likelihood is not yet \
                                   implemented. Set .use_gibbs = True. or \
                                   disable the lowl TT likelihood."
        )

    def setup_for_tt_te_covariance(self) -> None:
        """
        Get TT/TE/TB diagonals
        Note that all the files are 2-indexed, but to make it easier on
        python coding (and to maintain the sanity of whomever needs to
        write/read this code), the datasets are re-ordered to be 0-indexed.
        """
        tt_filename = os.path.normpath(os.path.join(self.data_folder, self.tt_filename))
        te_filename = os.path.normpath(os.path.join(self.data_folder, self.te_filename))
        tb_filename = os.path.normpath(os.path.join(self.data_folder, self.tb_filename))

        ttoff_filename = os.path.normpath(
            os.path.join(self.data_folder, self.ttoff_filename)
        )
        teoff_filename = os.path.normpath(
            os.path.join(self.data_folder, self.teoff_filename)
        )

        ls, cltt_dat, ntt, fskytt = np.loadtxt(tt_filename, unpack=True, dtype=float)
        ls = ls.astype(int)
        self.cltt_dat = zeros((ls.max() + 1))
        self.ntt = zeros((ls.max() + 1))
        self.fskytt = zeros((ls.max() + 1))

        if is_jax:
            self.cltt_dat = self.cltt_dat.at[ls].set(cltt_dat)
            self.ntt = self.ntt.at[ls].set(ntt)
            self.fskytt = self.fskytt.at[ls].set(fskytt)

        else:
            self.cltt_dat[ls] = cltt_dat
            self.ntt[ls] = ntt
            self.fskytt[ls] = fskytt

        ls, clte_dat, ntt_te, nee_te, fskyte = np.loadtxt(
            te_filename, usecols=(0, 1, 3, 4, 5), unpack=True, dtype=float
        )
        ls = ls.astype(int)
        self.clte_dat = zeros((ls.max() + 1))
        self.ntt_te = zeros((ls.max() + 1))
        self.nee_te = zeros((ls.max() + 1))
        self.fskyte = zeros((ls.max() + 1))

        if is_jax:
            self.clte_dat = self.clte_dat.at[ls].set(clte_dat)
            self.ntt_te = self.ntt_te.at[ls].set(ntt_te)
            self.nee_te = self.nee_te.at[ls].set(nee_te)
            self.fskyte = self.fskyte.at[ls].set(fskyte)

        else:
            self.clte_dat[ls] = clte_dat
            self.ntt_te[ls] = ntt_te
            self.nee_te[ls] = nee_te
            self.fskyte[ls] = fskyte

        ls, cltb_dat, ntt_tb, nbb_tb, fskytb = np.loadtxt(
            tb_filename, usecols=(0, 1, 3, 4, 5), unpack=True, dtype=float
        )
        ls = ls.astype(int)
        self.cltb_dat = zeros((ls.max() + 1))
        self.ntt_tb = zeros((ls.max() + 1))
        self.nbb_tb = zeros((ls.max() + 1))
        self.fskytb = zeros((ls.max() + 1))

        if is_jax:
            self.cltb_dat = self.cltb_dat.at[ls].set(cltb_dat)
            self.ntt_tb = self.ntt_tb.at[ls].set(ntt_tb)
            self.nbb_tb = self.nbb_tb.at[ls].set(nbb_tb)
            self.fskytb = self.fskytb.at[ls].set(fskytb)

        else:
            self.cltb_dat[ls] = cltb_dat
            self.ntt_tb[ls] = ntt_tb
            self.nbb_tb[ls] = nbb_tb
            self.fskytb[ls] = fskytb

        self.r_off_tttt = zeros((self.tt_lmax + 1, self.tt_lmax + 1))
        self.epsilon = zeros_like(self.r_off_tttt)
        self.r_off_tete = zeros((self.te_lmax + 1, self.te_lmax + 1))

        """ Get TT off-diagonal """
        i, j, epsilon, r_off_tttt = np.loadtxt(
            ttoff_filename, usecols=(0, 1, 2, 3), unpack=True
        )
        i = i.astype(int)
        j = j.astype(int)

        if is_jax:
            self.r_off_tttt = self.r_off_tttt.at[i, j].set(r_off_tttt)
            self.r_off_tttt = self.r_off_tttt.at[j, i].set(r_off_tttt)
            self.epsilon = self.epsilon.at[i, j].set(epsilon)
            self.epsilon = self.epsilon.at[j, i].set(epsilon)

        else:
            self.r_off_tttt[i, j] = r_off_tttt
            self.r_off_tttt[j, i] = r_off_tttt
            self.epsilon[i, j] = epsilon
            self.epsilon[j, i] = epsilon

        """ Get TE off-diagonal """
        i, j, r_off_tete = np.loadtxt(teoff_filename, usecols=(0, 1, 2), unpack=True)
        i = i.astype(int)
        j = j.astype(int)

        if is_jax:
            self.r_off_tete = self.r_off_tete.at[i, j].set(r_off_tete)
            self.r_off_tete = self.r_off_tete.at[j, i].set(r_off_tete)

        else:
            self.r_off_tete[i, j] = r_off_tete
            self.r_off_tete[j, i] = r_off_tete

        print("Initialized high-l TT/TE.")

    def load_fits_archive(self, filename):
        fp = fits.open(
            os.path.normpath(os.path.join(self.data_folder, self.TEEEBB_maskfile))
        )
        a = asarray([x for x, _ in fp[1].data])
        b = asarray([y for _, y in fp[1].data])
        fp.close()
        return a, b

    def setup_for_lowl_te_ee_bb(self):
        """
        Initialize the TE/EE/BB low-l likelihood. Note that it is not
        implemented and should be disabled.
        """
        raise NotImplementedError(
            "The Low-l polarization likelihood is not \
                                   implemented. Set .use_lowl_pol = False. to \
                                   disable this component. It may be \
                                   implemented in the future."
        )

        """
        Below here follows a start at copying the code from the
        original WMAP likelihood. We ended up not needing it (since there are
        much better low-l pol datasets available.
        """

        self.ires = 3
        self.nsmax = 2**self.ires
        self.nlmax = 3 * self.nsmax - 1
        self.np = 12 * self.nsmax**2

        self.mp = 0
        _, self.mask_r3 = self.load_fits_archive(self.TEEEBB_maskfile)

        self.ngood = zeros((self.np,), dtype=int)
        for i in arange(self.np):
            if self.mask_r3[i] != 0:
                self.ngood[self.mp] = i
                self.mp += 1

        self.wl = np.loadtxt(
            os.path.normpath(os.path.join(self.data_folder, self.TEEEBB_filename_9)),
            usecols=(0,),
        )

        self.ninvplninv2 = zeros((self.mp * (2 * self.mp + 1), 2 * self.nlmax))

        tmp = fits.open(
            os.path.normpath(
                os.path.join(self.data_folder, "lowlP/std", self.TEEEBB_filename_0)
            )
        )
        ninvplninv3 = tmp[0].data.T
        tmp.close()

        k = 0
        for j in arange(self.mp * 2):
            for i in arange(j + 1):
                ip = self.ngood[i % self.mp] + self.np * (i // self.mp)
                jp = self.ngood[j % self.mp] + self.np * (j // self.mp)
                self.ninvplninv2[k, 1 : self.nlmax - 1] = ninvplninv3[
                    2 : self.nlmax, ip, jp
                ]
                k += 1

        tmp = fits.open(
            os.path.normpath(
                os.path.join(self.data_folder, "lowlP/std", self.TEEEBB_filename_1)
            )
        )
        ninvplninv3 = tmp[0].data.T
        tmp.close()

        k = 0
        for j in arange(self.mp * 2):
            for i in arange(j + 1):
                ip = self.ngood[i % self.mp] + self.np * (i // self.mp)
                jp = self.ngood[j % self.mp] + self.np * (j // self.mp)
                self.ninvplninv2[k, self.nlmax : 2 * self.nlmax - 2] = ninvplninv3[
                    2 : self.nlmax, ip, jp
                ]
                k += 1

        tmp = fits.open(
            os.path.normpath(
                os.path.join(self.data_folder, "lowlP/std", self.TEEEBB_filename_2)
            )
        )
        NinvQUr3 = tmp[0].data.T
        tmp.close()

        self.w_r3 = zeros((2 * self.mp,))

        ip = arange(self.mp)

        T, N = self.load_fits_archive(self.TEEEBB_filename_3)

        self.w_r3[ip] = T[self.ngood[ip]] * self.mask_r3[self.ngood[ip]]

        T, N = self.load_fits_archive(self.TEEEBB_filename_4)

        self.w_r3[ip + self.mp] = T[self.ngood[ip]] * self.mask_r3[self.ngood[ip]]

        self.Dp0 = zeros((2 * self.mp, 2 * self.mp))

        ip = arange(self.mp)
        jp = arange(self.mp)
        ip, jp = meshgrid(ip, jp)

        self.Dp0[ip, jp] = NinvQUr3[self.ngood[ip], self.ngood[jp]]
        self.Dp0[ip, jp + self.mp] = NinvQUr3[self.ngood[ip], self.ngood[jp] + self.np]
        self.Dp0[ip + self.mp, jp] = NinvQUr3[self.ngood[ip] + self.np, self.ngood[jp]]
        self.Dp0[ip + self.mp, jp + self.mp] = NinvQUr3[
            self.ngood[ip] + self.np, self.ngood[jp] + self.np
        ]

        self.alm_tt = zeros((self.nlmax, self.nlmax))
        tmp_alm_tt = np.genfromtxt(
            os.path.normpath(os.path.join(self.data_folder, self.TEEEBB_filename_5)),
            converters={0: lambda x: complex(*safe_eval(x))},
        )
        self.alm_tt[triu_indices(self.alm_tt.shape[0])] = tmp_alm_tt[
            : self.nlmax * (self.nlmax + 1) // 2
        ]

        tmp = fits.open(
            os.path.normpath(
                os.path.join(self.data_folder, "lowlP/std", self.TEEEBB_filename_6)
            )
        )
        ninvy = tmp[0].data.T
        self.NinvY = zeros((ninvy.shape[1], ninvy.shape[2]), dtype=complex)
        self.NinvY[:, :] = ninvy[0, :, :] + 1.0j * ninvy[1, :, :]
        tmp.close()

        print("Initialized low-l TE/EE/BB.")

    def setup_for_lowl_te_tb_ee_bb_eb(self):
        """Initialize the TE/TB/EE/BB/EB low-l likelihood"""
        raise NotImplementedError(
            "TE/TB/EE/BB/EB low-l likelihood is not yet \
                                   implemented. Set .use_lowl_TBEB = False. \
                                   or .use_lowl_pol = False. to avoid this \
                                   error."
        )

    def compute_largest_term(self, cltt):
        print("First evaluation: Calculating largest term...")

        ls = arange(self.lmin, self.lmax + 1)
        self.sigmas = self.sigmas[ls, ...]
        self.logsigma = log(self.sigmas)

        x = self.sigmas / cltt[ls, newaxis, newaxis]
        s = nansum(
            0.5 * (2.0 * ls[..., newaxis, newaxis] + 1) * (log(x) - x) - self.logsigma,
            axis=0,
        )

        self.offset = nanmax(s)

        # print(f"Done finding largest term [{self.offset:g}].")

    def compute_br_estimator(self, cltt):
        """
        Compute the Blackwell-Rao estimator for each element in the batch.
        """
        start = time.time()

        if self.first_eval:
            self.compute_largest_term(cltt)
            self.first_eval = False
            cltt = array(
                [cltt]
            )  # if first eval make this 2d array for latter compatibility

        ls = arange(self.lmin, self.lmax + 1)

        start = time.time()
        x = (
            self.sigmas[newaxis, :] / cltt[:, ls, newaxis, newaxis]
        )  # Broadcasting sigmas across the batch dimension
        s = sum(
            0.5 * (2.0 * ls[:, ..., newaxis, newaxis] + 1) * (log(x) - x)
            - self.logsigma,
            axis=1,
        )
        if is_jax:
            logl = logsumexp(s - self.offset, axis=(1, 2))
        else:
            logl = numpy_logsumexp(s - self.offset, axis=(1, 2))
        end = time.time()
        print(f"Done evaluating br estimator in [{end - start:.2f}s].")

        return logl

    def get_requirements(self):
        return {"Cl": {k: 2000 for k in self.use_cl}}

    def loglike_lowl_TT(self, cltt):
        """
        Calculate the low l TT likelihood (with either the gibbs or
        pixel method).
        """

        logl = 0.0
        if self.use_gibbs:
            cltt_dummy = zeros(
                (
                    cltt.shape[0],
                    self.gibbs_ell_max + 1,
                )
            )

            # this condition should not be met
            if self.gibbs_ell_max > self.lowl_max:
                cltt_dummy[self.lowl_max + 1 : self.gibbs_ell_max] = self.cltt_fiducial[
                    self.lowl_max + 1 : self.gibbs_ell_max
                ]

            if is_jax:
                cltt_dummy = cltt_dummy.at[:, self.ttmin : self.lowl_max + 1].set(
                    cltt[:, self.ttmin : self.lowl_max + 1]
                )

            else:
                cltt_dummy[:, self.ttmin : self.lowl_max + 1] = cltt[
                    :, self.ttmin : self.lowl_max + 1
                ]

            logl = self.compute_br_estimator(cltt_dummy)
        else:
            raise NotImplementedError(
                "Low l TT pixel likelihood not \
                                       implemented. Set .use_gibbs = True. or \
                                       disable low l TT likelihood."
            )

        return logl

    def loglike_lowl_TEEEBB(self, cltt, clte, clee, clbb):
        raise NotImplementedError(
            "The Low-l polarization likelihood is not \
                                   implemented. Set .use_lowl_pol = False. to \
                                   disable this component. It may be \
                                   implemented in the future."
        )

        ls = 2 + arange(self.nlmax - 1)

        lfac = (2.0 * pi) * 1e-6 * self.wl[ls] ** 2.0 / (ls * (ls + 1.0))

        xxx = zeros(2 * self.nlmax)

        xxx[ls - 1] = (clee[ls] - clte[ls] ** 2.0 / cltt[ls]) * lfac
        xxx[ls - 2 + self.nlmax] = clbb[ls] * lfac

        yyy = self.ninvplninv2 @ xxx
        Dp1 = zeros_like(self.Dp0)
        Dp1[tril_indices(Dp1.shape[0])] = yyy
        Dp1 = Dp1 + Dp1.T - diag(diag(Dp1))

        Dpu = linalg.cholesky(self.Dp0 + Dp1, lower=False)
        _, lndet = linalg.slogdet(Dpu)

        p_r3 = zeros((2 * self.mp,))

        ip = arange(self.mp)
        ls = arange(2, 23)

        p_r3[ip] = nansum(
            clte[newaxis, ls]
            / cltt[newaxis, ls]
            * self.wl[newaxis, ls]
            * self.alm_tt[newaxis, ls, 0]
            * self.NinvY[self.ngood[ip], 3, newaxis],
            axis=1,
        )
        p_r3[ip + self.mp] = nansum(
            clte[newaxis, ls]
            / cltt[newaxis, ls]
            * self.wl[newaxis, ls]
            * self.alm_tt[newaxis, ls, 0]
            * self.NinvY[self.ngood[ip] + self.np, 3, newaxis],
            axis=1,
        )

        chisqr = -0.5 * (self.w_r3 - p_r3) @ Dpu @ (self.w_r3 - p_r3)

        return chisqr, (self.TEEEBB_pixlike_logdet_offset - lndet * 2.0) / 2.0

    def loglike_lowl_TETBEEBBEB(self, cltt, clte, cltb, clee, clbb, cleb):
        return 0.0, 0.0

    def loglike_init_covterms(self, cls, TTTT, TTTE, TETE, TTTB, TBTB):
        """
        Calculate the TTTT/TTTE/TETE/TTTB/TBTB diagonal components of the
        high l TT/TE/TB likelihood.
        """
        ls = arange(self.ttmin, self.ttmax + 1)

        if "tb" not in cls:
            cls["tb"] = zeros((cls["tt"].shape[0], self.temax + 1))
        if "ee" not in cls:
            cls["ee"] = zeros((cls["tt"].shape[0], self.temax + 1))
        if "bb" not in cls:
            cls["bb"] = zeros((cls["tt"].shape[0], self.temax + 1))

        if is_jax:
            TTTT = TTTT.at[:, ls].set(
                2.0
                * power(cls["tt"][:, ls] + self.ntt[ls], 2.0)
                / ((2.0 * ls + 1.0) * power(self.fskytt[ls], 2.0))
            )

        else:
            TTTT[:, ls] = (
                2.0
                * power(cls["tt"][:, ls] + self.ntt[ls], 2.0)
                / ((2.0 * ls + 1.0) * power(self.fskytt[ls], 2.0))
            )

        ls = arange(self.temin, self.temax + 1)

        if is_jax:
            TETE = TETE.at[:, ls].set(
                (
                    (cls["tt"][:, ls] + self.ntt_te[ls])
                    * (cls["ee"][:, ls] + self.nee_te[ls])
                    + power(cls["te"][:, ls], 2.0)
                )
                / ((2.0 * ls + 1.0) * power(self.fskyte[ls], 2.0))
            )

            TTTE = TTTE.at[:, ls].set(
                2.0
                * (cls["tt"][:, ls] + self.ntt[ls])
                * cls["te"][:, ls]
                / ((2.0 * ls + 1.0) * self.fskytt[ls] * self.fskyte[ls])
            )

            TBTB = TBTB.at[:, ls].set(
                (
                    (cls["tt"][:, ls] + self.ntt_tb[ls])
                    * (cls["bb"][:, ls] + self.nbb_tb[ls])
                    + power(cls["tb"][:, ls], 2.0)
                )
                / ((2.0 * ls + 1.0) * power(self.fskytb[ls], 2.0))
            )

        else:
            TETE[:, ls] = (
                (cls["tt"][:, ls] + self.ntt_te[ls])
                * (cls["ee"][:, ls] + self.nee_te[ls])
                + power(cls["te"][:, ls], 2.0)
            ) / ((2.0 * ls + 1.0) * power(self.fskyte[ls], 2.0))
            TTTE[:, ls] = (
                2.0
                * (cls["tt"][:, ls] + self.ntt[ls])
                * cls["te"][:, ls]
                / ((2.0 * ls + 1.0) * self.fskytt[ls] * self.fskyte[ls])
            )

            TBTB[:, ls] = (
                (cls["tt"][:, ls] + self.ntt_tb[ls])
                * (cls["bb"][:, ls] + self.nbb_tb[ls])
                + power(cls["tb"][:, ls], 2.0)
            ) / ((2.0 * ls + 1.0) * power(self.fskytb[ls], 2.0))

        return TTTT, TTTE, TETE, TTTB, TBTB

    def get_fisher_TTTT(self, cltt, TTTT, TTTE, TETE):
        """
        Calculate the TTTT fisher matrix from the TTTT/TTTE/TETE components.
        """
        # Ignore unsupported zero-variance multipoles; retain zero cross-covariance.

        TTTT = where(TTTT == 0, ones_like(TTTT), TTTT)
        TETE = where(TETE == 0, ones_like(TETE), TETE)

        # Each off-diagonal entry depends on both multipoles, as in pyWMAP.
        variance_product = TTTT[:, :, newaxis] * TTTT[:, newaxis, :]
        fisher = (-self.r_off_tttt / sqrt(variance_product)
                  + self.epsilon / variance_product)

        ls = arange(self.ttmin, self.ttmax + 1)
        ls_te = arange(self.temin, self.te_lmax + 1)

        diag = zeros_like(TTTT)

        if is_jax:
            diag = diag.at[:, ls].set(1.0 / TTTT[:, ls])
            diag = diag.at[:, ls_te].set(
                TETE[:, ls_te]
                / (TTTT[:, ls_te] * TETE[:, ls_te] - TTTE[:, ls_te] * TTTE[:, ls_te])
            )

            indices = arange(fisher.shape[-1])
            fisher = fisher.at[:, indices, indices].set(diag)
        else:
            diag[:, ls] = 1.0 / TTTT[:, ls]
            diag[:, ls_te] = TETE[:, ls_te] / (
                TTTT[:, ls_te] * TETE[:, ls_te] - TTTE[:, ls_te] * TTTE[:, ls_te]
            )

            for i in range(fisher.shape[0]):
                fill_diagonal(fisher[i], diag[i])

        return fisher

    def loglike_highl_TT(self, cltt, TTTT, TTTE, TETE):
        off_log_curv = zeros((cltt.shape[0], self.tt_lmax + 1, self.tt_lmax + 1))

        ls = arange(self.ttmin, self.ttmax + 1)
        z = log(self.cltt_dat[ls] + self.ntt[ls])
        zbar = log(cltt[:, ls] + self.ntt[ls])

        fisher = self.get_fisher_TTTT(cltt, TTTT, TTTE, TETE)

        # O[i, j] = ( cltt[i] + ntt[i] ) * fisher[i, j] * (cltt[j] + ntt[j])
        if is_jax:
            off_log_curv = off_log_curv.at[
                :, self.ttmin : self.ttmax + 1, self.ttmin : self.ttmax + 1
            ].set(
                (cltt[:, newaxis, self.ttmin:self.ttmax + 1] + self.ntt[newaxis, ls])
                * fisher[:, self.ttmin:self.ttmax + 1, self.ttmin:self.ttmax + 1]
                * (cltt[:, ls, newaxis] + self.ntt[ls, newaxis])
            )
            fisher = fisher.at[:, : self.tt_highl_start, : self.tt_highl_start].set(0.0)
            off_log_curv = off_log_curv.at[
                :, : self.tt_highl_start, : self.tt_highl_start
            ].set(0.0)

        else:
            off_log_curv[
                :, self.ttmin : self.ttmax + 1, self.ttmin : self.ttmax + 1
            ] = (
                (cltt[:, newaxis, ls] + self.ntt[newaxis, ls])
                * fisher[:, self.ttmin:self.ttmax + 1, self.ttmin:self.ttmax + 1]
                * (cltt[:, ls, newaxis] + self.ntt[ls, newaxis])
            )
            fisher[:, : self.tt_highl_start, : self.tt_highl_start] = 0.0
            off_log_curv[:, : self.tt_highl_start, : self.tt_highl_start] = 0.0

        dtt = cltt[:, ls] - self.cltt_dat[ls]

        logl = zeros((cltt.shape[0],))
        diff = z - zbar
        chi2 = einsum("bi, bij, bj -> b", diff, off_log_curv[:, self.ttmin:self.ttmax + 1, self.ttmin:self.ttmax + 1], diff)
        logl += (2.0 / 3.0) * chi2
        logl += (1.0 / 3.0) * einsum(
            "bi, bij, bj -> b", dtt, fisher[:, self.ttmin:self.ttmax + 1, self.ttmin:self.ttmax + 1], dtt
        )

        return -logl / 2.0

    def loglike_TT_beam_ptsrc(self, cltt, fisher):
        """
        Add in the TT beam & point-source corrections term.
        """
        mode = zeros((self.bptsrc_lmax + 1, self.nmodes))
        ls = arange(self.bptsrc_lmin, self.bptsrc_lmax + 1)

        if self.beam_include_beam_modes:
            if self.beam_fixed_fiducial_spectrum:
                mode[ls, : self.n_beam_modes] = (
                    self.beam_mode[ls, :] * self.fiducial_cltt[ls, newaxis]
                )
            else:
                mode[ls, : self.n_beam_modes] = (
                    self.beam_mode[ls, :] * cltt[ls, newaxis]
                )

        if self.beam_include_ptsrc_mode:
            mode[ls, self.ptsrc_mode_index] = self.ptsrc_err * self.ptsrc_mode[ls]

        a = zeros((self.nmodes,))
        b = identity(self.nmodes)
        F_mode = zeros((self.bptsrc_lmax + 1, self.nmodes))

        if self.beam_diagonal_sigma:
            F_mode[ls, :] = diag(fisher)[ls, newaxis] * mode[ls, :]
        else:
            # I havent found a good way to unloop this.
            for l1 in ls:
                for l2 in ls:
                    if l2 < l1:
                        F_mode[l1, :] += fisher[l2, l1] * mode[l2, :]
                    else:
                        F_mode[l1, :] += fisher[l1, l2] * mode[l2, :]

        a[: self.nmodes] = nansum(
            (self.cltt_dat[ls] - cltt[ls])[..., newaxis] * F_mode[ls, :], axis=0
        )
        b += dot(mode.T, F_mode)

        b = linalg.cholesky(b, lower=True)
        c = linalg.solve_triangular(b, a, lower=True)
        dgauss = dot(a, c)

        if self.beam_gaussian_likelihood:
            dlndet = log(nanprod(diag(b) ** 2.0))

            return (dgauss - dlndet) / 2.0

        return -dgauss

    def get_fisher_TETE(self, clte, TTTT, TTTE, TETE):
        """
        Calculate the TETE fisher matrix from the TTTT/TTTE/TETE components.
        """
        # Ignore unsupported zero-variance multipoles; retain zero cross-covariance.
        TTTT = where(TTTT == 0, ones_like(TTTT), TTTT)
        TETE = where(TETE == 0, ones_like(TETE), TETE)
        fisher = -self.r_off_tete / sqrt(TETE[:, :, newaxis] * TETE[:, newaxis, :])

        ls = arange(self.temin, self.temax + 1)
        ls_lo = arange(self.temin, self.te_lmax + 1)

        diag = zeros((clte.shape[0], fisher.shape[1]))

        if is_jax:
            diag = diag.at[:, ls].set(1.0 / TETE[:, ls])
            diag = diag.at[:, ls_lo].set(
                TTTT[:, ls_lo]
                / (TTTT[:, ls_lo] * TETE[:, ls_lo] - TTTE[:, ls_lo] * TTTE[:, ls_lo])
            )

            indices = arange(fisher.shape[-1])
            fisher = fisher.at[:, indices, indices].set(diag)

        else:
            diag[:, ls] = 1.0 / TETE[:, ls]
            diag[:, ls_lo] = TTTT[:, ls_lo] / (
                TTTT[:, ls_lo] * TETE[:, ls_lo] - TTTE[:, ls_lo] * TTTE[:, ls_lo]
            )
            # fill_diagonal(fisher, diag)
            # loop over the batch index and fill with diag
            for i in range(clte.shape[0]):
                fill_diagonal(fisher[i], diag[i])

        return fisher

    def loglike_highl_TE(self, clte, TTTT, TTTE, TETE):
        """
        Calculate the TETE log-likelihood.
        Returns both the model chi2 and the log(det) of the fisher matrix.
        """
        ls = arange(self.temin, self.temax + 1)

        fisher = self.get_fisher_TETE(clte, TTTT, TTTE, TETE)

        if is_jax:
            fisher = fisher.at[:, : self.te_highl_start, : self.te_highl_start].set(0.0)

        else:
            fisher[:, : self.te_highl_start, : self.te_highl_start] = 0.0

        dte = clte[:, ls] - self.clte_dat[ls]

        fisher_reshaped = fisher[:, self.temin:self.temax + 1, self.temin:self.temax + 1]
        logl = einsum("bi, bij, bj -> b", dte, fisher_reshaped, dte)

        logdet = zeros(
            dte.shape[0]
        )  # Initialize an array to store logdet for each batch

        for i in range(dte.shape[0]):
            # Compute the logarithm of the determinant of the submatrix of fisher for each batch
            _, logdet_batch = linalg.slogdet(
                fisher[i, self.te_highl_start :, self.te_highl_start :]
            )

            if is_jax:
                logdet = logdet.at[i].set((logdet_batch + self.TE_logdet_offset) / 2.0)
            else:
                logdet[i] = (logdet_batch + self.TE_logdet_offset) / 2.0

        # logl and logdet are now arrays with one entry per batch

        return -logl / 2.0, logdet

    def loglike_highl_TB(self, cltb, TTTT, TTTB, TBTB):
        """
        Calculate the TBTB log-likelihood.
        Returns both the model chi2 and the log(det) of the fisher matrix.
        """
        ls = arange(self.temin, self.temax + 1)

        fisher = zeros((self.te_lmax + 1, self.te_lmax + 1))

        diag = zeros((fisher.shape[0]))
        diag[ls] = 1.0 / TBTB[ls]
        fill_diagonal(fisher, diag)
        fisher[: self.te_highl_start, : self.te_highl_start] = 0.0

        dtb = cltb[ls] - self.cltb_dat[ls]

        logl = dtb @ fisher[ls, :][:, ls] @ dtb
        _, logdet = linalg.slogdet(fisher[self.te_highl_start :, self.te_highl_start :])

        return -logl / 2.0, (logdet + self.TB_logdet_offset) / 2.0

    def loglike(self, Cls, params_dict):
        components = {}

        if self.use_sz:
            # vectorize this later
            sz_spec = zeros(
                self.tt_lmax + 1,
            )
            lmax = self.tt_lmax + 1

            if is_jax:
                sz_spec = sz_spec.at[:lmax].set(self.sz_spec[:lmax])

            else:
                sz_spec[:lmax] = self.sz_spec[:lmax]

            Cls["tt"] = Cls.get("tt") + sz_spec[Cls.get("ell")] * array(
                params_dict["A_sz"][:, newaxis]
            )

        if self.use_lowl_TT:
            start = time.time()
            components["lowl_TT_gibbs"] = self.loglike_lowl_TT(Cls.get("tt"))
            # print("lowl_TT_gibbs", time.time() - start)

        TTTT = zeros(
            (
                Cls["tt"].shape[0],
                self.tt_lmax + 1,
            )
        )
        TETE = zeros(
            (
                Cls["tt"].shape[0],
                self.te_lmax + 1,
            )
        )
        TTTE = zeros(
            (
                Cls["tt"].shape[0],
                self.te_lmax + 1,
            )
        )
        TBTB = zeros(
            (
                Cls["tt"].shape[0],
                self.te_lmax + 1,
            )
        )
        TTTB = zeros(
            (
                Cls["tt"].shape[0],
                self.te_lmax + 1,
            )
        )

        if self.use_highl_TT or self.use_highl_TE or self.use_highl_TB:
            start = time.time()
            TTTT, TTTE, TETE, TTTB, TBTB = self.loglike_init_covterms(
                Cls, TTTT, TTTE, TETE, TTTB, TBTB
            )
            # print("init_covterms", time.time() - start)

        if self.use_highl_TT:
            start = time.time()
            components["MASTER_TTTT"] = self.loglike_highl_TT(
                Cls.get("tt"), TTTT, TTTE, TETE
            )
            print("MASTER_TTTT", time.time() - start)

        if self.use_highl_TE:
            start = time.time()
            logchi, logdet = self.loglike_highl_TE(Cls.get("te"), TTTT, TTTE, TETE)
            components["MASTER_TETE_chi2"] = logchi
            components["MASTER_TETE_det"] = logdet
            print("MASTER_TETE", time.time() - start)

        if self.use_highl_TB:
            logchi, logdet = self.loglike_highl_TB(Cls.get("tb"), TTTT, TTTB, TBTB)
            components["MASTER_TBTB_chi2"] = logchi
            components["MASTER_TBTB_det"] = logdet

        if self.use_highl_TT and self.use_highl_TT_beam_ptsrc:
            fisher = self.get_fisher_TTTT(Cls.get("tt"), TTTT, TTTE, TETE)
            components["beamptsrc_TT"] = self.loglike_TT_beam_ptsrc(
                Cls.get("tt"), fisher
            )

        logp = 0.0
        for c in components:
            logp += components[c]

        return logp, components

    def calculate(self, state, want_derived=True, **params):
        Cls = self.provider.get_Cl(ell_factor=True)

        logp, components = self.loglike(Cls, params)

        state["logp"] = logp
        if state["derived"] is None:
            state["derived"] = {}
        for c in components:
            state["derived"]["chi2_wmap_" + c] = -2.0 * components[c]
            state["derived"]["logp_" + c] = components[c]

        chi2 = -2.0 * logp

        for c in components:
            lp = components[c]
            chi2 = -2.0 * lp

        return True

    def compute_like(self, p):
        # get theory Cls from emulator
        units_factor = 1e12  # units factor to convert to microKelvin

        # Transform parameters and apply all necessary conversions (omega_m->omega_cdm, h/sigma8->theta_s/A_s, etc.)
        params_dict = utils_likelihood.prepare_params_dict(
            p, self.dtype, self.cosmology_params_fixed, self.additional_args
        )

        tt_emu_model, te_emu_model, ee_emu_model = self.emu_model_list

        if self.additional_args.get("integrated_model") is True:
            Cltt = tt_emu_model.predict(params_dict) * units_factor
            Clte = te_emu_model.predict(params_dict) * units_factor
            Clee = ee_emu_model.predict(params_dict) * units_factor

        else:
            Cltt = exp(tt_emu_model.predictions_np(params_dict)) * units_factor
            Clte = te_emu_model.predictions_np(params_dict) * units_factor
            Clee = exp(ee_emu_model.predictions_np(params_dict)) * units_factor

        if self.apply_wmap_temperature_rescaling:
            Cltt = Cltt * (1 + params_dict["m_t_wmap"][:, None]) ** 2  # TT
            Clte = Clte * (1 + params_dict["m_t_wmap"][:, None])
        if self.apply_wmap_polarization_rescaling:
            Clee = Clee * (1 + params_dict["m_e_wmap"][:, None]) ** 2
            Clte = Clte * (1 + params_dict["m_e_wmap"][:, None])

        # add two extra zeros at the front for ell=0,1
        rows = Cltt.shape[0]
        zeros_column = zeros(rows)  # Single column of zeros

        Cltt = column_stack((zeros_column, zeros_column, Cltt))
        Clte = column_stack((zeros_column, zeros_column, Clte))
        Clee = column_stack((zeros_column, zeros_column, Clee))

        # Cls = self.provider.get_Cl(ell_factor=True)

        ell = arange(self.tt_lmax + 1)
        dlth = {}
        dlth["tt"] = Cltt[:, : self.tt_lmax + 1] * (ell * (ell + 1)) / (2 * pi)
        dlth["ee"] = Clee[:, : self.tt_lmax + 1] * (ell * (ell + 1)) / (2 * pi)
        dlth["te"] = (
            Clte[:, : self.te_lmax + 1]
            * (ell[: self.te_lmax + 1] * (ell[: self.te_lmax + 1] + 1))
            / (2 * pi)
        )
        dlth["et"] = dlth["te"]
        dlth["ell"] = ell

        if self.additional_args.get("test_wmap") is not None:
            dlth = {}
            test_cls = np.genfromtxt(
                os.path.join(self.data_folder, "test_cls_v5.dat")
            )

            ell = test_cls[:, 0].astype(int)
            dlth["tt"] = array([test_cls[:, 1]])
            dlth["ee"] = array([test_cls[:, 3]])
            dlth["te"] = array([test_cls[:, 2]])
            dlth["et"] = dlth["te"]
            dlth["ell"] = ell

        else:
            pass

        loglike, _ = self.loglike(dlth, params_dict)

        return loglike
