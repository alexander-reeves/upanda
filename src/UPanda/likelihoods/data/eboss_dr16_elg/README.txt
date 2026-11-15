eBOSS DR16 ELG (z_eff = 0.85) likelihoods, as distributed by SDSS and used by Alam et al. 2021 (arXiv:2007.08991).
Copied 2026-10-02.

sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt
    Official BAO+full-shape consensus likelihood on a 100^3 grid: columns DM/rd, DH/rd, fsigma8, L (linear, peak 1).
    Source: https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-plus/
    (de Mattia et al. 2021, arXiv:2007.09008: interpolated consensus MCMC posterior incl. systematics). Used by the
    eBOSS CosmoMC fork (3D trilinear in L) and cobaya bao.sdss_dr16_baoplus_elg (trilinear in ln L).
sdss_DR16_ELG_rsd_consensus.dat
    Official 1D fsigma8 likelihood (columns fsigma8, L) used for the "RSD-only" fits of Alam et al. 2021
    (dataset DR16ELGrsd). Source: eBOSS CosmoMC fork github.com/evamariam/CosmoMC_SDSS2020 (data/DR16/).
    Mean 0.3146, sd 0.0946 = Table III's 0.315 +- 0.095.
sdss_DR16_ELG_BAO_DVtable.txt
    Official BAO-only DV/rd likelihood table (z = 0.845). Same SDSS source. Not used by UPanda.
eboss_elg_grid_processed.npz
    Derived by UPanda.likelihoods.eboss_elg.build_grid_cache() from the grid file (light Gaussian smoothing of L,
    sigma = 0.7 cells, to remove MCMC-histogram noise; ln L floored at -60; plus the fsigma8-marginal 2D grid).
