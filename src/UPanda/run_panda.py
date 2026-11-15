#!/usr/bin/env python
"""MCMC sampling and parameter space exploration utilities.

This module provides functions for running MCMC sampling and other
parameter space exploration methods using various samplers (emcee,
nautilus, HMC, ultranest, Adam optimization). It includes utilities for:

- Prior distribution handling (flat, uniform, Gaussian)
- Parameter space sampling with various sequences (Sobol, Halton, etc.)
- MCMC configuration and execution
- Gaussian prior handling with file-based means/covariances

The module supports both NumPy and JAX backends, with JAX enabling
JIT compilation and automatic differentiation for gradient-based
optimization and HMC sampling.

Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics
"""

import chaospy
import emcee
import numpy as np

import UPanda.utils as utils
from UPanda.module import *  # noqa

is_jax = config.get_jax_enabled()
if is_jax:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import debug, jit, lax, value_and_grad


# import provider
from UPanda.provider import Provider

# Supported distributions and sampling sequences
DISTRIBUTIONS = ["flat", "uniform"]
SEQUENCES = [
    "random",
    "korobov",
    "latin_hypercube",
    "sobol",
    "halton",
    "hammersley",
]


def _sampled_parameters(conf):
    """Return cosmology then nuisance parameters, excluding fixed names."""
    fixed = {p["name"] for p in conf.get("fixed_cosmology_parameters") or []}
    return [p for p in (conf.get("variable_cosmology_parameters") or [])
            + (conf.get("nuisance_parameters") or []) if p["name"] not in fixed]


def log_gauss(x, mu, cov):
    """
    Compute the log of a Gaussian distribution with mean mu and covariance cov.

    Args:
        x (array): Input values
        mu (array): Mean of the Gaussian
        cov (array): Covariance matrix or scalar variance

    Returns:
        float: Log of the Gaussian PDF
    """
    if ndim(cov) == 2:
        cov = asarray(cov)
    inv_C_gauss = utils.invert_input(
        cov
    )  # inverting the "covariance" which can be a scalar for a single param
    diff_gauss = x - array(mu).T

    if ndim(cov) == 2:  # if matrix
        return -0.5 * (diff_gauss.dot(inv_C_gauss) * diff_gauss).sum(axis=1)
    else:
        return -0.5 * diff_gauss * inv_C_gauss * diff_gauss


def get_param_lims(conf):
    """
    Extract parameter limits from configuration.

    Args:
        conf (dict): Configuration dictionary

    Returns:
        tuple: (list of parameter names, array of limits)
    """
    params = _sampled_parameters(conf)
    names = [param["name"] for param in params]
    lims = np.asarray([[param["prior_min"], param["prior_max"]] for param in params])
    return names, lims.reshape(-1, 2)


def get_gaussian_priors(conf):
    """
    Extract Gaussian priors from configuration.

    Args:
        conf (dict): Configuration dictionary

    Returns:
        list: List of tuples containing (indices, means, covariances)
    """
    priors = []
    seen = set()
    for i, param in enumerate(_sampled_parameters(conf)):
        distribution = param.get("distribution", "flat")
        if distribution in ("flat", "uniform"):
            continue
        if distribution != "gaussian":
            raise ValueError("must specify gaussian or flat priors")
        spec = param["distribution_kwargs"]
        index = spec.get("indices", i)
        key = tuple(index) if isinstance(index, (list, tuple, np.ndarray)) else index
        if key in seen:
            continue
        seen.add(key)
        mu = load(spec["mu"]) if isinstance(spec["mu"], str) else spec["mu"]
        covariance = (array(np.loadtxt(spec["cov"]))
                      if isinstance(spec["cov"], str) else spec["cov"])
        priors.append((index, mu, covariance))
    return priors


def get_chaospy_distribution(spec):
    """
    Create a chaospy distribution from specification.

    Args:
        spec (dict): Distribution specification

    Returns:
        chaospy.Distribution: Created distribution
    """

    d = None

    # uniform sampling to start MCMC

    d = chaospy.Uniform(lower=spec["prior_min"], upper=spec["prior_max"])
    return d


def get_chaospy_joint(list_distributions):
    """
    Create a joint distribution from list of distributions.

    Args:
        list_distributions (list): List of chaospy distributions

    Returns:
        chaospy.Distribution: Joint distribution
    """
    return chaospy.J(*list_distributions)


def sample_sequence(d, n_samples, n_gridpoints=None, sequence="sobol"):
    """
    Sample from a distribution using specified sequence.

    Args:
        d (chaospy.Distribution): Distribution to sample from
        n_samples (int): Number of samples
        n_gridpoints (int, optional): Number of grid points
        sequence (str, optional): Sampling sequence

    Returns:
        array: Sampled points
    """
    if sequence not in SEQUENCES:
        raise Exception(
            "sequence {} not supported, use {}".format(sequence, str(SEQUENCES))
        )

    # decide whether to load full sequcence or continue it
    if n_gridpoints is None:
        n_gridpoints = n_samples

    if n_gridpoints < n_samples:
        raise ValueError("n_gridpoints must be at least n_samples")
    samples = np.atleast_2d(d.sample(size=n_gridpoints, rule=sequence)).T

    return samples[0:n_samples, :]


def get_mcmc_start_and_fixed_params(
    conf, n_samples, n_gridpoints=None, sequence="sobol"
):
    """
    Get starting points and fixed parameters for MCMC.

    Args:
        conf (dict): Configuration dictionary
        n_samples (int): Number of samples
        n_gridpoints (int, optional): Number of grid points
        sequence (str, optional): Sampling sequence

    Returns:
        tuple: (dtype, starting points, fixed parameters)
    """
    params = _sampled_parameters(conf)
    cosmology_params_fixed = conf.get("fixed_cosmology_parameters")

    n_params = len(params)
    list_param_names = [p["name"] for p in params]

    list_distrib = [get_chaospy_distribution(params[ip]) for ip in range(n_params)]
    prior = get_chaospy_joint(list_distrib)

    samp = sample_sequence(
        d=prior,
        sequence=sequence,
        n_gridpoints=n_gridpoints,
        n_samples=n_samples,
    )

    dtype = dict(formats=["f4"] * n_params, names=list_param_names)

    params_start = np.zeros(len(samp), dtype=dtype)

    for i, par_name in enumerate(list_param_names):
        params_start[par_name] = samp[:, i]

    par = np.array(params_start.dtype.names)

    if cosmology_params_fixed is not None:
        fix = [p["name"] for p in cosmology_params_fixed]
        print("COSMO PARAMS FIXED:", fix)
        var = ~np.isin(par, fix)

    else:
        print("no fixed params")
        var = np.isin(par, par)

    print("VAR:", par[var])

    mcmc_start = np.array([params_start[c] for c in par[var]]).T

    return params_start.dtype, mcmc_start, cosmology_params_fixed


def _start_from_chain(chain_file, names, n_walkers, lims, logl_cut=50.0, seed=0):
    """Starting walkers drawn from an existing chain (a warm start) instead of the prior.

    Takes n_walkers distinct samples of `chain_file` (a structured .npy with the columns `names`), after
    dropping stuck walkers (log-likelihood more than `logl_cut` below the maximum, read from the sibling
    loglikes_*.npy) and anything outside the prior box. Burn-in is then a few autocorrelation times
    instead of the relaxation from a Sobol spread over the whole prior.
    """
    import os

    chain = np.load(chain_file)
    missing = [n for n in names if n not in chain.dtype.names]
    if missing:
        raise ValueError(f"init chain {chain_file} lacks parameters {missing}")
    x = np.stack([np.asarray(chain[n], dtype=np.float64) for n in names], axis=1)
    keep = np.ones(len(x), dtype=bool)
    ll_file = os.path.join(os.path.dirname(chain_file), os.path.basename(chain_file).replace("chains_", "loglikes_", 1))
    if os.path.exists(ll_file):
        ll = np.load(ll_file)
        if len(ll) == len(x):
            keep &= ll >= ll.max() - logl_cut
    keep &= np.all((x > lims[:, 0]) & (x < lims[:, 1]), axis=1)
    x = np.unique(x[keep], axis=0)
    if len(x) < n_walkers:
        raise ValueError(f"init chain {chain_file} has only {len(x)} usable distinct samples for {n_walkers} walkers")
    idx = np.random.default_rng(seed).choice(len(x), n_walkers, replace=False)
    print(f"warm start: {n_walkers} walkers drawn from {len(x)} distinct samples of {chain_file}", flush=True)
    return x[idx]


def _run_emcee_backend(backend_file, fingerprint, x0, n_walkers, n_par, n_iter, thin,
                       burn_in_frac, names, log_prob_fn, pool):
    """emcee with a live HDF5 backend: the chain is on disk while it runs, and a killed or timed-out
    job resumes where it stopped when resubmitted with the same arguments.

    Only every `thin`-th step is stored (emcee thin_by), which is exactly the thinning the returned chain
    has always used, so the file is n_walkers//4 times smaller than the full ensemble history. The
    returned chain is flattened step-major like before (sample i belongs to walker i % n_walkers) after
    discarding the first `burn_in_frac` of the stored steps.

    Resume rule: an existing file is continued only if its recorded fingerprint, walker number,
    dimension, thinning and parameter names all match this run; otherwise it is renamed
    <file>.superseded-<time> and a fresh chain starts. A larger n_iter than before extends the chain.
    """
    import builtins
    import os
    import time

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # Lustre/GPFS; lets readers peek at a live file
    import h5py

    n_store = max(1, n_iter // thin)
    meta = dict(fingerprint=str(fingerprint or ""), n_walkers=int(n_walkers), n_par=int(n_par),
                thin=int(thin), param_names=",".join(names))
    done = 0
    if os.path.exists(backend_file):
        try:
            with h5py.File(backend_file, "r") as f:
                old = {k: f.attrs.get(k) for k in meta}
                done = int(f["mcmc"].attrs.get("iteration", 0)) if "mcmc" in f else 0
            # builtins.all: this module's `from UPanda.module import *` shadows all/any/sum with the numpy
            # versions, and numpy's all() of a generator is always True
            match = builtins.all(str(old[k]) == str(v) for k, v in meta.items())
        except Exception as e:  # noqa: BLE001
            print(f"backend {backend_file} unreadable ({e!r}); starting fresh", flush=True)
            match, done = False, 0
        if not match or done == 0:
            if match:
                print("backend exists but holds no stored steps; starting fresh", flush=True)
            else:
                print(f"backend {backend_file} belongs to a different run: {old} vs {meta}", flush=True)
            aside = f"{backend_file}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"
            os.rename(backend_file, aside)
            print(f"moved it to {aside}", flush=True)
            done = 0

    backend = emcee.backends.HDFBackend(backend_file, name="mcmc")
    if done == 0:
        backend.reset(n_walkers, n_par)
        with h5py.File(backend_file, "a") as f:
            f.attrs.update(meta)
            f.attrs["n_iter"] = int(n_iter)
            f.attrs["burn_in_frac"] = float(burn_in_frac)
    else:
        with h5py.File(backend_file, "a") as f:
            f.attrs["n_iter"] = int(max(n_iter, int(f.attrs.get("n_iter", 0))))

    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=n_par, log_prob_fn=log_prob_fn,
                                    vectorize=True, pool=pool, backend=backend)
    if done == 0:
        print(f"live backend {backend_file}: fresh chain, {n_store} stored steps x thin {thin}", flush=True)
        state = sampler.run_mcmc(initial_state=x0, nsteps=10, store=False, progress=True)
    else:
        print(f"live backend {backend_file}: RESUMING at stored step {done} of {n_store} "
              f"(= {done * thin} of {n_store * thin} steps)", flush=True)
        state = backend.get_last_sample()
    if n_store > done:
        sampler.run_mcmc(state, nsteps=n_store - done, thin_by=thin, progress=True)

    rows = backend.iteration
    discard = int(burn_in_frac * rows)
    samples = backend.get_chain(flat=True, discard=discard)
    lnprobs = backend.get_log_prob(flat=True, discard=discard)
    try:
        print("autocorr time [steps]", backend.get_autocorr_time(discard=discard, quiet=True) * thin, flush=True)
    except Exception as e:  # noqa: BLE001
        print("could not get autocorr time. Exception:", e, flush=True)
    return samples, lnprobs


def run_mcmc(
    total_emu_model_list=[],
    experiments_list=[],
    compression_list=[],
    filename_config="",
    n_samples=1e5,
    n_walkers=512,
    burn_in_frac=0.8,
    pool=None,
    additional_args={},
    sampler="emcee",
    log_dir=None,
    resume_un="subfolder",
    point_check=None,
    backend_file=None,
    backend_fingerprint=None,
):
    """
    Run MCMC sampling for parameter estimation.

    Args:
        total_emu_model_list (list): List of emulator models
        experiments_list (list): List of experiments
        compression_list (list): List of compression vectors
        filename_config (str): Path to configuration file
        n_samples (int, optional): Number of samples
        n_walkers (int, optional): Number of walkers
        burn_in_frac (float, optional): Fraction of samples to discard as burn-in
        pool (object, optional): Multiprocessing pool
        additional_args (dict, optional): Additional arguments
        sampler (str, optional):
            "emcee"      ensemble MCMC  (default)
            "hmc"        NUTS via NumPyro  (requires is_jax=True)
            "nested"     JAX-NS nested sampling  (requires is_jax=True)
            "ultranest"  UltraNest nested sampling  (requires is_jax=True)
            "adam"       MAP finder using the optax Adam optimiser (requires is_jax=True)
            "nautilus"   Nautilus nested sampling  (requires is_jax=True)
        log_dir (str, optional): Directory for logging
        resume_un (str, optional): Subfolder for resuming
        point_check (array, optional): Point or points to check loglike at
        backend_file (str, optional): emcee only. Write the chain LIVE to this HDF5 file (every
            n_walkers//4-th step, the thinning the returned chain uses anyway) instead of holding it in
            memory until the end; the run then resumes from the file after a timeout or crash.
        backend_fingerprint (str, optional): identifies the run (config + arguments). An existing
            backend_file is resumed only if its fingerprint, walker number, dimension, thinning and
            length all match; otherwise it is renamed *.superseded-<time> and a fresh chain starts.
    Returns:
        tuple: (Parameter chain, log probabilities)
    """
    n_samples = int(n_samples)

    conf = utils.load_config(filename_config)

    # Rewrite these function to read from config and get starting point of chain:
    dtype, mcmc_start_x0, cosmology_params_fixed = get_mcmc_start_and_fixed_params(
        conf=conf,
        n_samples=n_walkers,
        n_gridpoints=n_walkers,
        sequence="sobol",
    )

    params_names_list, lims = get_param_lims(conf)

    if additional_args.get("init_chain"):
        mcmc_start_x0 = _start_from_chain(additional_args["init_chain"], dtype.names, n_walkers, lims)

    gauss_priors = get_gaussian_priors(conf)

    n_par = mcmc_start_x0.shape[1]

    likelihoods = []

    provider = Provider()
    for i, experiment in enumerate(experiments_list):
        # Get the class from the module
        module = utils.get_likelihood_module(experiment)
        LikelihoodClass = getattr(module, experiment)
        likelihood = LikelihoodClass(
            emu_model_list=total_emu_model_list[i],
            dtype=dtype,
            cosmology_params_fixed=cosmology_params_fixed,
            compression_vecs=compression_list[i],
            additional_args=additional_args,
        )
        if hasattr(likelihood, "set_provider"):
            likelihood.set_provider(provider)
        likelihoods.append(likelihood)

    kwargs = {
        "likelihoods": likelihoods,
        "lims": lims,
        "gaussian_priors": gauss_priors,
    }

    def log_like(p, likelihoods, lims, gaussian_priors, sampler="emcee"):
        like_total = zeros(p.shape[0])

        # Determine which samples are within bounds
        within_bounds = logical_and(p > lims[:, 0], p < lims[:, 1])

        if sampler in ["emcee", "ultranest", "anneal", "nautilus"]:
            select = all(within_bounds, 1)

            # Initialize a variable to accumulate likelihoods conditionally
            conditional_likes = zeros_like(like_total)

            # Assuming likelihood.compute_like can handle the full array efficiently
            for likelihood in likelihoods:
                like_individual = likelihood.compute_like(p)
                # Use where to add like_individual conditionally
                conditional_likes = where(
                    select, conditional_likes + like_individual, conditional_likes
                )

            # Adding a term for Gaussian priors, assuming log_gauss is vectorized and JAX compatible
            gaussian_add = zeros_like(like_total)

            for i, g in enumerate(gaussian_priors):
                if isinstance(g[0], int):
                    gaussian_add = where(
                        select,
                        gaussian_add + log_gauss(p[:, g[0]], g[1], g[2]),
                        gaussian_add,
                    )
                else:
                    gaussian_add = where(
                        select,
                        gaussian_add + log_gauss(p[:, array(g[0])], g[1], g[2]),
                        gaussian_add,
                    )

            like_total = conditional_likes + gaussian_add

            # if like total is 0 i.e. outside of param ranges make it -inf
            # (keeps walkers inside the prior — proposals outside bounds are always rejected)
            like_total = where(select, like_total, -inf)

        elif sampler in ["nested", "jaxns", "hmc", "adam", "profile"]:
            like_total = 0.0

            # Initialize a variable to accumulate likelihoods conditionally
            # Assuming likelihood.compute_like can handle the full array efficiently
            for likelihood in likelihoods:
                like_individual = likelihood.compute_like(array([p]))[0]
                # Use where to add like_individual conditionally
                like_total += like_individual
            # Adding a term for Gaussian priors, assuming log_gauss is vectorized and JAX compatible
            gaussian_add = 0.0

            # start_gauss_add = time.time()
            for i, g in enumerate(gaussian_priors):
                if isinstance(g[0], int):
                    gaussian_add += log_gauss(array([p[g[0]]]), g[1], g[2])[0]

                else:
                    gaussian_add += log_gauss(array([p[array(g[0])]]), g[1], g[2])[0]

            like_total += gaussian_add

        # neginf=-inf: preserves -inf from the out-of-bounds check above (walkers
        # can never escape the prior → no coordinate overflow).
        # nan=-1e30: emulator NaN (within bounds, outside training range) gets a large
        # finite penalty so walkers stuck there can still escape via the ensemble move
        # without triggering -inf - (-inf) = NaN arithmetic in emcee's red_blue.py.
        return nan_to_num(
            like_total, neginf=-inf, nan=-1e30, posinf=-1e30
        )

    if is_jax:

        def _log_like(p):
            return log_like(p, likelihoods, lims, gauss_priors, sampler=sampler)

        # JIT with static arguments to avoid recompilation when object states change
        log_func_final = jax.jit(_log_like, static_argnames=())
        # log_func_final = _log_like
    else:
        # Non-JAX branch: create a wrapper that closes over the arguments
        def _log_like(p):
            return log_like(p, likelihoods, lims, gauss_priors, sampler=sampler)

        log_func_final = _log_like

    # Add a diagnostic: evaluate loglike at a specific point or points if 'point_check' is provided

    if point_check is not None:
        loglikes = []
        for pt in point_check:
            if sampler in ["emcee", "ultranest", "anneal", "nautilus"]:
                pt = asarray(pt).reshape(-1, n_par)
            loglikes.append(log_func_final(pt))
        return point_check, np.array(loglikes)

    if sampler == "emcee":
        if is_jax:
            sampler_object = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=n_par,
                log_prob_fn=log_func_final,
                vectorize=True,
                pool=pool,
            )

        else:
            sampler_object = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=n_par,
                log_prob_fn=log_func_final,
                vectorize=True,
                pool=pool,
            )

        kw_emcee = {"progress": True}

        print("starting emcee sampler")

        n_iter = int(ceil(n_samples / n_walkers))
        thin = int(additional_args.get("thin") or max(1, n_walkers // 4))
        if backend_file is not None:
            samples, lnprobs = _run_emcee_backend(
                backend_file, backend_fingerprint, mcmc_start_x0, n_walkers, n_par, n_iter, thin,
                burn_in_frac, dtype.names, log_func_final, pool,
            )
        else:
            state = sampler_object.run_mcmc(
                initial_state=mcmc_start_x0, nsteps=10, **kw_emcee
            )
            sampler_object.reset()

            sampler_object.run_mcmc(state, nsteps=n_iter, **kw_emcee)

            samples = sampler_object.get_chain(
                flat=True, discard=int(burn_in_frac * n_iter), thin=thin
            )
            lnprobs = sampler_object.get_log_prob(
                flat=True, discard=int(burn_in_frac * n_iter), thin=thin
            )
        samples = samples[:n_samples, :]
        lnprobs = lnprobs[:n_samples]

        # try to get convergence information (the live-backend path prints its own, from the file)
        try:
            if backend_file is not None:
                raise RuntimeError("reported by _run_emcee_backend")
            tau = sampler_object.get_autocorr_time()
            print("autocorr time", tau)
            print("tau", tau)

        except Exception as e:
            print("could not get autocorr time. Exception:", e)

        params_chain = np.empty(len(samples), dtype=dtype)
        for i, c in enumerate(params_chain.dtype.names):
            params_chain[c] = samples[:, i]

        return params_chain, lnprobs

    elif sampler == "hmc" and is_jax:
        print("USING HMC")
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            # Uniform priors for each parameter
            params = []
            for i, param_limit in enumerate(lims):
                params.append(
                    numpyro.sample(
                        dtype.names[i], dist.Uniform(param_limit[0], param_limit[1])
                    )
                )
            # Combine parameters into a vector  (if needed)
            params = array(params)
            likelihood_value = log_func_final(params)
            numpyro.factor("likelihood", likelihood_value)

        nuts_kernel = NUTS(model)
        # MCMC sampling
        init_values = {}
        for i, param_limit in enumerate(lims):
            initial_value = (lims[i][0] + lims[i][1]) / 2
            init_values[dtype.names[i]] = initial_value

        mcmc = MCMC(nuts_kernel, num_samples=n_samples, num_warmup=100)
        mcmc.run(jax.random.PRNGKey(0), init_params=init_values)
        samples = mcmc.get_samples()

        return samples

    elif sampler == "nautilus":
        # ---  NAUTILUS nested-importance sampler  ---
        # This branch uses the NAUTILUS package (https://github.com/johannesulf/nautilus)
        # to draw posterior samples.  We assume **flat top-hat (uniform)** priors inside
        # the box defined by `lims` because Gaussian priors are already added explicitly
        # in the log-likelihood.

        from nautilus import Prior, Sampler

        low = lims[:, 0]
        high = lims[:, 1]

        # 1. Build a Prior object with uniform bounds for each parameter
        prior = Prior()
        for name, lo, hi in zip(params_names_list, low, high):
            # nautilus's Prior.add_parameter(key, dist) takes the uniform bounds as
            # a single (low, high) TUPLE. Passing them as two separate positional
            # args (`add_parameter(name, lo, hi)`) raises TypeError, and the old
            # fallback then added the parameter with the DEFAULT dist=(0, 1) — so
            # every parameter was silently sampled in [0, 1] instead of its real
            # prior range. log_like's in-bounds check (`p > lims[:,0]` etc.) then
            # rejected essentially every draw, so like_total collapsed to -inf and
            # nautilus returned all samples pinned at the -1e9 floor (the nautilus
            # "all -1e9 / log Z stuck" bug). `set_bounds` does not exist on the
            # nautilus>=1.0 Prior, so that fallback branch was a no-op anyway.
            prior.add_parameter(name, dist=(float(lo), float(hi)))

        # 2. Likelihood wrapper.  We run NAUTILUS in **vectorized** mode: it passes a
        #    (n_batch, n_dim) array and expects a (n_batch,) array back.  This lets the
        #    JAX/emulator likelihood (`log_func_final`, already batch-capable — see the
        #    ultranest branch) evaluate thousands of points per GPU call instead of one,
        #    which is the dominant cost in nested sampling here.  Column order matches the
        #    Prior parameter order = `params_names_list`, so no per-call dict rebuild.
        penalty = -1e9  # finite floor for non-finite log-likelihoods

        def _loglike_nautilus(theta_batch):
            loglike = np.asarray(log_func_final(theta_batch))
            return np.nan_to_num(loglike, neginf=penalty, nan=penalty, posinf=penalty)

        # 3. Instantiate the sampler – allow the caller to tweak hyper-parameters.
        #    n_live ~1500-2000 gives reliable log Z; n_batch sets the GPU batch size.
        n_live = int(additional_args.get("nlive", max(1500, 25 * n_par)))
        n_batch = int(additional_args.get("n_batch", 2048))
        n_networks = int(additional_args.get("n_networks", 4))
        print(
            f"[nautilus] vectorized | n_live={n_live} n_batch={n_batch} "
            f"n_networks={n_networks} for {n_par} parameters"
        )
        print(
            f"[nautilus] Parameter bounds: {dict(zip(params_names_list, zip(lims[:, 0], lims[:, 1])))}"
        )

        print("[nautilus] Initializing sampler...")
        sampler_naut = Sampler(
            prior,
            _loglike_nautilus,
            n_live=n_live,
            n_batch=n_batch,
            n_networks=n_networks,
            vectorized=True,
            pass_dict=False,
            pool=additional_args.get("nautilus_pool", None),
        )

        # 4. Run the sampler.  For *evidence* we mostly need the exploration phase that
        #    determines log Z, so `n_eff` (effective posterior sample target) can be modest
        #    to cut the extra sampling phase.  `n_like_max` caps total likelihood calls.
        max_calls = int(additional_args.get("max_ncalls", n_samples))
        n_eff = int(additional_args.get("n_eff", 5000))
        discard_exploration = bool(additional_args.get("discard_exploration", False))
        print(
            f"[nautilus] Starting nested sampling (n_eff={n_eff}, "
            f"n_like_max={max_calls}, discard_exploration={discard_exploration})..."
        )
        sampler_naut.run(
            verbose=True,
            n_eff=n_eff,
            n_like_max=max_calls,
            discard_exploration=discard_exploration,
        )
        print("[nautilus] Sampling completed!")

        # 5. Extract posterior samples (weighted) and return them in the same format
        #    as the other samplers used in this function.
        print("[nautilus] Extracting posterior samples...")
        pts, log_weights, log_l = sampler_naut.posterior()
        print(f"[nautilus] Retrieved {len(pts)} posterior samples")
        print(
            f"[nautilus] Log-likelihood range: [{log_l.min():.2f}, {log_l.max():.2f}]"
        )

        # Convert to a structured NumPy array with the expected `dtype`
        samples = np.empty(pts.shape[0], dtype=dtype)
        for idx, nm in enumerate(dtype.names):
            samples[nm] = pts[:, idx]

        # Return samples, log-likelihoods, and the sampler object for additional info
        return samples, log_l, sampler_naut

    elif sampler == "ultranest" and is_jax:
        import ultranest
        import ultranest.stepsampler as unst

        low = lims[:, 0]
        high = lims[:, 1]
        span = high - low  # shape (D,)

        # ---------- prior transform (batched) ----------
        def prior_transform(u):
            # u:  (N,D) or (D,) in [0,1]  →  θ in prior volume
            return u * span + low

        prior_transform.vectorized = True  # UltraNest flag

        def loglike_np(theta_batch):
            loglike = np.asarray(log_func_final(theta_batch))
            # return finite number
            penalty = -1e9  # large negative value for non-finite likelihoods
            return np.nan_to_num(loglike, neginf=penalty, nan=penalty, posinf=penalty)

        loglike_np.vectorized = True  # UltraNest flag

        n_live = int(additional_args.get("nlive", max(400, 4 * n_par)))
        sampler_ultra = ultranest.ReactiveNestedSampler(
            params_names_list,
            loglike_np,
            prior_transform,
            log_dir=log_dir,
            resume=resume_un,  # "overwrite" | "resume" | "subfolder"
            vectorized=True,
        )

        # SliceSampler with correct keyword names
        sampler_ultra.stepsampler = unst.SliceSampler(
            nsteps=4,
            generate_direction=unst.generate_mixture_random_direction,
            adaptive_nsteps="move-distance",  # ← valid option
        )

        # 4. Run
        result = sampler_ultra.run(
            max_ncalls=int(n_samples),
            min_num_live_points=n_live,
            dlogz=additional_args.get("dlogz", 0.3),
        )

        return result, result["weighted_samples"].logl

    elif sampler == "adam":
        if not is_jax:
            raise RuntimeError("sampler='adam' requires is_jax=True")

        jax_init_points = additional_args.get("jax_init_points", None)
        opts = additional_args.copy()

        def adam_minimizer_single(init_point):
            """
            Adam + early-stopping minimiser, JAX-friendly but **not jitted here**.
            Returns
                best_struct  – jnp.array shaped like `dtype.names`
                best_log_like (float)
            """

            # ---------------- hyper-parameters ----------------
            steps = int(opts.get("adam_steps", 10_000))
            lr0 = float(opts.get("adam_lr", 1e-3))
            rtol_obj = float(opts.get("adam_rtol", 1e-8))
            patience = int(opts.get("adam_patience", 500))
            clip_norm = float(opts.get("adam_clip_norm", 1.0))
            _PENALTY = 1e9
            _GRAD_EPS = 1e-9

            low, high = lims[:, 0], lims[:, 1]  # projection limits

            # -------------- helper functions -----------------
            def _neg_log_post(theta):
                # no jit – will be traced later by outer jitting
                return -log_func_final(theta)

            _val_grad = value_and_grad(_neg_log_post)

            lr_sched = optax.exponential_decay(
                init_value=lr0,
                transition_steps=patience,
                decay_rate=0.5,
                staircase=True,
            )

            opt_trans = optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adam(lr0),
                optax.scale_by_schedule(lr_sched),
            )

            def _project(x):
                return reflect(x, low, high)

            params = jnp.asarray(init_point, dtype=float)
            opt_state = opt_trans.init(params)
            best_val = _neg_log_post(params)
            no_improve = jnp.array(0, dtype=jnp.int32)
            step_idx = jnp.array(0, dtype=jnp.int32)

            # state carried by the loop
            state = (step_idx, params, opt_state, best_val, no_improve)

            # continue while we have steps left **and** have not exceeded patience
            def cond_fun(s):
                step, _, _, _, no_improve = s
                return (step < steps) & (no_improve < patience)

            def body_fun(s):
                step, params, opt_state, best_val, no_improve = s

                val, grads = _val_grad(params)
                # If non-finite objective or gradient, skip update and continue
                if not jnp.isfinite(val) or not jnp.all(jnp.isfinite(grads)):
                    debug.print(
                        "[adam] step {}: non-finite objective/gradient encountered, skipping update. Params: {}",
                        step,
                        params,
                    )
                    # Do not update params, best_val, or no_improve; just increment step
                    return (step + 1, params, opt_state, best_val, no_improve)

                # Adam update
                updates, opt_state = opt_trans.update(grads, opt_state, params)
                params = _project(optax.apply_updates(params, updates))

                # Improvement bookkeeping
                diff = best_val - val
                rtol_val = diff / (jnp.abs(best_val) + _GRAD_EPS)
                improve = (diff > 0) & (rtol_val > rtol_obj)

                best_val = jnp.where(improve, val, best_val)
                no_improve = jnp.where(improve, 0, no_improve + 1)

                debug.print("step {}/{}  val={:.6e}", step, steps, val)
                jax.block_until_ready(val)
                return (step + 1, params, opt_state, best_val, no_improve)

            _, best_params, _, best_val, _ = lax.while_loop(cond_fun, body_fun, state)

            best_struct = jnp.zeros((len(dtype.names),), dtype=float)
            best_struct = best_struct.at[:].set(best_params)
            best_log_like = -best_val

            return best_struct, best_log_like

        if jax_init_points is not None:
            adam_minimizer_vmap = jax.vmap(adam_minimizer_single)
            print("starting minimization with many points...")
            best_structs, best_loglikes = adam_minimizer_vmap(jax_init_points)
            results = np.zeros((best_structs.shape[0],), dtype=dtype)
            for i, name in enumerate(dtype.names):
                results[name] = np.array(best_structs[:, i])
            return results, np.array(best_loglikes)
        else:
            init = additional_args.get("adam_init", None)
            if init is None:
                params = asarray((lims[:, 0] + lims[:, 1]) * 0.5)
            elif isinstance(init, dict):
                params = asarray((lims[:, 0] + lims[:, 1]) * 0.5)
                for nm, val in init.items():
                    if nm not in dtype.names:
                        continue
                    params = params.at[dtype.names.index(nm)].set(float(val))
            else:
                params = np.asarray(init, dtype=float)
                if params.shape != (len(dtype.names),):
                    raise ValueError("adam_init must be a dict or an array of length D")
            best_struct, best_log_like = adam_minimizer_single(params)
            result = np.zeros((1,), dtype=dtype)
            for i, name in enumerate(dtype.names):
                result[name] = np.array(best_struct[i])
            return result, np.array([best_log_like])

    elif sampler == "anneal":
        """
        additional_args (all optional except *anneal_chain*):
          • anneal_chain         – structured array or (N,D) array from an
                                   earlier MCMC run (required → gives Σ & start)
          • anneal_chain_lnprob  – 1-D array of log-posteriors for that chain;
                                   if omitted we start from the *mean* of the
                                   chain instead of its best point.
          • anneal_init          – explicit warm start (dict or array); overrides
                                   the chain-based start if given.
          • anneal_n_rungs       – ladder rungs       (default 15 - improved from 5)
          • anneal_init_temp     – first temperature  (default 10.0 - improved from 1.5)
          • anneal_final_temp    – last  temperature  (default 0.001 - improved from 0.01)
          • anneal_rung_steps    – Metropolis steps per rung (default 30000 - improved from 20000)
          • anneal_accept_target – desired acceptance (default 0.25)
          • anneal_sigma0        – jump σ in whitened space (default 1.0 - improved from 0.8)
          • anneal_n_restarts    – number of independent restarts (default 20 - improved from 10)
          • anneal_adaptive_temp – use adaptive temperature scheduling (default True)
          • anneal_min_improve_steps – minimum steps before early termination (default 10000)
        """

        # ------------------ sanity: need JAX for log-posterior ------------------
        if not is_jax:
            raise RuntimeError("sampler='anneal' requires is_jax=True")

        # ------------------ read & validate run‑time options -------------------
        opts: Dict[str, object] = additional_args  # shorthand

        # -- required: need a chain to estimate the covariance
        chain = opts.get("anneal_chain")
        if chain is None:
            raise ValueError("You must pass additional_args['anneal_chain'].")
        chain = np.asarray(chain)
        print("chain_shape", chain.shape)
        if chain.ndim == 1:
            chain = chain.reshape(1, -1)

        # -- optional: log-posteriors for the chain
        chain_lnprob = opts.get("anneal_chain_lnprob", None)
        if chain_lnprob is not None:
            chain_lnprob = np.asarray(chain_lnprob)
            if chain_lnprob.ndim == 0:
                chain_lnprob = chain_lnprob.reshape(1)

        # -- optional with optimized defaults for cosmological likelihoods
        n_rungs = int(opts.get("anneal_n_rungs", 5))  # Exactly 6 well-spread rungs
        T0 = float(
            opts.get("anneal_init_temp", 1.5)
        )  # Higher initial temp for better exploration
        Tf = float(
            opts.get("anneal_final_temp", 0.05)
        )  # Lower final temp for precision
        rung_steps = int(
            opts.get("anneal_rung_steps", 25000)
        )  # More steps for better convergence
        acc_target = float(opts.get("anneal_accept_target", 0.25))
        σ = float(
            opts.get("anneal_sigma0", 0.15)
        )  # Smaller initial step size since starting near bestfit
        n_restarts = int(
            opts.get("anneal_n_restarts", 12)
        )  # Good coverage, reasonable time
        adaptive_temp = bool(
            opts.get("anneal_adaptive_temp", True)
        )  # Adaptive scheduling
        min_improve_steps = int(
            opts.get("anneal_min_improve_steps", 5000)
        )  # Faster response

        min_sigma = 0.01  # Reduced from 0.05 for finer control
        max_sigma = 5.0  # Increased from 2.0 for better exploration

        # ------------------ estimate initial covariance from chain ---------------------
        # Handle structured arrays properly
        if chain.dtype.names is not None:
            # Convert structured array to regular array for calculations
            chain_array = np.vstack([chain[name] for name in dtype.names]).T
            μ_global = np.mean(chain_array, axis=0)
            Σ_global = np.cov(chain_array, rowvar=False)
        else:
            # Regular array
            μ_global = np.mean(chain, axis=0)
            Σ_global = np.cov(chain, rowvar=False)
        # Add regularization to ensure positive definiteness
        Σ_global += (
            np.eye(Σ_global.shape[0]) * 1e-6 * np.trace(Σ_global) / Σ_global.shape[0]
        )

        # Initial Cholesky decomposition for whitening
        try:
            L_global = np.linalg.cholesky(Σ_global)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            L_global = np.diag(np.sqrt(np.diag(Σ_global)))

        # ------------------ simple global whitening --------------------------
        # Use global statistics for stable whitening
        μ = μ_global
        L = L_global

        # Simple coordinate transformation functions
        to_white = lambda θ: np.linalg.solve(L, θ - μ)  # θ→z
        from_white = lambda z: μ + L @ z  # z→θ

        # ─────────── choose the starting point ─────────────────────────────────
        init_points = opts.get("anneal_init_points", None)
        if init_points is None:
            # get init points from the chain
            num_init_points = int(opts.get("anneal_num_init_points", n_restarts))
            init_points = np.random.choice(chain, size=num_init_points, replace=False)
            print(
                f"[anneal] Using {num_init_points} random points from the chain as initial points..."
            )

        init_points = np.asarray(init_points)
        n_starts = init_points.shape[0]
        print(
            f"[anneal] Running {n_starts} annealing restarts from user-supplied initial points..."
        )

        # ─────────── objective helpers (already JIT-compiled elsewhere) ────────
        logp_fn = log_func_final  # maximise this (log-posterior)

        # Pre-compile the likelihood function for better performance
        if is_jax:
            logp_fn = jax.jit(logp_fn)

        # ─────────── improved annealing ladder settings ───────────────────────
        if adaptive_temp:
            # Exactly 6 well-spread temperature rungs for optimal exploration
            # Each rung serves a distinct purpose in the annealing process
            Ts = np.array(
                [
                    T0,  # Rung 1: High temp global exploration
                    T0 * 0.5,  # Rung 2: Medium-high exploration
                    T0 * 0.25,  # Rung 3: Medium refinement
                    T0 * 0.1,  # Rung 4: Low temp fine-tuning
                    T0 * 0.05,  # Rung 5: Very low temp precision
                    Tf,  # Rung 6: Final ultra-precision
                ]
            )
        else:
            # Simple geometric spacing for exactly 6 rungs
            Ts = np.geomspace(T0, Tf, num=6)

        sigma = σ
        low, high = lims[:, 0], lims[:, 1]
        rng = np.random.default_rng(42)

        # ------------- helper: improved boundary handling ----------------------
        def reflect(x, lo, hi):
            """Improved boundary handling with gradual penalty"""
            L = hi - lo
            reflected = lo + np.mod(x - lo, L)

            # Add small penalty for points near boundaries to discourage boundary artifacts
            boundary_penalty = 0.0
            boundary_width = 0.05 * L  # 5% of parameter range

            for i in range(len(x)):
                if x[i] < lo[i] + boundary_width[i]:
                    boundary_penalty += (
                        0.1
                        * ((lo[i] + boundary_width[i] - x[i]) / boundary_width[i]) ** 2
                    )
                elif x[i] > hi[i] - boundary_width[i]:
                    boundary_penalty += (
                        0.1
                        * ((x[i] - (hi[i] - boundary_width[i])) / boundary_width[i])
                        ** 2
                    )

            return reflected, boundary_penalty

        @jax.jit
        def reflect_vectorized(x, lo, hi):
            """Vectorized boundary handling"""
            L = hi - lo
            reflected = lo + mod(x - lo, L)

            # Vectorized boundary penalties
            boundary_width = 0.05 * L
            boundary_penalties = zeros(x.shape[0])

            # Vectorized penalty calculation
            near_lower = x < (lo + boundary_width)
            near_upper = x > (hi - boundary_width)

            boundary_penalties += sum(
                0.1 * ((lo + boundary_width - x) / boundary_width) ** 2 * near_lower,
                axis=1,
            )
            boundary_penalties += sum(
                0.1 * ((x - (hi - boundary_width)) / boundary_width) ** 2 * near_upper,
                axis=1,
            )

            return reflected, boundary_penalties

        from_white_vmap = jax.vmap(from_white)

        import time

        total_start_time = time.time()

        print(
            f"[anneal] Running {n_starts} restarts with {len(Ts)} temperature rungs..."
        )

        # Initialize all restarts
        all_z = np.array([to_white(init_point) for init_point in init_points])
        all_theta = init_points.copy()
        all_logp = np.array(logp_fn(init_points))

        # Track best values for each restart
        all_best_theta = init_points.copy()
        all_best_logp = all_logp.copy()

        # Track efficiency
        total_eval_time = 0.0
        total_evals = 0

        # Improved convergence tracking
        last_improvement_step = np.zeros(n_starts, dtype=int)
        global_step = 0
        convergence_history = []

        # Optimization: Use larger batch size for likelihood evaluation if memory allows
        batch_size = int(opts.get("anneal_batch_size", n_starts))
        if batch_size > n_starts:
            batch_size = n_starts
        print(f"[anneal] Using batch size of {batch_size} for likelihood evaluation")

        # Pre-allocate arrays outside the loop
        theta_props = np.zeros((n_starts, len(dtype.names)))
        boundary_penalties = np.zeros(n_starts)

        # Pre-allocate these arrays outside the main loop
        z_props = np.zeros((n_starts, len(dtype.names)))
        accept = np.zeros(n_starts, dtype=bool)
        ln_a = np.zeros(n_starts)

        for rung_idx, T in enumerate(Ts):
            print(f"[anneal] Temperature {rung_idx + 1}/{len(Ts)} = {T:.6f}")

            accepts = np.zeros(n_starts, dtype=int)

            # Create a new RNG for this temperature rung with a different seed
            rng = np.random.default_rng(int(T * 1e6 + rung_idx * 1000))

            # Progress bar for this temperature rung
            from tqdm import tqdm

            pbar = tqdm(range(rung_steps), desc=f"T={T:.4f}", leave=False)

            for step in pbar:
                global_step += 1

                # Generate proposals with adaptive step size
                z_props = all_z + sigma * rng.normal(size=all_z.shape)
                theta_props_raw = from_white_vmap(z_props)

                # Apply improved boundary handling
                theta_props_raw = from_white_vmap(z_props)
                theta_props[:], boundary_penalties[:] = reflect_vectorized(
                    theta_props_raw, low, high
                )

                # Evaluate likelihood for all proposals at once
                start_time = time.time()
                logp_props = np.array(logp_fn(theta_props))
                eval_time = time.time() - start_time

                # Apply boundary penalties
                logp_props -= boundary_penalties

                # Track efficiency metrics
                total_eval_time += eval_time
                total_evals += n_starts

                # Metropolis acceptance (optimized)
                ln_a = (logp_props - all_logp) / T
                accept = rng.uniform(size=n_starts) < np.exp(
                    np.minimum(ln_a, 0.0)
                )  # min is faster than clip

                # Update accepted restarts
                accept_mask = accept.astype(bool)
                all_z[accept_mask] = z_props[accept_mask]
                all_theta[accept_mask] = theta_props[accept_mask]
                all_logp[accept_mask] = logp_props[accept_mask]
                accepts += accept.astype(int)

                # Mid-rung sigma adjustment (every 2000 steps)
                if step % 2000 == 0 and step > 0:
                    current_acc_rate = np.mean(accepts) / (
                        step + 1
                    )  # Use mean for consistency

                    # Temperature-dependent targets (cosmology-optimized)
                    if T > T0 / 2:  # High temperature
                        target_low, target_high = (
                            0.15,
                            0.35,
                        )  # Lower targets for high dimensions
                    else:  # Low temperature
                        target_low, target_high = 0.20, 0.30  # Optimal 23-25% range

                    # Adjust sigma based on acceptance rate (cosmology-optimized)
                    if current_acc_rate < target_low:
                        sigma *= 0.8  # DECREASE step size for low acceptance
                        sigma = np.clip(sigma, min_sigma, max_sigma)
                        print(
                            f"[anneal] Mid-rung: low acceptance {current_acc_rate:.3f}, DECREASING sigma to {sigma:.3f}"
                        )
                    elif current_acc_rate > target_high:
                        sigma *= 1.1  # INCREASE step size for high acceptance
                        sigma = np.clip(sigma, min_sigma, max_sigma)
                        print(
                            f"[anneal] Mid-rung: high acceptance {current_acc_rate:.3f}, INCREASING sigma to {sigma:.3f}"
                        )
                    else:
                        print(
                            f"[anneal] Mid-rung: good acceptance {current_acc_rate:.3f}, keeping sigma at {sigma:.3f}"
                        )

                # Update best values and track improvement (optimized)
                improve_mask = logp_props > all_best_logp
                all_best_theta[improve_mask] = theta_props[
                    improve_mask
                ]  # Remove .copy() for speed
                all_best_logp[improve_mask] = logp_props[improve_mask]
                last_improvement_step[improve_mask] = global_step

                # Optimized early termination for cosmological likelihoods
                if (
                    global_step >= min_improve_steps and global_step % 3000 == 0
                ):  # Less frequent checks
                    steps_since_improvement = global_step - last_improvement_step
                    avg_improvement_rate = np.mean(steps_since_improvement < 1000)

                    # Terminate if few restarts improving and at moderate temperature
                    if (
                        avg_improvement_rate < 0.2 and T < T0 / 5
                    ):  # Less aggressive, earlier termination
                        print(
                            f"[anneal] Early termination at T={T:.4f}, step {global_step}"
                        )
                        print(
                            f"[anneal] Average improvement rate: {avg_improvement_rate:.3f}"
                        )
                        break

                # Update progress bar (less frequent for speed)
                if step % 5000 == 0:  # Much less frequent updates
                    avg_acc_rate = np.mean(accepts) / (step + 1)
                    avg_logp = np.mean(all_logp)
                    best_logp_overall = np.max(all_best_logp)
                    pbar.set_postfix(
                        avg_logp=f"{avg_logp:.2f}",
                        best=f"{best_logp_overall:.2f}",
                        acc=f"{avg_acc_rate:.3f}",
                    )

            pbar.close()

            # Improved Robbins–Monro adaptation
            avg_acc_rate = np.mean(accepts) / rung_steps
            if avg_acc_rate < 1e-6:
                sigma *= 0.5  # More aggressive reduction
            else:
                # Adaptive step size based on acceptance rate and temperature
                target_rate = acc_target * (
                    1 + 0.5 * (T / T0)
                )  # Higher target at high T
                sigma *= np.exp((avg_acc_rate - target_rate) / 1.5)

            sigma = np.clip(sigma, min_sigma, max_sigma)

            # Track convergence history (simplified for speed)
            convergence_history.append(
                (T, np.mean(all_logp), np.max(all_best_logp), sigma, avg_acc_rate)
            )

            print(
                f"[anneal] T={T:.4f}: avg_logp={np.mean(all_logp):.2f} best={np.max(all_best_logp):.2f} sigma={sigma:.3f} acc={avg_acc_rate:.3f}"
            )

        total_time = time.time() - total_start_time
        print(f"[anneal] Total annealing time: {total_time:.2f} seconds")
        print(
            f"[anneal] Efficiency: {total_evals} evals in {total_eval_time:.2f}s = {total_evals / total_eval_time:.1f} evals/s"
        )

        best_thetas = all_best_theta
        best_logps = all_best_logp

        # pick the restart with the HIGHEST log-posterior
        idx_best = int(np.argmax(np.asarray(best_logps)))
        best_theta = np.asarray(best_thetas[idx_best])
        best_logp = float(best_logps[idx_best])

        # sample-to-sample spread (useful diagnostics only)
        std_theta = np.std(np.asarray(best_thetas), axis=0)

        print(f"[anneal] Best log-posterior: {best_logp:.4f}")
        print(f"[anneal] Best theta: {best_theta}")
        print(f"[anneal] Std theta: {std_theta}")

        # Enhanced diagnostics
        print(f"[anneal] All log-posteriors: {best_logps}")
        print(
            f"[anneal] Log-posterior range: [{best_logps.min():.4f}, {best_logps.max():.4f}]"
        )
        print(f"[anneal] Log-posterior std: {best_logps.std():.4f}")
        print(
            f"[anneal] Convergence history: {len(convergence_history)} temperature rungs completed"
        )

        # ─────────── return best point in structured dtype ─────────────────────
        best_struct = np.zeros(1, dtype=dtype)
        for i, nm in enumerate(dtype.names):
            best_struct[nm] = best_theta[i]

        # Enhanced Adam refinement with better parameters
        refine_with_adam = opts.get("anneal_refine_with_adam", True)
        if refine_with_adam:
            print("Refining with enhanced Adam from best annealing point...")
            adam_opts = opts.copy()
            # Use improved Adam parameters for cosmological optimization
            adam_opts["adam_init"] = {
                nm: best_theta[i] for i, nm in enumerate(dtype.names)
            }
            adam_opts["adam_steps"] = (
                10000  # Fewer steps since we're already near optimum
            )
            adam_opts["adam_lr"] = 1e-3  # Higher learning rate for faster convergence
            adam_opts["adam_atol"] = (
                0.001  # Chi² absolute tolerance of 0.1 for cosmological precision
            )
            adam_opts["adam_patience"] = 500  # Much less patience for early stopping

            # Call the Adam branch logic directly
            steps = int(adam_opts.get("adam_steps", 5000))
            lr0 = float(adam_opts.get("adam_lr", 1e-3))
            atol_obj = float(adam_opts.get("adam_atol", 0.1))
            patience = int(adam_opts.get("adam_patience", 200))
            clip_norm = float(adam_opts.get("adam_clip_norm", 1.0))
            print_interval = int(adam_opts.get("adam_print_interval", 10))
            _PENALTY = 1e9
            _GRAD_EPS = 1e-9
            params = asarray([best_theta[i] for i in range(len(dtype.names))])

            # Create a wrapper for the vectorized likelihood function
            def loss_fn_adam(x):
                # The vectorized likelihood expects 2D array, so reshape and take [0]
                return -logp_fn(x.reshape(1, -1))[0]

            _val_grad = jit(value_and_grad(loss_fn_adam))
            print("initial chi2 (Adam refinement):", loss_fn_adam(params))
            lr_sched = optax.exponential_decay(
                init_value=lr0,
                transition_steps=patience,
                decay_rate=0.5,
                staircase=True,
            )
            opt_trans = optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adam(lr0),
                optax.scale_by_schedule(lr_sched),
            )
            opt_state = opt_trans.init(params)
            low, high = lims[:, 0], lims[:, 1]

            def _project(x):
                return reflect(x, low, high)[0]  # Only return the reflected point

            best_val = loss_fn_adam(params)
            no_improve = 0
            bar = tqdm(range(steps), desc="Adam refinement", leave=True, position=0)
            for step in bar:
                val, grads = _val_grad(params)
                if not jnp.isfinite(val) or not jnp.all(jnp.isfinite(grads)):
                    print("non-finite objective/gradient encountered → skipping step")
                    val = jnp.array(_PENALTY, dtype=float)
                    grads = jnp.zeros_like(params)
                updates, opt_state = opt_trans.update(grads, opt_state, params)
                params = _project(optax.apply_updates(params, updates))
                if step % print_interval == 0:
                    bar.set_postfix(loss=f"{float(val):.2f}")
                diff = best_val - val
                if diff > atol_obj:  # Absolute tolerance check
                    best_val = val
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        bar.write(
                            f"[adam] early stop at step {step} after {patience} stale steps (improvement < {atol_obj})"
                        )
                        break
            bar.close()
            best_struct = np.zeros(1, dtype=dtype)
            for i, name in enumerate(dtype.names):
                best_struct[name] = np.asarray(params[i])
            best_log_like = logp_fn(params.reshape(1, -1))[0]
            return best_struct, np.asarray([best_log_like])

    elif sampler == "profile":
        _PENALTY = 1e9  # very large χ² that safely lives above any plausible maximum
        _GRAD_EPS = 1e-9  # tiny value to avoid division by zero in relative tolerance
        if not is_jax:
            raise RuntimeError("sampler='profile' requires is_jax=True")

        # ────────────────────────────────────────────────────────────────────────────
        # Read & validate run‑time options
        # ────────────────────────────────────────────────────────────────────────────
        opts: Dict[str, object] = additional_args  # shorthand

        # -- required
        p_name: str | None = opts.get("profile_param")
        if p_name is None:
            raise ValueError("You must pass additional_args['profile_param'].")
        if p_name not in dtype.names:
            raise ValueError(f"Parameter '{p_name}' not found in dtype.names.")
        p_index: int = dtype.names.index(p_name)

        # -- optional with sensible defaults
        ngrid: int = int(opts.get("profile_npoints", 40))
        steps: int = int(opts.get("profile_adam_steps", 3000))
        lr0: float = float(opts.get("profile_adam_lr", 1e-2))
        rtol_obj: float = float(opts.get("profile_adam_rtol", 1e-3))
        patience: int = int(opts.get("profile_adam_patience", 1500))
        print_interval: int = int(opts.get("profile_print_interval", 10))
        clip_norm: float = float(opts.get("profile_clip_norm", 1.0))

        # user‑supplied callback:  callback(step, theta_full, val)
        callback: Callable[[int, ndarray, float], None] | None = opts.get(
            "profile_callback", None
        )

        # build grid in the profiled parameter
        lim_lo, lim_hi = lims[p_index]
        if (grid_spec := opts.get("profile_grid")) is None:
            grid = np.linspace(lim_lo, lim_hi, ngrid, dtype=float)
        else:
            grid = np.asarray(grid_spec, dtype=float)
            ngrid = grid.size

        # --------------------------------------------------------------------------
        # Optimiser:  clip ▸ Adam ▸ LR‑schedule (halve on plateau)
        # --------------------------------------------------------------------------
        lr_schedule = optax.exponential_decay(
            init_value=lr0,
            transition_steps=patience,
            decay_rate=0.5,
            staircase=True,
        )
        opt_trans = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(lr0),
            optax.scale_by_schedule(lr_schedule),
        )

        # --------------------------------------------------------------------------
        # Convenience arrays & masks
        # --------------------------------------------------------------------------
        D: int = len(dtype.names)
        mask_free = np.array([i != p_index for i in range(D)], dtype=bool)
        low, high = np.asarray(lims[:, 0][mask_free]), np.asarray(lims[:, 1][mask_free])

        def _project(x: ndarray) -> ndarray:  # hard projection into the box prior
            return np.clip(x, low, high)

        mids: ndarray = (lims[:, 0] + lims[:, 1]) * 0.5  # mid‑box defaults

        # --------------------------------------------------------------------------
        # Warm‑start initialisation (deterministic, but user‑overridable)
        # --------------------------------------------------------------------------
        warm_spec = opts.get("profile_warm_start", None)
        if warm_spec is None:
            warm_full = mids.copy()
        elif isinstance(warm_spec, dict):
            warm_full = mids.copy()
            for nm, val in warm_spec.items():
                if nm not in dtype.names:
                    print(f"Unknown parameter in warm start: {nm}. Skipping …")
                    continue
                warm_full[dtype.names.index(nm)] = float(val)
        else:  # assume iterable of shape (D,)
            warm_full = np.asarray(warm_spec, dtype=float)
            if warm_full.shape != (D,):
                raise ValueError(
                    "profile_warm_start must be a dict or an array of length D"
                )
        warm_free0 = warm_full[mask_free]

        # --------------------------------------------------------------------------
        # Robust objective wrapper: converts NaN/Inf to a large finite penalty
        # --------------------------------------------------------------------------
        nan_pen = np.array(_PENALTY, dtype=float)

        @jit
        def _neg_log_post_fixed(theta_free: ndarray, fixed_val: float) -> float:
            full = jnp.empty(D, dtype=jnp.float64)
            full = full.at[mask_free].set(theta_free)
            full = full.at[p_index].set(fixed_val)
            val = -log_func_final(full)
            return jnp.nan_to_num(val, nan=nan_pen, posinf=nan_pen, neginf=nan_pen)

        # value‑and‑grad factory that bakes in the *current* fixed parameter value
        def _make_obj_grad(fixed_val: float):
            return value_and_grad(lambda theta: _neg_log_post_fixed(theta, fixed_val))

        # --------------------------------------------------------------------------
        # Pre‑allocate outputs
        # --------------------------------------------------------------------------
        chain = np.zeros(ngrid, dtype=dtype)
        lnprobs = np.empty(ngrid, dtype=float)

        # --------------------------------------------------------------------------
        # Outer loop over the grid (tqdm #1)
        # --------------------------------------------------------------------------
        for i_grid, fixed_val in enumerate(tqdm(grid, desc=f"Profiling {p_name}")):
            obj_grad = _make_obj_grad(float(fixed_val))
            theta_free = np.asarray(warm_free0, dtype=float)
            opt_state = opt_trans.init(theta_free)

            best_val = None
            no_improve = 0

            # Inner minimisation (tqdm #2)
            inner_bar = tqdm(
                range(steps),
                desc=f"{p_name}={fixed_val:.4g}",
                leave=True,
                position=0,
            )
            for step in inner_bar:
                params = _project(theta_free)
                val, grads = obj_grad(params)
                if not np.all(np.isfinite(val)) or not np.all(np.isfinite(grads)):
                    print("non‑finite objective/gradient encountered → skipping step")
                    val = nan_pen
                    grads = np.zeros_like(theta_free)

                updates, opt_state = opt_trans.update(grads, opt_state, params)
                theta_free = _project(optax.apply_updates(theta_free, updates))

                # gradient‑norm early exit
                if np.linalg.norm(updates) < _GRAD_EPS:
                    break

                # relative improvement criterion
                if best_val is None:
                    best_val = val

                diff = best_val - val  # positive if val is an improvement
                rtol_val = diff / (abs(best_val) + _GRAD_EPS)

                if diff > 0 and rtol_val > rtol_obj:
                    # significant improvement → reset patience
                    best_val = val
                    no_improve = 0
                else:
                    # either no improvement or improvement within tolerance
                    no_improve += 1
                    if no_improve >= patience:
                        print(
                            f"[adam] early stop at step {step} after {patience} stale steps"
                        )
                        break

                if step % print_interval == 0:
                    chi2 = 2.0 * float(val)
                    inner_bar.set_postfix(chi2=f"{chi2:.2f}")
                    if callback is not None:
                        full_vec = mids.copy()
                        full_vec[mask_free] = theta_free
                        full_vec[p_index] = fixed_val
                        callback(step, full_vec, float(val))
            inner_bar.close()

            # ------------------------------------------------------------------
            # Store result for this grid point & prepare warm start for the next
            # ------------------------------------------------------------------
            full_vec = mids.copy()
            full_vec[mask_free] = theta_free
            full_vec[p_index] = fixed_val

            chain[i_grid] = tuple(full_vec)
            lnprobs[i_grid] = -float(val)
            warm_free0 = theta_free  # deterministic warm‑start

        # After the full sweep, return the profile chain & ln ℒ
        return chain, lnprobs

    else:
        raise ValueError(f"Sampler '{sampler}' is not supported or not implemented.")
