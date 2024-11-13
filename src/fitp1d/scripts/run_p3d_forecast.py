import argparse
import datetime
import multiprocessing as mp

import astropy.io.fits
import iminuit
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

import fitp1d.plotting
from fitp1d.p3d_model import LyaP3DArinyoModel

mycosmopowerdir = "/Users/nk452/repos/cosmopower"
mydatadir = "OSC-v5/no-contaminants/eboss-combined.fits"
use_mp = False  # This is not supported with vectorized log_prob.
progbar = False
vectorize = False
nl = 4
nwalkers = 32
nproc = 4
nsamples = 5000

prior_cosmo = {
    'omega_b': 0.00014 * 1.0,
    'omega_cdm': 0.00091 * 5.0,
    'h': 0.0042 * 1.0,
    'n_s': 0.0038 * 5.0,
    'ln10^{10}A_s': 0.014 * 1.0,
    'k_p': 4.8, 'q_1': 0.2
}

model, base_cosmo = None, None
k, p2fit, fisher = None, None, None
mcmc_package = None
free_params, fix_params, all_params = [], [], []


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mydatadir", help="Data directory",
                        default=mydatadir)
    parser.add_argument(
        "--mycosmopowerdir", help="CosmoPower trained emulator",
        default=mycosmopowerdir)
    parser.add_argument("--zeff", type=float, help="Effective redshift",
                        default=2.4)
    parser.add_argument("--fix-cosmology", action="store_true",
                        help="Do not sample cosmology.")
    parser.add_argument("--use-camb", action="store_true",
                        help="Use CAMB instead of cosmopower emu.")
    parser.add_argument("--mock-truth", action="store_true",
                        help="Replace data with exact p3d model.")
    parser.add_argument("--nwalkers", type=int, default=nwalkers,
                        help="Number of walkers.")
    parser.add_argument("--nproc", type=int, default=nproc,
                        help="Number of processes.")
    parser.add_argument("--nsamples", type=int, default=nsamples,
                        help="Number of samples.")
    parser.add_argument("--progbar", action="store_true")
    parser.add_argument("--vectorize", action="store_true",
                        help="Use vectorized log_prob")
    parser.add_argument("--mcmc", choice=['emcee', 'zeus'], default='emcee',
                        help="MCMC package to use.")

    return parser


def readP3dData(fname, nl):
    with astropy.io.fits.open(fname) as hdul:
        k = hdul['K'].data
        pfid = hdul['PFID'].data
        fp = hdul['FPOWER'].data
        p3d = hdul['P3D_EST'].data
        fisher = hdul['FISHER'].data
        cov = hdul['COV'].data

    nk = k.size // nl
    k = k.reshape(nl, nk)[0]
    njobs = p3d.shape[0]
    p3d_all = p3d.reshape(njobs, nl, nk)
    pfid = pfid.reshape(nl, nk)
    p3d = p3d_all[4]
    fp = fp[4].reshape(nl, nk)
    err = np.sqrt(cov.diagonal()).reshape(nl, nk)
    return nk, k, njobs, p3d_all, pfid, p3d, fp, err, fisher, cov


def setGlobals(args):
    global nwalkers, nsamples, progbar, nproc, vectorize, use_mp
    global model, base_cosmo, free_params, fix_params, all_params
    global k, p2fit, fisher
    global mcmc_package

    nwalkers = args.nwalkers
    nproc = args.nproc
    nsamples = args.nsamples
    progbar = args.progbar
    vectorize = args.vectorize
    use_mp = (nproc > 1) & (not vectorize)
    if vectorize and (nproc > 1):
        print("Vectorization disables multiprocessing.")

    _, k, _, _, _, p2fit, _, _, fisher, _ = readP3dData(args.mydatadir, nl)
    model = LyaP3DArinyoModel(args.zeff, args.mycosmopowerdir,
                              use_camb=args.use_camb)
    model.cacheK(k)
    p2fit = p2fit.ravel()
    base_cosmo = model.broadcastKwargs(**model.initial.copy())
    if args.mock_truth:
        p2fit = np.hstack(model.getPls(**base_cosmo))[0]

    if args.fix_cosmology:
        free_params = ['b_F', 'beta_F', 'q_1', 'k_p']
        model.fixCosmology(**base_cosmo)
    else:
        free_params = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s',
            'b_F', 'beta_F', 'q_1', 'k_p'
        ]

    fix_params = [_ for _ in model._broadcasted_params if _ not in free_params]
    all_params = list(model.initial.keys())

    if args.mcmc == "emcee":
        import emcee as mcmc_package
    elif args.mcmc == "zeus":
        import zeus as mcmc_package
    else:
        raise Exception("Unsupported MCMC package")


def cost(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args):
        new_cosmo[all_params[i]] = np.array([x])

    # new_cosmo = model.broadcastKwargs(**new_cosmo)
    prior = 0
    for key, s in prior_cosmo.items():
        prior += ((new_cosmo[key][0] - base_cosmo[key][0]) / s)**2

    y = np.hstack(model.getPls(**new_cosmo))[0] - p2fit

    return y.dot(fisher.dot(y)) + prior


def log_prob_vectorized(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args.T):
        new_cosmo[free_params[i]] = x

    new_cosmo = model.broadcastKwargs(**new_cosmo)

    w = np.ones(args.shape[0], dtype=bool)
    for key, (b1, b2) in model.boundary.items():
        w &= (b1 < new_cosmo[key]) & (new_cosmo[key] < b2)

    for key in new_cosmo:
        new_cosmo[key] = new_cosmo[key][w]

    prior = np.zeros(w.sum())
    for key, s in prior_cosmo.items():
        prior += ((new_cosmo[key] - model.initial[key][0]) / s)**2

    result = np.empty(args.shape[0])

    y = np.hstack(model.getPls(**new_cosmo)) - p2fit
    y = -0.5 * (np.sum(y.dot(fisher) * y, axis=1) + prior)

    result[w] = y
    result[~w] = -np.inf
    return result


def log_prob_nonvectorized(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args.T):
        new_cosmo[free_params[i]] = x

    new_cosmo = model.broadcastKwargs(**new_cosmo)

    for key, (b1, b2) in model.boundary.items():
        if (new_cosmo[key] <= b1) | (b2 <= new_cosmo[key]):
            return -np.inf

    prior = 0
    for key, s in prior_cosmo.items():
        prior += ((new_cosmo[key][0] - model.initial[key][0]) / s)**2

    y = np.hstack(model.getPls(**new_cosmo))[0] - p2fit
    return -0.5 * (y.dot(fisher.dot(y)) + prior)


def minimize():
    p0 = np.array([model.initial[_] for _ in all_params])
    # for i in np.nonzero(np.isin(all_params, free_params))[0]:
    #     p0[i] += np.random.normal(1e-4)

    mini = iminuit.Minuit(cost, p0, name=all_params)
    mini.errordef = 1
    mini.print_level = 1

    for key, item in model.boundary.items():
        if key in free_params:
            mini.limits[key] = item

    for key in fix_params:
        mini.fixed[key] = True

    print(mini.migrad())
    nplots = len(free_params)
    nplots *= nplots - 1
    nplots //= 2
    ncols = 4
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows),
        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})

    j = 0
    for i, key1 in enumerate(free_params[:-1]):
        for key2 in free_params[i + 1:]:
            fitp1d.plotting.plotEllipseMinimizer(
                mini, key1, key2, model.param_labels, 'tab:blue',
                ax=axs[j // ncols, j % ncols],
                alpha=0.6, box=True, truth=model.initial, prior=prior_cosmo
            )
            j += 1

    plt.savefig(f"minimizer-ellipses.pdf",
                dpi=200, bbox_inches='tight')
    plt.close()


def sample(drop=500, thin=10):
    if vectorize:
        log_prob = log_prob_vectorized
    else:
        log_prob = log_prob_nonvectorized

    ndim = len(free_params)
    rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
    p0 = np.array([base_cosmo[_] for _ in free_params]).T + rshift

    if use_mp:
        with mp.Pool(processes=nproc) as pool:
            sampler = mcmc_package.EnsembleSampler(
                nwalkers, ndim, log_prob_nonvectorized,
                vectorize=False, pool=pool
            )
            sampler.run_mcmc(p0, nsamples, progress=progbar)
    else:
        sampler = mcmc_package.EnsembleSampler(
            nwalkers, ndim, log_prob, vectorize=vectorize)
        sampler.run_mcmc(p0, nsamples, progress=progbar)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    samples = sampler.get_chain()
    np.savetxt(f"chains_{timestamp}_p3d.txt",
               np.column_stack((samples.reshape((-1, ndim), order='F'),
                                sampler.get_log_prob(flat=True))),
               header=' '.join(free_params) + ' log_like')

    samples = samples[drop::thin].reshape((-1, ndim), order='F')
    samples = MCSamples(samples=samples, names=free_params, label="P3D")

    prior_keys = list(prior_cosmo.keys())
    prior_chains = np.array([
        np.random.default_rng().normal(
            loc=base_cosmo[key], scale=prior_cosmo[key], size=nsamples
        ) for key in prior_keys
    ]).T
    np.savetxt(f"chains_{timestamp}_prior.txt",
               prior_chains, header=' '.join(free_params))
    prior_chains = MCSamples(
        samples=prior_chains, names=prior_keys, label="Prior")

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        [samples, prior_chains],
        filled=[True, False],
        contour_colors=["tab:blue", 'k'])
    plt.savefig(f"corner_plot.pdf", dpi=200, bbox_inches='tight')
    plt.close()


def main():
    mp.set_start_method('fork')
    args = getParser().parse_args()
    setGlobals(args)
    minimize()
    sample()
