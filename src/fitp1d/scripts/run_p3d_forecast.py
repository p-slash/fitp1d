import argparse
import datetime
import json
import multiprocessing as mp

import astropy.io.fits
import iminuit
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

import fitp1d.plotting
from fitp1d.p3d_model import LyaP3DArinyoModel

mydatadir = "OSC-v5/no-contaminants/eboss-combined.fits"
use_mp = False  # This is not supported with vectorized log_prob.
progbar = False
vectorize = False
nl = 4
nwalkers = 40
nproc = 5
nsamples = 5000

model, base_cosmo = None, None
k, p2fit, fisher, invcov = None, None, None, None
fit_ftimesmodel = False
mcmc_package = None
free_params, fix_params, all_params = ['b_F', 'beta_F', 'q_1', 'k_p'], [], []
timestamp = "000_000"


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mydatadir", help="Data directory",
                        default=mydatadir)
    parser.add_argument("--prior-file",
                        help="Prior updates. Non-present keys "
                             "will be existing Planck18 defaults")
    parser.add_argument("--zeff", type=float, help="Effective redshift",
                        default=2.4)
    parser.add_argument("--fix-cosmology", action="store_true",
                        help="Do not sample cosmology.")
    parser.add_argument("--fit-hcds", action="store_true",
                        help="Fit for b_HCD and beta_HCD")
    parser.add_argument("--fit-SiIII", action="store_true",
                        help="Fit for b_SiIII")
    parser.add_argument("--fit-sky", action="store_true",
                        help="Fit the sky model")
    parser.add_argument("--use-camb", action="store_true",
                        help="Use CAMB instead of cosmopower emu.")
    parser.add_argument("--mock-truth", action="store_true",
                        help="Replace data with exact p3d model.")
    parser.add_argument("--fit-pfid", action="store_true",
                        help="Fit the fiducial power spectrum.")
    parser.add_argument("--fit-FtimesModel", action="store_true",
                        help="Fit Fisher . model")
    parser.add_argument("--marg-p1d", action="store_true")
    parser.add_argument("--nwalkers", type=int, default=nwalkers,
                        help="Number of walkers.")
    parser.add_argument("--nproc", type=int, default=nproc,
                        help="Number of processes.")
    parser.add_argument("--nsamples", type=int, default=nsamples,
                        help="Number of samples.")
    parser.add_argument("--progbar", action="store_true")
    parser.add_argument("--vectorize", action="store_true",
                        help="Use vectorized log_prob")
    parser.add_argument("--mcmc", choices=['emcee', 'zeus'], default='emcee',
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
    global k, p2fit, fisher, invcov, fit_ftimesmodel
    global mcmc_package

    nwalkers = args.nwalkers
    nproc = args.nproc
    nsamples = args.nsamples
    progbar = args.progbar
    vectorize = args.vectorize
    fit_ftimesmodel = args.fit_FtimesModel
    use_mp = (nproc > 1) & (not vectorize)
    if vectorize and (nproc > 1):
        print("Vectorization disables multiprocessing.")

    _, k, _, _, pfid, p2fit, _, _, fisher, _ = readP3dData(args.mydatadir, nl)
    model = LyaP3DArinyoModel(args.zeff, use_camb=args.use_camb)
    model.cacheK(k)
    p2fit = p2fit.ravel()

    if args.marg_p1d:
        model.margLyaP1D()
        free_params.append('a_p1d')

    base_cosmo = model.broadcastKwargs(**model.initial.copy())
    if args.mock_truth:
        p2fit = model.getPls(**base_cosmo)[0].ravel()
    elif args.fit_pfid:
        p2fit = pfid.ravel()

    if fit_ftimesmodel:
        invcov = np.linalg.inv(fisher)
        p2fit = fisher.dot(p2fit)
    else:
        invcov = fisher

    if args.fix_cosmology:
        model.fixCosmology(**base_cosmo)
    else:
        free_params += ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']

    if args.fit_hcds:
        free_params += ['b_hcd', 'beta_hcd']

    if args.fit_SiIII:
        free_params.append('b_SiIII_1207')

    if args.fit_sky:
        free_params += ['a_sky', 'sigma_sky']

    fix_params = [_ for _ in model._broadcasted_params if _ not in free_params]
    all_params = list(model.initial.keys())

    if args.prior_file:
        with open(args.prior_file, 'r') as f:
            model.prior.update(json.load(f))

    if args.mcmc == "emcee":
        import emcee as mcmc_package
    elif args.mcmc == "zeus":
        import zeus as mcmc_package
    else:
        raise Exception("Unsupported MCMC package")


def getModel(**cosmo):
    r = model.getPls(**cosmo)[0].ravel()
    if fit_ftimesmodel:
        return fisher.dot(r)
    return r


def cost(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args):
        new_cosmo[all_params[i]] = np.array([x])

    # new_cosmo = model.broadcastKwargs(**new_cosmo)
    y = getModel(**new_cosmo) - p2fit

    return y.dot(invcov.dot(y)) + model.getPrior(**new_cosmo)


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

    ndim = w.sum()
    result = np.empty(args.shape[0])

    y = getModel(**new_cosmo) - p2fit
    y = -0.5 * (np.sum(y.dot(invcov) * y, axis=1)
                + model.getPriorVector(ndim, **new_cosmo))

    result[w] = y
    result[~w] = -np.inf
    return result


def log_prob_nonvectorized(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args.T):
        new_cosmo[free_params[i]] = x

    new_cosmo = model.broadcastKwargs(**new_cosmo)

    for key, (b1, b2) in model.boundary.items():
        if (new_cosmo[key][0] <= b1) | (b2 <= new_cosmo[key][0]):
            return -np.inf

    y = getModel(**new_cosmo) - p2fit
    return -0.5 * (y.dot(invcov.dot(y)) + model.getPrior(**new_cosmo))


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

    # plot data vs best-fit
    p2plot = p2fit.reshape(nl, k.size)

    bestfit_cosmo = base_cosmo.copy()
    for key, x in mini.values.to_dict().items():
        bestfit_cosmo[key] = np.array([x])
    p_bestfit = getModel(**bestfit_cosmo).reshape(nl, k.size)
    err = np.sqrt(np.linalg.inv(invcov).diagonal()).reshape(nl, k.size)
    for i in range(nl):
        s = (i % 2) * 3e-3 + k
        plt.errorbar(
            s, k * p2plot[i], k * err[i], fmt='.', c=plt.cm.tab10(i),
            label=r"$\ell=$"f"{i*2}", alpha=1)
        plt.plot(s, k * p_bestfit[i], '-', c=plt.cm.tab10(i))
    plt.legend(fontsize='x-large', ncol=2)
    plt.xlabel(r"$k~$[Mpc$^{-1}$]")
    plt.ylabel(r"$k P_\ell(k)~$[Mpc$^{2}$]")
    plt.savefig(f"minimizer_data_vs_bestfit_{timestamp}.pdf",
                dpi=200, bbox_inches='tight')
    plt.close()

    nplots = len(free_params)
    nplots *= nplots - 1
    nplots //= 2
    ncols = round(np.sqrt(nplots))
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows),
        gridspec_kw={'hspace': 0.2, 'wspace': 0.3})

    j = 0
    for i, key1 in enumerate(free_params[:-1]):
        for key2 in free_params[i + 1:]:
            fitp1d.plotting.plotEllipseMinimizer(
                mini, key1, key2, model.param_labels, 'tab:blue',
                ax=axs[j // ncols, j % ncols],
                alpha=0.6, box=True, truth=model.initial, prior=model.prior
            )
            j += 1

    plt.savefig(f"minimizer-ellipses_{timestamp}.pdf",
                dpi=200, bbox_inches='tight')
    plt.close()


def sample(drop=400, thin=40):
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

    samples = sampler.get_chain(flat=True)
    np.savetxt(f"chains_{timestamp}_p3d.txt",
               np.column_stack((samples, sampler.get_log_prob(flat=True))),
               header=' '.join(free_params) + ' log_like')

    samples = sampler.get_chain(flat=True, thin=thin, discard=drop)
    samples = MCSamples(samples=samples, names=free_params, label="P3D",
                        labels=[model.param_labels[_] for _ in free_params])

    prior_keys = list(model.prior.keys())
    prior_chains = np.array([
        np.random.default_rng().normal(
            loc=base_cosmo[key], scale=model.prior[key], size=20000
        ) for key in prior_keys
    ]).T
    np.savetxt(f"chains_{timestamp}_prior.txt",
               prior_chains, header=' '.join(free_params))
    prior_chains = MCSamples(
        samples=prior_chains, names=prior_keys, label="Prior",
        labels=[model.param_labels[_] for _ in prior_keys])

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        [samples, prior_chains],
        filled=[True, False],
        contour_colors=["tab:blue", 'k'])
    plt.savefig(f"corner_plot_{timestamp}.pdf", dpi=200, bbox_inches='tight')
    plt.close()

    lya_nuis = [_ for _ in model._lya_nuis if _ in free_params]
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        [samples, prior_chains], lya_nuis,
        filled=[True, False],
        contour_colors=["tab:blue", 'k'])
    plt.savefig(f"corner_plot_lya_nuis_{timestamp}.pdf",
                dpi=200, bbox_inches='tight')
    plt.close()

    cosmo_p = [_ for _ in model._cosmo_names if _ in free_params]
    if cosmo_p:
        g = plots.get_subplot_plotter()
        g.triangle_plot(
            [samples, prior_chains], cosmo_p,
            filled=[True, False],
            contour_colors=["tab:blue", 'k'])
        plt.savefig(f"corner_plot_cosmo_{timestamp}.pdf",
                    dpi=200, bbox_inches='tight')
        plt.close()


def main():
    global timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mp.set_start_method('fork')
    args = getParser().parse_args()
    setGlobals(args)
    minimize()
    if nsamples > 0:
        sample()
