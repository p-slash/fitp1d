import argparse
import multiprocessing as mp

import numpy as np
from astropy.cosmology import Planck18
from fitp1d.xcmb import getMpc2Kms

import emcee

myb1ddatadir = "/dvs_ro/cfs/cdirs/desicollab/users/naimgk/CMBxPLya/v3x/forecast"
mycosmopowerdir = "/dvs_ro/u1/n/naimgk/Repos/cosmopower"
mywienerfilter = "/dvs_ro/u1/n/naimgk/Repos/plyacmb/src/plyacmb/data/mv_wiener_filter_planck18.txt"
zeff = 2.4
nwalkers = 128
nproc = 128
nsteps = 4000
scale_invcov_b1d = 5
scale_invcov_p1d = 1
progbar = True
model = None
boundary = None

free_params = [
    'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s',
    'b_F', 'beta_F', 'k_p', 'q_1', 'log10T', 'nu0_th', 'nu1_th'
]

base_cosmo = {
    'omega_b': np.array([Planck18.Ob0 * Planck18.h**2]),
    'omega_cdm': np.array([Planck18.Odm0 * Planck18.h**2]),
    'h': np.array([Planck18.h]),
    'n_s': np.array([Planck18.meta['n']]),
    'ln10^{10}A_s': np.array([3.044]),
    'z': np.array([zeff]),
    'b_F': [-0.12], 'beta_F': [1.7],
    'k_p': np.array([8.]),  # Mpc^-1
    'q_1': np.array([0.7]), 'q_2': np.array([0]),
    'log10T': np.array([4.]),
    'nu0_th': np.array([1]), 'nu1_th': np.array([1.5])
}

planck_prior = {
    'omega_b': 0.00014,
    'omega_cdm': 0.00091,
    'h': 0.0042,
    'n_s': 0.0038,
    'ln10^{10}A_s': 0.014,
    'beta_F': 0.01
}

prior_cosmo = planck_prior.copy()
prior_cosmo.update({
    'omega_cdm': 0.005,
    'n_s': 0.02,
    'ln10^{10}A_s': 0.1,
    'k_p': 10.,
    'log10T': 0.1,
    'q_1': 0.1,
    'nu0_th': 0.1,
    'nu1_th': 0.1
})


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--myb1ddatadir", help="Data directory", default=myb1ddatadir)
    parser.add_argument(
        "--mycosmopowerdir", help="CosmoPower trained emulator", default=mycosmopowerdir)
    parser.add_argument(
        "--mywienerfilter", help="Wiener filter file", default=mywienerfilter)
    parser.add_argument("--zeff", type=float, help="Effective redshift", default=zeff)
    parser.add_argument("--nwalkers", type=int, default=nwalkers)
    parser.add_argument("--nsteps", type=int, default=nsteps)
    parser.add_argument("--nproc", type=int, default=nproc)
    parser.add_argument("--scale-invcov-b1d", type=float, default=scale_invcov_b1d)
    parser.add_argument("--scale-invcov-p1d", type=float, default=scale_invcov_p1d)
    parser.add_argument("--progbar", action="store_true")

    return parser


def read_data(myb1ddatadir=myb1ddatadir):
    k_b1d, base_b1d = np.loadtxt(f"{myb1ddatadir}/forecast_base_b1d_24_mpc.txt", unpack=True)
    b1d_invcov = np.loadtxt(f"{myb1ddatadir}/forecast_base_b1d_invcov_24_mpc.txt")

    k_p1d, base_p1d = np.loadtxt(f"{myb1ddatadir}/forecast_base_p1d_24_mpc.txt", unpack=True)
    p1d_invcov = np.loadtxt(f"{myb1ddatadir}/forecast_base_p1d_invcov_24_mpc.txt")

    mpc2kms = getMpc2Kms(zeff, **base_cosmo)
    k_b1d /= mpc2kms
    k_p1d /= mpc2kms
    base_b1d *= mpc2kms
    base_p1d *= mpc2kms
    b1d_invcov /= mpc2kms**2
    p1d_invcov /= mpc2kms**2

    return k_b1d, base_b1d, b1d_invcov, k_p1d, base_p1d, p1d_invcov


def log_prob_kms_p1d(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args):
        new_cosmo[free_params[i]] = np.array([x])

    for key, (b1, b2) in boundary.items():
        v = new_cosmo[key][0]
        if (v < b1) or (v > b2):
            return -np.inf

    prior = 0
    for key, s in prior_cosmo.items():
        prior += ((new_cosmo[key][0] - base_cosmo[key][0]) / s)**2

    c = model.getMpc2Kms(**new_cosmo)
    y = model.getP1dTrapz(k_p1d * c, **new_cosmo)[0] * c - base_p1d
    return -0.5 * (y.dot(p1d_invcov.dot(y)) + prior)


def log_prob_kms_joint(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args):
        new_cosmo[free_params[i]] = np.array([x])

    for key, (b1, b2) in boundary.items():
        v = new_cosmo[key][0]
        if (v < b1) or (v > b2):
            return -np.inf

    prior = 0
    for key, s in prior_cosmo.items():
        prior += ((new_cosmo[key][0] - base_cosmo[key][0]) / s)**2

    c = model.getMpc2Kms(**new_cosmo)
    y = model.getP1dTrapz(k_p1d * c, **new_cosmo)[0] * c - base_p1d
    cost = y.dot(p1d_invcov.dot(y)) + prior

    y = model.integrateB3dTrapz(k_b1d * c, **new_cosmo)[0] * c - base_b1d
    cost += y.dot(b1d_invcov.dot(y))
    return -0.5 * cost


def setGlobals(args):
    global zeff, nwalkers, nproc, nsteps, scale_invcov_p1d, scale_invcov_b1d
    global progbar, model, boundary
    global k_b1d, base_b1d, b1d_invcov, k_p1d, base_p1d, p1d_invcov

    zeff = args.zeff
    nwalkers = args.nwalkers
    nproc = args.nproc
    nsteps = args.nsteps
    scale_invcov_p1d = args.scale_invcov_p1d
    scale_invcov_b1d = args.scale_invcov_b1d
    progbar = args.progbar

    from fitp1d.xcmb import LyaxCmbModel
    model = LyaxCmbModel(
        args.zeff, args.mycosmopowerdir, args.mywienerfilter,
        nlnkbins=100, nwbins=10
    )

    boundary = model.boundary.copy()
    k_b1d, base_b1d, b1d_invcov, k_p1d, base_p1d, p1d_invcov = read_data(args.myb1ddatadir)
    b1d_invcov *= scale_invcov_b1d
    p1d_invcov *= scale_invcov_p1d


def main():
    args = getParser().parse_args()
    setGlobals(args)
    ndim = len(free_params)
    rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
    p0 = np.array([base_cosmo[_] for _ in free_params]).T + rshift

    with mp.Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_kms_p1d, pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=progbar)

        np.savetxt(
            f"chains_p1d_x{scale_invcov_p1d}.txt",
            sampler.get_chain(flat=True), header=' '.join(free_params))

        print("Joint likelihood.")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_kms_joint, pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=progbar)

        np.savetxt(
            f"chains_joint_px{scale_invcov_p1d}_bx{scale_invcov_b1d}.txt",
            sampler.get_chain(flat=True), header=' '.join(free_params))
