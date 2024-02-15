import multiprocessing as mp

import numpy as np
from astropy.cosmology import Planck18
from fitp1d.xcmb import LyaxCmbModel

import zeus

myb1ddatadir = "/global/cfs/cdirs/desicollab/users/naimgk/CMBxPLya/v3x/forecast"
mycosmopowerdir = "/global/cfs/cdirs/desicollab/users/naimgk/CMBxPLya/v3x/forecast"
mywienerfilter = "/global/homes/n/naimgk/Repos/plyacmb/src/plyacmb/data/mv_wiener_filter_planck18.txt"
zeff = 2.4
nwalkers = 64
nproc = 64
nsteps = 10000
usePool = True
useKms = True
scale_cov = 5

base_cosmo = {
    'omega_b': np.array([Planck18.Ob0 * Planck18.h**2]),
    'omega_cdm': np.array([Planck18.Odm0 * Planck18.h**2]),
    'h': np.array([Planck18.h]),
    'n_s': np.array([Planck18.meta['n']]),
    'ln10^{10}A_s': np.array([3.044]),
    'z': np.array([zeff]),
    'b_F': np.array([-0.081]), 'beta_F': np.array([1.67]),
    'k_p': np.array([11.2])  # Mpc^-1
}

planck_prior = {
    'omega_b': 0.00014,
    'omega_cdm': 0.00091,
    'h': 0.0042,
    'n_s': 0.0038,
    'ln10^{10}A_s': 0.014,
    'k_p': 50.,
    'beta_F': 0.01
}

prior_cosmo = planck_prior.copy()
prior_cosmo.update({
    'omega_cdm': 0.02,
    'n_s': 0.05
})

model = LyaxCmbModel(
    zeff, mycosmopowerdir, mywienerfilter,
    nlnkbins=100, nwbins=10
)

boundary = model.boundary.copy()

k, base_b1d = np.loadtxt(f"{myb1ddatadir}/base_b1d_24_mpc.txt", unpack=True)
invcov = scale_cov * np.loadtxt(f"{myb1ddatadir}/base_invcov_24_mpc.txt")

free_params = [
    'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s',
    'b_F', 'beta_F', 'k_p'
]


def _log_prob_mpc(args):
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

    y = model.integrateB3dTrapz(k, **new_cosmo)[0] - base_b1d
    return -0.5 * (y.dot(invcov.dot(y)) + prior)


def _log_prob_kms(args):
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
    y = model.integrateB3dTrapz(k * c, **new_cosmo)[0] * c - base_b1d
    return -0.5 * (y.dot(invcov.dot(y)) + prior)


def log_prob_vectorized(args):
    new_cosmo = base_cosmo.copy()
    for i, x in enumerate(args.T):
        new_cosmo[free_params[i]] = x

    new_cosmo = model.broadcastKwargs(**new_cosmo)

    w = np.ones(args.shape[0], dtype=bool)
    for key, (b1, b2) in boundary.items():
        w &= (b1 < new_cosmo[key]) & (new_cosmo[key] < b2)

    for key in new_cosmo:
        new_cosmo[key] = new_cosmo[key][w]

    result = np.empty(args.shape[0])
    y = model.integrateB3dTrapz(k, **new_cosmo) - base_b1d
    y = -0.5 * np.sum(y.dot(invcov) * y, axis=1)
    result[w] = y
    result[~w] = -np.inf
    return result


if useKms:
    mpc2kms = model.getMpc2Kms(**base_cosmo)
    k /= mpc2kms
    base_b1d *= mpc2kms
    invcov /= mpc2kms**2
    log_prob = _log_prob_kms
else:
    log_prob = _log_prob_mpc


def main():
    ndim = len(free_params)
    rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
    p0 = np.array([base_cosmo[_] for _ in free_params]).T + rshift

    if usePool:
        with mp.Pool(processes=nproc) as pool:
            sampler = zeus.EnsembleSampler(
                nwalkers, ndim, log_prob, vectorize=False, pool=pool
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, log_prob_vectorized, vectorize=True
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    np.savetxt(f"chains_x{scale_cov}.txt", sampler.get_chain(), header=' '.join(free_params))
    tau = sampler.get_autocorr_time()
    np.savetxt(f"autocorr_x{scale_cov}.txt", tau)
    print(
        f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}"
    )
    print(f"Mean autocorrelation time: {np.mean(tau):.1f} steps")


if __name__ == '__main__':
    main()
