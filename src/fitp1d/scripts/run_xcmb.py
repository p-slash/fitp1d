import multiprocessing as mp

import numpy as np
from astropy.cosmology import Planck18
from fitp1d.xcmb import LyaxCmbModel

import zeus

myb1ddatadir = "/global/cfs/cdirs/desicollab/users/naimgk/CMBxPLya/v3x/forecast"
mycosmopowerdir = "/global/cfs/cdirs/desicollab/users/naimgk/CMBxPLya/v3x/forecast"
mywienerfilter = "/global/homes/n/naimgk/Repos/plyacmb/src/plyacmb/data/mv_wiener_filter_planck18.txt"
zeff = 2.4
nwalkers = 128
nproc = 128
nsteps = 4000
scale_cov = 5
progbar = True

base_cosmo = {
    'omega_b': np.array([Planck18.Ob0 * Planck18.h**2]),
    'omega_cdm': np.array([Planck18.Odm0 * Planck18.h**2]),
    'h': np.array([Planck18.h]),
    'n_s': np.array([Planck18.meta['n']]),
    'ln10^{10}A_s': np.array([3.044]),
    'z': np.array([zeff]),
    'b_F': np.array([-0.15]), 'beta_F': np.array([1.67]),
    'k_p': np.array([8.0])  # Mpc^-1
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
    'omega_cdm': 0.005,
    'n_s': 0.02,
    'ln10^{10}A_s': 0.1
})

model = LyaxCmbModel(
    zeff, mycosmopowerdir, mywienerfilter, nlnkbins=100, nwbins=10
)

boundary = model.boundary.copy()

k, base_b1d = np.loadtxt(f"{myb1ddatadir}/forecast_base_b1d_24_mpc.txt", unpack=True)
b1d_invcov = scale_cov * np.loadtxt(f"{myb1ddatadir}/forecast_base_b1d_invcov_24_mpc.txt")

_, base_p1d = np.loadtxt(f"{myb1ddatadir}/forecast_base_p1d_24_mpc.txt", unpack=True)
p1d_invcov = np.loadtxt(f"{myb1ddatadir}/forecast_base_p1d_invcov_24_mpc.txt") / scale_cov

mpc2kms = model.getMpc2Kms(**base_cosmo)
k /= mpc2kms
base_b1d *= mpc2kms
base_p1d *= mpc2kms
b1d_invcov /= mpc2kms**2
p1d_invcov /= mpc2kms**2

free_params = [
    'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s',
    'b_F', 'beta_F', 'k_p'
]


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
    y = model.getP1dTrapz(k * c, **new_cosmo)[0] * c - base_p1d
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
    y = model.getP1dTrapz(k * c, **new_cosmo)[0] * c - base_p1d
    cost = y.dot(p1d_invcov.dot(y)) + prior

    y = model.integrateB3dTrapz(k * c, **new_cosmo)[0] * c - base_b1d
    cost += y.dot(b1d_invcov.dot(y))
    return -0.5 * cost


def main():
    ndim = len(free_params)
    rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
    p0 = np.array([base_cosmo[_] for _ in free_params]).T + rshift

    with mp.Pool(processes=nproc) as pool:
        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, log_prob_kms_p1d, pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=progbar)

        np.savetxt(
            f"chains_p1d_x{scale_cov}.txt",
            sampler.get_chain(flat=True), header=' '.join(free_params))
        tau = sampler.get_autocorr_time()
        np.savetxt(f"autocorr_p1d_x{scale_cov}.txt", tau)
        print(
            f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}"
        )
        print(f"Mean autocorrelation time: {np.mean(tau):.1f} steps")

        print("Joint likelihood.")
        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, log_prob_kms_joint, pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=progbar)

        np.savetxt(
            f"chains_joint_x{scale_cov}.txt",
            sampler.get_chain(flat=True), header=' '.join(free_params))
        tau = sampler.get_autocorr_time()
        np.savetxt(f"autocorr_joint_x{scale_cov}.txt", tau)
        print(
            f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}"
        )
        print(f"Mean autocorrelation time: {np.mean(tau):.1f} steps")


if __name__ == '__main__':
    main()
