import numpy as np
import emcee
import iminuit
from getdist import MCSamples

from fitp1d.data import PSData
from fitp1d.model import CombinedModel


class P1DLikelihood():
    def readData(self, fname_power, fname_cov=None, cov=None):
        self.psdata = PSData(fname_power)
        self.ndata = self.psdata.size

        if fname_cov:
            self.psdata.readCovariance(fname_cov)
        elif cov is not None:
            self.psdata.cov = cov.copy()

    def __init__(
            self, add_reso_bias, add_var_reso,
            use_simple_lya_model=False,
            fname_power=None, fname_cov=None, cov=None,
            fname_noise=None
    ):
        self.p1dmodel = CombinedModel(add_reso_bias, add_var_reso)
        if use_simple_lya_model:
            self.p1dmodel.useSimpleLyaModel()

        self.names = self.p1dmodel.names
        self.free_params = self.names.copy()
        self.initial = self.p1dmodel.initial
        self.boundary = self.p1dmodel.boundary
        self.prior = self.p1dmodel.prior
        self.param_labels = self.p1dmodel.param_labels

        if fname_power is not None:
            self.readData(fname_power, fname_cov, cov)

        if fname_noise is not None:
            self.noisedata = PSData(fname_noise)
        else:
            self.noisedata = None

        self._data = None
        self._cov = None
        self._invcov = None
        self._mini = iminuit.Minuit(self.chi2, name=self.names, **self.initial)
        self._mini.errordef = 1
        self._mini.print_level = 1

        for key, boun in self.boundary.items():
            self._mini.limits[key] = boun

        if use_simple_lya_model:
            self.fixParam("B", 0)
            self.fixParam("beta", 0)
            self.fixParam("k1", 1e6)

        if fname_noise is None:
            self.fixParam("eta_noise")

    def fixParam(self, key, value=None):
        self.free_params = [x for x in self.free_params if x != key]

        self._mini.fixed[key] = True
        if value is not None:
            self._mini.values[key] = value

    def releaseParam(self, key):
        self._mini.fixed[key] = False
        self.free_params.append(key)

    def sample(self, label, nwalkers=32, nsamples=20000):
        ndim = len(self.free_params)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)

        rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
        p0 = self._mini.values[self.free_params] + rshift
        self.setPrior()

        _ = sampler.run_mcmc(p0, nsamples, progress=True)

        tau = sampler.get_autocorr_time()
        print("Auto correlations:", tau)

        samples = MCSamples(
            samples=sampler.get_chain(discard=1000, thin=15, flat=True),
            names=self.free_params, label=label
        )

        samples.paramNames.setLabels(
            [self.param_labels[_] for _ in self.free_params])

        return samples

    def fitDataBin(self, z, kmin=0, kmax=10):
        self._data, kedges, self._cov = self.psdata.getZBinVals(z, kmin, kmax)
        if self._cov is None:
            self._invcov = self._data['e']**-2
        else:
            self._invcov = np.linalg.inv(self._cov)

        self.p1dmodel.cache(kedges, z)
        if self.noisedata:
            ndat = self.noisedata.getZBinVals(
                z, kmin, kmax, get_pnoise=True, k2match=self._data['k']
            )[0]
            self.p1dmodel.setNoiseModel(ndat['p'])

        print(self._mini.migrad())

        chi2 = self._mini.fval
        ndof = self._data.size - self._mini.nfit
        print(f"Chi2 / dof= {chi2:.1f} / {ndof:d}")

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}

        pmodel = self.p1dmodel.getIntegratedModel(**kwargs)
        diff = pmodel - self._data['p']
        chi2 = 0
        for k, s in self.prior.items():
            chi2 += ((kwargs[k] - self.initial[k]) / s)**2

        if self._cov is None:
            chi2 += np.sum(diff**2 * self._invcov)
        else:
            chi2 += diff @ self._invcov @ diff

        return chi2

    def setPrior(self, gp=5.0):
        centers = self._mini.values.to_dict()
        sigmas = self._mini.errors.to_dict()

        for i, par in enumerate(self.names):
            x1 = max(self.boundary[par][0], centers[par] - gp * sigmas[par])
            x2 = min(self.boundary[par][1], centers[par] + gp * sigmas[par])

            self.boundary[par] = (x1, x2)

    def logPrior(self, *args):
        for i, par in enumerate(self.free_params):
            x1, x2 = self.boundary[par]

            if args[i] < x1 or args[i] > x2:
                return -np.inf

        return 0.

    def likelihood(self, args):
        lp = self.logPrior(*args)
        if not np.isfinite(lp):
            return -np.inf

        new_args = []
        j = 0
        for par in self.names:
            if par in self.free_params:
                new_args.append(args[j])
                j += 1
            else:
                new_args.append(self._mini.values[par])

        return -0.5 * self.chi2(*new_args)
