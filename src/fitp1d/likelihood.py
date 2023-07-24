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
            fname_power=None, fname_cov=None, cov=None
    ):
        self.p1dmodel = CombinedModel(add_reso_bias, add_var_reso)

        self.names = self.p1dmodel.names
        self.free_params = self.names.copy()
        self.initial = self.p1dmodel.initial
        self.boundary = self.p1dmodel.boundary
        self.param_labels = self.p1dmodel.param_labels

        if fname_power is not None:
            self.readData(fname_power, fname_cov, cov)

        self._data = None
        self._cov = None
        self._invcov = None
        self._mini = iminuit.Minuit(self.chi2, name=self.names, **self.initial)
        self._mini.errordef = 1
        self._mini.print_level = 1

        self.fixParam("B", 0)
        self.fixParam("beta", 0)
        self.fixParam("k1", 1e6)

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
        print(self._mini.migrad())

        chi2 = self._mini.fval
        ndof = self._data.size - self._mini.nfit
        print(f"Chi2 / dof= {chi2:.1f} / {ndof:d}")

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}

        pmodel = self.p1dmodel.getIntegratedModel(**kwargs)
        diff = pmodel - self._data['p']

        if self._cov is None:
            return np.sum(diff**2 * self._invcov)

        return diff @ self._invcov @ diff

    def setPrior(self, gp=6.0):
        centers = self._mini.values.to_dict()
        sigmas = self._mini.errors.to_dict()

        for i, par in enumerate(self.names):
            x1 = centers[par] - gp * sigmas[par]
            x2 = centers[par] + gp * sigmas[par]

            if par == 'k1':
                x1 = max(1e-6, x1)
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
