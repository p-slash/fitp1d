import numpy as np
import emcee
import iminuit
from getdist import MCSamples

from fitp1d.data import DetailedData
import fitp1d.model


class P1DLikelihood():
    def readData(self, fname_power, fname_cov=None, cov=None):
        if fname_power.endswith(".txt") or fname_power.endswith(".dat"):
            self.psdata = DetailedData.fromFile(fname_power)
            self.ndata = self.psdata.size

            if fname_cov:
                self.psdata.readCovariance(fname_cov, skiprows=0)
            elif cov is not None:
                self.psdata.setCovariance(cov)

        if fname_power.endswith(".fits") or fname_power.endswith(".fits.gz"):
            self.psdata = DetailedData.fromP1dFitsFile(fname_power)
            self.ndata = self.psdata.size

        else:
            raise Exception("File type not recognized")

    def __init__(
            self, fname_power, use_simple_lya_model=False,
            fname_cov=None, cov=None, forecast=False
    ):
        self.readData(fname_power, fname_cov, cov)

        self.p1dmodel = fitp1d.model.CombinedModel(
            self.psdata.data_table.dtype.names)
        if use_simple_lya_model:
            self.p1dmodel.useSimpleLyaModel()

        self.names = self.p1dmodel.names
        self.fixed_params = []
        self.initial = self.p1dmodel.initial
        self.boundary = self.p1dmodel.boundary
        self.prior = self.p1dmodel.prior
        self.param_labels = self.p1dmodel.param_labels
        self._new_args = np.empty(len(self.names))
        self._free_idx = list(np.arange(self._new_args.size))

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
            self.fixParam("k1", np.inf)

    @property
    def free_params(self):
        return [x for x in self.names if x not in self.fixed_params]

    def fixParam(self, key, value=None):
        self.fixed_params.append(key)
        idx = self.names.index(key)
        self._free_idx.remove(idx)

        self._mini.fixed[key] = True
        if value is not None:
            self._mini.values[key] = value

        self._new_args[idx] = self._mini.values[key]

    def releaseParam(self, key):
        self._mini.fixed[key] = False
        self.fixed_params.remove(key)
        idx = self.names.index(key)
        self._free_idx.append(idx)
        self._free_idx.sort()

    def sample(self, label, nwalkers=32, nsamples=20000):
        ndim = len(self.free_params)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)

        rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
        p0 = self._mini.values[self.free_params] + rshift
        self.resetBoundary()

        _ = sampler.run_mcmc(p0, nsamples, progress=True)

        try:
            tau = sampler.get_autocorr_time()
            print("Auto correlations:", tau)
        except Exception:
            pass

        samples = MCSamples(
            samples=sampler.get_chain(discard=1000, thin=15, flat=True),
            names=self.free_params, label=label
        )

        samples.paramNames.setLabels(
            [self.param_labels[_] for _ in self.free_params])

        return samples

    def fitDataBin(self, z, kmin=1e-3, kmax=None, print_info=False):
        if kmax is None:
            rkms = (
                fitp1d.model.LIGHT_SPEED * 0.8
                / (1 + z) / fitp1d.model.LYA_WAVELENGTH
            )
            kmax = np.pi / rkms / 2

        self._data, self._cov = self.psdata.getZBinVals(z, kmin, kmax)
        if self._cov is None:
            self._invcov = self._data['e_stat']**-2
        else:
            self._invcov = np.linalg.inv(self._cov)

        kedges = (self._data['k1'], self._data['k2'])
        self.p1dmodel.cache(kedges, z, self._data)

        self._mini.migrad()

        if print_info:
            chi2 = self._mini.fval
            ndof = self._data.size - self._mini.nfit
            print(f"Chi2 / dof= {chi2:.1f} / {ndof:d}")
            print(self._mini)

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}

        pmodel = self.p1dmodel.getIntegratedModel(**kwargs)
        diff = pmodel - self._data['p_final']
        chi2 = 0
        for k, s in self.prior.items():
            chi2 += ((kwargs[k] - self.initial[k]) / s)**2

        if self._cov is None:
            chi2 += np.sum(diff**2 * self._invcov)
        else:
            chi2 += diff @ self._invcov @ diff

        return chi2

    def resetBoundary(self, gp=5.0):
        centers = self._mini.values.to_dict()
        sigmas = self._mini.errors.to_dict()

        for i, par in enumerate(self.names):
            x1 = max(self.boundary[par][0], centers[par] - gp * sigmas[par])
            x2 = min(self.boundary[par][1], centers[par] + gp * sigmas[par])

            self.boundary[par] = (x1, x2)

    def likelihood(self, args):
        for i, par in enumerate(self.free_params):
            x1, x2 = self.boundary[par]

            if args[i] < x1 or args[i] > x2:
                return -np.inf

        self._new_args[self._free_idx] = args

        return -0.5 * self.chi2(*self._new_args)
