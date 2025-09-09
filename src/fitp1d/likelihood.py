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

        elif fname_power.endswith(".fits") or fname_power.endswith(".fits.gz"):
            self.psdata = DetailedData.fromP1dFitsFile(fname_power)
            self.ndata = self.psdata.size

        else:
            raise Exception("File type not recognized")

    @property
    def names(self):
        return self.p1dmodel.names

    @property
    def initial(self):
        return self.p1dmodel.initial

    @property
    def boundary(self):
        return self.p1dmodel.boundary

    @property
    def prior(self):
        return self.p1dmodel.prior

    @property
    def param_labels(self):
        return self.p1dmodel.param_labels

    def __init__(
            self, fname_power,
            use_camb=False, use_cosmopower=False, cp_model_dir=None,
            use_simple_lya_model=False,
            model_ions=["Si-II", "Si-III", "O-I"], doublet_ions=['C-IV'],
            turn_off_x_ion_terms=False, free_ion_boost=False,
            per_transition_bias=False,
            add_reso_bias=False, add_reso_var=False,
            fname_cov=None, cov=None, forecast=False,
            fit_scaling_systematics=False,
            fit_poly_order=-1,
            hcd_systems=None
    ):
        self.readData(fname_power, fname_cov, cov)

        if fit_scaling_systematics:
            syst = self.psdata.data_table.dtype.names
        else:
            syst = []
        self.p1dmodel = fitp1d.model.CombinedModel(
            syst, use_camb, use_cosmopower, cp_model_dir,
            model_ions=model_ions, per_transition_bias=per_transition_bias,
            turn_off_x_ion_terms=turn_off_x_ion_terms,
            free_ion_boost=free_ion_boost, doublet_ions=doublet_ions,
            hcd_systems=hcd_systems,
            add_reso_bias=add_reso_bias, add_reso_var=add_reso_var)
        if use_simple_lya_model:
            self.p1dmodel.useSimpleLyaModel()

        if fit_poly_order > 0:
            self.p1dmodel.addPolynomialP1dTerms(fit_poly_order)

        self.fixed_params = []
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
            self.initial.update({'B': 0, 'beta': 0, 'k1': np.inf})
            self.fixParam("B")
            self.fixParam("beta")
            self.fixParam("k1")

        if fit_poly_order > 0:
            self.initial["PMC0"] = 1.0
            self.fixParam("PMC0")

        self._fixed_cosmology = False

    @property
    def free_params(self):
        return [x for x in self.names if x not in self.fixed_params]

    def resetMini(self):
        for key, boun in self.boundary.items():
            self._mini.limits[key] = boun

        for key in self.free_params:
            self._mini.values[key] = self.initial[key]

    def resetBoundary(self, gp=5.0):
        centers = self._mini.values.to_dict()
        sigmas = self._mini.errors.to_dict()

        for i, par in enumerate(self.names):
            x1 = max(self.boundary[par][0], centers[par] - gp * sigmas[par])
            x2 = min(self.boundary[par][1], centers[par] + gp * sigmas[par])

            self.boundary[par] = (x1, x2)

    def fixCosmology(self):
        self._fixed_cosmology = True
        for key in self.p1dmodel._models['lya']._cosmo_names:
            self.fixParam(key)

    def fixParam(self, key):
        idx = self.names.index(key)

        if key not in self.fixed_params:
            self.fixed_params.append(key)
            self._free_idx.remove(idx)

        self._mini.fixed[key] = True

    def releaseParam(self, key):
        self._mini.fixed[key] = False
        self.fixed_params.remove(key)
        idx = self.names.index(key)
        self._free_idx.append(idx)
        self._free_idx.sort()

    def setupZbin(self, z, kmin=1e-3, kmax=None):
        if z is None:
            assert self.p1dmodel is not None
            return

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

        if self._fixed_cosmology:
            self.p1dmodel.fixCosmology(**self.initial)

        for key in self.fixed_params:
            idx = self.names.index(key)
            self._mini.values[key] = self.initial[key]
            self._new_args[idx] = self._mini.values[key]

    def fitDataBin(
            self, z, kmin=1e-3, kmax=None, print_info=False,
            interm_fix_keys=[], auto_inflate_errs=False
    ):
        print(f"Fitting redshift bin {z:.1f}")
        self.setupZbin(z, kmin, kmax)

        if interm_fix_keys:
            for _ in interm_fix_keys:
                self._mini.fixed[_] = True

            self._mini.migrad()

            for _ in reversed(self.free_params):
                self._mini.fixed[_] = False
                self._mini.migrad()

        self._mini.migrad()
        chi2 = self._mini.fval
        ndof = self._data.size - self._mini.nfit
        chi2_nu = chi2 / ndof
        print(f"Chi2 / dof= {chi2:.1f} / {ndof:d} = {chi2_nu:.2f}")

        if print_info:
            print(self._mini)

        if auto_inflate_errs and chi2_nu > 1.0:
            if self._cov is None:
                self._invcov = self._data['e_stat']**-2 / chi2_nu
                f = chi2_nu
            else:
                f = ((chi2_nu - 1.0) / 2.0)
                di = self._cov.diagonal()
                self._cov += f * np.diag(di)
                self._invcov = np.linalg.inv(self._cov)

            print(f"High chi-square auto inflating diagonals by {f:.4f}")
            self._mini.migrad()

        chi2 = self._mini.fval
        ndof = self._data.size - self._mini.nfit
        chi2_nu_2 = chi2 / ndof

        print(f"Chi2 / dof= {chi2:.1f} / {ndof:d} = {chi2_nu_2:.2f}")
        if print_info:
            print(self._mini)
        print("------------------------------")

        return chi2_nu

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
            chi2 += diff.dot(self._invcov.dot(diff))

        return chi2

    def likelihood(self, args):
        for i, par in enumerate(self.free_params):
            x1, x2 = self.boundary[par]

            if args[i] <= x1 or args[i] >= x2:
                return -np.inf

        self._new_args[self._free_idx] = args

        return -0.5 * self.chi2(*self._new_args)

    def likelihood_noprior_no_bounds(self, kwargs):
        # self._new_args[self._free_idx] = args
        # kwargs = {par: args[i] for i, par in enumerate(self.names)}
        x = self.p1dmodel.getIntegratedModel(**kwargs) - self._data['p_final']

        if self._cov is None:
            return -0.5 * np.sum(x**2 * self._invcov)
        else:
            return -0.5 * x.dot(self._invcov.dot(x))

    def sample(
            self, label, z=None, kmin=1e-3, kmax=None,
            nwalkers=8, nsamples=4000, discard=1000, thin=15
    ):
        self.setupZbin(z, kmin, kmax)
        ndim = len(self.free_params)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)

        rshift = 1e-2 * np.random.default_rng().normal(size=(nwalkers, ndim))
        p0 = self._mini.values[self.free_params] + rshift
        # self.resetBoundary()

        _ = sampler.run_mcmc(p0, nsamples, progress=True)

        try:
            tau = sampler.get_autocorr_time()
            print("Auto correlations:", tau)
        except Exception as e:
            print(e)
            pass

        samples = MCSamples(
            samples=sampler.get_chain(discard=discard, thin=thin, flat=True),
            names=self.free_params, label=label
        )

        samples.paramNames.setLabels(
            [self.param_labels[_] for _ in self.free_params])

        return samples

    def sampleNautilus(
            self, label, z, kmin=1e-3, kmax=None, nlive=1000, pool=4,
            verbose=True
    ):
        from nautilus import Prior, Sampler
        from scipy.stats import truncnorm

        self.setupZbin(z, kmin, kmax)

        prior = Prior()
        for key in self.names:
            loc = self.initial[key]
            if key in self.fixed_params:
                prior.add_parameter(key, dist=loc)

            elif key in self.prior:
                x1, x2 = self.boundary[key]
                s = self.prior[key]
                a, b = (x1 - loc) / s, (x2 - loc) / s
                prior.add_parameter(key, dist=truncnorm(
                    loc=loc, scale=s, a=a, b=b)
                )

            elif key in self.boundary:
                prior.add_parameter(key, dist=self.boundary[key])

            else:
                prior.add_parameter(key)

        sampler = Sampler(
            prior, self.likelihood_noprior_no_bounds,
            n_live=nlive, pass_dict=True, pool=pool)
        sampler.run(verbose=verbose)
        points, log_w, log_l = sampler.posterior()

        samples = MCSamples(
            samples=points, weights=np.exp(log_w), loglikes=log_l,
            names=self.free_params, label=label
        )

        samples.paramNames.setLabels(
            [self.param_labels[_] for _ in self.free_params])

        return samples
