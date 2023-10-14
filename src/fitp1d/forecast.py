from astropy.cosmology import Planck18
import camb
import numpy as np
import emcee
import iminuit
from getdist import MCSamples

from fitp1d.data import DetailedData
import fitp1d.model


def getHubbleZ(z, H0, Ode0):
    return H0 * np.sqrt(Ode0 + (1 - Ode0) * (1 + z)**3)


class LyaP1DArinyoModel2(fitp1d.model.Model):
    def __init__(self):
        super().__init__()

        self._cosmo_names = ['ln10As', 'ns', 'mnu', 'Ode0', 'H0']
        self.names = [
            'blya_A', 'blya_n',
            'beta_A', 'beta_n',
            'q1', 'kv', 'av', 'bv',
            'kp_A', 'kp_n'
        ] + self._cosmo_names

        # Table 6 of Bourboux et al. 2020, http://arxiv.org/abs/2007.08995
        self._bias_zeff = 2.334
        self.initial = {
            'blya_A': -0.136,
            'blya_n': 3.231,
            'beta_A': 1.82,
            'beta_n': -2.0,
            'q1': 0.50,
            'kv': 0.2,
            'av': 0.37,
            'bv': 1.0,
            'kp_A': 45.0,
            'kp_n': -0.1,
            'ln10As': 3.044,
            'ns': Planck18.meta['n'],
            'mnu': 8.,
            'Ode0': Planck18.Ode0,
            'H0': Planck18.H0.value
        }

        self.param_labels = {
            "blya_A": r"b_\mathrm{Lya, 0}", "blya_n": r"b_\mathrm{Lya, 1}",
            "beta_A": r"\beta_\mathrm{Lya, 0}", "beta_n": r"\beta_\mathrm{Lya, 1}",
            "q1": r"q_1", "kv": r"k_\nu", "av": r"a_\nu", "bv": r"b_\nu",
            "kp_A": r"k_{p, 0}", "kp_n": r"k_{p, 1}",
            "ln10As": r"$\ln(10^{10} A_s)$",
            "ns": r"$n_s$", "mnu": r"$\sum m_\nu \times 10^2$",
            "Ode0": r"$\Omega_\Lambda$", "H0": r"$H_0$"
        }

        self.boundary = {
            'blya_A': (-1, 0), 'blya_n': (0, 5),
            'beta_A': (0.5, 3.0), 'beta_n': (-5, -1),
            'q1': (0.0, 5.),
            'kv': (0., 5.),
            'av': (0.0, 5.),
            'bv': (0.1, 5.),
            'kp_A': (0., 200.), 'kp_n': (-2.0, 0.1),
            'ln10As': (2., 4.),
            'ns': (0.94, 1.),
            'mnu': (0., 100.),
            'Ode0': (0.5, 0.9),
            'H0': (50., 100.)
        }

        self._kperp, self._dlnkperp = np.linspace(-4, 3, 700, retstep=True)
        self._kperp = np.exp(self._kperp)[:, np.newaxis, np.newaxis]
        self._kperp2pi = self._kperp**2 / (2 * np.pi)

        self.zlist = None
        self._p3dlin = None
        self._Delta2 = None
        # self._k1d_Mpc = None
        self._k3d = None
        self._mu = None
        self.Mpc2kms = None
        self.ndata = None

        self._cCosmo = {}

    def cacheZAndK(self, kedges_tuple_list, zlist):
        if self.zlist is not None and np.allclose(self.zlist, zlist):
            return

        self.zlist = zlist
        self.kedges_tuple_list = kedges_tuple_list

    def newcosmo(self, **kwargs):
        if (
                self._cCosmo and all((
                    np.isclose(cC, kwargs[key], rtol=1e-8)
                    for key, cC in self._cCosmo.items()
                ))
        ):
            return

        for key in self._cosmo_names:
            self._cCosmo[key] = kwargs[key]

        H0, Ode0 = kwargs['H0'], kwargs['Ode0']
        h = H0 / 100.

        self.Mpc2kms = np.asarray([
            getHubbleZ(z, H0, Ode0) / (1 + z) for z in self.zlist
        ])

        # self._k1d_Mpc = []
        self._k3d_Mpc = []
        self._mu = []
        for i, (k1, k2) in enumerate(self.kedges_tuple_list):
            kfine = np.linspace(k1, k2, fitp1d.model._NSUB_K_, endpoint=False).T
            _k1d_Mpc = (kfine * self.Mpc2kms[i])[np.newaxis, :]
            _k3d_Mpc = np.sqrt(self._kperp**2 + _k1d_Mpc**2)

            self._mu.append(_k1d_Mpc / _k3d_Mpc)
            # self._k1d_Mpc.append(_k1d_Mpc)
            self._k3d_Mpc.append(_k3d_Mpc)

        camb_params = camb.set_params(
            redshifts=sorted(self.zlist, reverse=True),
            WantCls=False, WantScalars=False,
            WantTensors=False, WantVectors=False,
            WantDerivedParameters=False,
            WantTransfer=True, kmax=20,
            omch2=(1 - Ode0 - Planck18.Ob0) * h**2,
            ombh2=Planck18.Ob0 * h**2,
            omk=0.,
            H0=H0,
            ns=kwargs['ns'],
            As=np.exp(kwargs['ln10As']) * 1e-10,
            mnu=kwargs['mnu'] / 100.
        )
        camb_results = camb.get_results(camb_params)

        camb_interp = camb_results.get_matter_power_interpolator(
            nonlinear=False,
            hubble_units=False,
            k_hunit=False)

        self._p3dlin = []
        self._Delta2 = []
        for i, z in enumerate(self.zlist):
            _p3dlin = camb_interp.P(z, self._k3d_Mpc[i])

            self._p3dlin.append(_p3dlin)
            self._Delta2.append(_p3dlin * self._k3d_Mpc[i]**3 / 2 / np.pi**2)

    def getP1DListKms(self, **kwargs):
        self.newcosmo(**kwargs)
        p1d_list = []

        for i, z in enumerate(self.zlist):
            xx = ((1 + z) / (1 + self._bias_zeff))
            blya = kwargs['blya_A'] * xx**kwargs['blya_n']
            beta = kwargs['beta_A'] * xx**kwargs['beta_n']
            kp = kwargs['kp_A'] * xx**kwargs['kp_n']
            bias_rsd = (blya * (1 + beta * self._mu[i]**2))**2
            t1 = (
                (self._k3d_Mpc[i] / kwargs['kv'])**kwargs['av']
                * self._mu[i]**kwargs['bv']
            )
            t2 = (self._k3d_Mpc[i] / kp)**2
            Fnl = np.exp(kwargs['q1'] * self._Delta2[i] * (1 - t1) - t2)

            p3d = self._p3dlin[i] * bias_rsd * Fnl
            p1d_Mpc = np.trapz(p3d * self._kperp2pi, dx=self._dlnkperp, axis=0)
            p1d_list.append(p1d_Mpc.mean(axis=1) * self.Mpc2kms[i])

        return p1d_list


class P1DLikelihood2():
    def readData(self, fname_power, fname_cov=None, cov=None):
        self.psdata = DetailedData(fname_power)
        self.ndata = self.psdata.size

        if fname_cov:
            self.psdata.readCovariance(fname_cov, skiprows=0)
        elif cov is not None:
            self.psdata.setCovariance(cov)

        rkms = (
            fitp1d.model.LIGHT_SPEED * 0.8
            / (1 + self.psdata.data_table['z'])
            / fitp1d.model.LYA_WAVELENGTH
        )
        kmax = np.pi / rkms / 2

        self.zlist, self._data, self._cov = self.psdata.getZBinAsList(5e-4, kmax)

        self._invcov = []
        for i, c in enumerate(self._cov):
            if self._cov is None:
                self._invcov.append(self._data[i]['e_stat']**-2)
            else:
                self._invcov.append(np.linalg.inv(c))

    def __init__(
            self, fname_power, fname_cov=None, cov=None
    ):
        self.readData(fname_power, fname_cov, cov)

        self.p1dmodel = LyaP1DArinyoModel2()

        self.names = self.p1dmodel.names
        self.fixed_params = []
        self.initial = self.p1dmodel.initial
        self.boundary = self.p1dmodel.boundary
        self.prior = self.p1dmodel.prior
        self.param_labels = self.p1dmodel.param_labels
        self._new_args = np.empty(len(self.names))
        self._free_idx = list(np.arange(self._new_args.size))

        kedges_tuple_list = [(x['k1'], x['k2']) for x in self._data]
        self.p1dmodel.cacheZAndK(kedges_tuple_list, self.zlist)

    def setFiducial(self):
        pmodel_fid = self.p1dmodel.getP1DListKms(**self.initial)
        for i, pm in enumerate(pmodel_fid):
            self._data[i]['p_final'] = pm

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}

        chi2 = 1e-8
        for k, s in self.prior.items():
            chi2 += ((kwargs[k] - self.initial[k]) / s)**2

        pmodel = self.p1dmodel.getP1DListKms(**kwargs)
        for i, (d, icov, pm) in enumerate(zip(self._data, self._invcov, pmodel)):
            diff = pm - d['p_final']
            if icov.ndim == 1:
                chi2 += np.dot(diff**2, icov)
            else:
                chi2 += diff.dot(icov.dot(diff))

        return chi2

    def setMinimizer(self):
        mini_initials = {
            key: value + 1e-3 * np.random.default_rng().normal()
            for key, value in self.initial.items()
        }
        self._mini = iminuit.Minuit(self.chi2, name=self.names, **mini_initials)
        self._mini.errordef = 1
        self._mini.print_level = 1

        for key, boun in self.boundary.items():
            self._mini.limits[key] = boun

        for key in self.fixed_params:
            self._mini.fixed[key] = True
            self._mini.values[key] = self.initial[key]

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

    def fit(self, print_info=True):
        self._mini.migrad()

        if print_info:
            chi2 = self._mini.fval
            ndof = np.sum([x.size for x in self._data]) - self._mini.nfit
            print(f"Chi2 / dof= {chi2:.1f} / {ndof:d}")
            print(self._mini)

    def sample(
            self, label, nwalkers=32, nsamples=20000, discard=1000, thin=10,
            check_autocorr=False, pool=None
    ):
        ndim = len(self.free_params)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.likelihood, pool=pool)

        rshift = 1e-4 * np.random.default_rng().normal(size=(nwalkers, ndim))
        p0 = self._mini.values[self.free_params] + rshift
        self.resetBoundary()

        _ = sampler.run_mcmc(p0, nsamples, progress=True)

        if check_autocorr:
            tau = sampler.get_autocorr_time()
            print("Auto correlations:", tau)

        samples = MCSamples(
            samples=sampler.get_chain(discard=discard, thin=thin, flat=True),
            names=self.free_params, label=label
        )

        samples.paramNames.setLabels(
            [self.param_labels[_] for _ in self.free_params])

        return samples

    def sampleUnest(
            self, log_dir, use_slice=True, nlive=200, slice_steps=50,
            resume='resume', mpi_rank=0
    ):
        from ultranest import ReactiveNestedSampler
        from ultranest.stepsampler import (
            SliceSampler, generate_mixture_random_direction)

        def _uniform(c, x1, x2):
            return c * (x2 - x1) + x1

        def _prior_transform(cube):
            params = cube.copy()
            for i, key in enumerate(self.free_params):
                params[i] = _uniform(cube[i], *self.boundary[key])
            return params

        sampler = ReactiveNestedSampler(
            self.free_params, self.likelihood, _prior_transform,
            log_dir=log_dir, resume=resume)

        if use_slice:
            sampler.stepsampler = SliceSampler(
                nsteps=slice_steps,
                generate_direction=generate_mixture_random_direction,
            )

        sampler.run(min_num_live_points=nlive)

        if mpi_rank == 0:
            sampler.print_results()
            sampler.plot()
            sampler.plot_trace()
