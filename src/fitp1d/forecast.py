from astropy.cosmology import Planck18
import camb
import numpy as np
import emcee
import iminuit
from getdist import MCSamples
from scipy.interpolate import CubicSpline

from fitp1d.data import DetailedData
from fitp1d.model import (
    Model, _NSUB_K_, LIGHT_SPEED, LYA_WAVELENGTH, getHubbleZ)


def plotEllipse(
        fpl, covariance, key1, key2, ax, color, label=None, **kwargs
):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import copy

    std_x = np.sqrt(covariance[(key1, key1)])
    mean_x = fpl.initial[key1]
    std_y = np.sqrt(covariance[(key2, key2)])
    mean_y = np.mean(fpl.initial[key2])

    if (key1, key2) not in covariance:
        key1, key2 = key2, key1

    pearson = covariance[(key1, key2)] / (std_x * std_y)

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        fc=color, ec=color, lw=2, **kwargs
    )

    ax.plot(mean_x, mean_y, 'kx')

    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.4
    for n, ls in zip([1, 2], ['-', '--']):
        scale_x = n * std_x
        scale_y = n * std_y
        transf = transforms.Affine2D().rotate_deg(45).scale(
            scale_x, scale_y).translate(mean_x, mean_y)
        ell = copy.deepcopy(ellipse)

        ell.set_transform(transf + ax.transData)
        ell.set_linestyle(ls)
        if 'fill' not in kwargs or kwargs['fill']:
            ell.set_alpha(alpha / n)

        if n == 1:
            ell.set_label(label)
        ax.add_patch(ell)

    ax.set_xlabel(fpl.param_labels[key1])
    ax.set_ylabel(fpl.param_labels[key2])


class LyaP1DArinyoModel2(Model):
    CAMB_KMAX = 2e2
    CAMB_KMIN = 1e-3

    def addZInits(self):
        for i, z in enumerate(self.zlist):
            xx = ((1 + z) / (1 + self._bias_zeff))
            for key in self.per_z_params:
                keyz = f"{key}{i}"
                self.initial[keyz] = self.initial_z[key] * xx**self.evo_z[key]
                low, up = -200., 200

                if key in ['kp', '10kv']:
                    low = 0.
                if key in ['blya', 'bbeta']:
                    up = 0.

                self.boundary[keyz] = (low, up)
                self.param_labels[keyz] = f"{key}({z:.1f})"

        self.names = list(self.initial.keys())

    def __init__(self):
        super().__init__()

        self._cosmo_names = ['ln10As', 'ns', 'mnu', 'Ode0', 'H0']
        self.per_z_params = ['blya', 'bbeta', 'kp', '10kv']
        self.no_z_evo_params = ['q1', 'av', 'bv']
        self.names = None

        # Table 6 of Bourboux et al. 2020, http://arxiv.org/abs/2007.08995
        self._bias_zeff = 2.334
        self.initial_z = {
            'blya': -0.136, 'bbeta': -0.24752, 'kp': 45., '10kv': 2.
        }
        self.evo_z = {
            'blya': 3.231, 'bbeta': 1.231, 'kp': -0.1, '10kv': 0
        }

        self.initial = {
            'q1': 0.50,
            'av': 0.37,
            'bv': 1.0,
            'ln10As': 3.044,
            'ns': Planck18.meta['n'],
            'mnu': 8.,
            'Ode0': Planck18.Ode0,
            'H0': Planck18.H0.value
        }

        self.boundary = {
            'q1': (0.0, 1.),
            'av': (0.0, 1.),
            'bv': (0., 2.0),
            'ln10As': (2., 4.),
            'ns': (0.94, 1.),
            'mnu': (0., 50.),
            'Ode0': (0.5, 0.9),
            'H0': (50., 100.)
        }

        self.param_labels = {
            "q1": r"$q_1$", "10kv": r"$k_\nu \times 10$", "av": r"$a_\nu$",
            "bv": r"$b_\nu$", "ln10As": r"$\ln(10^{10} A_s)$",
            "ns": r"$n_s$", "mnu": r"$\sum m_\nu$ [$10^{-2}~$eV]",
            "Ode0": r"$\Omega_\Lambda$", "H0": r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
        }

        self._kperp, self._dlnkperp = np.linspace(
            np.log(LyaP1DArinyoModel2.CAMB_KMIN),
            np.log(LyaP1DArinyoModel2.CAMB_KMAX), 500, retstep=True)
        self._kperp = np.exp(self._kperp)[:, np.newaxis, np.newaxis]
        print("kperp range", self._kperp[[0, -1], 0, 0])
        self._kperp2pi = self._kperp**2 / (2 * np.pi)

        self.zlist = None
        self._p3dlin = None
        self._Delta2 = None
        self._k3d = None
        self._mu = None
        self.Mpc2kms = None
        self.ndata = None
        self.fixedCosmology = False

        self._cCosmo = {}

    def cacheZAndK(self, kedges_tuple_list, zlist):
        if self.zlist is not None and np.allclose(self.zlist, zlist):
            return

        self.zlist = zlist
        self.kedges_tuple_list = kedges_tuple_list
        self.addZInits()

        self._k3d_Mpc = []
        self._mu = []
        self._p3dlin = []
        self._Delta2 = []
        for i, (k1, k2) in enumerate(self.kedges_tuple_list):
            shape = (self._kperp.shape[0], k1.size, _NSUB_K_)
            self._k3d_Mpc.append(np.empty(shape))
            self._mu.append(np.empty(shape))
            self._p3dlin.append(np.empty(shape))
            self._Delta2.append(np.empty(shape))

    def newcosmo(self, **kwargs):
        # if (
        #         self._cCosmo and all((
        #             np.isclose(cC, kwargs[key], atol=1e-12, rtol=1e-12)
        #             for key, cC in self._cCosmo.items()
        #         ))
        # ):
        #     return

        for key in self._cosmo_names:
            self._cCosmo[key] = kwargs[key]

        H0, Ode0 = kwargs['H0'], kwargs['Ode0']
        h = H0 / 100.

        self.Mpc2kms = np.asarray([
            getHubbleZ(z, H0, Ode0) / (1 + z) for z in self.zlist
        ])

        # 2.3 ms
        for i, (k1, k2) in enumerate(self.kedges_tuple_list):
            kfine = np.linspace(k1, k2, _NSUB_K_, endpoint=False).T
            _k1d_Mpc = (kfine * self.Mpc2kms[i])[np.newaxis, :]
            self._k3d_Mpc[i][:] = self._kperp**2 + _k1d_Mpc**2
            np.sqrt(self._k3d_Mpc[i], out=self._k3d_Mpc[i])
            np.reciprocal(self._k3d_Mpc[i], out=self._mu[i])
            self._mu[i] *= _k1d_Mpc

        # 120 ms +- 3
        camb_params = camb.set_params(
            redshifts=sorted(self.zlist, reverse=True),
            WantCls=False, WantScalars=False,
            WantTensors=False, WantVectors=False,
            WantDerivedParameters=False,
            WantTransfer=True,
            omch2=(1 - Ode0 - Planck18.Ob0) * h**2,
            ombh2=Planck18.Ob0 * h**2,
            omk=0.,
            H0=H0,
            ns=kwargs['ns'],
            As=np.exp(kwargs['ln10As']) * 1e-10,
            mnu=kwargs['mnu'] / 100.
        )
        camb_results = camb.get_results(camb_params)

        khs, _, pk = camb_results.get_linear_matter_power_spectrum(
            nonlinear=False, hubble_units=False)
        khs *= h
        np.log(pk, out=pk)
        np.log(khs, out=khs)

        # Add extrapolation data points as done in camb
        logextrap = np.log(2 * LyaP1DArinyoModel2.CAMB_KMAX)
        delta = logextrap - khs[-1]

        pk0 = pk[:, -1]
        dlog = (pk0 - pk[:, -2]) / (khs[-1] - khs[-2])
        pk = np.column_stack((pk, pk0 + dlog * delta * 0.9, pk0 + dlog * delta))
        khs = np.hstack((khs, logextrap - delta * 0.1, logextrap))

        # 16.4 ms
        for i, z in enumerate(self.zlist):
            self._p3dlin[i][:] = np.exp(CubicSpline(
                khs, pk[i], bc_type='natural', extrapolate=True
            )(np.log(self._k3d_Mpc[i])))
            self._Delta2[i][:] = self._p3dlin[i] * self._k3d_Mpc[i]**3 / 2 / np.pi**2

    def getP1DListKms(self, **kwargs):
        """ 10.6 ms """
        if not self.fixedCosmology:
            self.newcosmo(**kwargs)
        p1d_list = []

        for i, z in enumerate(self.zlist):
            blya = kwargs[f'blya{i}']
            bbeta = kwargs[f'bbeta{i}']
            kp = kwargs[f'kp{i}']
            kv10 = kwargs[f'10kv{i}']
            t1 = self._Delta2[i] * (
                (self._k3d_Mpc[i] / kv10)**kwargs['av']
                * self._mu[i]**kwargs['bv']
            ) * 10**-kwargs['av']
            t2 = (self._k3d_Mpc[i] / kp)**2
            p3d = np.exp(kwargs['q1'] * (1 - t1) - t2)

            p3d *= self._p3dlin[i]
            p3d *= (blya + bbeta * self._mu[i]**2)**2
            p3d *= self._kperp2pi
            p1d_Mpc = np.trapz(p3d, dx=self._dlnkperp, axis=0).mean(axis=1)
            p1d_list.append(p1d_Mpc * self.Mpc2kms[i])

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
            LIGHT_SPEED * 0.8
            / (1 + self.psdata.data_table['z'])
            / LYA_WAVELENGTH
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
        kedges_tuple_list = [(x['k1'], x['k2']) for x in self._data]
        self.p1dmodel.cacheZAndK(kedges_tuple_list, self.zlist)

        self.names = self.p1dmodel.names
        self.fixed_params = []
        self.initial = self.p1dmodel.initial
        self.boundary = self.p1dmodel.boundary
        self.prior = self.p1dmodel.prior
        self.param_labels = self.p1dmodel.param_labels
        self._new_args = np.empty(len(self.names))
        self._free_idx = list(np.arange(self._new_args.size))

    def setFiducial(self):
        pmodel_fid = self.p1dmodel.getP1DListKms(**self.initial)
        for i, pm in enumerate(pmodel_fid):
            self._data[i]['p_final'] = pm

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}

        chi2 = 0.
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

    def setMinimizer(self, randomize_init=False):
        if randomize_init:
            mini_initials = {
                key: value + 1e-3 * np.random.default_rng().normal()
                for key, value in self.initial.items()
            }
        else:
            mini_initials = self.initial.copy()

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

    def fisherForecast(self, dx=1e-2):
        deriv_p1d = {}
        if not isinstance(dx, dict):
            dx_dict = {key: dx for key in self.names}
        else:
            dx_dict = dx.copy()

        for key, value in self.initial.items():
            if key in self.fixed_params:
                continue

            dx = dx_dict[key]

            kwargs = self.initial.copy()

            kwargs[key] = value + dx
            p1d_p1 = self.p1dmodel.getP1DListKms(**kwargs)
            kwargs[key] = value + 2 * dx
            p1d_p2 = self.p1dmodel.getP1DListKms(**kwargs)

            kwargs[key] = value - dx
            p1d_m1 = self.p1dmodel.getP1DListKms(**kwargs)
            kwargs[key] = value - 2 * dx
            p1d_m2 = self.p1dmodel.getP1DListKms(**kwargs)

            p1d_deriv = [
                (-yp2 + 8 * yp1 - 8 * ym1 + ym2) / (12. * dx)
                for (yp1, yp2, ym1, ym2) in zip(p1d_p1, p1d_p2, p1d_m1, p1d_m2)
            ]
            deriv_p1d[key] = p1d_deriv

        nparam = len(list(deriv_p1d.keys()))
        fisher = np.empty((nparam, nparam))
        for i, d1 in enumerate(deriv_p1d.values()):
            for j, d2 in enumerate(deriv_p1d.values()):
                fisher[i, j] = np.sum([
                    d1[_].dot(icov.dot(d2[_]))
                    for _, icov in enumerate(self._invcov)
                ])

        return fisher, deriv_p1d

    def fixParam(self, key, value=None):
        if key not in self.fixed_params:
            self.fixed_params.append(key)
            idx = self.names.index(key)
            self._free_idx.remove(idx)

        self._mini.fixed[key] = True
        if value is not None:
            self._mini.values[key] = value

        self._new_args[idx] = self._mini.values[key]

    def releaseParam(self, key):
        if key in self.fixed_params:
            self._mini.fixed[key] = False
            self.fixed_params.remove(key)
            idx = self.names.index(key)
            self._free_idx.append(idx)
            self._free_idx.sort()

        if key in self.p1dmodel._cosmo_names:
            self.p1dmodel.fixedCosmology = False

    def releaseAll(self):
        cpy = self.fixed_params.copy()
        for key in cpy:
            self.releaseParam(key)

        self.p1dmodel.fixedCosmology = False

    def fixCosmology(self):
        for key in self.p1dmodel._cosmo_names:
            self.fixParam(key)

        self.p1dmodel.fixedCosmology = True

    def freeCosmology(self):
        for key in self.p1dmodel._cosmo_names:
            self.releaseParam(key)

        self.p1dmodel.fixedCosmology = False

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
