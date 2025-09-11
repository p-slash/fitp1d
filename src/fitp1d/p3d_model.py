import numpy as np
from astropy.cosmology import Planck18

from fitp1d.model import Model, LyaP1DSimpleModel, LYA_WAVELENGTH
from fitp1d.basic_cosmo import MyPlinInterp

cosmo_package = None


def getPowerSkyModel(k, mu, a_sky, sigma_sky):
    # This multiplication is preferable when broadcasting
    x = k**2 * (1.0 - mu**2)
    y = -0.5 * sigma_sky**2
    return a_sky * np.exp(x * y)


class MetalModel3D(Model):
    def __init__(self, z):
        super().__init__()
        self.names = ['b_SiIII_1207', 'beta_metal', 'sigma_v']
        self.initial = {
            'b_SiIII_1207': np.array([-9.8e-3]), 'beta_metal': np.array([0.5]),
            'sigma_v': np.array([5.0])
        }
        self.boundary = {
            'b_SiIII_1207': (-0.5, 0), 'beta_metal': (0.0, 2.0),
            'sigma_v': (0.0, 40.0)
        }
        self.param_labels = {
            'b_SiIII_1207': 'b_\\mathrm{Si~III(1207)}',
            'beta_metal': '\\beta_M', 'sigma_v': '\\sigma_v'
        }

        alpha_si = LYA_WAVELENGTH / 1206.52
        z_si = (1.0 + z) * alpha_si - 1.0
        self.dr_SiIII = (Planck18.comoving_distance(z_si)
                         - Planck18.comoving_distance(z)).to("Mpc").value

    def getTransfer3D(self, bbeta_lya, d_lya, kz, mu2, **kwargs):
        b_SiIII1207 = kwargs['b_SiIII_1207'][:, None, None]
        beta_metal = kwargs['beta_metal'][:, None, None]
        bbeta_siIII = b_SiIII1207 * (1.0 + beta_metal * mu2)
        dfog = 1.0 / (1.0 + kz**2 * (kwargs['sigma_v']**2)[:, None, None])

        return (2.0 * bbeta_lya * bbeta_siIII * np.cos(kz * self.dr_SiIII)
                    * np.sqrt(d_lya * dfog)
                + bbeta_siIII * bbeta_siIII * dfog)


class LyaP3DArinyoModel(Model):
    def __init__(
            self, z, emu="PKLIN_NN", use_camb=False,
            nmu=512, nl=4
    ):
        global cosmo_package
        super().__init__()

        self.z = z
        self._muarr, self._dmu = np.linspace(0, 1, nmu, retstep=True)
        self._k, self._kk, self._mmuu = None, None, None
        self._k3, self.apo_halo = None, None
        self._kmax_halo = 1.5

        self._use_camb = use_camb
        self.pls = []
        for _ in range(nl):
            pl = np.polynomial.legendre.Legendre.basis(2 * _)
            self.pls.append(pl(self._muarr))

        if not use_camb:
            import cosmopower_slim as cosmo_package
            self._cp_emulator = cosmo_package.cosmopower_NN()
            self._cp_log10k = self._cp_emulator.log10k
        else:
            import camb as cosmo_package
            self._cp_emulator, self._cp_log10k = None, None

        self._cosmo_names = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']
        self._cosmo_fixed = False
        self._plin = None

        self.metal_model = MetalModel3D(z)

        self._lya_nuis = [
            'b_F', 'beta_F', 'q_1', '10kv', 'nu_0', 'nu_1', 'k_p',
            'b_hcd', 'beta_hcd', 'L_hcd',
            'a_sky', 'sigma_sky'] + self.metal_model.names
        self._broadcasted_params = self._cosmo_names + self._lya_nuis

        self.initial = {
            'omega_b': np.array([Planck18.Ob0 * Planck18.h**2]),
            'omega_cdm': np.array([Planck18.Odm0 * Planck18.h**2]),
            'h': np.array([Planck18.h]),
            'n_s': np.array([Planck18.meta['n']]),
            'ln10^{10}A_s': np.array([3.044]),
            'b_F': np.array([-0.1195977]), 'beta_F': np.array([1.6633]),
            'q_1': np.array([0.796]), '10kv': np.array([3.922]),
            'nu_0': np.array([1.267]), 'nu_1': np.array([1.65]),
            'k_p': np.array([16.802]),  # Mpc^-1
            'b_hcd': np.array([-0.05]), 'beta_hcd': np.array([0.7]),
            'L_hcd': np.array([14.8]),
            'a_sky': np.array([0.0]), 'sigma_sky': np.array([45.0])
        } | self.metal_model.initial

        self.boundary = {
            'omega_b': (0.01875, 0.02625),
            'omega_cdm': (0.05, 0.255),
            'n_s': (0.84, 1.1),
            'h': (0.64, 0.82),
            'ln10^{10}A_s': (1.61, 3.91),
            'b_F': (-2.0, 0.0), 'beta_F': (1.0, 3.0), 'q_1': (0.0, 4.0),
            '10kv': (0.0, 1e2), 'nu_0': (0.0, 10.0), 'nu_1': (0.0, 10.0),
            'k_p': (0.0, 1e2),
            'b_hcd': (-0.2, 0.0), 'beta_hcd': (0.0, 2.0), 'L_hcd': (0.0, 40.0),
            'a_sky': (0.0, 10.0), 'sigma_sky': (10.0, 125.0)
        } | self.metal_model.boundary

        self.param_labels = {
            'omega_b': '\\Omega_b h^2', 'omega_cdm': '\\Omega_c h^2',
            'h': 'h', 'n_s': 'n_s', 'ln10^{10}A_s': 'ln(10^{10} A_s)',
            'b_F': 'b_F', 'beta_F': '\\beta_F', 'k_p': 'k_p',
            '10kv': 'k_\\nu [10^{-1}~Mpc]',
            'q_1': 'q_1', 'nu_0': '\\nu_0', 'nu_1': '\\nu_1',
            'b_hcd': 'b_{HCD}', 'beta_hcd': '\\beta_{HCD}', 'L_hcd': 'L_{HCD}',
            'a_sky': 'a_\\mathrm{sky}', 'sigma_sky': '\\sigma_\\mathrm{sky}'
        } | self.metal_model.param_labels

        self.prior = {
            'omega_b': 0.00014,
            'omega_cdm': 0.00091,
            'h': 0.0042,
            'n_s': 0.0038,
            'ln10^{10}A_s': 0.014,
            'k_p': 4.8, 'q_1': 0.2, 'beta_hcd': 0.09
        } | self.metal_model.prior

        if "mnu" in emu:
            self._cosmo_names.append("m_nu")
            self.initial['m_nu'] = 0.06
            self.param_labels['m_nu'] = 'm_{\nu}'

        self._templates = {}

    def margLyaP1D(self):
        p1d_m = LyaP1DSimpleModel()
        p1d_m.z = self.z
        p1d_k = p1d_m.evaluate(self._kk * self._mmuu, **p1d_m.initial)

        res = []
        for l, pl in enumerate(self.pls):
            res.append(
                (4 * l + 1) * np.trapz(p1d_k * pl, dx=self._dmu, axis=-1))

        self.names.append("a_p1d")
        self._lya_nuis.append("a_p1d")
        self.initial['a_p1d'] = np.array([0.0])
        self.boundary['a_p1d'] = (-3.0, 3.0)
        self.param_labels['a_p1d'] = 'a_{p1d}'
        self._templates['a_p1d'] = np.array(res)

    def _apodize(self, k):
        return np.cos((2.0 * k / self._kmax_halo - 1.0) * np.pi / 2.0)**2

    def cacheK(self, k):
        self._k = k.copy()
        self._k3 = self._k**3 / (2 * np.pi**2)
        w = (self._kmax_halo / 2.0) < self._k
        if np.any(w):
            self.apo_halo = np.piecewise(
                self._k, [~w, w & (self._k < self._kmax_halo)],
                [1.0, self._apodize, 0.0])
            self.apo_halo = self.apo_halo[:, None, None]
        else:
            self.apo_halo = 1
        self._kk, self._mmuu = np.meshgrid(k, self._muarr, indexing='ij')

    def broadcastKwargs(self, **kwargs):
        for key in kwargs:
            if isinstance(kwargs[key], (float, np.float64)):
                kwargs[key] = [kwargs[key]]

        ndim = np.max([
            len(kwargs[key]) for key in kwargs
            if key in self._broadcasted_params
        ])

        for key in self._broadcasted_params:
            if key not in kwargs:
                kwargs[key] = np.ones(ndim) * self.initial[key]
            elif isinstance(kwargs[key], (float, np.float64)):
                kwargs[key] = np.ones(ndim) * kwargs[key]
            elif len(kwargs[key]) == 1:
                kwargs[key] = np.ones(ndim) * kwargs[key][0]
            elif len(kwargs[key]) != ndim:
                raise Exception(f"Wrong dimensions in kwargs[{key}]!")
            else:
                kwargs[key] = np.array(kwargs[key])

        kwargs['z'] = self.z * np.ones(ndim)
        return kwargs

    def fixCosmology(self, **kwargs):
        self._plin = self.getPlinInterp(**kwargs)
        self._cosmo_fixed = True

    def releaseCosmology(self):
        self._plin = None
        self._cosmo_fixed = False

    def getPrior(self, **kwargs):
        prior = 0
        for key, s in self.prior.items():
            prior += ((kwargs[key][0] - self.initial[key][0]) / s)**2
        return prior

    def getPriorVector(self, ndim, **kwargs):
        prior = np.zeros(ndim)
        for key, s in self.prior.items():
            prior += ((kwargs[key] - self.initial[key][0]) / s)**2
        return prior

    def getCambInterpolator(self, **kwargs):
        """ Vectorization is not possible """
        for key in self._cosmo_names:
            assert len(kwargs[key]) == 1

        camb_params = cosmo_package.set_params(
            redshifts=[self.z],
            WantCls=False, WantScalars=False,
            WantTensors=False, WantVectors=False,
            WantDerivedParameters=False, NonLinear="NonLinear_none",
            WantTransfer=True,
            omch2=kwargs['omega_cdm'][0],
            ombh2=kwargs['omega_b'][0],
            omk=0.,
            H0=100.0 * kwargs['h'][0],
            ns=kwargs['n_s'][0],
            As=np.exp(kwargs['ln10^{10}A_s'][0]) * 1e-10,
            mnu=0.0, nnu=3.046, kmax=10.0
        )
        camb_results = cosmo_package.get_results(camb_params)

        khs, _, pk = camb_results.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)
        np.log10(pk, out=pk)
        np.log10(khs, out=khs)
        return MyPlinInterp(khs, pk)

    def getPlinInterp(self, **kwargs):
        if self._use_camb:
            return self.getCambInterpolator(**kwargs)

        emu_params = {key: kwargs[key] for key in self._cosmo_names}
        emu_params['z'] = kwargs['z']

        # shape (ndim, nkmodes) in Mpc
        return MyPlinInterp(
            self._cp_log10k, self._cp_emulator.predictions_np(emu_params),
            h=emu_params['h'])

    def getP3D(self, k, mu, **kwargs):
        if self._cosmo_fixed:
            plin = self._plin(self._k)
        else:
            plin_interp = self.getPlinInterp(**kwargs)
            plin = plin_interp(self._k)

        delta2 = plin * self._k3
        tk = self._getTransfer3D(self._kk, self._mmuu, delta2, **kwargs)
        result = plin[:, :, None] * tk

        if any(kwargs['a_sky'] > 0):
            result += getPowerSkyModel(
                self._kk[None, :, :], self._mmuu[None, :, :],
                kwargs['a_sky'][:, None, None],
                kwargs['sigma_sky'][:, None, None])

        return result

    def _getTransfer3D(self, k, mu, delta2, **kwargs):
        b_F = kwargs['b_F'][:, None, None]
        b_HCD = kwargs['b_hcd'][:, None, None]
        beta_F = kwargs['beta_F'][:, None, None]
        beta_HCD = kwargs['beta_hcd'][:, None, None]
        L_HCD = kwargs['L_hcd'][:, None, None]
        q_1 = kwargs['q_1'][:, None, None]
        k_nu = kwargs['10kv'][:, None, None]
        nu_0 = kwargs['nu_0'][:, None, None]
        nu_1 = kwargs['nu_1'][:, None, None]
        mu = mu[None, :, :]

        mu2 = mu**2
        bbeta_lya = b_F * (1.0 + beta_F * mu2)
        kz = k[None, :, :] * mu
        bbeta_hcd_kz = b_HCD * (1 + beta_HCD * mu2) * np.exp(-L_HCD * kz)

        k_knu = k[None, :, :] / k_nu
        lnD = delta2[:, :, None] * q_1 * (
            1.0 - ((kz / k_nu)**nu_1 / k_knu**nu_0) * 10**(nu_1 - nu_0))
        lnD -= (k**2)[None, :, :] / (kwargs['k_p']**2)[:, None, None]
        np.exp(lnD, out=lnD)

        result = bbeta_lya * bbeta_lya * lnD + self.apo_halo * (
            + 2.0 * bbeta_lya * bbeta_hcd_kz
            + bbeta_hcd_kz * bbeta_hcd_kz
            + self.metal_model.getTransfer3D(
                bbeta_lya, lnD, kz, mu2, **kwargs)
        )

        return result

    def getPls(self, **kwargs):
        if self._cosmo_fixed:
            plin = self._plin(self._k)
        else:
            plin_interp = self.getPlinInterp(**kwargs)
            plin = plin_interp(self._k)
        ncosmo = plin.shape[0]
        delta2 = plin * self._k3
        tk = self._getTransfer3D(self._kk, self._mmuu, delta2, **kwargs)

        res = np.empty((ncosmo, len(self.pls), self._k.size))
        for l, pl in enumerate(self.pls):
            res[:, l] = plin * (4 * l + 1) * np.trapz(
                tk * pl, dx=self._dmu, axis=-1)

        if any(kwargs['a_sky'] > 0):
            psky = getPowerSkyModel(
                self._kk[None, :, :], self._mmuu[None, :, :],
                kwargs['a_sky'][:, None, None],
                kwargs['sigma_sky'][:, None, None])

            for l, pl in enumerate(self.pls):
                res[:, l] += (4 * l + 1) * np.trapz(
                    psky * pl, dx=self._dmu, axis=-1)

        for key, item in self._templates.items():
            res += kwargs[key] * item

        return res
