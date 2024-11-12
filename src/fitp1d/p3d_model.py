import numpy as np
from astropy.cosmology import Planck18

from fitp1d.model import Model
from fitp1d.xcmb import MyPlinInterp

cosmo_package = None


class LyaP3DArinyoModel(Model):
    """docstring for LyaxCmbModel"""

    def __init__(
            self, z, cp_model_dir, emu="PKLIN_NN", use_camb=False,
            nmu=512, nl=4
    ):
        global cosmo_package
        super().__init__()

        self.z = z
        self._muarr, self._dmu = np.linspace(0, 1, nmu, retstep=True)
        self._k, self._kk, self._mmuu = None, None, None
        self._use_camb = use_camb
        self.pls = []
        for _ in range(nl):
            pl = np.polynomial.legendre.Legendre.basis(2 * _)
            self.pls.append(pl(self._muarr))

        if not use_camb:
            import cosmopower as cosmo_package
            self._cp_emulator = cosmo_package.cosmopower_NN(
                restore=True,
                restore_filename=f'{cp_model_dir}/PKLIN_NN')
            self._cp_log10k = np.log10(
                np.loadtxt(f"{cp_model_dir}/k_modes.txt"))
        else:
            import camb as cosmo_package
            self._cp_emulator, self._cp_log10k = None, None

        self._cosmo_names = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']
        self._cosmo_fixed = False
        self._plin = None

        self._lya_nuis = [
            'b_F', 'beta_F', 'q_1', '10kv', 'nu_0', 'nu_1', 'k_p']
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
        }

        self.boundary = {
            'omega_b': (0.01875, 0.02625),
            'omega_cdm': (0.05, 0.255),
            'n_s': (0.84, 1.1),
            'h': (0.64, 0.82),
            'ln10^{10}A_s': (1.61, 3.91),
            'b_F': (-2, 0), 'beta_F': (1, 3), 'q_1': (0, 4), '10kv': (0, 1e2),
            'nu_0': (0, 10), 'nu_1': (0, 10), 'k_p': (0, 1e3)
        }

        self.param_labels = {
            'omega_b': '\\Omega_b h^2', 'omega_cdm': '\\Omega_c h^2',
            'h': 'h', 'n_s': 'n_s', 'ln10^{10}A_s': 'ln(10^{10} A_s)',
            'b_F': 'b_F', 'beta_F': '\\beta_F', 'k_p': 'k_p',
            '10kv': 'k_\\nu [10^{-1}~Mpc]',
            'q_1': 'q_1', 'nu_0': '\\nu_0', 'nu_1': '\\nu_1'
        }

        if "mnu" in emu:
            self._cosmo_names.append("m_nu")
            self.initial['m_nu'] = 0.06
            self.param_labels['m_nu'] = 'm_{\nu}'

    def cacheK(self, k):
        self._k = k.copy()
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

    def getP3D(self, k, mu, plin, **kwargs):
        b_F = kwargs['b_F'][:, None, None]
        beta_F = kwargs['beta_F'][:, None, None]
        q_1 = kwargs['q_1'][:, None, None]
        k_nu = kwargs['10kv'][:, None, None]
        nu_0 = kwargs['nu_0'][:, None, None]
        nu_1 = kwargs['nu_1'][:, None, None]
        k_p = kwargs['k_p'][:, None, None]
        mu = mu[None, :, :]
        k = k[None, :, :]

        k_kp = k / k_p
        # k_kp = np.multiply.outer(k_p**-1, k)
        k_knu = k / k_nu
        # k_knu = np.multiply.outer(k_nu**-1, k)
        mu2 = mu**2
        bbeta_lya = b_F * (1.0 + beta_F * mu2)
        # bbeta_hcd_kz = b_HCD * (1 + beta_HCD * mu2) * np.exp(-L_HCD * k * mu)

        lnD = plin * k**3 / (2 * np.pi**2) * q_1 * (
            1.0 - ((k_knu * mu)**nu_1 / k_knu**nu_0) * 10**(nu_1 - nu_0))
        lnD -= k_kp**2

        return plin * bbeta_lya * bbeta_lya * np.exp(lnD)

    def getPls(self, **kwargs):
        if self._cosmo_fixed:
            plin = self._plin(self._kk)
        else:
            plin_interp = self.getPlinInterp(**kwargs)
            plin = plin_interp(self._kk)
        p3d = self.getP3D(self._kk, self._mmuu, plin, **kwargs)

        res = []
        for l, pl in enumerate(self.pls):
            res.append((4 * l + 1) * np.trapz(p3d * pl, dx=self._dmu, axis=-1))

        return res
