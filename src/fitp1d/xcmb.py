import functools

import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import spherical_jn

import cosmopower

from fitp1d.model import Model, LIGHT_SPEED, BOLTZMANN_K, M_PROTON


HUBBLE_DISTANCE_Mpch = LIGHT_SPEED / 100  # Mpc / h


def efunc(z, Om0, Or0=Planck18.Ogamma0):
    return np.sqrt(1 - Om0 - Or0 + Om0 * (1 + z)**3 + Or0 * (1 + z)**4)


def invEfunc(z, Om0, Or0=Planck18.Ogamma0):
    return np.sqrt(1 - Om0 - Or0 + Om0 * (1 + z)**3 + Or0 * (1 + z)**4)**-1


def getLinearGrowth(z1, z2, Om0, Or0=Planck18.Ogamma0):
    def _integ(x):
        return (1 + x) * invEfunc(x, Om0, Or0)**3
    integ1 = quad(_integ, z2, z1)[0]
    norm = quad(_integ, z1, np.inf)[0]
    return (1 + integ1 / norm) * efunc(z2, Om0, Or0) * invEfunc(z1, Om0, Or0)


def getMpc2Kms(zarr, **kwargs):
    h = kwargs['h']
    Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
    return 100. * h * efunc(zarr, Om0) / (1 + zarr)


def comovingDistanceMpch(Om0, z, dz=0.01):
    npoints = min(1000, int(z / dz))
    zinteg, dz = np.linspace(0, z, npoints, retstep=True)
    return HUBBLE_DISTANCE_Mpch * np.trapz(invEfunc(zinteg, Om0), dx=dz)


def comovingDistanceMpch_d2z(Om0, z, Or0=Planck18.Ogamma0):
    inv_e = invEfunc(z, Om0)
    exde_dz = (1 + z)**2 * (1.5 * Om0 + 2 * Or0 * (1 + z))
    return -HUBBLE_DISTANCE_Mpch * inv_e**3 * exde_dz


def kappaKernel(Om0, z, z_source=1100):
    """ This is W_kappa(chi), so it returns in h / Mpc units
    """
    chi = comovingDistanceMpch(Om0, z)
    chi_S = comovingDistanceMpch(Om0, z_source)
    A = 1.5 * Om0 / HUBBLE_DISTANCE_Mpch**2

    return A * (1 + z) * chi * (1. - chi / chi_S)


def kappaKerneldChi(Om0, z, z_source=1100):
    """ This is W'_kappa(chi), so it returns in h^2 / Mpc^2 units
    """
    chi = comovingDistanceMpch(Om0, z)
    chi_S = comovingDistanceMpch(Om0, z_source)
    A = 1.5 * Om0 / HUBBLE_DISTANCE_Mpch**2

    return A * (1 + z) * (1. - 2 * chi / chi_S)


def getF2Kernel(k, q, w):
    """ 2 x F_2 """
    r = q / k + k / q
    return 10. / 7. + w * r + (4. / 7. * w**2)


def getBispectrumTree(k, q, w, plin_interp):
    """
    plin_interp returns (ncosmo, *k.shape) size array

    Returns:
        B3d : nd array of shape (ncosmo, *k.shape) in Mpc^6
    """
    # assert k.shape == q.shape
    # assert w.shape == k.shape

    t = k**2 + q**2 + 2 * k * q * w
    np.fmax(1e-32, t, out=t)
    np.sqrt(t, out=t)

    plink = plin_interp(k)
    plinq = plin_interp(q)
    plint = plin_interp(t)

    result = (
        plink * plinq * getF2Kernel(k, q, w)
        + plinq * plint * getF2Kernel(q, t, -(q + k * w) / t)
        + plint * plink * getF2Kernel(t, k, -(k + q * w) / t)
    )

    return result


class MyPlinInterp():
    LOG10_KMIN, LOG10_KMAX = np.log10(1e-7), np.log10(1e3)
    S8_k = np.logspace(-3.5, 1.5, 750)
    S8_log10k = np.log10(S8_k)
    S8_dlnk = np.log(S8_k[-1] / S8_k[0]) / (S8_k.size - 1)
    S8_norm = 3 / np.pi / np.sqrt(2)

    def __init__(self, log10k, log10p, h=1):
        self.h = np.ones(log10p.shape[0])
        self.h *= h
        # Add extrapolation data points as done in camb
        # Low k
        delta1 = log10k[0] - MyPlinInterp.LOG10_KMIN
        pk1 = log10p[:, 0]
        dlog1 = (log10p[:, 1] - pk1) / (log10k[1] - log10k[0])

        # High k
        delta2 = MyPlinInterp.LOG10_KMAX - log10k[-1]
        pk2 = log10p[:, -1]
        dlog2 = (pk2 - log10p[:, -2]) / (log10k[-1] - log10k[-2])

        log10pk_pad = np.column_stack((
            pk1 - dlog1 * delta1, pk1 - dlog1 * delta1 * 0.9,
            log10p, pk2 + dlog2 * delta2 * 0.9, pk2 + dlog2 * delta2))
        log10k_pad = np.hstack((
            MyPlinInterp.LOG10_KMIN, MyPlinInterp.LOG10_KMIN + delta1 * 0.1,
            log10k, MyPlinInterp.LOG10_KMAX - delta2 * 0.1, MyPlinInterp.LOG10_KMAX))
        self._interp = CubicSpline(
            log10k_pad, log10pk_pad, bc_type='natural', extrapolate=True,
            axis=1)
        self._delta2_interp = CubicSpline(
            log10k_pad, log10pk_pad + 3 * log10k_pad - np.log10(2 * np.pi**2),
            bc_type='natural', extrapolate=True,
            axis=1)

    def __call__(self, k):
        return 10**(self._interp(np.log10(k)))

    def sigma8(self):
        P = 10**self._interp(MyPlinInterp.S8_log10k)
        x = np.multiply.outer(8. * self.h**-1, MyPlinInterp.S8_k)
        W8_2 = (spherical_jn(1, x) / x)**2 * MyPlinInterp.S8_k**3
        return MyPlinInterp.S8_norm * np.sqrt(
            np.trapz(P * W8_2, dx=MyPlinInterp.S8_dlnk, axis=-1)
        )

    def getDelta2(self, k):
        return 10**(self._delta2_interp(np.log10(k)))


class MyWienerInterp():
    def __init__(self, wiener_fname, fit=False):
        lvals, wiener = np.loadtxt(wiener_fname, unpack=True)
        self.lvals = np.append(lvals, np.linspace(lvals[-1] + 1, 2**15, 256))
        wiener = np.append(wiener, np.zeros(256))

        self._interp = CubicSpline(
            np.log(1 + self.lvals), wiener, bc_type='natural', extrapolate=True)

        self._pfit = None
        if fit:
            self.fitSmooth()

    def fitSmooth(self):
        l1 = np.concatenate((
            np.linspace(0, 10, 100), np.logspace(1, 2, 100), np.logspace(2, 3, 200)
        ))
        w1 = self(l1)

        def wfit_fn(ll, A, a, b, n, m1, m2):
            x = np.log(1 + ll)
            return A * x**n * np.exp(a * x**m1 - b * x**m2)

        self._pfit = curve_fit(wfit_fn, l1, w1, bounds=(0, np.inf))[0]
        wiener = wfit_fn(self.lvals, *self._pfit)

        self._interp = CubicSpline(
            np.log(1 + self.lvals), wiener, bc_type='natural', extrapolate=True)

    def __call__(self, x):
        return self._interp(np.log(1 + x))

    def derivative(self, nu=1):
        f = self._interp.derivative(nu)

        def deriv(x):
            return f(np.log(1 + x)) / (1 + x)

        return deriv


class LyaxCmbModel(Model):
    """docstring for LyaxCmbModel"""

    def setKappaOm0Interp(self):
        Om0s = np.linspace(0.1, 0.7, 300)
        kappas = np.array([kappaKernel(o, self.z) for o in Om0s])
        self.kappa_om0_interp_hMpc = CubicSpline(Om0s, kappas, bc_type='natural')

        kappas = np.array([kappaKerneldChi(o, self.z) for o in Om0s])
        self.kappa_dchi_om0_interp_hMpc2 = CubicSpline(Om0s, kappas, bc_type='natural')

    def setRedshift(self, z, dz):
        self.z = z
        self.dz = dz
        self.setKappaOm0Interp()
        self.chiz_om0_mpch_fn = np.vectorize(
            functools.partial(comovingDistanceMpch, z=self.z))
        self.chiz_d2z_om0_mpch_fn = np.vectorize(
            functools.partial(comovingDistanceMpch_d2z, z=self.z))

    def setIntegrationArrays(self, nlnkbins=None, nwbins=None, klimits=None):
        if klimits:
            self.klimits = klimits.copy()

        if nlnkbins:
            self.nlnkbins = nlnkbins

        if klimits or nlnkbins:
            lnk_integ_array, self.dlnk = np.linspace(
                *np.log(self.klimits), self.nlnkbins, retstep=True)
            self.qb_1d = np.exp(lnk_integ_array)

            tmax = 4096. / self.chiz_om0_mpch_fn(0.3) / 0.65
            lnk_integ_array, self.dlntb = np.linspace(
                *np.log([self.klimits[0], tmax]), self.nlnkbins, retstep=True)
            self.tb_1d = np.exp(lnk_integ_array)

            self.qb2_1d_p1d, self.dlnk_p1d = np.linspace(
                *np.log([5e-5, 1e2]), 500, retstep=True)
            self.qb2_1d_p1d = np.exp(2 * self.qb2_1d_p1d)

        if nwbins:
            self.w_arr, self.w_weight = np.polynomial.chebyshev.chebgauss(nwbins)
            self.nwbins = nwbins

        if klimits or nlnkbins or nwbins:
            self.qb2_2d, self.tb_2d = np.meshgrid(
                self.qb_1d**2, self.tb_1d**2, indexing='ij', copy=True)

            # self.qb2_tb2_sum_2d = self.qb2_2d + self.tb_2d
            np.sqrt(self.tb_2d, out=self.tb_2d)

            self.qb2_3d, self.pb2_3d, self.ww_3d = np.meshgrid(
                self.qb_1d**2, self.tb_1d**2, self.w_arr,
                indexing='ij', copy=True)

            self.qtb_2d = np.outer(self.qb_1d, self.tb_1d)
            self.ww_3d *= self.qtb_2d[:, :, np.newaxis]
            self.pb2_3d += self.qb2_3d + 2 * self.ww_3d
            self._pb2_3d_exp = None
            self._wchi_exp_mult = None

    def __init__(
            self, z, cp_model_dir, wiener_fname,
            nlnkbins=100, nwbins=10, klimits=[5e-5, 1e2], dz=0.4,
            emu="PKLIN_NN"
    ):
        super().__init__()
        self.setRedshift(z, dz)
        self.setIntegrationArrays(nlnkbins, nwbins, klimits)

        self._cp_emulator = cosmopower.cosmopower_NN(
            restore=True,
            restore_filename=f'{cp_model_dir}/PKLIN_NN')
        self._cp_log10k = np.log10(np.loadtxt(f"{cp_model_dir}/k_modes.txt"))

        self.wiener = MyWienerInterp(wiener_fname, fit=True)
        self.wiener_deriv = self.wiener.derivative()

        self._cosmo_names = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']

        self._lya_nuis = [
            'b_F', 'beta_F', 'k_p', 'q_1', 'q_2', 'log10T', 'nu0_th', 'nu1_th']
        self._broadcasted_params = self._cosmo_names + self._lya_nuis

        self.initial = {
            'omega_b': np.array([Planck18.Ob0 * Planck18.h**2]),
            'omega_cdm': np.array([Planck18.Odm0 * Planck18.h**2]),
            'h': np.array([Planck18.h]),
            'n_s': np.array([Planck18.meta['n']]),
            'ln10^{10}A_s': np.array([3.044]),
            'b_F': np.array([-0.15]), 'beta_F': np.array([1.67]),
            'k_p': np.array([8.7]),  # Mpc^-1
            'q_1': np.array([0.25]), 'q_2': np.array([0.25]),
            'log10T': np.array([4.]),
            'nu0_th': np.array([1.5]), 'nu1_th': np.array([1])
        }
        self._sigma_th_pivot_kms = LIGHT_SPEED * np.sqrt(
            BOLTZMANN_K * 10000. / M_PROTON)

        self.boundary = {
            'omega_b': (0.01875, 0.02625),
            'omega_cdm': (0.05, 0.255),
            'n_s': (0.84, 1.1),
            'h': (0.64, 0.82),
            'ln10^{10}A_s': (1.61, 3.91),
            'b_F': (-2, 0), 'beta_F': (1, 3), 'k_p': (0, 1e3),
            'q_1': (0, 4), 'q_2': (0, 4), 'log10T': (-2, 10),
            'nu0_th': (0, 10), 'nu1_th': (0, 10)
        }

        self.param_labels = {
            'omega_b': '\\Omega_b h^2', 'omega_cdm': '\\Omega_c h^2',
            'h': 'h', 'n_s': 'n_s', 'ln10^{10}A_s': 'ln(10^{10} A_s)',
            'b_F': 'b_F', 'beta_F': '\\beta_F', 'k_p': 'k_p',
            'q_1': 'q_1', 'q_2': 'q_2', 'log10T': '\\log_{10}T',
            'nu0_th': '\\nu_{0, th}', 'nu1_th': '\\nu_{1, th}'
        }

        if "mnu" in emu:
            self._cosmo_names.append("m_nu")
            self.initial['m_nu'] = 0.06
            self.param_labels['m_nu'] = 'm_{\nu}'

    def broadcastKwargs(self, **kwargs):
        ndim = np.max([
            len(kwargs[key]) for key in kwargs
            if key in self._broadcasted_params
        ])

        for key in self._broadcasted_params:
            if key not in kwargs:
                kwargs[key] = np.ones(ndim) * self.initial[key]
            elif isinstance(kwargs[key], float):
                kwargs[key] = np.ones(ndim) * kwargs[key]
            elif len(kwargs[key]) == 1:
                kwargs[key] = np.ones(ndim) * kwargs[key][0]
            elif len(kwargs[key]) != ndim:
                raise Exception(f"Wrong dimensions in kwargs[{key}]!")
            else:
                kwargs[key] = np.array(kwargs[key])

        kwargs['z'] = self.z * np.ones(ndim)
        return kwargs

    def getPlinInterp(self, **kwargs):
        emu_params = {key: kwargs[key] for key in self._cosmo_names}
        emu_params['z'] = kwargs['z']

        # shape (ndim, nkmodes) in Mpc
        return MyPlinInterp(
            self._cp_log10k, self._cp_emulator.predictions_np(emu_params),
            h=emu_params['h'])

    def getMpc2Kms(self, zarr=None, **kwargs):
        h = kwargs['h']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        if zarr is None:
            zarr = self.z
        return 100. * h * efunc(zarr, Om0) / (1 + zarr)

    def getKms2Mpc(self, **kwargs):
        return self.getMpc2Kms(**kwargs)**-1

    def _integrandB3dPhi(self, k2, plin_interp, beta_F):
        q = self.qb2_3d + k2
        ww = q + self.ww_3d
        p = self.pb2_3d + k2

        b3d = (1 + np.divide.outer(beta_F * k2, p)) * self._pb2_3d_exp
        np.sqrt(q, out=q)
        np.sqrt(p, out=p)
        ww /= q
        ww /= p
        ww *= -1

        b3d *= getBispectrumTree(q, p, ww, plin_interp)

        return b3d

    def _integrateB3dFncTrapzUnnorm(self, k2, plin_interp, beta_F):
        """ Gauss-Hermite quadrature does not work
        k: Mpc^-1
        B1d: Mpc
        """
        # Mpc^4. shape: (ncosmo, qb.size, tb.size)
        b3d = (
            self._integrandB3dPhi(k2, plin_interp, beta_F).dot(self.w_weight)
            * self._wchi_exp_mult
            * (1 + np.divide.outer(beta_F * k2, self.qb2_2d + k2))
        )

        b3d = np.trapz(b3d, dx=self.dlnk, axis=-1)
        b3d *= self.qb_1d
        return np.trapz(b3d, dx=self.dlnk, axis=-1)

    def integrateB3dTrapz(self, k, limber_o1=False, **kwargs):
        # kwargs = self.broadcastKwargs(**kwargs)
        h = kwargs['h']
        kp = kwargs['k_p']
        b_F = kwargs['b_F']
        nu = kwargs['nu1_th'][:, np.newaxis]
        sigma_th = (
            self.getKms2Mpc(**kwargs) * self._sigma_th_pivot_kms
            * 10**(kwargs['log10T'] / 2 - 2)
        )
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        plin_interp = self.getPlinInterp(**kwargs)

        invkp2 = -kp**-2
        k2 = k**2

        chiz_tb_3d = np.multiply.outer(self.chiz_om0_mpch_fn(Om0) / h, self.tb_2d)
        self._wchi_exp_mult = (
            self.qtb_2d * self.wiener(chiz_tb_3d)
            * np.exp(np.multiply.outer(invkp2, self.qb2_2d))
            * self.tb_1d
        )
        self._pb2_3d_exp = np.exp(np.multiply.outer(invkp2, self.pb2_3d))

        # Mpc^-1
        norm = b_F**2 * h * self.kappa_om0_interp_hMpc(Om0) / (4. * np.pi**3)
        norm = norm[:, np.newaxis] * np.exp(
            np.multiply.outer(2 * invkp2, k2)
            - np.power(np.multiply.outer(sigma_th, k), nu)
        )

        b1d = norm * np.fromiter((
            self._integrateB3dFncTrapzUnnorm(_, plin_interp, kwargs['beta_F'])
            for _ in k2), dtype=np.dtype((float, (h.size, )))).T

        if not limber_o1:
            return b1d

        chi_d2z = self.chiz_d2z_om0_mpch_fn(Om0) * self.dz**2 / 8.
        b1d *= 1 + chi_d2z * self.kappa_dchi_om0_interp_hMpc2(
            Om0) / self.kappa_om0_interp_hMpc(Om0)

        self._wchi_exp_mult = (
            self.qtb_2d * self.wiener_deriv(chiz_tb_3d)
            * np.exp(np.multiply.outer(invkp2, self.qb2_2d))
            * self.tb_1d**2
        )

        b1d += norm * chi_d2z * np.fromiter((
            self._integrateB3dFncTrapzUnnorm(_, plin_interp, kwargs['beta_F'])
            for _ in k2), dtype=np.dtype((float, (h.size, )))).T

        return b1d

    def getP1dTrapz(self, k, **kwargs):
        kp = kwargs['k_p']
        b_F = kwargs['b_F']
        beta_F = kwargs['beta_F']
        q_1 = kwargs['q_1'][:, np.newaxis, np.newaxis]
        q_2 = kwargs['q_2'][:, np.newaxis, np.newaxis]
        nu1 = kwargs['nu1_th'][:, np.newaxis, np.newaxis] / 2
        nu0 = kwargs['nu0_th'][:, np.newaxis, np.newaxis]
        sigma_th = (
            self.getKms2Mpc(**kwargs) * self._sigma_th_pivot_kms
            * 10**(kwargs['log10T'] / 2 - 2)
        )
        plin_interp = self.getPlinInterp(**kwargs)

        invkp2 = -2 * kp**-2
        k2 = k**2

        k2_2d, qb2_2d = np.meshgrid(k2, self.qb2_1d_p1d, indexing='ij', copy=True)
        q_2d = k2_2d + qb2_2d
        ww2 = k2_2d / q_2d
        np.sqrt(q_2d, out=q_2d)

        p3d = (1 + np.multiply.outer(beta_F, ww2))**2 * plin_interp(q_2d)
        Delta2 = plin_interp.getDelta2(q_2d)
        nonlinear = q_1 * Delta2
        if not np.allclose(q_2, 0):
            nonlinear += q_2 * Delta2**2

        p3d *= np.exp(
            np.multiply.outer(invkp2, qb2_2d)
            + nonlinear * (
                1 - np.power(
                    np.multiply.outer(sigma_th**2, k2_2d), nu1
                ) / np.power(np.multiply.outer(sigma_th, q_2d), nu0)
            )
        )
        p1d = np.trapz(p3d * qb2_2d, dx=self.dlnk_p1d, axis=-1)

        norm = b_F**2 / (2 * np.pi)
        norm = norm[:, np.newaxis] * np.exp(np.multiply.outer(invkp2, k2))
        return norm * p1d
