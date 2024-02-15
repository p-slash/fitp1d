import functools

import numpy as np
from astropy.cosmology import Planck18
from scipy.interpolate import CubicSpline, interp1d

from fitp1d.model import Model, LIGHT_SPEED


HUBBLE_DISTANE_Mpch = LIGHT_SPEED / 100  # Mpc / h


def efunc(z, Om0, Or0=Planck18.Ogamma0):
    return np.sqrt(1 - Om0 - Or0 + Om0 * (1 + z)**3 + Or0 * (1 + z)**4)


def invEfunc(z, Om0, Or0=Planck18.Ogamma0):
    return np.sqrt(1 - Om0 - Or0 + Om0 * (1 + z)**3 + Or0 * (1 + z)**4)**-1


def comovingDistanceMpch(Om0, z, npoints=1000):
    zinteg, dz = np.linspace(0, z, npoints, retstep=True)
    return HUBBLE_DISTANE_Mpch * np.trapz(invEfunc(zinteg, Om0), dx=dz)


def kappaKernel(Om0, z, z_source=1100):
    """ This W_kappa(chi), so it returns in h / Mpc units
    """
    chi = comovingDistanceMpch(Om0, z)
    chi_S = comovingDistanceMpch(Om0, z_source)
    A = 1.5 * Om0 / HUBBLE_DISTANE_Mpch**2

    return A * (1 + z) * chi * (1. - chi / chi_S)


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

    t = np.sqrt(k**2 + q**2 + 2 * k * q * w)

    plink = plin_interp(k)
    plinq = plin_interp(q)
    plint = plin_interp(t)

    result = (
        plink * plinq * getF2Kernel(k, q, w)
        + plinq * plint * getF2Kernel(q, t, -(q + k * w) / t)
        + plint * plink * getF2Kernel(t, k, -(k + q * w) / t)
    )

    return result


class MyPlinInterp(CubicSpline):
    LOG10_KMIN, LOG10_KMAX = np.log10(1e-7), np.log10(1e3)

    def __init__(self, log10k, log10p):
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

    def __call__(self, k):
        return 10**(self._interp(np.log10(k)))


class LyaxCmbModel(Model):
    """docstring for LyaxCmbModel"""

    def setKappaOm0Interp(self):
        Om0s = np.linspace(0.1, 0.5, 100)
        kappas = np.array([kappaKernel(o, self.z) for o in Om0s])
        self.kappa_om0_interp_hMpc = CubicSpline(Om0s, kappas, bc_type='natural')

    def setRedshift(self, z):
        self.z = z
        self.setKappaOm0Interp()
        self.chiz_om0_mpch_fn = np.vectorize(
            functools.partial(comovingDistanceMpch, z=self.z))

    def setIntegrationArrays(self, nlnkbins=None, nwbins=None, klimits=None):
        if klimits:
            self.klimits = klimits.copy()

        if nlnkbins:
            self.nlnkbins = nlnkbins

        if klimits or nlnkbins:
            lnk_integ_array, self.dlnk = np.linspace(
                *np.log(self.klimits), self.nlnkbins, retstep=True)
            self.qb_1d = np.exp(lnk_integ_array)
            self.tb_1d = np.exp(lnk_integ_array)

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
            nlnkbins=100, nwbins=10, klimits=[1e-3, 20.],
            emu="PKLIN_NN"
    ):
        super().__init__()
        self.setRedshift(z)
        self.setIntegrationArrays(nlnkbins, nwbins, klimits)

        import cosmopower
        self._cp_emulator = cosmopower.cosmopower_NN(
            restore=True,
            restore_filename=f'{cp_model_dir}/PKLIN_NN')
        self._cp_log10k = np.log10(np.loadtxt(f"{cp_model_dir}/k_modes.txt"))

        lvals, wiener = np.loadtxt(wiener_fname, unpack=True)
        self.wiener = interp1d(
            lvals, wiener, 'cubic', copy=False, bounds_error=False,
            fill_value=0, assume_sorted=True)

        self._cosmo_names = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']

        self._lya_nuis = ['b_F', 'beta_F', 'k_p']
        self._broadcasted_params = self._cosmo_names + self._lya_nuis

        self.initial = {
            'omega_b': Planck18.Ob0 * Planck18.h**2,
            'omega_cdm': Planck18.Odm0 * Planck18.h**2,
            'h': Planck18.h,
            'n_s': Planck18.meta['n'],
            'ln10^{10}A_s': 3.044,
            'b_F': -0.136,
            'beta_F': 1.82,
            'k_p': 4.  # in Mpc^-1 by Misha's paper ~5 h/Mpc
        }

        self.boundary = {
            'omega_b': (0.01875, 0.02625),
            'omega_cdm': (0.05, 0.255),
            'n_s': (0.84, 1.1),
            'h': (0.64, 0.82),
            'ln10^{10}A_s': (1.61, 3.91),
            'b_F': (-2, 0),
            'beta_F': (1, 3),
            'k_p': (0, 1e3),
        }

        self.param_labels = {
            'omega_b': '\\Omega_b h^2', 'omega_cdm': '\\Omega_c h^2',
            'h': 'h', 'n_s': 'n_s', 'ln10^{10}A_s': 'ln(10^{10} A_s)',
            'b_F': 'b_F', 'beta_F': '\\beta_F', 'k_p': 'k_p'
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
        return MyPlinInterp(self._cp_log10k, self._cp_emulator.predictions_np(emu_params))

    def getMpc2Kms(self, zarr=None, **kwargs):
        h = kwargs['h']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        if zarr is None:
            zarr = self.z
        return 100. * h * efunc(zarr, Om0) / (1 + zarr)

    def getKms2Mpc(self, **kwargs):
        return self.getMpc2Kms(**kwargs)**-1

    def _integrandB3dPhi(
            self, k2, plin_interp, Om0, h, invkp2, beta_F
    ):
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

    def _integrateB3dFncTrapzUnnorm(self, k, plin_interp, Om0, h, invkp2, b_F, beta_F):
        """ Gauss-Hermite quadrature does not work
        k: Mpc^-1
        B1d: Mpc
        """

        k2 = k**2

        # shape: (ncosmo, qb.size, tb.size)
        b3d = self._integrandB3dPhi(
            k2, plin_interp, Om0, h, invkp2, beta_F
        ).dot(self.w_weight)  # Mpc^4

        b3d *= self._wchi_exp_mult
        b3d *= (1 + np.divide.outer(beta_F * k2, self.qb2_2d + k2))

        b3d = np.trapz(b3d, dx=self.dlnk, axis=-1)
        b3d *= self.qb_1d
        return np.trapz(b3d, dx=self.dlnk, axis=-1)

    def integrateB3dTrapz(self, k, **kwargs):
        # kwargs = self.broadcastKwargs(**kwargs)
        h = kwargs['h']
        kp = kwargs['k_p']
        b_F = kwargs['b_F']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        chiz = self.chiz_om0_mpch_fn(Om0) / h
        plin_interp = self.getPlinInterp(**kwargs)

        invkp2 = -kp**-2
        k2 = k**2

        _wiener_chiz = self.qtb_2d * self.wiener(np.multiply.outer(chiz, self.tb_2d))
        _qb2_2d_exp = np.exp(np.multiply.outer(invkp2, self.qb2_2d))
        self._pb2_3d_exp = np.exp(np.multiply.outer(invkp2, self.pb2_3d))
        self._wchi_exp_mult = _wiener_chiz * _qb2_2d_exp * self.tb_1d

        # Mpc^-1
        norm = h * b_F**2 * self.kappa_om0_interp_hMpc(Om0) / (4. * np.pi**3)
        norm = norm[:, np.newaxis] * np.exp(np.multiply.outer(2 * invkp2, k2))

        return norm * np.fromiter((
            self._integrateB3dFncTrapzUnnorm(
                _, plin_interp, Om0, h, invkp2, b_F, kwargs['beta_F']
            ) for _ in k), dtype=np.dtype((float, (h.size, )))).T

    def getP1dTrapz(self, k, **kwargs):
        kp = kwargs['k_p']
        b_F = kwargs['b_F']
        beta_F = kwargs['beta_F']
        plin_interp = self.getPlinInterp(**kwargs)

        invkp2 = -2 * kp**-2
        k2 = k**2

        lnk_integ_array, dlnk = np.linspace(
            *np.log(self.klimits), 400, retstep=True)
        k2_2d, qb2_2d = np.meshgrid(k2, self.qb_1d**2, indexing='ij', copy=True)
        q_2d = k2_2d + qb2_2d
        ww2 = k2_2d / q_2d
        np.sqrt(q_2d, out=q_2d)

        p3d = (1 + np.multiply.outer(beta_F, ww2))**2 * plin_interp(q_2d)
        p3d *= np.exp(np.multiply.outer(invkp2, qb2_2d))
        p1d = np.trapz(p3d * qb2_2d, dx=self.dlnk, axis=-1)

        norm = b_F**2 / (2 * np.pi)
        norm = norm[:, np.newaxis] * np.exp(np.multiply.outer(invkp2, k2))
        return norm * p1d
