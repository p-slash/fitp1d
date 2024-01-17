import functools

import numpy as np
from astropy.cosmology import Planck18
from scipy.interpolate import CubicSpline, interp1d

from fitp1d.model import (
    Model, LIGHT_SPEED, LYA_WAVELENGTH, getHubbleZ)


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
    r = q / k + k / q
    return 5. / 7. + (w * r / 2) + (2. / 7. * w**2)


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

    result = 2 * (
        plink * plinq * getF2Kernel(k, q, w)
        + plinq * plint * getF2Kernel(q, t, -(q + k * w) / t)
        + plint * plink * getF2Kernel(t, k, -(k + q * w) / t)
    )

    return result


class MyPlinInterp(CubicSpline):
    LOG10_KMAX = np.log10(5e2)

    def __init__(self, log10k, log10p):
        # Add extrapolation data points as done in camb
        delta = MyPlinInterp.LOG10_KMAX - log10k[-1]

        pk0 = log10p[:, -1]
        dlog = (pk0 - log10p[:, -2]) / (log10k[-1] - log10k[-2])
        log10pk_pad = np.column_stack(
            (log10p, pk0 + dlog * delta * 0.9, pk0 + dlog * delta))
        log10k_pad = np.hstack(
            (log10k, MyPlinInterp.LOG10_KMAX - delta * 0.1, MyPlinInterp.LOG10_KMAX))
        self._interp = CubicSpline(
            log10k_pad, log10pk_pad, bc_type='natural', extrapolate=True,
            axis=1)

    def __call__(self, k):
        return 10**(self._interp(np.log10(k)))


class LyaxCmbModel(Model):
    """docstring for LyaxCmbModel"""
    K_INTEG_LIMITS = 5e-2, 2e2
    LNK_INTEG_LIMITS = np.log(5e-2), np.log(2e2)
    LNK_INTEG_ARRAY, DLNK_INTEG = np.linspace(
        *LNK_INTEG_LIMITS, 200, retstep=True)
    K_INTEG_ARRAY = np.exp(LNK_INTEG_ARRAY)

    W_INTEG_ARRAY, W_WEIGHT_ARRAY = np.polynomial.chebyshev.chebgauss(40)

    def setKappaOm0Interp(self):
        Om0s = np.linspace(0.1, 0.5, 100)
        kappas = np.array([kappaKernel(o, self.z) for o in Om0s])
        self.kappa_om0_interp_hMpc = CubicSpline(Om0s, kappas, bc_type='natural')

    def __init__(self, z, cp_model_dir, wiener_fname):
        super().__init__()
        self.z = z

        import cosmopower
        self._cp_emulator = cosmopower.cosmopower_NN(
            restore=True,
            restore_filename=f'{cp_model_dir}/PKLIN_NN')
        self._cp_log10k = np.log10(np.loadtxt(f"{cp_model_dir}/k_modes.txt"))

        lvals, wiener = np.loadtxt(wiener_fname, unpack=True)
        self.wiener = interp1d(
            lvals, wiener, 'cubic', copy=False, bounds_error=False,
            fill_value=0, assume_sorted=True)
        self.setKappaOm0Interp()
        self.chiz_om0_mpch_fn = np.vectorize(
            functools.partial(comovingDistanceMpch, z=self.z))

        self._cosmo_names = [
            'omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s']
        self._lya_nuis = ['b_F', 'beta_F', 'k_p']

        self.initial = {
            'omega_b': Planck18.Ob0 * Planck18.h**2,
            'omega_cdm': Planck18.Odm0 * Planck18.h**2,
            'h': Planck18.h,
            'n_s': Planck18.meta['n'],
            'ln10^{10}A_s': 3.044,
            'b_F': -0.136,
            'beta_F': 1.82,
            'k_p': 15.  # fixed
        }

        self.boundary = {
            'omega_b': (0.0, 0.05),
            'omega_cdm': (0.0, 0.3),
            'h': (0.5, 0.9),
            'n_s': (0.94, 1.),
            'ln10^{10}A_s': (2., 4.),
            'b_F': (-2, 0),
            'beta_F': (0, 5),
            'k_p': (0, 200),
        }

        # self.param_labels = {
        # }

    def getPlinInterp(self, **kwargs):
        ndim = np.max([len(kwargs[key]) for key in self._cosmo_names])

        # create a dict of cosmological parameters
        emu_params = {}
        for key in self._cosmo_names:
            emu_params[key] = kwargs[key]
            if len(emu_params[key]) == 1:
                emu_params[key] = np.ones(ndim) * emu_params[key][0]
            elif len(emu_params[key]) != ndim:
                raise Exception("Wrong dimensions in emu_params!")

        emu_params['z'] = self.z * np.ones(ndim)

        # shape (ndim, nkmodes) in Mpc
        return MyPlinInterp(self._cp_log10k, self._cp_emulator.predictions_np(emu_params))

    def getMpc2Kms(self, **kwargs):
        h = kwargs['h']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        return 100. * h * efunc(self.z, Om0) / (1 + self.z)

    def getKms2Mpc(self, **kwargs):
        return self.getMpc2Kms(**kwargs)**-1

    def _integrandB3d(self, qb, pb, w, k, plin_interp, Om0, h, kp, is_gh=False):
        """
        Om0, h, k_p are ncosmo size arrays.
        plin_interp returns (ncosmo, nk) size array.
        Returned array needs to be normalized by dividing by (2pi)^3
        units: Mpc^4
        """
        k2 = k**2
        qb2 = qb**2
        pb2 = pb**2

        # Meshgrid
        qb2, pb2, w = np.meshgrid(qb2, pb2, w, indexing='ij')
        qpb = np.outer(qb, pb)[:, :, np.newaxis]
        tb = np.sqrt(qb2 + pb2 + 2 * qpb * w)

        q = np.sqrt(qb2 + k2)
        p = np.sqrt(pb2 + k2)
        w = (qpb * w - k2) / (q * p)

        # Absorb pressure smoothing of qb, pb into Gauss-Hermite quadrature
        b3d = getBispectrumTree(q, p, w, plin_interp) * np.exp(-2 * k2 / kp**2)

        if not is_gh:
            b3d *= np.exp(- (qb2 + pb2) / kp**2)

        chiz = self.chiz_om0_mpch_fn(Om0) / h
        return b3d * qpb * self.wiener(np.multiply.outer(chiz, tb))

    def _integrateB3dFncTrapz(self, k, **kwargs):
        """
        k: Mpc^-1
        B1d: Mpc
        """
        h = kwargs['h']
        kp = kwargs['k_p']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        plin_interp = self.getPlinInterp(**kwargs)

        qb = LyaxCmbModel.K_INTEG_ARRAY
        pb = LyaxCmbModel.K_INTEG_ARRAY

        # shape: (ncosmo, qb.size, pb.size, x_w.size)
        integrand = self._integrandB3d(
            qb, pb, LyaxCmbModel.W_INTEG_ARRAY, k, plin_interp, Om0, h, kp
        ).dot(LyaxCmbModel.W_WEIGHT_ARRAY)  # Mpc^4

        integrand = np.trapz(
            integrand * pb, dx=LyaxCmbModel.DLNK_INTEG, axis=-1)
        result = np.trapz(
            integrand * qb, dx=LyaxCmbModel.DLNK_INTEG, axis=-1)
        norm = h * self.kappa_om0_interp_hMpc(Om0) / (4. * np.pi**3)  # Mpc^-1
        return norm * result

    def integrateB3dTrapz(self, k, **kwargs):
        return np.vectorize(
            functools.partial(self._integrateB3dFncTrapz, **kwargs),
            signature='()->(m)')(k).T

    def _integrateB3dFncGH(self, k, **kwargs):
        """
        k: Mpc
        """
        h = kwargs['h']
        kp = kwargs['k_p']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        plin_interp = self.getPlinInterp(**kwargs)

        x_bot, weight_bot = np.polynomial.hermite.hermgauss(2 * 30)
        qb = x_bot[30:] * kp
        weight_bot = weight_bot[30:]
        pb = qb

        # shape: (ncosmo, qb.size, pb.size, x_w.size)
        integrand = self._integrandB3d(
            qb, pb, LyaxCmbModel.W_INTEG_ARRAY, k, plin_interp, Om0, h, kp,
            is_gh=True).dot(LyaxCmbModel.W_WEIGHT_ARRAY)
        integrand = integrand.dot(weight_bot)
        result = integrand.dot(weight_bot)
        norm = kp**2 * self.kappa_om0_interp_hMpc(Om0) / (4. * np.pi**3)
        return norm * result

    def integrateB3dGH(self, k, **kwargs):
        return np.vectorize(
            functools.partial(self._integrateB3dFncGH, **kwargs),
            signature='()->(m)')(k).T
