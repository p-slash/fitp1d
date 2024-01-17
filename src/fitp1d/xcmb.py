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

    t = np.clip(np.sqrt(k**2 + q**2 + 2 * k * q * w), 1e-5, None)

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
    K_INTEG_LIMITS = 1e-3, 5e2
    LNK_INTEG_LIMITS = np.log(1e-3), np.log(5e2)
    LNK_INTEG_ARRAY, DLNK_INTEG = np.linspace(
        *LNK_INTEG_LIMITS, 500, retstep=True)
    K_INTEG_ARRAY = np.exp(LNK_INTEG_ARRAY)

    W_INTEG_ARRAY, W_WEIGHT_ARRAY = np.polynomial.chebyshev.chebgauss(40)

    @staticmethod
    def setChebyshev(n):
        LyaxCmbModel.W_INTEG_ARRAY, LyaxCmbModel.W_WEIGHT_ARRAY = \
            np.polynomial.chebyshev.chebgauss(n)

    @staticmethod
    def setLnkIntegration(n):
        LyaxCmbModel.LNK_INTEG_ARRAY, LyaxCmbModel.DLNK_INTEG = np.linspace(
            *LyaxCmbModel.LNK_INTEG_LIMITS, n, retstep=True)

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
        self._broadcasted_params = self._cosmo_names + ['b_F', 'beta_F']

        self.initial = {
            'omega_b': Planck18.Ob0 * Planck18.h**2,
            'omega_cdm': Planck18.Odm0 * Planck18.h**2,
            'h': Planck18.h,
            'n_s': Planck18.meta['n'],
            'ln10^{10}A_s': 3.044,
            'b_F': -0.136,
            'beta_F': 1.82,
            'k_p': 15.  # in Mpc^-1
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

    def broadcastKwargs(self, **kwargs):
        assert isinstance(kwargs['k_p'], float)

        ndim = np.max([len(kwargs[key]) for key in self._cosmo_names])

        for key in self._broadcasted_params:
            if key not in kwargs:
                kwargs[key] = np.ones(ndim) * self.initial[key]
            elif isinstance(kwargs[key], float):
                kwargs[key] = np.ones(ndim) * kwargs[key]
            elif len(kwargs[key]) == 1:
                kwargs[key] = np.ones(ndim) * kwargs[key][0]
            elif len(kwargs[key]) != ndim:
                raise Exception("Wrong dimensions in kwargs!")

        kwargs['z'] = self.z * np.ones(ndim)
        return kwargs

    def getPlinInterp(self, **kwargs):
        emu_params = {key: kwargs[key] for key in self._cosmo_names}
        emu_params['z'] = kwargs['z']

        # shape (ndim, nkmodes) in Mpc
        return MyPlinInterp(self._cp_log10k, self._cp_emulator.predictions_np(emu_params))

    def getMpc2Kms(self, **kwargs):
        h = kwargs['h']
        Om0 = (kwargs['omega_b'] + kwargs['omega_cdm']) / h**2
        return 100. * h * efunc(self.z, Om0) / (1 + self.z)

    def getKms2Mpc(self, **kwargs):
        return self.getMpc2Kms(**kwargs)**-1

    def _integrandB3dPhi(self, qb, pb, w, k, plin_interp, Om0, h):
        # Meshgrid
        qb2, pb2, ww = np.meshgrid(qb**2, pb**2, w, indexing='ij', copy=True)
        qpb = np.outer(qb, pb)[:, :, np.newaxis]
        ww *= qpb
        tb = np.sqrt(qb2 + pb2 + 2 * ww)
        chiz = self.chiz_om0_mpch_fn(Om0) / h
        b3d = qpb * self.wiener(np.multiply.outer(chiz, tb))

        k2 = k**2
        q = np.sqrt(qb2 + k2)
        p = np.sqrt(pb2 + k2)
        ww -= k2
        ww /= q
        ww /= p

        # Absorb pressure smoothing of qb, pb into Gauss-Hermite quadrature
        b3d *= getBispectrumTree(q, p, ww, plin_interp)

        return b3d

    def _integrandB3dPerp(self, qb, pb, w, k, plin_interp, Om0, h, kp, betaF):
        # shape: (ncosmo, qb.size, pb.size)
        b3d = self._integrandB3dPhi(
            qb, pb, LyaxCmbModel.W_INTEG_ARRAY, k, plin_interp, Om0, h
        ).dot(LyaxCmbModel.W_WEIGHT_ARRAY)  # Mpc^4

        qb2, pb2 = np.meshgrid(qb**2, pb**2, indexing='ij', copy=True)
        k2 = k**2
        q = qb2 + k2
        p = pb2 + k2
        b3d *= (1 + np.divide.outer(betaF * k2, q)) * (1 - np.divide.outer(betaF * k2, p))
        b3d *= np.exp(-(qb2 + pb2) / kp**2)
        return b3d

    def _integrandB3d(self, qb, pb, w, k, plin_interp, Om0, h, kp, betaF):
        """
        Om0, h, k_p are ncosmo size arrays.
        plin_interp returns (ncosmo, nk) size array.
        Returned array needs to be normalized by dividing by (2pi)^3
        units: Mpc^4
        """
        # Meshgrid
        qb2, pb2, ww = np.meshgrid(qb**2, pb**2, w, indexing='ij', copy=True)
        qpb = np.outer(qb, pb)[:, :, np.newaxis]
        ww *= qpb
        tb = np.sqrt(qb2 + pb2 + 2 * ww)
        chiz = self.chiz_om0_mpch_fn(Om0) / h
        b3d = qpb * self.wiener(np.multiply.outer(chiz, tb))

        k2 = k**2
        q = qb2 + k2
        p = pb2 + k2
        bb = (1 + np.divide.outer(betaF * k2, q)) * (1 - np.divide.outer(betaF * k2, p))
        b3d *= bb
        np.sqrt(q, out=q)
        np.sqrt(p, out=p)
        ww -= k2
        ww /= q
        ww /= p

        # Absorb pressure smoothing of qb, pb into Gauss-Hermite quadrature
        b3d *= getBispectrumTree(q, p, ww, plin_interp) * np.exp(-2 * k2 / kp**2)
        b3d *= np.exp(-(qb2 + pb2) / kp**2)

        return b3d

    def _integrateB3dFncTrapz(self, k, **kwargs):
        """ Gauss-Hermite quadrature does not work
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
        integrand = self._integrandB3dPerp(
            qb, pb, LyaxCmbModel.W_INTEG_ARRAY, k, plin_interp, Om0, h, kp,
            kwargs['beta_F']
        )  # Mpc^4

        integrand = np.trapz(
            integrand * pb, dx=LyaxCmbModel.DLNK_INTEG, axis=-1)
        result = np.trapz(
            integrand * qb, dx=LyaxCmbModel.DLNK_INTEG, axis=-1)
        norm = h * kwargs['b_F']**2 * self.kappa_om0_interp_hMpc(Om0) / (4. * np.pi**3)  # Mpc^-1
        return norm * result * np.exp(-2 * k**2 / kp**2)

    def integrateB3dTrapz(self, k, **kwargs):
        kwargs = self.broadcastKwargs(**kwargs)
        return np.vectorize(
            functools.partial(self._integrateB3dFncTrapz, **kwargs),
            signature='()->(m)')(k).T
