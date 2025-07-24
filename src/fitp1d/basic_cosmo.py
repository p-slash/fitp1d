import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

LIGHT_SPEED = 299792.458  # km / s
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


def getLinearGrowthMasai(z, Om0):
    # https://arxiv.org/abs/1012.2671
    a = 1.0 / (1 + z)
    x = (1 - Om0) / Om0 * a**3

    bs = np.array([0.005355, 0.3064, 1.175, 1.0])
    cs = np.array([0.1530, 1.021, 1.857, 1.0])
    return a * np.sqrt(1 + x) * np.polyval(bs, x) / np.polyval(cs, x)


def getGrowthFactorMasaiDeriv(z, Om0):
    # Analytic derivative of getLinearGrowthMasai
    a = 1.0 / (1 + z)
    x = (1 - Om0) / Om0 * a**3

    bs = np.array([0.005355, 0.3064, 1.175, 1.0])
    bsp = np.array([0.016065, 0.6128, 1.175])
    cs = np.array([0.1530, 1.021, 1.857, 1.0])
    csp = np.array([0.459, 2.042, 1.857])

    fpf = (1.0 + 2.5 * x) / (3.0 * x * (1.0 + x))
    ppp = np.polyval(bsp, x) / np.polyval(bs, x)
    qpq = np.polyval(csp, x) / np.polyval(cs, x)
    return 3.0 * x * (fpf + ppp - qpq)


def getGrowthFactorOmApprox(z, Om0, nu=0.547):
    Om = Om0 * (1.0 + z)**3
    up = 1.0 + (1.0 - Om0) / Om
    return up**-nu


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
            log10k, MyPlinInterp.LOG10_KMAX - delta2 * 0.1,
            MyPlinInterp.LOG10_KMAX))
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
