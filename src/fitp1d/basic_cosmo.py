import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import quad

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
