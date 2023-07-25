import numpy as np
from astropy.io import ascii

accepted_keys_k = ['k', 'kc']
accepted_keys_z = ['z', 'redshift', 'zLya']
accepted_keys_p = ['Pest', 'P-FFT', 'P1D', 'P_1D', 'PLya', 'p_final']
accepted_keys_e = [
    'ErrorP', 'error', 'sigma', 'e', 'e_stat', 'stat', 'e_total']


def _findAcceptedKeys(keys):
    set_keys = set(keys)
    kkey = set_keys.intersection(accepted_keys_k).pop()
    zkey = set_keys.intersection(accepted_keys_z).pop()
    pkey = set_keys.intersection(accepted_keys_p).pop()
    ekey = set_keys.intersection(accepted_keys_e).pop()

    return kkey, zkey, pkey, ekey


class PSData():
    """PowerSpectrum class read an astropy.io.ascii table with
    fixed_width format.

    Attributes
    ----------
    k : np.array(double)
    z : np.array(double)
    p : np.array(double)
    e : np.array(double)

    z_bins, k_bins : np.array(double)
    nz, nk = int

    k1, k2 : optional, np.array(double)
    cov : after setCovariance
    """

    def __init__(self, fname, fmt=None):
        self.fname = fname

        if fmt:
            power_table = ascii.read(fname, format=fmt)
        else:
            power_table = ascii.read(fname)

        kkey, zkey, pkey, ekey = _findAcceptedKeys(power_table.keys())

        self.k = np.array(power_table[kkey], dtype=float)
        self.z = np.array(power_table[zkey], dtype=float)
        self.p = np.array(power_table[pkey], dtype=float)
        self.e = np.array(power_table[ekey], dtype=float)

        if "p_noise" in power_table.keys():
            self.p_noise = np.array(power_table["p_noise"], dtype=float)

        self.z_bins = np.unique(self.z)
        self.nz = self.z_bins.size
        self.size = self.k.size

        # Read k edges
        if 'k1' in power_table.keys() and 'k2' in power_table.keys():
            self.k1 = np.array(power_table['k1'], dtype=float)
            self.k2 = np.array(power_table['k2'], dtype=float)
        else:
            self.k1 = self.k2 = None

        self.cov = None
        self.reso_tbl = None

    def trim(self, kmin=0, kmax=10, zmin=0, zmax=10):
        keep = (self.k > kmin) & (self.k < kmax)
        keep &= (self.z > zmin) & (self.z < zmax)
        keep &= self.e > 0

        if all(keep):
            return

        self.k = self.k[keep]
        self.z = self.z[keep]
        self.p = self.p[keep]
        self.e = self.e[keep]

        self.z_bins = np.unique(self.z)
        self.nz = self.z_bins.size

        if self.k1 is not None:
            self.k1 = self.k1[keep]
            self.k2 = self.k2[keep]

        if self.cov is not None:
            self.cov = self.cov[:, keep][keep, :]

    def readCovariance(self, fname, skiprows=1):
        self.cov = np.loadtxt(fname, skiprows=skiprows)
        assert self.cov.shape[0] == self.size

    def setResolution(self, fname, fmt='fixed_width'):
        self.reso_tbl = ascii.read(fname, format=fmt)

    def getResolution(self, z):
        return np.interp(z, self.reso_tbl['z'], self.reso_tbl['R'])

    def getZBinVals(self, z, kmin=0, kmax=10, get_pnoise=False, k2match=None):
        w = np.isclose(self.z, z)
        w &= (self.k >= kmin) & (self.k <= kmax) & (self.e > 0)

        result = np.empty(
            w.sum(), dtype=[('k', 'f8'), ('p', 'f8'), ('e', 'f8')])

        result['k'] = self.k[w]
        if get_pnoise:
            result['p'] = self.p_noise[w]
        else:
            result['p'] = self.p[w]
        result['e'] = self.e[w]
        kedges = (self.k1[w], self.k2[w])

        if self.cov is not None:
            cov = self.cov[:, w][w, :]
        else:
            cov = None

        if k2match is not None:
            idx = np.intersect1d(
                result['k'], k2match, assume_unique=True, return_indices=True
            )[1]

            result = result[idx]
            kedges = (kedges[0][idx], kedges[1][idx])
            if cov is not None:
                cov = cov[:, idx][idx, :]

        return result, kedges, cov
