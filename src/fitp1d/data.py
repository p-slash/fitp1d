import fitsio
import numpy as np
import numpy.lib.recfunctions as nplr
from astropy.io import ascii


class P1dFitsFile():
    def __init__(self, fname):
        self.fname = fname
        with fitsio.FITS(fname) as fts:
            if 'P1D_BLIND' in fts:
                self._blind = True
                self.p1d_data = fts['P1D_BLIND'].read()
            else:
                self._blind = False
                self.p1d_data = fts['P1D'].read()
            self.esyst = fts['SYSTEMATICS'].read()
            self.cov = fts['COVARIANCE'].read()
            self.cov_stat = fts['COVARIANCE_STAT'].read()

    @property
    def size(self):
        return self.p1d_data.size


class DetailedData():
    """Class to read detailed results."""
    _expected_keys = (
        "z k1 k2 kc Pfid ThetaP p_final e_stat p_raw p_noise p_fid_qmle "
        "p_smooth e_total"
    ).split(' ')
    # e_n_syst e_res_syst e_cont_syst e_dla_syst p_sb1 e_sb1_stat

    _map_from_fts = {
        "Z": "z", "K1": "k1", "K2": "k2", "K": "kc", "PINPUT": "Pfid",
        "PLYA": "p_final", "E_PK": "e_total", "PRAW": "p_raw",
        "PNOISE": "p_noise", "PFID": "p_fid_qmle",
        "E_STAT": "e_stat", "PSMOOTH": "p_smooth", "E_SYST": "e_syst_total"
    }

    def _check_keys(self):
        data_keys = self.data_table.dtype.names
        for key in data_keys:
            if key not in DetailedData._expected_keys:
                print(f"WARNING: Column {key} in file is not expected.")

        for key in DetailedData._expected_keys:
            if key not in data_keys:
                print(f"WARNING: Column {key} is not found in file.")

    @classmethod
    def fromP1dFitsFile(cls, p1d_fits):
        if isinstance(p1d_fits, str):
            p1d_fits = P1dFitsFile(p1d_fits)

        dsyst = nplr.drop_fields(
            p1d_fits.esyst, ["Z", "K", "E_SYST"], usemask=False)
        dsyst_keys = dsyst.dtype.names
        syst_map = {
            key: f"{key.lower()}_syst" for key in dsyst_keys
            if key.startswith("E_")
        }

        res = nplr.merge_arrays(
            (p1d_fits.p1d_data, dsyst), flatten=True, usemask=False)
        res = nplr.rename_fields(res, DetailedData._map_from_fts | syst_map)
        p = cls(res, p1d_fits.fname)
        p.setCovariance(p1d_fits.cov)
        return p

    def fromFile(cls, fname, fmt=None):
        if fmt:
            data_table = ascii.read(fname, format=fmt)
        else:
            data_table = ascii.read(fname)

        return cls(np.array(data_table), fname)

    def __init__(self, data_table, fname=None):
        self.data_table = data_table.copy()
        self.fname = fname
        self._check_keys()
        self.z_bins = np.unique(self.data_table['z'])

    @property
    def size(self):
        return self.data_table.size

    def readCovariance(self, fname, skiprows=1):
        self.cov = np.loadtxt(fname, skiprows=skiprows)
        assert self.cov.shape[0] == self.size

    def setCovariance(self, cov):
        assert cov.shape[0] == self.size
        assert cov.shape[1] == self.size
        self.cov = cov.copy()

    def getZBinVals(self, z, kmin=0, kmax=10):
        w = np.isclose(self.data_table['z'], z)
        w &= (self.data_table['kc'] >= kmin) & (self.data_table['kc'] <= kmax)
        w &= (self.data_table['e_total'] > 0)

        result = self.data_table[w]

        if self.cov is not None:
            cov = self.cov[:, w][w, :]
        else:
            cov = None

        return result, cov

    def getZBinAsList(self, kmin=0, kmax=10):
        w = (self.data_table['kc'] >= kmin) & (self.data_table['kc'] <= kmax)
        w &= (self.data_table['e_total'] > 0)

        this_data = self.data_table[w]
        if self.cov is not None:
            cov = self.cov[:, w][w, :]
        else:
            cov = None

        zbins = np.unique(this_data['z'])
        output_power = []
        output_cov = []

        for zb in zbins:
            w2 = np.isclose(zb, this_data['z'])
            output_power.append(this_data[w2])
            if cov is not None:
                output_cov.append(cov[:, w2][w2, :])

        return zbins, output_power, output_cov
