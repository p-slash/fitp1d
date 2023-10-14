import numpy as np
from astropy.io import ascii


class DetailedData():
    """Class to read detailed results."""
    _expected_keys = (
        "z k1 k2 kc Pfid ThetaP p_final e_stat p_raw p_noise p_fid_qmle "
        "p_smooth e_n_syst e_res_syst e_cont_syst e_dla_syst p_sb1 e_sb1_stat "
        "e_total"
    ).split(' ')

    def _check_keys(self):
        data_keys = self.data_table.dtype.names
        for key in data_keys:
            if key not in DetailedData._expected_keys:
                print(f"WARNING: Column {key} in file is not expected.")

        for key in DetailedData._expected_keys:
            if key not in data_keys:
                print(f"WARNING: Column {key} is not found in file.")

    def __init__(self, fname, fmt=None):
        self.fname = fname

        if fmt:
            self.data_table = ascii.read(fname, format=fmt)
        else:
            self.data_table = ascii.read(fname)

        # self.data_table.rename_column("p_final", "p")
        self.data_table = np.array(self.data_table)
        self._check_keys()

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
