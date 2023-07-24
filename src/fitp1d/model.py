import itertools

import numpy as np
from scipy.interpolate import CubicSpline

LYA_WAVELENGTH = 1215.67
LIGHT_SPEED = 299792.458

PDW_FIT_AMP = 7.63089e-02
PDW_FIT_N = -2.52054e+00
PDW_FIT_APH = -1.27968e-01
PDW_FIT_B = 3.67469e+00
PDW_FIT_BETA = 2.85951e-01
PDW_FIT_LMD = 7.33473e+02

PDW_FIT_PARAMETERS = (
    PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, PDW_FIT_BETA, PDW_FIT_LMD)
PDW_FIT_PARAMETERS_0BETA = (
    PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, 0, PDW_FIT_LMD)
PD13_PIVOT_K = 0.009
PD13_PIVOT_Z = 3.0


def meanFluxFG08(z):
    tau = 0.001845 * np.power(1. + z, 3.924)

    return np.exp(-tau)


def evaluatePD13Lorentz(k, z, A, n, alpha, B, beta, k1):
    q0 = k / PD13_PIVOT_K + 1e-10

    result = (
        (A * np.pi / PD13_PIVOT_K)
        * np.power(q0, 2 + n + alpha * np.log(q0)) / (1 + (k / k1)**2)
    )

    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)

    return result


class Model():
    def __init__(self):
        self.names = []
        self.boundary = {}
        self.initial = {}
        self.param_labels = {}


class IonModel(Model):
    Transitions = {
        "Si-II": [(1190.42, 2.77e-01), (1193.28, 5.75e-01),
                  (1194.50, 7.37e-01), (1197.39, 1.50e-01)],
        "Si-III": [(1206.53, 1.52e+00), (1207.52, 5.30e-01)]
    }

    PivotF = {"Si-II": 7.37e-01, "Si-III": 1.52e+00}

    def _setConstA2Terms(self):
        self._splines['const_a2'] = {}
        for ion, transitions in IonModel.Transitions.items():
            self._splines['const_a2'][f"a_{ion}"] = 0
            fpivot = IonModel.PivotF[ion]

            for wave, fn in transitions:
                self._splines['const_a2'][f"a_{ion}"] += (fn / fpivot)**2

    def _setLinearATerms(self, kmin=0, kmax=10, nkpoints=1000000):
        self._splines['linear_a'] = {}

        karr = np.linspace(kmin, kmax, nkpoints)
        for ion, transitions in IonModel.Transitions.items():
            result = np.zeros(nkpoints)
            fpivot = IonModel.PivotF[ion]

            for wave, fn in transitions:
                vn = LIGHT_SPEED * np.log(LYA_WAVELENGTH / wave)
                r = fn / fpivot
                result += 2 * r * np.cos(karr * vn)

            self._splines['linear_a'][f"a_{ion}"] = CubicSpline(karr, result)

    def _setOneionA2Terms(self, kmin=0, kmax=10, nkpoints=1000000):
        self._splines['oneion_a2'] = {}

        karr = np.linspace(kmin, kmax, nkpoints)
        for ion, transitions in IonModel.Transitions.items():
            result = np.zeros(nkpoints)
            fpivot = IonModel.PivotF[ion]

            for p1, p2 in itertools.combinations(transitions, 2):
                vmn = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))
                r = (p1[1] / fpivot) * (p2[1] / fpivot)
                result += 2 * r * np.cos(karr * vmn)

            self._splines['oneion_a2'][f"a_{ion}"] = CubicSpline(karr, result)

    def _setTwoionA2Terms(self, kmin=0, kmax=10, nkpoints=1000000):
        self._splines['twoion_a2'] = {}

        karr = np.linspace(kmin, kmax, nkpoints)
        ions = list(IonModel.Transitions.keys())
        for i1, i2 in itertools.combinations(ions, 2):
            result = np.zeros(nkpoints)
            t1 = IonModel.Transitions[i1]
            t2 = IonModel.Transitions[i2]
            fp1 = IonModel.PivotF[i1]
            fp2 = IonModel.PivotF[i2]

            for (p1, p2) in itertools.product(t1, t2):
                vmn = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))
                r = (p1[1] / fp1) * (p2[1] / fp2)
                result += 2 * r * np.cos(karr * vmn)

            self._splines['twoion_a2'][f"a_{i1}-a_{i2}"] = \
                CubicSpline(karr, result)

    def __init__(self):
        super().__init__()
        self.names = ["a_Si-II", "a_Si-III"]
        self._name_combos = itertools.combinations(self.names, 2)
        self.param_labels = {
            "a_Si-II": r"a-{\mathrm{Si~II}}",
            "a_Si-III": r"a-{\mathrm{Si~III}}"
        }

        self.initial = {k: 1e-2 for k in self.names}
        self.boundary = {k: (-1, 1) for k in self.names}

        self._splines = {}
        self._integrated_model = {}
        self.kfine = None

        self._setConstA2Terms()
        self._setLinearATerms()
        self._setOneionA2Terms()
        self._setTwoionA2Terms()

    def integrate(self, kedges, nsubk=20):
        k1, k2 = kedges
        nkbins = k1.size
        self.kfine = np.linspace(k1, k2, nsubk, endpoint=False).T

        self._integrated_model['const_a2'] = self._splines['const_a2'].copy()
        self._integrated_model['linear_a'] = {}
        self._integrated_model['oneion_a2'] = {}
        self._integrated_model['twoion_a2'] = {}

        for term in ['linear_a', 'oneion_a2', 'twoion_a2']:
            for ionkey, interp in self._splines[term].items():
                v = interp(self.kfine).reshape(nkbins, nsubk).mean(axis=1)
                self._integrated_model[term][ionkey] = v

    def getCachedModel(self, **kwargs):
        result = np.ones_like(self.kfine)

        for key in self.names:
            asi = kwargs[key]
            result += asi * self._integrated_model['linear_a'][key]
            result += (
                self._integrated_model['const_a2'][key]
                + self._integrated_model['oneion_a2'][key]
            ) * asi**2

        for (key1, key2) in self._name_combos:
            a1 = kwargs[key1]
            a2 = kwargs[key2]
            m = self._integrated_model['twoion_a2'][f"{key1}-{key2}"]
            result += a1 * a2 * m

        return result

    def cache(self, kfine):
        self._integrated_model['const_a2'] = self._splines['const_a2'].copy()
        self._integrated_model['linear_a'] = {}
        self._integrated_model['oneion_a2'] = {}
        self._integrated_model['twoion_a2'] = {}
        self.kfine = kfine

        for term in ['linear_a', 'oneion_a2', 'twoion_a2']:
            for ionkey, interp in self._splines[term].items():
                self._integrated_model[term][ionkey] = interp(kfine)


class ResolutionModel(Model):
    def __init__(self, add_bias=True, add_variance=True):
        super().__init__()

        # self.names = ["b_reso", "var_reso"]
        # self.boundary = {'b_reso': (-0.1, 0.1), 'var_reso': (-0.0001, 0.1)}
        # self.initial = {'b_reso': 0, 'var_reso': 0}
        # self.param_labels = {'b_reso': r"b_R", 'var_reso': r"\sigma^2_R"}

        if add_bias:
            self.names.append("b_reso")
            self.boundary['b_reso'] = (-0.1, 0.1)
            self.initial['b_reso'] = 0
            self.param_labels['b_reso'] = r"b_R"

        if add_variance:
            self.names.append("var_reso")
            self.boundary['var_reso'] = (-0.0001, 0.1)
            self.initial['var_reso'] = 0
            self.param_labels['var_reso'] = r"\sigma^2_R"

        self.rkms = None
        self.kfine = None
        self._cached_model = None

    def cache(self, kfine, rkms):
        self.kfine = kfine
        self.rkms = rkms
        self._cached_model = {}

        if "b_reso" in self.names:
            self._cached_model["b_reso"] = -(kfine * rkms)**2

        if "var_reso" in self.names:
            self._cached_model["var_reso"] = (kfine * rkms)**4 / 2

    def getCachedModel(self, **kwargs):
        if not self.names:
            return 1

        result = np.zeros_like(self.kfine)
        for rkey in self.names:
            result += kwargs[rkey] * self._cached_model[rkey]

        return np.exp(result)


class LyaP1DSimpleModel(Model):
    def __init__(self):
        super().__init__()

        self.names = [
            'A', 'n', 'alpha', 'B', 'beta', 'k1'
        ]

        self.initial = {
            'A': PDW_FIT_AMP,
            'n': PDW_FIT_N,
            'alpha': PDW_FIT_APH,
            'B': PDW_FIT_B,
            'beta': PDW_FIT_BETA,
            'k1': 1 / np.sqrt(PDW_FIT_LMD)
        }

        self.param_labels = {
            "A": "A", "n": "n", "alpha": r"\alpha",
            "B": "B", "beta": r"\beta", "k1": r"k_1"
        }

        for par in self.names:
            x1, x2 = -100, 100
            if par == "k1":
                x1 = 1e-6
            self.boundary[par] = (x1, x2)

        self.nsubk = 20
        self.z = None
        self.kfine = None
        self.ndata = None

    def cache(self, kedges, z):
        assert isinstance(kedges, tuple)
        k1, k2 = kedges
        self.z = z
        self.kfine = np.linspace(k1, k2, self.nsubk, endpoint=False).T
        self.ndata = k1.size

    def getCachedModel(self, **kwargs):
        A, n, alpha, B, beta, k1 = (
            kwargs['A'], kwargs['n'], kwargs['alpha'], kwargs['B'],
            kwargs['beta'], kwargs['k1']
        )

        result = evaluatePD13Lorentz(
            self.kfine, self.z, A, n, alpha, B, beta, k1)

        return result


class LyaP1DArinyoModel(Model):
    def __init__(self):
        super().__init__()

        self.names = ['blya', 'beta', 'q1', 'kv', 'av', 'bv', 'kp']

        self.initial = {
            'blya': -0.2,
            'beta': 1.67,
            'q1': 0.65,
            'kv': 0.8,
            'av': 0.5,
            'bv': 1.55,
            'kp': 13.0
        }

        self.param_labels = {
            "blya": r"b_\mathrm{Lya}", "beta": r"\beta_\mathrm{Lya}",
            "q1": r"q_1", "kv": r"k_\nu", "av": r"a_\nu", "bv": r"b_\nu",
            "kp": r"k_p"
        }
        self.boundary = {
            'blya': (-1, 0),
            'beta': (0.5, 2.5),
            'q1': (0.1, 5.),
            'kv': (0.1, 5.),
            'av': (0.1, 5.),
            'bv': (0.1, 5.),
            'kp': (1., 50.)
        }

        self.nsubk = 20
        self._kperp, self._dlnkperp = np.linspace(-4, 3, 1000, retstep=True)
        self._kperp = np.exp(self._kperp)[:, np.newaxis, np.newaxis]
        self._kperp2pi = self._kperp**2 / (2 * np.pi)

        self.z = None
        self._p3dlin = None
        self.kfine_skm = None
        self._k1d_Mpc = None
        self._k3d = None
        self._mu = None
        self.Mpc2kms = None
        self.ndata = None

    def cache(self, kedges, z):
        assert isinstance(kedges, tuple)
        k1, k2 = kedges
        self.z = z
        self.kfine = np.linspace(k1, k2, self.nsubk, endpoint=False).T
        self.ndata = k1.size

        import camb
        from astropy.cosmology import Planck18

        camb_params = camb.set_params(
            redshifts=[z],
            WantTransfer=True, kmax=1e4,
            omch2=Planck18.Odm0 * Planck18.h**2,
            ombh2=Planck18.Ob0 * Planck18.h**2,
            omk=0.,
            H0=Planck18.H0.value,
            ns=Planck18.meta['n']
        )
        camb_results = camb.get_results(camb_params)

        camb_interp = camb_results.get_matter_power_interpolator(
            nonlinear=False,
            hubble_units=False,
            k_hunit=False)

        self.Mpc2kms = Planck18.H(z).value / (1 + z)
        self._k1d_Mpc = self.kfine * self.Mpc2kms
        self._k1d_Mpc = self._k1d_Mpc[np.newaxis, :]
        self._k3d_Mpc = np.sqrt(self._kperp**2 + self._k1d_Mpc**2)
        self._mu = self._k1d_Mpc / self._k3d_Mpc
        self._p3dlin = camb_interp.P(z, self._k3d_Mpc)
        self._Delta2 = self._p3dlin * self._k3d_Mpc**3 / 2 / np.pi**2

    def getP3D(self, **kwargs):
        bias_rsd = (kwargs['blya'] * (1 + kwargs['beta'] * self._mu**2))**2
        t1 = (
            (self._k3d_Mpc / kwargs['kv'])**kwargs['av']
            * self._mu**kwargs['bv']
        )
        t2 = (self._k3d_Mpc / kwargs['kp'])**2
        Fnl = np.exp(kwargs['q1'] * self._Delta2 * (1 - t1) - t2)

        return self._p3dlin * bias_rsd * Fnl

    def getCachedModel(self, **kwargs):
        p3d_flux = self.getP3D(**kwargs) * self._kperp2pi
        p1d_Mpc = np.trapz(p3d_flux, dx=self._dlnkperp, axis=0)
        return p1d_Mpc * self.Mpc2kms


class CombinedModel(Model):
    def _setAttr(self):
        self.names = []
        self.boundary = {}
        self.initial = {}
        self.param_labels = {}

        for M in self._models.values():
            self.names += M.names
            self.initial |= M.initial
            self.param_labels |= M.param_labels
            self.boundary |= M.boundary

    def __init__(self, add_reso_bias, add_var_reso):
        super().__init__()
        self._models = {
            'lya': LyaP1DArinyoModel(),
            'ion': IonModel(),
            'reso': ResolutionModel(add_reso_bias, add_var_reso)
        }

        self._setAttr()

    @property
    def ndata(self):
        return self._models['lya'].ndata

    @property
    def nsubk(self):
        return self._models['lya'].nsubk

    def cache(self, kedges, z):
        self._models['lya'].cache(kedges, z)
        kfine = self._models['lya'].kfine
        self._models['ion'].cache(kfine)

        rkms = LIGHT_SPEED * 0.8 / (1 + z) / LYA_WAVELENGTH
        self._models['reso'].cache(kfine, rkms)

    def getIntegratedModel(self, **kwargs):
        result = self._models['lya'].getCachedModel(**kwargs)
        result *= self._models['ion'].getCachedModel(**kwargs)
        result *= self._models['reso'].getCachedModel(**kwargs)
        result = result.reshape(self.ndata, self.nsubk).mean(axis=1)

        return result
