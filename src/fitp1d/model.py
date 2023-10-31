import itertools

from astropy.cosmology import Planck18
import camb
import numpy as np
from scipy.interpolate import CubicSpline

from fitp1d.forecast import getHubbleZ

LYA_WAVELENGTH = 1215.67
LIGHT_SPEED = 299792.458

_NSUB_K_ = 5

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
        self.prior = {}


class IonModel(Model):
    Transitions = {
        "Si-II": [
            (1190.42, 0.277), (1193.28, 0.575), (1260.42, 1.22),
            (1304.37, 0.0928), (1526.72, 0.133)
        ],
        "Si-III": [(1206.52, 1.67)],
        "O-I": [(1302.168, 0.0520)]
    }

    PivotF = {"Si-II": 1.22, "Si-III": 1.67, "O-I": 0.0520}
    VMax = LIGHT_SPEED * np.log(1180. / 1050.) / 2.

    def _setConstA2Terms(self):
        self._splines['const_a2'] = {}
        for ion in self._ions:
            transitions = IonModel.Transitions[ion]
            self._splines['const_a2'][f"a_{ion}"] = 0
            fpivot = IonModel.PivotF[ion]

            for wave, fn in transitions:
                self._splines['const_a2'][f"a_{ion}"] += (fn / fpivot)**2

    def _setLinearATerms(self, kmin=0, kmax=1, nkpoints=int(1e6)):
        self._splines['linear_a'] = {}

        karr = np.linspace(kmin, kmax, nkpoints)
        for ion in self._ions:
            transitions = IonModel.Transitions[ion]
            result = np.zeros(nkpoints)
            fpivot = IonModel.PivotF[ion]

            for wave, fn in transitions:
                vn = np.abs(LIGHT_SPEED * np.log(LYA_WAVELENGTH / wave))

                if vn > self._vmax:
                    continue

                r = fn / fpivot
                result += 2 * r * np.cos(karr * vn)

                print(f"_setLinearATerms({ion}, {wave}): {vn:.0f}")

            self._splines['linear_a'][f"a_{ion}"] = CubicSpline(karr, result)

    def _setOneionA2Terms(self, kmin=0, kmax=1, nkpoints=int(1e6)):
        self._splines['oneion_a2'] = {}

        karr = np.linspace(kmin, kmax, nkpoints)
        for ion in self._ions:
            transitions = IonModel.Transitions[ion]
            result = np.zeros(nkpoints)
            fpivot = IonModel.PivotF[ion]

            for p1, p2 in itertools.combinations(transitions, 2):
                vmn = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))

                if vmn > self._vmax:
                    continue

                r = (p1[1] / fpivot) * (p2[1] / fpivot)
                result += 2 * r * np.cos(karr * vmn)

                print(f"_setOneionA2Terms({ion}, {p1[0]}, {p2[0]}): {vmn:.0f}")

            self._splines['oneion_a2'][f"a_{ion}"] = CubicSpline(karr, result)

    def _setTwoionA2Terms(self, kmin=0, kmax=1, nkpoints=int(1e6)):
        i2a2 = {}
        self._name_combos = []

        karr = np.linspace(kmin, kmax, nkpoints)
        for i1, i2 in itertools.combinations(self._ions, 2):
            result = np.zeros(nkpoints)
            all_zero = True
            t1 = IonModel.Transitions[i1]
            t2 = IonModel.Transitions[i2]
            fp1 = IonModel.PivotF[i1]
            fp2 = IonModel.PivotF[i2]

            for (p1, p2) in itertools.product(t1, t2):
                vmn = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))

                if vmn > self._vmax:
                    continue

                r = (p1[1] / fp1) * (p2[1] / fp2)
                result += 2 * r * np.cos(karr * vmn)
                all_zero = False
                print(f"_setTwoionA2Terms({i1}, {p1[0]}, {i2}, {p2[0]}): {vmn:.0f}")

            if all_zero:
                continue

            i2a2[f"a_{i1}-a_{i2}"] = CubicSpline(karr, result)
            self._name_combos.append((f"a_{i1}", f"a_{i2}"))

        self._splines['twoion_a2'] = i2a2

    def __init__(self, model_ions=["Si-II", "Si-III"], vmax=0):
        super().__init__()
        self._ions = model_ions.copy()
        self.names = [f"a_{ion}" for ion in model_ions]
        self._name_combos = []
        self.param_labels = {
            "a_Si-II": r"a-{\mathrm{Si~II}}",
            "a_Si-III": r"a-{\mathrm{Si~III}}",
            "a_O-I": r"a-{\mathrm{O~I}}"
        }

        if vmax > 0:
            self._vmax = vmax
        else:
            self._vmax = IonModel.VMax

        self.initial = {k: 1e-2 for k in self.names}
        self.boundary = {k: (-1, 1) for k in self.names}

        self._splines = {}
        self._integrated_model = {}
        self.kfine = None

        self._setConstA2Terms()
        self._setLinearATerms()
        self._setOneionA2Terms()
        self._setTwoionA2Terms()

    def getAllVelocitySeparations(self):
        vseps = {}

        for ion, transitions in IonModel.Transitions.items():
            for wave, fn in transitions:
                key = f'Lya-{ion} ({wave:.0f})'
                vseps[key] = np.abs(LIGHT_SPEED * np.log(LYA_WAVELENGTH / wave))

        for ion, transitions in IonModel.Transitions.items():
            for p1, p2 in itertools.combinations(transitions, 2):
                key = f'{ion} ({p1[0]:.0f}-{p2[0]:.0f})'
                vseps[key] = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))

        ions = list(IonModel.Transitions.keys())
        for i1, i2 in itertools.combinations(ions, 2):
            t1 = IonModel.Transitions[i1]
            t2 = IonModel.Transitions[i2]

            for (p1, p2) in itertools.product(t1, t2):
                key = f"{i1} ({p1[0]:.0f}) - {i2} ({p2[0]:.0f})"
                vseps[key] = np.abs(LIGHT_SPEED * np.log(p2[0] / p1[0]))

        return vseps

    def integrate(self, kedges):
        k1, k2 = kedges
        nkbins = k1.size
        self.kfine = np.linspace(k1, k2, _NSUB_K_, endpoint=False).T

        self._integrated_model['const_a2'] = self._splines['const_a2'].copy()
        self._integrated_model['linear_a'] = {}
        self._integrated_model['oneion_a2'] = {}
        self._integrated_model['twoion_a2'] = {}

        for term in ['linear_a', 'oneion_a2', 'twoion_a2']:
            for ionkey, interp in self._splines[term].items():
                v = interp(self.kfine).reshape(nkbins, _NSUB_K_).mean(axis=1)
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

    def evaluate(self, k, **kwargs):
        result = np.zeros_like(k)

        for key in self.names:
            asi = kwargs[key]
            result += asi * self._splines['linear_a'][key](k)
            result += (
                self._splines['const_a2'][key]
                + self._splines['oneion_a2'][key](k)
            ) * asi**2

        for (key1, key2) in self._name_combos:
            a1 = kwargs[key1]
            a2 = kwargs[key2]
            m = self._splines['twoion_a2'][f"{key1}-{key2}"](k)
            result += a1 * a2 * m

        result += 1

        return result


class ResolutionModel(Model):
    def __init__(self, add_bias=True, add_variance=True):
        super().__init__()

        # self.names = ["b_reso", "var_reso"]
        # self.boundary = {'b_reso': (-0.1, 0.1), 'var_reso': (-0.0001, 0.1)}
        # self.initial = {'b_reso': 0, 'var_reso': 0}
        # self.param_labels = {'b_reso': r"b_R", 'var_reso': r"\sigma^2_R"}

        if add_bias:
            self.names.append("b_reso")
            self.boundary['b_reso'] = (-0.5, 0.5)
            self.initial['b_reso'] = 0
            self.param_labels['b_reso'] = r"b_R"

        if add_variance:
            self.names.append("var_reso")
            self.boundary['var_reso'] = (-0.0001, 0.5)
            self.initial['var_reso'] = 0
            self.param_labels['var_reso'] = r"\sigma^2_R"

        self.rkms = None
        self.kfine = None
        self._cached_model = None

        for k in self.names:
            self.prior[k] = 0.01

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


class NoiseModel(Model):
    def __init__(self):
        super().__init__()
        self.names = ['eta_noise']
        self.initial['eta_noise'] = 0
        self.param_labels['eta_noise'] = r"\eta_N"
        self.boundary['eta_noise'] = (-0.2, 0.2)
        self.prior['eta_noise'] = 0.01
        self._cached_noise = 0

    def cache(self, p_noise):
        self._cached_noise = p_noise.copy()

    def getCachedModel(self, **kwargs):
        return kwargs['eta_noise'] * self._cached_noise


class ScalingSystematicsModel(Model):
    def __init__(self, label):
        super().__init__()
        self.name = f'{label}_bias'
        self.names = [self.name]
        self.initial[self.name] = 0
        self.param_labels[self.name] = r"\eta_\mathrm{"f"{label}""}"
        self.boundary[self.name] = (-5, 5)
        self.prior[self.name] = 1
        self._cached_power = 0

    def cache(self, p_scale):
        self._cached_power = p_scale.copy()

    def getCachedModel(self, **kwargs):
        return kwargs[self.name] * self._cached_power


class LyaP1DSimpleModel(Model):
    def __init__(self):
        super().__init__()

        self.names = ['A', 'n', 'alpha', 'B', 'beta', 'k1']

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

        self.boundary = {
            "A": (0, 100), "n": (-10, 0), "alpha": (-10, 0),
            "B": (-10, 10), "beta": (-10, 10), "k1": (1e-6, np.inf)
        }

        self.z = None
        self.kfine = None
        self.ndata = None

    def cache(self, kedges, z):
        assert isinstance(kedges, tuple)
        k1, k2 = kedges
        self.z = z
        self.kfine = np.linspace(k1, k2, _NSUB_K_, endpoint=False).T
        self.ndata = k1.size

    def getCachedModel(self, **kwargs):
        A, n, alpha, B, beta, k1 = (
            kwargs['A'], kwargs['n'], kwargs['alpha'], kwargs['B'],
            kwargs['beta'], kwargs['k1']
        )

        result = evaluatePD13Lorentz(
            self.kfine, self.z, A, n, alpha, B, beta, k1)

        return result

    def evaluate(self, k, **kwargs):
        A, n, alpha, B, beta, k1 = (
            kwargs['A'], kwargs['n'], kwargs['alpha'], kwargs['B'],
            kwargs['beta'], kwargs['k1']
        )

        return evaluatePD13Lorentz(
            k, self.z, A, n, alpha, B, beta, k1)


class FiducialCorrectionModel(LyaP1DSimpleModel):
    def __init__(self, *args):
        super().__init__()
        self.initial = {par: args[i] for i, par in enumerate(self.names)}
        self.args = args
        self._cached_corr = 0

    def cache(self, kedges, z):
        super().cache(kedges, z)
        k1, k2 = kedges
        kcenter = (k1 + k2) / 2

        self._cached_corr = evaluatePD13Lorentz(
            self.kfine, self.z, *self.args
        ).reshape(self.ndata, _NSUB_K_).mean(axis=1)
        self._cached_corr -= evaluatePD13Lorentz(kcenter, self.z, *self.args)

    def getCachedModel(self):
        return self._cached_corr


class LyaP1DArinyoModel(Model):
    CAMB_KMAX = 2e2
    CAMB_KMIN = 1e-3

    def __init__(self):
        super().__init__()

        self._cosmo_names = ['ln10As', 'ns', 'mnu', 'Ode0', 'H0']
        self.names = ['blya', 'bbeta', 'q1', '10kv', 'av', 'bv', 'kp']

        self.initial = {
            'blya': -0.2,
            'bbeta': -0.334,
            'q1': 0.65,
            'kv': 8.,
            'av': 0.5,
            'bv': 1.55,
            'kp': 13.0,
            'ln10As': 3.044,
            'ns': Planck18.meta['n'],
            'mnu': 8.,
            'Ode0': Planck18.Ode0,
            'H0': Planck18.H0.value
        }

        self.param_labels = {
            "blya": r"b_\mathrm{Lya}", "bbeta": r"b\beta_\mathrm{Lya}",
            "q1": r"$q_1$", "10kv": r"$k_\nu$ [$10^{-1}~$Mpc]", "av": r"$a_\nu$",
            "bv": r"$b_\nu$", "kp": r"$k_p$ [Mpc]", "ln10As": r"$\ln(10^{10} A_s)$",
            "ns": r"$n_s$", "mnu": r"$\sum m_\nu$ [$10^{-2}~$eV]",
            "Ode0": r"$\Omega_\Lambda$", "H0": r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
        }

        self.boundary = {
            'blya': (-5, 0),
            'bbeta': (-5, 0),
            'q1': (0.1, 5.),
            '10kv': (0., 50.),
            'av': (0.1, 5.),
            'bv': (0.1, 5.),
            'kp': (1., 50.),
            'ln10As': (2., 4.),
            'ns': (0.94, 1.),
            'mnu': (0., 50.),
            'Ode0': (0.5, 0.9),
            'H0': (50., 100.)
        }

        self._kperp, self._dlnkperp = np.linspace(
            np.log(LyaP1DArinyoModel.CAMB_KMIN),
            np.log(LyaP1DArinyoModel.CAMB_KMAX), 500, retstep=True)
        self._kperp = np.exp(self._kperp)[:, np.newaxis, np.newaxis]
        self._kperp2pi = self._kperp**2 / (2 * np.pi)

        self.z = None
        self._p3dlin = None
        self.kfine = None
        self._k1d_Mpc = None
        self._k3d = None
        self._mu = None
        self.Mpc2kms = None
        self.ndata = None
        self.fixedCosmology = False
        self._cosmo_interp = None

    def cacheZAndK(self, kedges, z):
        assert isinstance(kedges, tuple)

        if self.z is not None and np.isclose(self.z, z):
            return

        k1, k2 = kedges
        self.kfine = np.linspace(k1, k2, _NSUB_K_, endpoint=False).T
        self.ndata = k1.size
        self.z = z

        # shape = (self._kperp.shape[0], k1.size, _NSUB_K_)

    def newCambInterp(self, **kwargs):
        H0, Ode0 = kwargs['H0'], kwargs['Ode0']
        h = H0 / 100.

        camb_params = camb.set_params(
            redshifts=[self.z],
            WantCls=False, WantScalars=False,
            WantTensors=False, WantVectors=False,
            WantDerivedParameters=False,
            WantTransfer=True,
            omch2=(1 - Ode0 - Planck18.Ob0) * h**2,
            ombh2=Planck18.Ob0 * h**2,
            omk=0.,
            H0=H0,
            ns=kwargs['ns'],
            As=np.exp(kwargs['ln10As']) * 1e-10,
            mnu=kwargs['mnu'] / 100.
        )
        camb_results = camb.get_results(camb_params)

        khs, _, pk = camb_results.get_linear_matter_power_spectrum(
            nonlinear=False, hubble_units=False)
        khs *= h
        np.log(pk, out=pk)
        np.log(khs, out=khs)

        # Add extrapolation data points as done in camb
        logextrap = np.log(2 * LyaP1DArinyoModel.CAMB_KMAX)
        delta = logextrap - khs[-1]

        pk0 = pk[:, -1]
        dlog = (pk0 - pk[:, -2]) / (khs[-1] - khs[-2])
        pk = np.column_stack((pk, pk0 + dlog * delta * 0.9, pk0 + dlog * delta))
        khs = np.hstack((khs, logextrap - delta * 0.1, logextrap))

        self._cosmo_interp = CubicSpline(
            khs, pk, bc_type='natural', extrapolate=True
        )

    def newKandP(self, k1d_skm, **kwargs):
        H0, Ode0 = kwargs['H0'], kwargs['Ode0']
        if k1d_skm is None:
            k1d_skm = self.kfine

        self.Mpc2kms = getHubbleZ(self.z, H0, Ode0) / (1 + self.z)

        _k1d_Mpc = (k1d_skm * self.Mpc2kms)[np.newaxis, :]
        self._k3d_Mpc = np.sqrt(self._kperp**2 + _k1d_Mpc**2)
        self._mu = _k1d_Mpc / self._k3d_Mpc

        self._p3dlin = np.exp(self._cosmo_interp(np.log(self._k3d_Mpc)))
        self._Delta2 = self._p3dlin * self._k3d_Mpc**3 / 2 / np.pi**2

    def evaluateP3D(self, k1d_skm=None, **kwargs):
        if not self.fixedCosmology:
            self.newCambInterp(**kwargs)
        if not self.fixedCosmology or k1d_skm is not None:
            self.newKandP(k1d_skm, **kwargs)

        t1 = (
            (self._k3d_Mpc / kwargs['10kv'])**kwargs['av']
            * self._mu**kwargs['bv']
        ) * 10**-kwargs['av']
        t2 = (self._k3d_Mpc / kwargs['kp'])**2
        p3d = np.exp(kwargs['q1'] * (1 - t1) - t2)
        p3d *= self._p3dlin
        p3d *= (kwargs['blya'] + kwargs['bbeta'] * self._mu**2)**2

        return p3d

    def evaluateP1D(self, k1d_skm, **kwargs):
        p3d_flux = self.evaluateP3D(k1d_skm, **kwargs) * self._kperp2pi
        p1d_kms = self.Mpc2kms * np.trapz(p3d_flux, dx=self._dlnkperp, axis=0)
        return p1d_kms

    def getCachedModel(self, **kwargs):
        p3d_flux = self.evaluateP3D(**kwargs) * self._kperp2pi
        p1d_kms = self.Mpc2kms * np.trapz(p3d_flux, dx=self._dlnkperp, axis=0)
        return p1d_kms


class CombinedModel(Model):
    def _setAttr(self):
        self.names = []
        self.boundary = {}
        self.initial = {}
        self.param_labels = {}
        self.prior = {}

        for M in self._models.values():
            self.names += M.names
            self.initial |= M.initial
            self.param_labels |= M.param_labels
            self.boundary |= M.boundary
            self.prior |= M.prior

    def __init__(
            self, syst_dtype_names, model_ions=["Si-II", "Si-III", "O-I"],
            xi1d=False
    ):
        super().__init__()
        self._models = {
            'lya': LyaP1DArinyoModel(),
            'ion': IonModel(model_ions=model_ions),
            # 'reso': ResolutionModel(add_reso_bias, add_var_reso),
            # 'noise': NoiseModel()
        }

        self._xi1d = xi1d
        self._syst_models = []

        if xi1d and syst_dtype_names:
            raise Exception("Xi1D is not supported with systematics.")

        for name in syst_dtype_names:
            if not name.endswith("_syst"):
                continue
            label = name[2:name.rfind("_syst")]
            self._models[f'{label}_syst'] = ScalingSystematicsModel(label)
            self._syst_models.append(f'{label}_syst')

        self._setAttr()
        self._additive_corrections = 0

    @property
    def ndata(self):
        return self._models['lya'].ndata

    @property
    def nsubk(self):
        return _NSUB_K_

    def useSimpleLyaModel(self):
        self._models['lya'] = LyaP1DSimpleModel()
        self._setAttr()

    def useArinyoLyaModel(self):
        self._models['lya'] = LyaP1DArinyoModel()
        self._setAttr()

    def setFiducialCorrectionModel(self, *args):
        self._models['fid'] = FiducialCorrectionModel(*args)

    def removeFiducialCorrectionModel(self):
        self._models.pop('fid', None)
        self._additive_corrections = 0

    # def setNoiseModel(self, p_noise):
    #     self._models['noise'].cache(p_noise)

    def cache(self, kedges, z, data):
        self._models['lya'].cache(kedges, z)
        if self._xi1d:
            Rkms = LIGHT_SPEED * 0.8 / (1 + z) / LYA_WAVELENGTH
            self._dv = np.mean(kedges[1] - kedges[0]) / _NSUB_K_

            N = int(np.round(64 * self._models['lya'].kfine.max() / self._dv))
            self._k = 2 * np.pi * np.fft.rfftfreq(N, d=self._dv)
            # self._v = np.arange(N // 2) * dv
            self._reso = np.exp(-(self._k * Rkms)**2)

            return

        kfine = self._models['lya'].kfine
        self._models['ion'].cache(kfine)

        # rkms = LIGHT_SPEED * 0.8 / (1 + z) / LYA_WAVELENGTH
        # self._models['reso'].cache(kfine, rkms)

        if 'fid' in self._models:
            self._models['fid'].cache(kedges, z)
            self._additive_corrections = self._models['fid'].getCachedModel()

        for name in self._syst_models:
            self._models[name].cache(data[f'e_{name}'])

    def getIntegratedModel(self, **kwargs):
        result = self._models['lya'].getCachedModel(**kwargs)
        result *= self._models['ion'].getCachedModel(**kwargs)
        # result *= self._models['reso'].getCachedModel(**kwargs)
        result = result.reshape(self.ndata, _NSUB_K_).mean(axis=1)
        result += self._additive_corrections
        for name in self._syst_models:
            m = self._models[name]
            result -= m.getCachedModel(**kwargs)

        return result

    def getIntegratedModelXi1D(self, **kwargs):
        result = self._models['lya'].evaluate(self._k, **kwargs)
        result *= self._models['ion'].evaluate(self._k, **kwargs)
        result *= self._reso
        xi1d = np.fft.irfft(result)[:self.ndata * _NSUB_K_] / self._dv
        xi1d = xi1d.reshape(self.ndata, _NSUB_K_).mean(axis=1)
        return xi1d
