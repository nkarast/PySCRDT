"""
PySCRDT
A module to calculate the resonance driving terms from the space charge potential.

Version : 1.1.0
Author  : F. Asvesta
Contact : foteini.asvesta@cern.ch
"""

import logging

import numpy as np
import sympy as sy

__version__ = "1.1.0"
__author__ = ["Foteini Asvesta"]
__contact__ = ["foteini.asvesta@cern.ch"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] - %(funcName)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class PySCRDT:
    """
    Class for the calculation of the resonance driving terms (RDTs)
    from the space charge potential.

    Returns: PySCRDT instance
    """

    def __init__(
        self,
        parameters: bool | str = False,
        mode: int | None = None,
        twissFile: str | None = None,
        order: list | None = None,
        twissTableXsuite: dict | None = None,
    ):
        """
        Initialise the PySCRDT instance.

        Args:
            parameters: if True use default values; if a str read parameters from that file.
            mode: resonance description mode (3 or 5).
            twissFile: path to a MAD-X Twiss file.
            order: resonance order and harmonic.
            twissTableXsuite: Twiss dictionary produced by an X-suite tracker.
        """
        self.x, self.y, self.t = sy.symbols("x y t")
        self.a = sy.Symbol("a", positive=True, real=True)
        self.b = sy.Symbol("b", positive=True, real=True)
        self.D = sy.Symbol("D", positive=True, real=True)
        self.fx = sy.Symbol("fx", positive=True, real=True)
        self.fy = sy.Symbol("fy", positive=True, real=True)
        self.V: sy.Expr | None = None
        self.K: float | None = None
        self.data: np.ndarray | None = None
        self.factor: float | None = None
        self.factor_d: float | None = None
        self.rdt: complex | None = None
        self.rdt_d: complex | None = None
        self.feed: bool = False
        self.mode: int | None = None
        self.order: list | None = None
        # Resonance order attributes — set by setOrder()
        self.m: int | None = None
        self.n: int | None = None
        self.h: int | None = None
        self.i: int | None = None
        self.j: int | None = None
        self.k: int | None = None
        self.l: int | str | None = None

        if isinstance(parameters, str):
            self.parameters = None
            self.readParameters(parameters)
        else:
            if parameters:
                self.setParameters()
            else:
                self.parameters = None
                logger.info("Set parameters with setParameters() or read them with readParameters()")

        if twissFile is None:
            if twissTableXsuite is None:
                logger.info(
                    "Load a MAD-X Twiss file with prepareData() or an X-suite Twiss with loadTwissFromXsuite()"
                )
            else:
                self.loadTwissFromXsuite(twissTableXsuite)
        else:
            self.prepareData(twissFile)

        if order is None:
            logger.info("Set resonance order with setOrder()")
        else:
            if mode is None:
                self.setMode(len(order))
            else:
                self.setMode(mode)
            self.setOrder(order)

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def setMode(self, mode: int | None) -> None:
        """
        Set the mode for characterising resonances.

        Args:
            mode: 3 or 5.
        """
        if mode in (3, 5):
            self.mode = mode
        elif mode is None:
            logger.info("Set resonance mode with setMode()")
        else:
            raise ValueError(f"mode must be 3 or 5, got {mode!r}")

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def setOrder(self, args: list) -> None:
        """
        Set the resonance orders.

        In mode 3:
            args = [m, n, l]   where m=H order, n=V order, l=harmonic (int or 'any')
        In mode 5:
            args = [h, i, j, k, l]

        Args:
            args: list of resonance parameters.
        """
        if len(args) == 3:
            if self.mode == 5:
                raise ValueError("mode is 5: use [h, i, j, k, l] format instead of [m, n, l]")
            if not isinstance(args[0], int):
                raise TypeError(f"resonance order m must be int, got {type(args[0]).__name__}")
            if not isinstance(args[1], int):
                raise TypeError(f"resonance order n must be int, got {type(args[1]).__name__}")
            if not (isinstance(args[2], int) or args[2] == "any"):
                raise TypeError(f"harmonic l must be an int or 'any', got {args[2]!r}")
            self.m = args[0]
            self.n = args[1]
            self.l = args[2]
            self.mode = 3

        elif len(args) == 5:
            if self.mode == 3:
                raise ValueError("mode is 3: use [m, n, l] format instead of [h, i, j, k, l]")
            for idx, label in enumerate(("h", "i", "j", "k")):
                if not isinstance(args[idx], int):
                    raise TypeError(
                        f"resonance order {label} must be int, got {type(args[idx]).__name__}"
                    )
            if not (isinstance(args[4], int) or args[4] == "any"):
                raise TypeError(f"harmonic l must be an int or 'any', got {args[4]!r}")
            self.h = args[0]
            self.i = args[1]
            self.j = args[2]
            self.k = args[3]
            self.l = args[4]
            self.mode = 5

        else:
            if self.mode == 3:
                raise ValueError("mode is 3: provide [m, n, l] (3 values)")
            if self.mode == 5:
                raise ValueError("mode is 5: provide [h, i, j, k, l] (5 values)")
            raise ValueError(
                f"args must have length 3 (mode 3) or 5 (mode 5), got {len(args)}"
            )

        self.factor = None
        self.factor_d = None

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def setParameters(
        self,
        intensity: float = 41e10,
        bunchLength: float = 5.96,
        ro: float = 1.5347e-18,
        emittance_x: float = 2e-6,
        emittance_y: float = 1.1e-6,
        dpp_rms: float = 0.5e-3,
        dpp: float = 0.0,
        bF: float | None = None,
        harmonic: int = 1,
    ) -> None:
        """
        Set the parameters for the calculation.

        Args:
            intensity:   bunch intensity in ppb (default 41e10).
            bunchLength: RMS bunch length in m (default 5.96).
            ro:          classical particle radius in m (default 1.5347e-18 for proton).
            emittance_x: normalised horizontal emittance in m·rad (default 2e-6).
            emittance_y: normalised vertical emittance in m·rad (default 1.1e-6).
            dpp_rms:     RMS Δp/p (default 0.5e-3).
            dpp:         single-particle Δp/p (default 0).
            bF:          bunching factor (default None).
            harmonic:    harmonic number / number of buckets (default 1).
        """
        self.parameters = {
            "intensity": intensity,
            "bunchLength": bunchLength,
            "ro": ro,
            "emittance_x": emittance_x,
            "emittance_y": emittance_y,
            "dpp_rms": dpp_rms,
            "dpp": dpp,
            "bF": bF,
            "harmonic": harmonic,
        }

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def readParameters(self, inputFile: str) -> None:
        """
        Read parameters from a plain-text file.

        Expected format (one parameter per line):
            intensity   = <float>
            bunchLength = <float>
            ...

        Args:
            inputFile: path to the parameter file.
        """
        params = np.genfromtxt(inputFile, dtype=str)
        if self.parameters is None:
            self.setParameters()
        assert self.parameters is not None
        if params.ndim == 1:
            key = params[0]
            if key not in self.parameters:
                raise ValueError(
                    f"unknown parameter '{key}' — valid keys are: {list(self.parameters)}"
                )
            self.parameters[key] = float(params[2])
        else:
            for row in params:
                key = row[0]
                if key not in self.parameters:
                    raise ValueError(
                        f"unknown parameter '{key}' — valid keys are: {list(self.parameters)}"
                    )
                self.parameters[key] = float(row[2])
        if self.data is not None:
            self.beamSize()
            self.ksc()

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def potential(self, feedDown: bool = False) -> None:
        """
        Calculate the space charge potential for the given resonance order.

        Args:
            feedDown: set True when single-particle Δp/p is non-zero (default False).
        """
        if self.mode == 5:
            assert self.h is not None and self.i is not None
            assert self.j is not None and self.k is not None
            self.m = self.h + self.i
            self.n = self.j + self.k
        if self.m is None or self.n is None:
            raise ValueError("resonance order not set — call setOrder() first")
        if self.m % 2 != 0 and not feedDown:
            raise ValueError(
                f"horizontal order m={self.m} is odd; the SC potential has only even orders "
                f"without Δp/p — enable feedDown=True or change m via setOrder()"
            )
        if self.n % 2 != 0:
            raise ValueError(
                f"vertical order n={self.n} is odd; the SC potential has only even orders "
                f"— change n via setOrder()"
            )
        V = (
            -1
            + sy.exp(
                -self.x**2 / (self.t + 2 * self.a**2)
                - self.y**2 / (self.t + 2 * self.b**2)
            )
        ) / sy.sqrt((self.t + 2 * self.a**2) * (self.t + 2 * self.b**2))
        if self.m > self.n:
            if feedDown:
                p1 = sy.series(V, self.x, 0, abs(self.m) + 2).removeO()
            else:
                p1 = sy.series(V, self.x, 0, abs(self.m) + 1).removeO()
            p2 = sy.series(p1, self.y, 0, abs(self.n) + 1).removeO()
            termy = sy.collect(p2, self.y, evaluate=False)
            termpowy = termy[self.y ** abs(self.n)]
            if feedDown:
                termpowy = sy.expand(termpowy.subs(self.x, self.x + self.D))
            termx = sy.collect(termpowy, self.x, evaluate=False)
            termpowx = termx[self.x ** abs(self.m)]
            sterm = sy.simplify(termpowx)
        else:
            p1 = sy.series(V, self.y, 0, abs(self.n) + 1).removeO()
            if feedDown:
                p2 = sy.series(p1, self.x, 0, abs(self.m) + 2).removeO()
            else:
                p2 = sy.series(p1, self.x, 0, abs(self.m) + 1).removeO()
            termx = sy.collect(p2, self.x, evaluate=False)
            if feedDown:
                termx = sy.expand(termx.subs(self.x, self.x + self.D))
            termpowx = termx[self.x ** abs(self.m)]
            termy = sy.collect(termpowx, self.y, evaluate=False)
            termpowy = termy[self.y ** abs(self.n)]
            sterm = sy.simplify(termpowy)
        res = sy.integrate(sterm, (self.t, 0, sy.oo)).doit()
        self.V = sy.simplify(res)
        self.f = sy.lambdify((self.a, self.b, self.D), self.V)

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def ksc(self) -> None:
        """
        Calculate the space charge perveance Ksc from the parameters dictionary.
        """
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() first")
        if self.data is None:
            raise ValueError("Twiss data not loaded — call prepareData() or loadTwissFromXsuite() first")
        if self.parameters["bF"]:
            self.K = (
                2
                * self.parameters["intensity"]
                * self.parameters["ro"]
                * (self.parameters["harmonic"] / self.parameters["bF"])
                / (
                    self.parameters["C"]
                    * self.parameters["b"] ** 2
                    * self.parameters["g"] ** 3
                )
            )
        else:
            self.K = (
                2
                * self.parameters["intensity"]
                * self.parameters["ro"]
                / (
                    np.sqrt(2 * np.pi)
                    * self.parameters["bunchLength"]
                    * self.parameters["b"] ** 2
                    * self.parameters["g"] ** 3
                )
            )

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def beamSize(self) -> None:
        """
        Calculate the transverse RMS beam sizes from parameters and Twiss data.
        """
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() first")
        if self.data is None:
            raise ValueError("Twiss data not loaded — call prepareData() or loadTwissFromXsuite() first")
        bg = self.parameters["b"] * self.parameters["g"]
        self.sx = np.sqrt(
            self.parameters["emittance_x"] * self.data[:, 1] / bg
            + (self.parameters["dpp_rms"] * self.data[:, 3]) ** 2
        )
        self.sy = np.sqrt(
            self.parameters["emittance_y"] * self.data[:, 2] / bg
            + (self.parameters["dpp_rms"] * self.data[:, 4]) ** 2
        )

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def prepareData(self, twissFile: str | None) -> None:
        """
        Load and interpolate Twiss data from a MAD-X Twiss file.

        The file must contain at least the columns: s, betx, bety, dx, dy, mux, muy, l.

        Args:
            twissFile: path to the MAD-X Twiss file.
        """
        if twissFile is None:
            raise ValueError("twissFile path must be provided")
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() first")

        skip_header_nr = 0
        skip_rows_nr = 0
        with open(twissFile, "r") as f:
            for line_nr, line in enumerate(f):
                if line[0] == "*":
                    skip_header_nr = line_nr
                elif line[0] == "$":
                    skip_rows_nr = line_nr + 1
                    break

        params = np.genfromtxt(twissFile, max_rows=40, dtype=str)
        for row in params:
            if row[1] == "GAMMA":
                self.parameters["g"] = float(row[3])
                self.parameters["b"] = np.sqrt(1 - 1 / self.parameters["g"] ** 2)
            elif row[1] == "LENGTH":
                self.parameters["C"] = float(row[3])
            elif row[1] == "Q1":
                self.actualQx = float(row[3])
            elif row[1] == "Q2":
                self.actualQy = float(row[3])

        header = np.genfromtxt(
            twissFile, skip_header=skip_header_nr, max_rows=1, dtype=str
        )
        cols = (
            np.where(header == "S")[0][0] - 1,
            np.where(header == "BETX")[0][0] - 1,
            np.where(header == "BETY")[0][0] - 1,
            np.where(header == "DX")[0][0] - 1,
            np.where(header == "DY")[0][0] - 1,
            np.where(header == "MUX")[0][0] - 1,
            np.where(header == "MUY")[0][0] - 1,
            np.where(header == "L")[0][0] - 1,
        )
        data = np.loadtxt(twissFile, skiprows=skip_rows_nr, usecols=cols)
        # Scale dispersion by relativistic beta before interpolation
        beta = self.parameters["b"]
        data[:, 3] *= beta
        data[:, 4] *= beta
        self._interpolate(data)

    def loadTwissFromXsuite(self, twissTableXsuite: dict | None) -> None:
        """
        Load Twiss data from an X-suite tracker dictionary.

        The dictionary must contain at least: s, betx, bety, dx, dy, mux, muy,
        particle_on_co, circumference, qx, qy.

        Args:
            twissTableXsuite: Twiss dictionary from an X-suite tracker.
        """
        if twissTableXsuite is None:
            raise ValueError("twissTableXsuite must be provided")
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() first")
        self.parameters["g"] = twissTableXsuite["particle_on_co"].gamma0[0]
        self.parameters["b"] = twissTableXsuite["particle_on_co"].beta0[0]
        self.parameters["C"] = twissTableXsuite["circumference"]
        self.actualQx = twissTableXsuite["qx"]
        self.actualQy = twissTableXsuite["qy"]

        beta = self.parameters["b"]
        s_raw = twissTableXsuite["s"]
        raw = np.column_stack(
            [
                s_raw,
                twissTableXsuite["betx"],
                twissTableXsuite["bety"],
                beta * np.asarray(twissTableXsuite["dx"]),
                beta * np.asarray(twissTableXsuite["dy"]),
                twissTableXsuite["mux"],
                twissTableXsuite["muy"],
                np.zeros(len(s_raw)),
            ]
        )
        self._interpolate(raw)

    def _interpolate(self, data: np.ndarray) -> None:
        """
        Interpolate Twiss data onto 100 000 uniformly-spaced points.

        Expects data columns [s, betx, bety, beta*dx, beta*dy, mux, muy, (unused)].
        Beta functions are interpolated via sqrt to avoid oscillatory artefacts.

        Args:
            data: array with columns as described above (beta scaling already applied
                  to dispersion columns 3 and 4 by the caller).
        """
        assert self.parameters is not None
        C = self.parameters["C"]
        s = np.linspace(0, C, 100_000)
        data2 = np.zeros((100_000, 8))
        data2[:, 0] = s
        data2[:, 1] = np.square(np.interp(s, data[:, 0], np.sqrt(data[:, 1])))
        data2[:, 2] = np.square(np.interp(s, data[:, 0], np.sqrt(data[:, 2])))
        data2[:, 3] = np.interp(s, data[:, 0], data[:, 3])
        data2[:, 4] = np.interp(s, data[:, 0], data[:, 4])
        data2[:, 5] = np.interp(s, data[:, 0], data[:, 5])
        data2[:, 6] = np.interp(s, data[:, 0], data[:, 6])
        data2[:, 7] = C / len(s)
        self.data = data2
        self.beamSize()
        self.ksc()

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def reIndexing(self, factor: dict) -> dict:
        """
        Re-index sympy collect() output so all keys are proper exp() expressions.

        Args:
            factor: dict returned by sy.collect(..., evaluate=False).

        Returns:
            dict with normalised keys.
        """
        dictionary: dict = {}
        for key, val in factor.items():
            if len(key.args) == 0:
                dictionary[key] = val
            else:
                dictionary[sy.exp(key.args[0] / 1.0)] = val
        return dictionary

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def calculateFactor(self, Detuning: bool = False) -> float:
        """
        Compute the combinatorial factor for the RDT or detuning integral.

        Args:
            Detuning: if True compute the detuning factor (default False).

        Returns:
            The computed factor as a float.
        """
        assert self.m is not None and self.n is not None
        if Detuning:
            if self.m == 0:
                det1 = sy.cos(self.fy) ** abs(self.n)  # type: ignore[operator]
                det2 = sy.expand(det1.rewrite(sy.exp))
                self.factor_d = float(
                    sy.collect(det2, sy.exp(1j * self.fy), evaluate=False)[1]
                )
            elif self.n == 0:
                det1 = sy.cos(self.fx) ** abs(self.m)
                det2 = sy.expand(det1.rewrite(sy.exp))
                self.factor_d = float(
                    sy.collect(det2, sy.exp(1j * self.fx), evaluate=False)[1]
                )
            else:
                det1 = sy.cos(self.fx) ** abs(self.m) * sy.cos(self.fy) ** abs(self.n)
                det2 = sy.expand(det1.rewrite(sy.exp))
                factor1 = sy.collect(det2, sy.exp(1j * self.fx), evaluate=False)[1]
                self.factor_d = float(
                    sy.collect(factor1, sy.exp(1j * self.fy), evaluate=False)[1]
                )
            return self.factor_d
        else:
            if self.mode == 5:
                assert self.h is not None and self.i is not None
                assert self.j is not None and self.k is not None
                self.m = self.h - self.i
                self.n = self.j - self.k
            if self.m == 0:
                det1 = sy.cos(self.fy) ** abs(self.n)
                det2 = sy.expand(det1.rewrite(sy.exp))
                factor = sy.collect(det2, sy.exp(1j * self.fy), evaluate=False)
                dictionary = self.reIndexing(factor)
                self.factor = float(
                    2.0 * dictionary[sy.exp(abs(self.n) * 1j * self.fy)]
                )
            elif self.n == 0:
                det1 = sy.cos(self.fx) ** abs(self.m)
                det2 = sy.expand(det1.rewrite(sy.exp))
                factor = sy.collect(det2, sy.exp(1j * self.fx), evaluate=False)
                dictionary = self.reIndexing(factor)
                self.factor = float(
                    2.0 * dictionary[sy.exp(abs(self.m) * 1j * self.fx)]
                )
            else:
                det1 = sy.cos(self.fx) ** abs(self.m) * sy.cos(self.fy) ** abs(self.n)
                det2 = sy.expand(det1.rewrite(sy.exp))
                factor1 = sy.collect(det2, sy.exp(1j * self.fx), evaluate=False)
                dictionary = self.reIndexing(factor1)
                factor1 = dictionary[sy.exp(abs(self.m) * 1j * self.fx)]
                factor2 = sy.collect(factor1, sy.exp(1j * self.fy), evaluate=False)
                dictionary = self.reIndexing(factor2)
                self.factor = float(
                    2.0 * dictionary[sy.exp(abs(self.n) * 1j * self.fy)]
                )
            return self.factor

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def resonanceDrivingTerms(self, feedDown: bool = False) -> None:
        """
        Calculate the resonance driving terms for the configured resonance.

        Args:
            feedDown: if True include feed-down from single-particle Δp/p (default False).
        """
        self.feed = feedDown
        if self.V is None:
            self.potential(feedDown=self.feed)
        if self.data is None:
            raise ValueError("Twiss data not loaded — call prepareData() or loadTwissFromXsuite() first")
        if self.K is None:
            self.ksc()
        assert self.K is not None
        assert self.parameters is not None
        if self.factor is None:
            self.calculateFactor()
        assert self.factor is not None
        assert self.m is not None and self.n is not None and self.l is not None
        d = self.data
        fsc = self.factor * d[:, 7] * self.K / 2.0 / (2 * np.pi)
        if self.mode == 3:
            amp = (
                fsc
                * (np.sqrt(2 * d[:, 1]) ** abs(self.m))
                * (np.sqrt(2 * d[:, 2]) ** abs(self.n))
            )
            amp *= self.f(self.sx, self.sy, self.parameters["dpp"] * d[:, 3])
            if self.l == "any":
                phase = np.exp(1j * 2 * np.pi * (self.m * d[:, 5] + self.n * d[:, 6]))
            else:
                assert isinstance(self.l, int)
                phase = np.exp(
                    1j
                    * (
                        self.m * 2 * np.pi * d[:, 5]
                        + self.n * 2 * np.pi * d[:, 6]
                        + (self.l - self.m * self.actualQx - self.n * self.actualQy)
                        * 2
                        * np.pi
                        * d[:, 0]
                        / self.parameters["C"]
                    )
                )
        else:
            assert self.h is not None and self.i is not None
            assert self.j is not None and self.k is not None
            m_eff = self.h + self.i
            n_eff = self.j + self.k
            m_ph = self.h - self.i
            n_ph = self.j - self.k
            amp = (
                fsc
                * (np.sqrt(2 * d[:, 1]) ** abs(m_eff))
                * (np.sqrt(2 * d[:, 2]) ** abs(n_eff))
            )
            amp *= self.f(self.sx, self.sy, self.parameters["dpp"] * d[:, 3])
            if self.l == "any":
                phase = np.exp(1j * 2 * np.pi * (m_ph * d[:, 5] + n_ph * d[:, 6]))
            else:
                assert isinstance(self.l, int)
                phase = np.exp(
                    1j
                    * (
                        m_ph * 2 * np.pi * d[:, 5]
                        + n_ph * 2 * np.pi * d[:, 6]
                        + (self.l - m_ph * self.actualQx - n_ph * self.actualQy)
                        * 2
                        * np.pi
                        * d[:, 0]
                        / self.parameters["C"]
                    )
                )
        self.rdt_s = amp * phase
        self.rdt = self.rdt_s.sum()

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def detuning(self) -> None:
        """
        Calculate the non-linear detuning terms.
        """
        if self.V is None:
            self.potential()
        if self.data is None:
            raise ValueError("Twiss data not loaded — call prepareData() or loadTwissFromXsuite() first")
        if self.K is None:
            self.ksc()
        assert self.K is not None
        assert self.parameters is not None
        if self.factor_d is None:
            self.calculateFactor(Detuning=True)
        assert self.factor_d is not None
        assert self.m is not None and self.n is not None
        d = self.data
        fsc = self.factor_d * d[:, 7] * self.K / 2.0 / (2 * np.pi)
        if self.mode == 3:
            self.rdt_s_d = (
                fsc
                * (np.sqrt(2 * d[:, 1]) ** abs(self.m))
                * (np.sqrt(2 * d[:, 2]) ** abs(self.n))
                * self.f(self.sx, self.sy, self.parameters["dpp"] * d[:, 3])
            )
        else:
            assert self.h is not None and self.i is not None
            assert self.j is not None and self.k is not None
            self.rdt_s_d = (
                fsc
                * (np.sqrt(2 * d[:, 1]) ** abs(self.h + self.i))
                * (np.sqrt(2 * d[:, 2]) ** abs(self.j + self.k))
                * self.f(self.sx, self.sy, self.parameters["dpp"] * d[:, 3])
            )
        self.rdt_d = self.rdt_s_d.sum()

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def updateParameters(self, **kwargs) -> None:
        """
        Update one or more entries in the parameters dictionary.

        Accepted keys: intensity, bunchLength, ro, emittance_x, emittance_y,
                       dpp_rms, dpp, b, g, bF, harmonic.

        Args:
            **kwargs: key=value pairs to update.
        """
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() first")
        for key, value in kwargs.items():
            if key not in self.parameters:
                raise ValueError(
                    f"unknown parameter '{key}' — valid keys are: {list(self.parameters)}"
                )
            self.parameters[key] = value
        if self.data is not None:
            self.beamSize()
            self.ksc()

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getParameters(self) -> dict:
        """
        Return the parameters dictionary.

        Returns:
            The current parameters dict.
        """
        if self.parameters is None:
            raise ValueError("parameters not set — call setParameters() or readParameters() first")
        return self.parameters

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getWorkingPoint(self) -> tuple[float, float]:
        """
        Return the tunes (Qx, Qy) read from the Twiss file.

        Returns:
            Tuple (Qx, Qy).
        """
        if self.data is None:
            raise ValueError("Twiss data not loaded — call prepareData() or loadTwissFromXsuite() first")
        return self.actualQx, self.actualQy

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getOrder(self) -> tuple:
        """
        Return the resonance orders.

        Returns:
            (m, n, l) for mode 3 or (h, i, j, k, l) for mode 5.
        """
        if self.mode is None:
            raise ValueError("resonance mode not set — call setMode() first")
        if self.mode == 3:
            if self.m is None:
                raise ValueError("resonance order not set — call setOrder() first")
            return self.m, self.n, self.l
        if self.mode == 5:
            if self.h is None:
                raise ValueError("resonance order not set — call setOrder() first")
            return self.h, self.i, self.j, self.k, self.l
        raise ValueError(f"unexpected mode value {self.mode!r}")

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getMode(self) -> int:
        """
        Return the resonance mode description.

        Returns:
            3 or 5.
        """
        if self.mode is None:
            raise ValueError("resonance mode not set — call setMode() first")
        return self.mode

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getKsc(self) -> float:
        """
        Return the space charge perveance Ksc.

        Returns:
            Ksc as a float.
        """
        if self.K is None:
            self.ksc()
        assert self.K is not None
        return self.K

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getPotential(self) -> sy.Expr:
        """
        Return the space charge potential V (sympy expression).

        Returns:
            V as a sympy expression.
        """
        if self.V is None:
            self.potential()
        assert self.V is not None
        return self.V

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getResonanceDrivingTerms(self, feedDown: bool = False) -> dict:
        """
        Return the resonance driving terms.

        Args:
            feedDown: if True include feed-down (only relevant if [resonanceDrivingTerms]
                      has not been called yet, default False).

        Returns:
            dict with keys 'RDT' (complex), 'Amplitude' (float), 'Phase' (float in radians).
        """
        self.feed = feedDown
        if self.rdt is None:
            self.resonanceDrivingTerms(feedDown=self.feed)
        assert self.rdt is not None
        return {
            "RDT": self.rdt,
            "Amplitude": abs(self.rdt),
            "Phase": np.angle(self.rdt),
        }

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def getDetuning(self) -> complex:
        """
        Return the non-linear detuning term.

        Returns:
            The accumulated detuning as a (generally real) number.
        """
        if self.rdt_d is None:
            self.detuning()
        assert self.rdt_d is not None
        return self.rdt_d

    # - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

    def checkWriting(self) -> dict:
        """
        Return the valid parameter names for [setParameters] and [updateParameters].

        Returns:
            dict with keys 'Set & Update' and 'Update only'.
        """
        return {
            "Set & Update": [
                "intensity",
                "bunchLength",
                "emittance_x",
                "emittance_y",
                "dpp_rms",
                "dpp",
                "ro",
            ],
            "Update only": ["b", "g"],
        }
