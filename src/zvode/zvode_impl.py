import warnings
import numpy as np

from scipy.integrate import OdeSolver, DenseOutput

from . import _zvode
from ._helpers import (
    MESSAGES,
    _validate_max_step,
    _validate_first_step,
    _wrapped_fun,
    _wrapped_jac,
    _check_tolerances,
    _validate_fun_shape,
    _validate_jac_shape,
    _determine_miter,
)


def _warn_extraneous(extraneous):
    """Warn about unexpected keyword arguments passed to a solver."""
    if extraneous:
        warnings.warn(
            "The following arguments have no effect for the chosen solver: {}.".format(
                ", ".join(f"`{k}`" for k in extraneous)
            ),
            stacklevel=3,
        )


class ZVODEDenseOutput(DenseOutput):
    """Dense output interpolant for ZVODE using the Nordsieck history array.

    Evaluates the interpolating polynomial via Horner's method applied to the
    snapshot of the Nordsieck array YH captured at the end of each accepted
    step.  The Nordsieck array column *j* holds ``H**j / j! * y^(j)(t)``, so
    the polynomial is evaluated by the recurrence

    .. math::

        p(t) = \\sum_{j=0}^{nq} \\binom{s}{j} \\, yh_j,
        \\quad s = (t - t_n) / h

    where the binomial weights collapse to 1 for plain interpolation (k=0),
    giving a simple Horner evaluation.  No internal Fortran state is required
    after construction.

    Parameters
    ----------
    t_old : float
        Start of the step.
    t : float
        End of the step.
    yh : ndarray, shape (n, nq+1), complex128
        Nordsieck history array, column-major copy taken at the end of the
        step and scaled to step size `h`.
    h : float
        Step size the Nordsieck array is scaled to (``HCUR`` in ZVODE).
    """

    def __init__(self, t_old, t, yh, h):
        """Capture a Nordsieck array snapshot for later polynomial evaluation."""
        super().__init__(t_old, t)

        # yh : (n, nq+1) complex128, column j holds H^j/j! * y^(j)(t)
        self.yh = yh
        self.nq = yh.shape[1] - 1
        self.h = h

    def _call_impl(self, t):
        """Evaluate the interpolant at time(s) `t`; returns shape ``(n,)`` or ``(n, m)``."""

        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # normalised position, shape (m,)
        s = (t - self.t) / self.h

        # Horner's method along the Nordsieck columns; for plain interpolation
        # all falling-factorial weights are 1, so the recurrence simplifies to:
        #   p = yh[:,nq]; for j = nq-1 ... 0: p = yh[:,j] + s*p
        # One (n, m) buffer is allocated upfront; each iteration is then two
        # in-place operations with no temporaries: dky *= s; dky += yh[:,j].
        # Starting from a view of yh would corrupt the stored Nordsieck array.

        n = self.yh.shape[0]
        dky = np.empty((n, len(t)), dtype=self.yh.dtype)
        dky[:] = self.yh[:, self.nq, np.newaxis]  # seed: broadcast (n,1) -> (n,m)
        for j in range(self.nq - 1, -1, -1):
            dky *= s  # dky = s * dky  (broadcasts m)
            dky += self.yh[:, j, np.newaxis]  # dky = yh[:,j] + s * dky

        return dky[:, 0] if scalar else dky


class ZVODE(OdeSolver):
    """Solver for complex-valued ODEs using ZVODE (Variable-coefficient, fixed-leading-coefficient).

    ZVODE solves the initial value problem for stiff or non-stiff systems of
    first-order complex ODEs::

        dy/dt = f(t, y),   y(t0) = y0

    where `y` is a complex vector.  It is based on the EPISODE/EPISODEB
    packages and implements Adams (non-stiff) and BDF (stiff) methods with
    orders up to 12 and 5 respectively.

    .. note::

        When using ZVODE for a stiff system, `f` must be analytic (i.e., each
        component f(i) must be an analytic function of each y(j)).  For a
        complex stiff system where `f` is not analytic, use a real-valued
        solver on the equivalent real system of doubled dimension.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system, ``f(t, y)``.  The output must be
        array-like with the same shape as `y`.
    t0 : float
        Initial value of the independent variable.
    y0 : array_like, shape (n,)
        Initial state; will be cast to ``complex128``.
    t_bound : float
        Boundary time.  Integration will not proceed past this value; also
        determines the direction of integration.
    lmm : {'BDF', 'Adams'}, optional
        Linear multistep method.  ``'BDF'`` (default) is recommended for stiff
        problems (max order 5); ``'Adams'`` is recommended for non-stiff
        problems (max order 12).
    rtol, atol : float or array_like, optional
        Relative and absolute local error tolerances.  The solver keeps the
        local error roughly below ``rtol * |y(i)| + atol`` for each
        component.  Scalar or per-component arrays are accepted.  Defaults
        are ``rtol=1e-3``, ``atol=1e-6``.
    first_step : float, optional
        Initial step size.  Chosen automatically if not given.
    min_step : float, optional
        Minimum allowed step size.  Default 0.
    max_step : float, optional
        Maximum allowed step size.  Default ``np.inf``.
    jac : callable or None, optional
        Jacobian matrix of `f` with respect to `y`, ``jac(t, y)``.
        For a full Jacobian, return an ``(n, n)`` array ``J[i, j] = df(i)/dy(j)``.
        For a banded Jacobian (when `lband` / `uband` are set), return an
        ``(lband + uband + 1, n)`` array where ``J[i-j+uband, j] = df(i)/dy(j)``.
        If not supplied, BDF approximates the Jacobian by finite differences
        (``miter=2``); Adams uses functional iteration and needs no Jacobian
        (``miter=0``).
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian.  Must be
        non-negative integers.  When either is set, the banded Jacobian path
        is used and the other defaults to 0.  The full band has width
        ``lband + uband + 1``.
    max_order : int, optional
        Maximum integration order.  Capped at 12 for Adams and 5 for BDF.
    miter : {0, 1, 2, 3, 4, 5}, optional
        Iteration method override.  Normally inferred from `lmm`, `jac`, and
        `lband`/`uband`: Adams without `jac` defaults to ``0``; BDF without
        `jac` defaults to ``2``; providing `jac` selects ``1`` (dense) or
        ``4`` (banded).

        * 0 – functional iteration (no Jacobian; default for Adams)
        * 1 – chord with user-supplied full Jacobian
        * 2 – chord with internally generated full Jacobian (default for BDF)
        * 3 – chord with diagonal Jacobian approximation
        * 4 – chord with user-supplied banded Jacobian
        * 5 – chord with internally generated banded Jacobian
    jsv : {1, -1}, optional
        Jacobian-saving flag.  ``1`` (default) retains a copy of the Jacobian
        to reuse when rebuilding the Newton iteration matrix.  ``-1`` does not
        retain a copy; the Jacobian is recomputed whenever the iteration matrix
        needs updating.  Ignored when no full Jacobian matrix is stored, i.e.
        for functional iteration (``miter=0``) and the diagonal approximation
        (``miter=3``).

    Attributes
    ----------
    n : int
        Number of equations.
    status : str
        Current solver status: ``'running'``, ``'finished'``, or ``'failed'``.
    t : float
        Current time.
    y : ndarray
        Current state vector.
    t_bound : float
        Boundary time.
    nfev : int
        Number of right-hand side evaluations.
    njev : int
        Number of Jacobian evaluations.
    nlu : int
        Number of LU decompositions.

    Notes
    -----
    **Thread safety:** ``ZVODE`` is *not* thread-safe.  The underlying Fortran
    library stores solver state in process-global COMMON blocks, so stepping
    any two instances concurrently from different threads — even distinct
    objects — will corrupt that shared state.  Protect all calls to
    :meth:`step` with a single process-wide ``threading.Lock``.  Running
    multiple independent integrations in separate *processes* (e.g. via
    ``multiprocessing``) is safe.

    References
    ----------
    .. [1] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A
       Variable-Coefficient ODE Solver," *SIAM J. Sci. Stat. Comput.*,
       10(5), pp. 1038-1051, 1989. https://doi.org/10.1137/0910062

    Examples
    --------
    Trace the unit circle: ``dy/dt = i*y``, ``y(0) = 1``, analytic solution
    ``y(t) = exp(i*t)``.  Pass ``ZVODE`` as the ``method`` argument to
    :func:`scipy.integrate.solve_ivp`:

    >>> import math
    >>> from scipy.integrate import solve_ivp
    >>> from zvode import ZVODE
    >>> sol = solve_ivp(lambda t, y: 1j*y, [0, 2*math.pi], [1+0j], method=ZVODE)
    >>> bool(abs(sol.y[0, -1] - 1.0) < 1e-2)   # back near start after one loop
    True
    """

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        *,
        lmm="BDF",
        rtol=1.0e-3,
        atol=1.0e-6,
        first_step=None,
        min_step=0.0,
        max_step=np.inf,
        jac=None,
        lband=None,
        uband=None,
        max_order=None,
        miter=None,
        jsv=1,
        **extraneous,
    ):
        """Validate arguments, allocate ZVODE workspaces, and prepare the initial state."""
        _warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=True)

        self.tout = self.t_bound
        if np.isrealobj(y0):
            warnings.warn(
                "y0 has a real dtype and will be cast to complex128. "
                "Pass a complex array to suppress this warning.",
                stacklevel=2,
            )
        self._ytmp = np.array(y0, dtype=np.complex128, order="C", copy=True)
        self.y = self._ytmp.copy()

        self.istate = 1  # start integration
        self.itask = 5  # take one step, without passing t_bound, then return

        # Select method
        if lmm == "Adams":
            self.meth = 1
            maxord_allowed = 12
        elif lmm == "BDF":
            self.meth = 2
            maxord_allowed = 5
        else:
            raise ValueError(
                f"Invalid linear multistep method (lmm) '{lmm}'. "
                "Valid options are 'Adams' or 'BDF'."
            )

        self.itol, self.rtol, self.atol = _check_tolerances(rtol, atol, self.n)

        self.wrap_fun = _wrapped_fun(fun)
        _validate_fun_shape(fun, self.n, t0, self._ytmp)

        self.miter, self.ml, self.mu = _determine_miter(
            jac, lband, uband, self.meth, miter
        )

        if self.miter in (4, 5):
            bandwidth = self.ml + self.mu + 1
            if bandwidth * 2 > self.n:
                warnings.warn(
                    f"Bandwidth lband + uband + 1 = {bandwidth} exceeds half "
                    f"the system size neq = {self.n}; verify that a banded "
                    "solver is appropriate for this problem.",
                    stacklevel=2,
                )

        self.wrap_jac = _wrapped_jac(jac, banded=(self.miter == 4)) if jac else None

        # Check for int32 overflow in Fortran workspace arithmetic before
        # doing anything that allocates memory proportional to neq (including
        # the jac shape probe below).
        _INT32_MAX = 2**31 - 1
        if self.miter in (1, 2):
            # worst case: LENWM = 2*N*N  (JSV=1, JCO=1)
            if self.n**2 > _INT32_MAX:
                raise ValueError(
                    f"neq = {self.n} exceeds the maximum of 46340 for dense "
                    f"Jacobian methods: neq**2 overflows the 32-bit integer "
                    f"arithmetic used internally by the Fortran library."
                )
        elif self.miter in (4, 5):
            # worst case: LENWM = (2*ML + MU + 1 + ML)*N = (3*ML + MU + 1)*N
            _lenwm_max = (3 * self.ml + self.mu + 1) * self.n
            if _lenwm_max > _INT32_MAX:
                raise ValueError(
                    f"Banded workspace size ({_lenwm_max:,}) overflows the "
                    f"32-bit integer arithmetic used internally by the Fortran library."
                )

        if jac is not None and self.miter in (1, 4):
            _validate_jac_shape(
                jac, self.miter, self.ml, self.mu, self.n, t0, self._ytmp
            )

        if jsv not in (1, -1):
            raise ValueError(
                "'jsv' must be 1 (save Jacobian) or -1 (recompute every step)."
            )
        self.jsv = jsv

        # Method Flag (MF)
        self.mf = self.jsv * (10 * self.meth + self.miter)

        if abs(self.mf) not in (10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25):
            # TODO: we may be able to get rid of this check if
            #       jsv, meth and miter have been checked before-hand
            raise RuntimeError("Error setting the method flag")

        if self.miter == 0:
            lwm = 0
        elif self.miter in (1, 2):
            if self.mf > 0:
                lwm = 2 * self.n**2
            elif self.mf < 0:
                lwm = self.n**2
            else:
                lwm = None
        elif self.miter == 3:
            lwm = self.n
        elif self.miter in (4, 5):
            if self.mf > 0:
                lwm = (3 * self.ml + 2 * self.mu + 2) * self.n
            elif self.mf < 0:
                lwm = (2 * self.ml + self.mu + 1) * self.n
            else:
                lwm = None
        else:
            assert False, f"Unhandled miter value {self.miter}."

        if lwm is None:
            raise ValueError()

        lzw = self.n * (maxord_allowed + 1) + 2 * self.n + lwm
        self.zwork = np.zeros(lzw, dtype=np.complex128)

        lrw = 20 + self.n
        self.rwork = np.zeros(lrw, dtype=np.float64)

        liw = 30 if self.miter in (0, 3) else 30 + self.n
        self.iwork = np.zeros(liw, dtype=np.int32)

        if self.miter in (4, 5):
            # Banded Jacobian
            self.iwork[0] = self.ml
            self.iwork[1] = self.mu

        # Optional input settings
        self.iopt = 1
        self.rwork[4:9] = 0.0
        self.iwork[4:9] = 0

        if self.itask == 5:
            self.rwork[0] = t_bound

        if first_step is not None:
            self.h0 = _validate_first_step(first_step, t0, t_bound)
            # ZVODE requires H0 to carry the sign of the integration direction.
            self.rwork[4] = self.h0 * np.sign(t_bound - t0)

        if max_step is not None:
            self.max_step = _validate_max_step(max_step)
            self.rwork[5] = self.max_step

        if min_step is not None:
            self.rwork[6] = float(min_step)

        if max_order is not None:
            if max_order <= 0:
                raise ValueError("'max_order' must be a positive integer.")

            max_allowed = 12 if self.meth == 1 else 5
            if max_order > max_allowed:
                warnings.warn(
                    f"'max_order' ({max_order}) exceeds the maximum allowed order "
                    f"({max_allowed}) for the selected method. The solver will "
                    f"automatically reduce it.",
                    stacklevel=2,
                )

            # Load the potentially "wrong" value; capping happens inside Fortran
            self.iwork[4] = max_order

    def _step_impl(self):
        """Advance one step; return (success, message)"""

        t, istate = _zvode.zvode(
            self.wrap_fun,
            self._ytmp,
            self.t,
            self.t_bound,
            self.itol,
            self.rtol,
            self.atol,
            self.itask,
            self.istate,
            self.iopt,
            self.zwork,
            self.rwork,
            self.iwork,
            self.wrap_jac,
            self.mf,
        )

        self.istate = istate
        self.t = t

        self.nfev = self.iwork[11]
        self.njev = self.iwork[12]
        self.nlu = self.iwork[19]

        if self.istate != 2:
            description = MESSAGES.get(self.istate, "Unknown error.")
            return False, f"zvode: istate = {self.istate}: {description}"

        self.y = self._ytmp.copy()

        # Succesful step
        return True, None

    def _dense_output_impl(self):
        """Capture the current Nordsieck array and return a dense interpolant."""

        nq = int(self.iwork[14])  # IWORK(15) = NQCUR
        h = float(self.rwork[10])  # RWORK(11) = HU: step size last used

        # YH occupies zwork[0 : n*(nq+1)] in Fortran column-major order.
        yh = self.zwork[: self.n * (nq + 1)].reshape((self.n, nq + 1), order="F").copy()

        return ZVODEDenseOutput(self.t_old, self.t, yh, h)


class ZVODE_Adams(ZVODE):
    """ZVODE with the Adams (non-stiff) linear multistep method.

    Recommended for non-stiff problems, typically combined with fixed-point
    (functional) iteration (``miter=0``, the default when no ``jac`` is given).

    For all parameters and attributes see :class:`ZVODE`.
    The ``lmm`` argument is fixed to ``'Adams'``.
    """

    def __init__(self, fun, t0, y0, t_bound, **kwargs):
        super().__init__(fun, t0, y0, t_bound, lmm="Adams", **kwargs)


class ZVODE_BDF(ZVODE):
    """ZVODE with the BDF (stiff) linear multistep method.

    Recommended for stiff problems, typically combined with Newton iteration
    using an internally generated Jacobian (``miter=2``, the default when no
    ``jac`` is given).

    For all parameters and attributes see :class:`ZVODE`.
    The ``lmm`` argument is fixed to ``'BDF'``.
    """

    def __init__(self, fun, t0, y0, t_bound, **kwargs):
        super().__init__(fun, t0, y0, t_bound, lmm="BDF", **kwargs)
