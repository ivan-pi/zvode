import warnings
import numpy as np

from scipy.integrate import OdeSolver, DenseOutput

from . import _zvode
from ._helpers import (
    MESSAGES,
    _LMM,
    _eval_nordsieck,
    _make_workspace,
    _validate_max_step,
    _validate_min_step,
    _check_tolerances,
    _validate_fun_shape,
    _validate_jac_shape,
    _resolve_miter,
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

        p(t) = \\sum_{j=0}^{nq} s^j \\, yh_j, \\quad s = (t - t_n) / h

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
        Step size the Nordsieck array is scaled to (HU in ZVODE).
    """

    def __init__(self, t_old, t, yh, h):
        """Capture a Nordsieck array snapshot for later polynomial evaluation."""
        super().__init__(t_old, t)
        self.yh = yh
        self.h = h

    def _call_impl(self, t):
        """Evaluate the interpolant at time(s) `t`; returns shape ``(n,)`` or ``(n, m)``."""
        return _eval_nordsieck(self.yh, self.h, t, self.t)


class ZVODE(OdeSolver):
    """Solver for complex-valued ODEs using ZVODE (Variable-coefficient, fixed-leading-coefficient).

    Implements the :class:`scipy.integrate.OdeSolver` interface so that it can be
    passed as the ``method`` argument to :func:`scipy.integrate.solve_ivp`::

        sol = scipy.integrate.solve_ivp(fun, tspan, y0, method=ZVODE)

    ZVODE solves the initial value problem for stiff or non-stiff systems of
    first-order complex ODEs::

        dy/dt = f(t, y),   y(t0) = y0

    where `y` is a complex vector.  It is based on the EPISODE/EPISODEB
    packages and implements Adams (non-stiff) and BDF (stiff) methods with
    orders up to 12 and 5 respectively.

    .. note::

        When using ZVODE for a stiff system, `fun` must be *analytic* (i.e.,
        each component f(i) must be an analytic function of each y(j)).  For
        a complex stiff system where `fun` is *not* analytic, use a
        real-valued solver on the equivalent real system of doubled dimension.

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
        component.  ``rtol`` controls relative accuracy (number of correct
        digits); ``atol`` controls absolute accuracy and guards against loss
        of significance when a component passes through zero.  Scalar or
        per-component arrays are accepted.  Defaults are ``rtol=1e-3``,
        ``atol=1e-6``.
    first_step : float, optional
        Initial step size.  Chosen automatically if not given.
    min_step : float, optional
        Minimum allowed step size.  Default 0.
    max_step : float, optional
        Maximum allowed step size.  Default ``np.inf``, i.e., the step size
        is not bounded and determined solely by the solver.
    jac : callable or None, optional
        Jacobian matrix of `f` with respect to `y`, ``jac(t, y)``.
        For a full Jacobian, return an ``(n, n)`` array ``J[i, j] = df(i)/dy(j)``.
        For a banded Jacobian (when `lband` / `uband` are set), return an
        ``(lband + uband + 1, n)`` array where ``J[i-j+uband, j] = df(i)/dy(j)``.
        If not supplied, BDF approximates the Jacobian by finite differences
        (``miter=2``); Adams uses functional iteration and needs no Jacobian
        (``miter=0``).
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian, i.e.,
        ``jac[i, j]`` is assumed zero unless ``i - lband <= j <= i + uband``.
        Must be non-negative integers.  When either is set, the banded Jacobian
        path is used and the other defaults to 0.  The full band has width
        ``lband + uband + 1``.  Can be used with ``jac=None`` to have the
        solver estimate the Jacobian by finite differences within the band
        only, reducing the number of function evaluations compared to a full
        finite-difference Jacobian.
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

    Raises
    ------
    ValueError
        On invalid input.

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

    See Also
    --------
    scipy.integrate.OdeSolver : Abstract base class implemented by this solver.
    scipy.integrate.solve_ivp : Driver function that accepts ``method=ZVODE``
        (or ``method=ZVODE_BDF`` / ``method=ZVODE_Adams``) to use this solver.

    Notes
    -----
    **Thread safety:** ``ZVODE`` is *not* thread-safe.  The underlying Fortran
    library stores solver state in process-global COMMON blocks, so stepping
    any two instances concurrently from different threads — even distinct
    objects — will corrupt that shared state.  Protect all calls to
    :meth:`step` with a single process-wide ``threading.Lock``.  Running
    multiple independent integrations in separate *processes* (e.g. via
    ``multiprocessing``) is safe.

    **Lifetime of the callback** ``y``\\ **:** the ``y`` passed to `fun` (and
    to `jac`) is a read-only view onto the solver's internal workspace, valid
    only for the duration of that call; its contents are overwritten as the
    integration advances.  Reading ``y`` and returning a freshly computed
    array is always safe; only retaining a reference to ``y`` past the call is
    not.  Copy it with ``y.copy()`` if you need to keep the state (e.g. to log
    a trajectory).

    References
    ----------
    .. [Brown1989ZVODE] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A
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

        if np.isrealobj(y0):
            warnings.warn(
                "y0 has a real dtype and will be cast to complex128. "
                "Pass a complex array to suppress this warning.",
                stacklevel=2,
            )
        # super().__init__ may leave self.y as float if y0 is real; override to
        # complex128.  np.array always allocates a fresh array regardless of input
        # dtype, so self.y is independent of whatever the caller passed.
        self.y = np.array(y0, dtype=np.complex128)
        self._ytmp = self.y.copy()  # mutable Fortran work buffer

        self.istate = 1  # start integration
        self.itask = 5  # take one step, without passing t_bound, then return

        self.itol, self.rtol, self.atol = _check_tolerances(rtol, atol, self.n)

        self.nfev = 0
        self.njev = 0
        self._nfe_last = 0
        self._nje_last = 0

        # Select method
        if lmm not in _LMM:
            raise ValueError(
                f"Invalid linear multistep method (lmm) {lmm!r}. "
                "Valid options are 'Adams' or 'BDF'."
            )
        self.meth, maxord_allowed = _LMM[lmm]

        # OdeSolver.__init__ already stored the (complex-coerced) RHS as
        # self.fun_single; reuse it as the C callback.  nfev is tracked from
        # ZVODE's NFE counter in _step_impl, so we deliberately use the
        # non-counting fun_single rather than self.fun (which increments nfev).
        _validate_fun_shape(fun, self.n, t0, self.y)
        self.nfev += 1

        self.miter, self.ml, self.mu = _resolve_miter(
            jac, lband, uband, self.meth, self.n, miter
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

        self.wrap_jac = jac if jac else None

        if jsv not in (1, -1):
            raise ValueError(
                "'jsv' must be 1 (save Jacobian) or -1 (recompute every step)."
            )
        self.jsv = jsv

        # Method Flag (MF)
        self.mf = self.jsv * (10 * self.meth + self.miter)

        assert abs(self.mf) in (10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25), (
            f"mf={self.mf!r} is invalid (jsv={self.jsv!r}, meth={self.meth!r}, "
            f"miter={self.miter!r}); this is a bug in zvode"
        )

        self.max_step = _validate_max_step(max_step)
        self.min_step = _validate_min_step(min_step)
        if max_order is not None:
            if max_order <= 0:
                raise ValueError("'max_order' must be a positive integer.")
            if max_order > maxord_allowed:
                warnings.warn(
                    f"'max_order' ({max_order}) exceeds the maximum allowed order "
                    f"({maxord_allowed}) for the selected method. The solver will "
                    f"automatically reduce it.",
                    stacklevel=2,
                )
        self.iopt = 1
        self.zwork, self.rwork, self.iwork = _make_workspace(
            self.n,
            self.miter,
            self.ml,
            self.mu,
            self.mf,
            t0,
            t_bound,
            first_step=first_step,
            min_step=self.min_step,
            max_step=self.max_step,
            max_order=max_order,
        )
        # Last: probing jac(t0, y0) may allocate an (neq, neq) array; validate
        # after _make_workspace so its overflow check fires first for large neq.
        if jac is not None and self.miter in (1, 4):
            _validate_jac_shape(jac, self.miter, self.ml, self.mu, self.n, t0, self.y)
            self.njev += 1

    def _step_impl(self):
        """Advance one step; return (success, message)"""

        # Python evaluates the full RHS before any assignment, so the current
        # self.t and self.istate are safely read as inputs before being overwritten.
        self.t, self.istate = _zvode.zvode(
            self.fun_single,
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

        nfe_new = int(self.iwork[11])  # NFE  IWORK(12): f evaluations
        nje_new = int(self.iwork[12])  # NJE  IWORK(13): Jacobian evaluations
        self.nfev += nfe_new - self._nfe_last
        self.njev += nje_new - self._nje_last
        self._nfe_last = nfe_new
        self._nje_last = nje_new
        self.nlu = self.iwork[19]  # NLU  IWORK(20): LU decompositions

        if self.istate != 2:
            description = MESSAGES.get(self.istate, "Unknown error.")
            return False, f"zvode: istate = {self.istate}: {description}"

        self.y = self._ytmp.copy()
        return True, None

    def _dense_output_impl(self):
        nq = int(self.iwork[13])  # IWORK(14) = NQU: order last used
        hu = float(self.rwork[10])  # RWORK(11) = HU: step size last used
        # Copy with order='F': the interpolant outlives this step's zwork.
        yh = (
            self.zwork[: self.n * (nq + 1)]
            .reshape((self.n, nq + 1), order="F")
            .copy(order="F")
        )
        return ZVODEDenseOutput(self.t_old, self.t, yh, hu)


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
