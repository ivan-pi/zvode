import warnings
import numpy as np

from scipy.integrate import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (
    warn_extraneous,
    validate_max_step,
    validate_first_step,
)

from . import _zvode

# ZVODE ISTATE error codes and their human-readable descriptions.
MESSAGES = {
    -1: "Excess work done on this call.",
    -2: "Excess accuracy requested.",
    -3: "Illegal input detected.",
    -4: "Repeated error test failures.",
    -5: "Repeated convergence failures.",
    -6: "Error weight became zero during problem integration.",
}


def _wrapped_fun(fun):
    """Adapt a SciPy-style ``f(t, y)`` to the in-place ZVODE signature."""

    def _zvode_fun(t, y, dy):
        dy[:] = fun(t, y)

    return _zvode_fun


def _wrapped_jac(jac, banded=False):
    """Adapt a SciPy-style ``jac(t, y)`` to the in-place ZVODE Jacobian signature.

    ZVODE passes an output array ``pd`` of shape ``(nrowpd, neq)`` in Fortran
    (column-major) order.  For the dense case ``nrowpd >= neq``.  For the banded
    case ``nrowpd >= 2*ml + mu + 1``: the extra ``ml`` rows beyond the user band
    ``ml + mu + 1`` are fill-in workspace that ZGBFA (or the equivalent
    LAPACK routines) need during LU factorisation and should be ignored
    by the Jacobian callback.

    Within the user band, ``df(i)/dy(j)`` goes into ``pd[i - j + mu, j]``.
    The triangular corner entries that correspond to nonexistent matrix elements
    (where the band extends beyond the matrix) can be set to any value.

    Note: ZVODE's native interface only requires callers to set the non-zero
    elements of ``pd``; unset entries are ignored.  Because this wrapper copies
    the return value of ``jac(t, y)`` into ``pd``, the full slice is always
    overwritten, which is a minor overhead paid for SciPy interface compatibility.
    """

    def _zvode_jac(t, y, pd):
        n = y.shape[0]
        pd[:n, :n] = jac(t, y)

    def _zvode_banded_jac(t, y, pd, ml, mu):
        n = y.shape[0]
        pd[: ml + mu + 1, :n] = jac(t, y)

    return _zvode_banded_jac if banded else _zvode_jac


def _check_tolerances(rtol, atol, n):
    """Validate rtol/atol, warn if too small, and return the ZVODE ITOL flag.

    ITOL encodes which combination of scalar/array tolerances is used:

    ======  ==========  ==========  ===========================
    ITOL    RTOL        ATOL        EWT(i)
    ======  ==========  ==========  ===========================
    1       scalar      scalar      RTOL*|Y(i)| + ATOL
    2       scalar      array       RTOL*|Y(i)| + ATOL(i)
    3       array       scalar      RTOL(i)*|Y(i)| + ATOL
    4       array       array       RTOL(i)*|Y(i)| + ATOL(i)
    ======  ==========  ==========  ===========================
    """
    rtol = np.asarray(rtol)
    atol = np.asarray(atol)

    # 1. Shape checks (Strict)
    if rtol.ndim > 0 and rtol.shape != (n,):
        raise ValueError(f"'rtol' must be a scalar or a 1D array of length {n}.")
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError(f"'atol' must be a scalar or a 1D array of length {n}.")

    # 2. Positivity checks
    if np.any(rtol < 0) or np.any(atol < 0):
        raise ValueError("'rtol' and 'atol' must be positive.")

    # 3. Auto-correction for impossibly small rtol (SciPy style)
    EPS = np.finfo(float).eps
    if np.any(rtol < 100 * EPS):
        warnings.warn(
            f"At least one element of 'rtol' is too small. "
            f"Setting 'rtol = np.maximum(rtol, {100 * EPS})'.",
            stacklevel=3,
        )
        rtol = np.maximum(rtol, 100 * EPS)

    # 4. Determine ITOL flag
    if rtol.ndim == 0 and atol.ndim == 0:
        itol = 1
    elif rtol.ndim == 0 and len(atol) == n:
        itol = 2
    elif len(rtol) == n and atol.ndim == 0:
        itol = 3
    elif len(rtol) == n and len(atol) == n:
        itol = 4
    else:
        raise RuntimeError("This should not occur.")

    return itol, rtol, atol


def _validate_jac_shape(jac, miter, ml, mu, n, t0, y0):
    """Evaluate *jac* once at ``(t0, y0)`` and verify its return shape.

    Only called for miter=1 (dense) and miter=4 (banded); skipped for
    internally generated Jacobians (miter=2,3,5) and functional iteration
    (miter=0) where no user callback is involved.
    """
    # FIXME: this evaluation should be counted toward njev, but the Fortran
    # library owns that counter inside iwork and it is only readable after
    # each accepted step, so incrementing it here would require duplicating
    # the counter in Python.
    trial = np.asarray(jac(t0, y0))
    if miter == 4:
        expected = (ml + mu + 1, n)
        if trial.shape != expected:
            raise ValueError(
                f"For miter=4 (banded Jacobian), 'jac' must return an array "
                f"of shape (lband + uband + 1, neq) = {expected}; "
                f"got shape {trial.shape}. "
                "Pass a dense Jacobian and use miter=1, or fix the banded format."
            )
    else:  # miter == 1
        expected = (n, n)
        if trial.shape != expected:
            raise ValueError(
                f"For miter=1 (dense Jacobian), 'jac' must return an array "
                f"of shape (neq, neq) = {expected}; "
                f"got shape {trial.shape}. "
                "Pass a banded Jacobian with lband/uband and use miter=4."
            )


def _determine_miter(jac, lband, uband, explicit_miter=None):
    """Determine the MITER iteration-method flag from the supplied jac/band arguments."""

    if jac is not None and not callable(jac):
        raise TypeError("'jac' must be callable or None.")
    if lband is not None and lband < 0:
        raise ValueError("'lband' must be a non-negative integer.")
    if uband is not None and uband < 0:
        raise ValueError("'uband' must be a non-negative integer.")

    is_banded = lband is not None or uband is not None
    lband = lband if lband is not None else 0
    uband = uband if uband is not None else 0

    if explicit_miter is not None:
        if explicit_miter not in range(6):
            raise ValueError("'miter' must be an integer between 0 and 5.")
        if explicit_miter in (1, 4) and not jac:
            raise ValueError(
                f"'jac' must be provided when 'miter' is {explicit_miter}."
            )
        if explicit_miter in (4, 5) and not is_banded:
            raise ValueError(
                f"'lband' and 'uband' must be provided when 'miter' is {explicit_miter}."
            )
        return explicit_miter, lband, uband

    if is_banded:
        miter = 4 if jac else 5
    else:
        miter = 1 if jac else 2
    return miter, lband, uband


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
        ``(ml + mu + 1, n)`` array where ``PD[i-j+mu, j] = df(i)/dy(j)``.
        If not supplied, ZVODE approximates the Jacobian by finite differences.
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian.  Must be
        non-negative integers.  When either is set, the banded Jacobian path
        is used and the other defaults to 0.  The full band has width
        ``lband + uband + 1``.
    max_order : int, optional
        Maximum integration order.  Capped at 12 for Adams and 5 for BDF.
    miter : {0, 1, 2, 3, 4, 5}, optional
        Iteration method override.  Normally inferred from `jac` and `lband`/`uband`:

        * 0 – functional iteration (no Jacobian, non-stiff only)
        * 1 – chord with user-supplied full Jacobian
        * 2 – chord with internally generated full Jacobian (default for BDF without *jac*)
        * 3 – chord with diagonal Jacobian approximation
        * 4 – chord with user-supplied banded Jacobian
        * 5 – chord with internally generated banded Jacobian
    jsv : {1, -1}, optional
        Jacobian-saving flag.  ``1`` (default) saves and reuses the Jacobian;
        ``-1`` recomputes it every step.

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
    .. [1] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A Variable
       Coefficient ODE Solver," SIAM J. Sci. Stat. Comput., 10(5), 1038–1051
       (1989).
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

        warn_extraneous(extraneous)
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

        self.miter, self.ml, self.mu = _determine_miter(jac, lband, uband, miter)

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

        if jac is not None and self.miter in (1, 4):
            _validate_jac_shape(jac, self.miter, self.ml, self.mu, self.n, t0, self._ytmp)

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

        # Complex workspace size
        #
        # The Fortran source computes LENWM = (1+JCO)*N*N and LENP = N*N using
        # int32 (LP64 calling convention).  For N >= 46341, N*N overflows int32,
        # which would corrupt internal array offsets even though Python allocates
        # the arrays correctly (Python integers are arbitrary-precision).
        # Detect this before calling into Fortran.
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
            self.h0 = validate_first_step(first_step, t0, t_bound)
            self.rwork[4] = self.h0

        if max_step is not None:
            self.max_step = validate_max_step(max_step)
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
