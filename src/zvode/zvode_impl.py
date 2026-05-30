
import warnings
import numpy as np

from scipy.integrate import (OdeSolver, DenseOutput)
from scipy.integrate._ivp.common import (warn_extraneous, validate_max_step,
                                         validate_first_step)

from . import _zvode

MESSAGES = {
    -1: "Excess work done on this call",
    -2: "Excess accuracy requested",
    -3: "Illegal input detected.",
    -4: "Repeated error test failures",
    -5: "Repeated convergence failues",
    -6: "Error weight become zero during problem integration",
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
        pd[:ml + mu + 1, :n] = jac(t, y)

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
        raise ValueError(f"`rtol` must be a scalar or a 1D array of length {n}.")
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError(f"`atol` must be a scalar or a 1D array of length {n}.")

    # 2. Positivity checks
    if np.any(rtol < 0) or np.any(atol < 0):
        raise ValueError("`rtol` and `atol` must be positive.")

    # 3. Auto-correction for impossibly small rtol (SciPy style)
    EPS = np.finfo(float).eps
    if np.any(rtol < 100 * EPS):
        warnings.warn(
            f"At least one element of `rtol` is too small. "
            f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.",
            stacklevel=3
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


def _determine_miter(jac, lband, uband, explicit_miter=None):
    """Determine the MITER iteration-method flag from the supplied jac/band arguments."""

    # --- 1. Validate Band Value ---
    if lband is not None and lband < 0:
        raise ValueError("`lband` must be a non-negative integer.")
    if uband is not None and uband < 0:
        raise ValueError("`uband` must be a non-negative integer.")

    # --- 2. Manual Override Logic ---
    if explicit_miter is not None:
        if explicit_miter not in (0, 1, 2, 3, 4, 5):
            raise ValueError("Explicit `miter` must be an integer between 0 and 5.")

        # ENFORCEMENT: miter 1 and 4 strictly require a user-supplied Jacobian
        if explicit_miter in (1, 4) and jac is None:
            raise ValueError(
                f"`jac` must be provided when `miter` is {explicit_miter} "
                "(user-supplied Jacobian method)."
            )

        # Normalize bands for banded methods
        if explicit_miter in (4, 5):
            lband = 0 if lband is None else lband
            uband = 0 if uband is None else uband

        return explicit_miter, lband, uband

    # --- 3. Automatic Deduction Logic ---
    is_banded = (lband is not None) or (uband is not None)

    if is_banded:
        # Normalize missing bands to 0
        lband = 0 if lband is None else lband
        uband = 0 if uband is None else uband

        miter = 4 if jac is not None else 5
    else:
        miter = 1 if jac is not None else 2

    return miter, lband, uband


def _falling_factorial(j, k):
    """Compute j*(j-1)*...*(j-k+1); returns 1 for k=0 (empty product)."""
    result = 1
    for m in range(j - k + 1, j + 1):
        result *= m
    return result

class ZVODEDenseOutput(DenseOutput):
    """Dense output interpolant for ZVODE using the Nordsieck history array.

    Evaluates the interpolating polynomial via Horner's method applied to the
    snapshot of the Nordsieck array YH captured at the end of each accepted
    step.  The Nordsieck array column *j* holds ``H**j / j! * y^(j)(t)``, so
    the polynomial is evaluated by the recurrence

    .. math::

        p(t) = \\sum_{j=0}^{nq} \\binom{s}{j}^{(k)} \\, yh_j,
        \\quad s = (t - t_n) / h

    where the falling-factorial weights are computed iteratively via Horner's
    method.  No internal Fortran state is required after construction.

    Parameters
    ----------
    t_old : float
        Start of the step.
    t : float
        End of the step.
    yh : ndarray, shape (n, nq+1), complex128
        Nordsieck history array, column-major copy taken at the end of the
        step and scaled to step size *h*.
    h : float
        Step size the Nordsieck array is scaled to (``HCUR`` in ZVODE).
    """

    def __init__(self, t_old, t, yh, h):
        super().__init__(t_old, t)

        # yh : (n, nq+1) complex128, column j holds H^j/j! * y^(j)(t)
        self.yh = yh
        self.nq = yh.shape[1] - 1
        self.h = h      # HCUR: step size the Nordsieck array is scaled to

    def _call_impl(self, t):
        """Evaluate the interpolant at time(s) *t*; returns shape (n,) or (n, m)."""
        nq, h = self.nq, self.h
        tn = self.t

        k = 0  # interpolation

        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # normalised position, shape (m,)
        s = (t - tn)/h

        # Seed Horner with the highest-order Nordsieck column
        c = _falling_factorial(nq, k)
        dky = c*np.outer(self.yh[:,nq], np.ones(t.shape[0])) # (n, m)
        for j in range(nq - 1, -1, -1):
            c = _falling_factorial(j, k)
            dky = c*self.yh[:,j,np.newaxis] + s*dky

        return dky[:,0] if scalar else dky

class ZVODE(OdeSolver):
    """Solver for complex-valued ODEs using ZVODE (Variable-coefficient, fixed-leading-coefficient).

    ZVODE solves the initial value problem for stiff or non-stiff systems of
    first-order complex ODEs::

        dy/dt = f(t, y),   y(t0) = y0

    where *y* is a complex vector.  It is based on the EPISODE/EPISODEB
    packages and implements Adams (non-stiff) and BDF (stiff) methods with
    orders up to 12 and 5 respectively.

    .. note::

        When using ZVODE for a stiff system, *f* must be analytic (i.e., each
        component f(i) must be an analytic function of each y(j)).  For a
        complex stiff system where *f* is not analytic, use a real-valued
        solver on the equivalent real system of doubled dimension.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system, ``f(t, y)``.  The output must be
        array-like with the same shape as *y*.
    t0 : float
        Initial value of the independent variable.
    y0 : array_like, shape (n,)
        Initial state; will be cast to ``complex128``.
    t_bound : float
        Boundary time.  Integration will not proceed past this value; also
        determines the direction of integration.
    zvode_method : {'BDF', 'Adams'}, optional
        Integration method.  ``'BDF'`` (default) uses the stiff
        Backward-Differentiation Formula method (max order 5).  ``'Adams'``
        uses the non-stiff Adams method (max order 12).
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
        Jacobian matrix of *f* with respect to *y*, ``jac(t, y)``.
        For a full Jacobian, return an ``(n, n)`` array ``J[i, j] = df(i)/dy(j)``.
        For a banded Jacobian (when *lband* / *uband* are set), return an
        ``(ml + mu + 1, n)`` array where ``PD[i-j+mu, j] = df(i)/dy(j)``.
        If not supplied, ZVODE approximates the Jacobian by finite differences.
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian.  Must be
        non-negative integers.  When either is set, the banded Jacobian path
        is used and the other defaults to 0.  The full band has width
        ``lband + uband + 1``.
    max_order : int, optional
        Maximum integration order.  Capped at 12 for Adams and 5 for BDF.
    max_steps : int, optional
        Maximum number of internal steps (currently ignored).
    miter : {0, 1, 2, 3, 4, 5}, optional
        Iteration method override.  Normally inferred from *jac* and *lband*/*uband*:

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

    References
    ----------
    .. [1] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A Variable
       Coefficient ODE Solver," SIAM J. Sci. Stat. Comput., 10(5), 1038–1051
       (1989).
    """

    def __init__(self, fun, t0, y0, t_bound, *,
                 zvode_method='BDF',
                 rtol=1.0e-3, atol=1.0e-6,
                 first_step=None,
                 min_step=0.0,
                 max_step=np.inf,
                 jac=None,
                 lband=None,uband=None,
                 max_order=None,
                 max_steps=None,
                 miter=None,
                 jsv=1,
                 **extraneous):

        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound,
                        vectorized=False,
                        support_complex=True)

        self.tout = self.t_bound
        self._ytmp = np.array(y0,dtype=np.complex128,order='C',copy=True)
        self.y = self._ytmp.copy()

        self.istate = 1 # Start integration
        self.itask = 5 # Take one step, without passing t_bound, and return

        # Select method
        if zvode_method == 'Adams':
            self.meth = 1
            maxord_allowed = 12
        elif zvode_method == 'BDF':
            self.meth = 2
            maxord_allowed = 5
        else:
            raise ValueError(
                f"Invalid method '{zvode_method}'. Valid options are 'Adams' or 'BDF'."
            )

        # Determine tolerance settings
        self.itol, self.rtol, self.atol = \
            _check_tolerances(rtol,atol,self.n)

        # Wrap the SciPy function callback to do in-place modification
        self.wrap_fun = _wrapped_fun(fun)

        # Determine iteration method
        self.miter, self.ml, self.mu = _determine_miter(
            jac, lband, uband, miter)

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

        # TODO: Jacobian-saving strategy checks
        self.jsv = jsv

        # Calculate the method flag
        self.mf = self.jsv*(10*self.meth + self.miter)

        if self.mf not in (10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25):
            raise RuntimeError("Error setting the method flag")

        # Complex workspace
        if self.miter == 0:
            lwm = 0
        elif self.miter in (1,2):
            if self.mf > 0:
                lwm = 2*self.n**2
            elif self.mf < 0:
                lwm = self.n**2
            else:
                lwm = None
        elif self.miter == 3:
            lwm = self.n
        elif self.miter in (4,5):
            if self.mf > 0:
                lwm = (3*self.ml + 2*self.mu + 2)*self.n
            elif self.mf < 0:
                lwm = (2*self.ml + self.mu + 1)*self.n
            else:
                lwm = None

        if lwm is None:
            raise ValueError()

        lzw = self.n*(maxord_allowed + 1) + 2*self.n + lwm
        self.zwork = np.empty(lzw,dtype=np.complex128)

        # Real workspace
        lrw = 20 + self.n
        self.rwork = np.empty(lrw,dtype=np.float64)

        # Integer work space
        liw = 30 if self.miter in (0,3) else 30 + self.n
        self.iwork = np.empty(liw,dtype=np.int32)

        if self.miter in (4,5):
            # Banded Jacobian
            self.iwork[0] = self.ml
            self.iwork[1] = self.mu

        # Optional input settings
        self.iopt = 1
        self.rwork[4:9] = 0.0
        self.iwork[4:9] = 0

        # TODO: domain checks for step-sizes
        if self.itask == 5:
            self.rwork[0] = t_bound

        if first_step is not None:
            self.h0 = validate_first_step(first_step,t0,t_bound)
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
                    f"'max_order' ({max_order}) exceeds the maximum allowed order ({max_allowed}) "
                    f"for the selected method. The solver will automatically reduce it.",
                    stacklevel=2
                )

            # Load the potentially "wrong" value; the capping
            # happens within the Fortran routine
            self.iwork[4] = max_order

        if max_steps is not None:
            if max_order <= 0:
                raise ValueError("'max_steps' must be a positive integer.")

            warnings.warn("'max_steps' are ignored currently")
            # self.iwork[5] = int(max_steps)


    def _step_impl(self):
        """Call ZVODE for one step"""

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
            self.mf)

        self.istate = istate
        self.t = t

        self.nfev = self.iwork[11]
        self.njev = self.iwork[12]
        self.nlu = self.iwork[19]

        if self.istate != 2:
            return False, f"ZVODE returned with istate = {self.istate}"

        self.y = self._ytmp.copy()

        # Succesful step
        return True, None

    def _dense_output_impl(self):
        """Capture the current Nordsieck array and return a ZVODEDenseOutput interpolant."""
        nq = int(self.iwork[14]) # IWORK(15) = NQCUR
        h = float(self.rwork[10]) # RWORK(11) = HU: step size last used

        # YH occupies zwork[0 : n*(nq+1)] in Fortran column-major order

        yh = self.zwork[:self.n * (nq + 1)].reshape((self.n,nq+1),order='F').copy()

        return ZVODEDenseOutput(self.t_old, self.t, yh, h)
