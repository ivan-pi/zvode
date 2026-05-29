
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
    """Wraps the ODE function into a mutating function"""
    def zvode_fun(t, y, dy):
        dy[:] = fun(t, y)

    return zvode_fun

def _wrapped_jac(jac,banded=False):
    """Wraps the Jacobian into a mutating function"""

    # pd will be F-contiguous here, and since we are copying the results
    # into it, jac() could be either C or F contiguous
    def zvode_jac(t,y,pd):
        assert y.shape[0] == pd.shape[1]
        # The pd array may be padded in the first dimension
        n = y.shape[0]
        pd[0:n,0:n] = jac(t,y)

    def zvode_banded_jac(t,y,pd,ml,mu):
        n = y.shape[0]
        pd[0:ml+mu+1,0:n] = jac(t,y)

    return zvode_banded_jac if banded else zvode_jac

#  ITOL    RTOL       ATOL          EWT(i)
#   1     scalar     scalar     RTOL*ABS(Y(i)) + ATOL
#   2     scalar     array      RTOL*ABS(Y(i)) + ATOL(i)
#   3     array      scalar     RTOL(i)*ABS(Y(i)) + ATOL
#   4     array      array      RTOL(i)*ABS(Y(i)) + ATOL(i)
def _check_tolerances(rtol, atol, n):
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
    """
    Determines the MITER flag, normalizes bandwidths, and enforces
    integer/Jacobian constraints.
    """

    # --- 1. Validate Band Types ---
    def _validate_band(band, name):
        if band is not None:
            # Check for integer types (including NumPy integers) and non-negativity
            if not isinstance(band, (int, np.integer)) or band < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        return band

    lband = _validate_band(lband, 'lband')
    uband = _validate_band(uband, 'uband')

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

    Evaluates the interpolating polynomial via Horner's method on the
    snapshot of YH taken at the end of the step.  No Fortran COMMON
    block state is needed after construction.
    """

    def __init__(self, t_old, t, yh, h):
        super().__init__(t_old, t)

        # yh : (n, nq+1) complex128, column j holds H^j/j! * y^(j)(t)
        self.yh = yh
        self.nq = yh.shape[1] - 1
        self.h = h      # HCUR: step size the Nordsieck array is scaled to

    def _call_impl(self, t):

        nq, h = self.nq, self.h
        tn = self.t

        k = 0  # interpolation

        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # normalised position, shape (m,)
        s = (t - tn)/h

        # Seed Horner with the highest-order Nordsieck column
        c = _falling_factorial(nq, k)
        dky = np.outer(self.yh[:,nq], np.ones(t.shape[0])) # (n, m)
        for j in range(nq - 1, -1, -1):
            c = _falling_factorial(j, k)
            dky = c*self.yh[:,j,np.newaxis] + s*dky

        return dky[:,0] if scalar else dky

class ZVODE(OdeSolver):
    """Wrapper of ZVODE

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial time
    y0 : array_like, shape(n,)
        Initial state
    t_bound : float
        Boundary time - the integration won't continue beyond it.
        Determines the driection of the integration
    method : string

    rtol, atol : float or array_like, optional
        Relative and absolute local error tolerances

    See Also
    --------


    References
    ----------
    .. [1] ...

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
        self.itask = 2 # Take one step and return

        # Select method
        if zvode_method == 'Adams':
            self.meth = 1
            maxord_allowed = 12
        elif zvode_method == 'BDF':
            self.meth = 2
            maxord_allowed = 5
        else:
            raise ValueError(
                f"Invalid method '{method}'. Valid options are 'Adams' or 'BDF'."
            )

        # Determine tolerance settings
        self.itol, self.rtol, self.atol = \
            _check_tolerances(rtol,atol,self.n)

        # Wrap the SciPy function callback to do in-place modification
        self.wrap_fun = _wrapped_fun(fun)

        # Determine iteration method
        self.miter, self.ml, self.mu = _determine_miter(
            jac, lband, uband, miter)

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
        #self.rwork[0] = t_bound

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
                # The Fortran solver will do this
            self.iwork[4] = max_order

        if max_steps is not None:
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

        if self.istate != 2:
            return False, f"ZVODE returned with istate = {self.istate}"

        self.y = self._ytmp.copy()

        # Succesful step
        return True, None

    def _dense_output_impl(self):

        nq = int(self.iwork[14]) # IWORK(15) = NQCUR
        h = float(self.rwork[11]) # RWORK(12) = HCUR

        # YH occupies zwork[0 : n*(nq+1)] in Fortran column-major order

        yh = self.zwork[:self.n * (nq + 1)].reshape((self.n,nq+1),order='F').copy()

        return ZVODEDenseOutput(self.t_old, self.t, yh, h)
