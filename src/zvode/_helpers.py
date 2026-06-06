"""SciPy-free helpers shared by solve.py and zvode_impl.py."""

import warnings

import numpy as np

# ZVODE ISTATE error codes and their human-readable descriptions.
MESSAGES = {
    -1: "Excess work done on this call.",
    -2: "Excess accuracy requested.",
    -3: "Illegal input detected.",
    -4: "Repeated error test failures.",
    -5: "Repeated convergence failures.",
    -6: "Error weight became zero during problem integration.",
}


def _validate_max_step(max_step):
    """Validate that max_step is a positive number."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def _validate_first_step(first_step, t0, t_bound):
    """Validate the user-supplied initial step size.

    Like ``max_step`` and ``min_step``, ``first_step`` is always a positive
    magnitude regardless of integration direction.  ZVODE's H0 (RWORK(5))
    must carry the sign of the direction, so the caller is responsible for
    applying ``np.sign(t_bound - t0)`` when writing the value into rwork[4].
    """
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > abs(t_bound - t0):
        raise ValueError("`first_step` exceeds `abs(t_bound - t0)`.")
    return first_step


def _wrapped_fun(fun):
    """Adapt a SciPy-compatible ``f(t, y) -> array`` callable to the in-place ZVODE signature."""

    def _zvode_fun(t, y, dy):
        dy[:] = fun(t, y)

    return _zvode_fun


def _wrapped_jac(jac, banded=False):
    """Adapt a ``jac(t, y)`` return-value callable to the in-place ZVODE Jacobian signature.

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
    overwritten.
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

    # 3. Auto-correction for impossibly small rtol
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


def _validate_fun_shape(fun, n, t0, y0):
    """Evaluate *fun* once at ``(t0, y0)`` and verify its return shape.

    The right-hand side must return a 1-D array of shape ``(neq,)``.  Common
    mistakes — returning a Python scalar, a 0-D ndarray, or a 2-D array —
    are caught here before any Fortran call is made.
    """
    # FIXME: this evaluation should be counted toward nfev, but the Fortran
    # library owns that counter inside iwork and it is only readable after
    # each accepted step, so incrementing it here would require duplicating
    # the counter in Python.
    trial = np.asarray(fun(t0, y0))
    expected = (n,)
    if trial.shape != expected:
        raise ValueError(
            f"'fun' must return a 1-D array of shape (neq,) = {expected}; "
            f"got shape {trial.shape}."
        )


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


def _eval_nordsieck(yh, h, t, tn):
    """Evaluate the Nordsieck interpolating polynomial at time(s) *t*.

    Implements the Horner recurrence for k=0 (plain interpolation):

    .. math::

        p(t) = \\sum_{j=0}^{nq} s^j \\, yh_j, \\quad s = (t - t_n) / h

    where column *j* of *yh* holds ``h^j / j! * y^(j)(t_n)``.

    Parameters
    ----------
    yh : ndarray, shape ``(n, nq+1)``, complex128
        Nordsieck history array captured at the end of the step.
    h : float
        Step size the array is scaled to (HU).
    t : float or ndarray
        Evaluation time(s).  A scalar returns shape ``(n,)``; an array of
        shape ``(m,)`` returns shape ``(n, m)``.
    tn : float
        Current solver time (end of step, TN/TCUR).

    Returns
    -------
    ndarray, shape ``(n,)`` or ``(n, m)``
    """
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))

    s = (t - tn) / h  # normalised position, shape (m,)

    nq = yh.shape[1] - 1
    n = yh.shape[0]
    # Allocate one (n, m) buffer; iterate in-place — no temporaries.
    dky = np.empty((n, len(t)), dtype=yh.dtype)
    dky[:] = yh[:, nq, np.newaxis]
    for j in range(nq - 1, -1, -1):
        dky *= s
        dky += yh[:, j, np.newaxis]

    return dky[:, 0] if scalar else dky


def _determine_miter(jac, lband, uband, meth, explicit_miter=None):
    """Determine the MITER iteration-method flag from the supplied jac/band/meth arguments."""

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
    elif jac:
        miter = 1
    else:
        miter = (
            0 if meth == 1 else 2
        )  # Adams: functional; BDF: chord with generated Jacobian
    return miter, lband, uband
