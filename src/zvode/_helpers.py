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

# Maps linear multistep method name to (ZVODE integer code, maximum order).
_LMM = {"Adams": (1, 12), "BDF": (2, 5)}
# Reverse map: ZVODE integer code → maximum order.  A dict, not a list: the
# codes are 1-based (1 = Adams, 2 = BDF), so list indexing would be off by one.
_METH_MAXORD = {m: o for m, o in _LMM.values()}


def _validate_step_bounds(min_step, max_step):
    """Validate the step-size bounds: ``min_step >= 0`` and ``max_step > 0``.

    Both are unsigned magnitudes, independent of integration direction; ZVODE
    carries the direction sign itself.  Results are not returned — the caller
    forwards the original values straight into the work arrays.
    """
    if min_step < 0:
        raise ValueError("`min_step` must be non-negative.")
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")


def _validate_first_step(first_step, t0, t_bound):
    """Validate the user-supplied initial step size (``None`` passes through).

    ``first_step`` is an unsigned magnitude: it must be positive and no larger
    than the total interval ``abs(t_bound - t0)``.
    """
    if first_step is None:
        return
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > abs(t_bound - t0):
        raise ValueError("`first_step` exceeds `abs(t_bound - t0)`.")


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
        assert False, (
            f"_check_tolerances: unhandled tolerance shape combination "
            f"(rtol.ndim={rtol.ndim}, rtol.shape={rtol.shape}, "
            f"atol.ndim={atol.ndim}, atol.shape={atol.shape}, n={n})"
        )

    return itol, rtol, atol


def _validate_fun_shape(fun, n, t0, y0):
    """Evaluate *fun* once at ``(t0, y0)`` and verify its return shape.

    The right-hand side must return a 1-D array of shape ``(neq,)``.  Common
    mistakes — returning a Python scalar, a 0-D ndarray, or a 2-D array —
    are caught here before any Fortran call is made.
    """
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
    # For plain interpolation (k=0) all falling-factorial weights are 1, so
    # the recurrence simplifies to:
    #   p = yh[:,nq]; for j = nq-1 ... 0: p = yh[:,j] + s*p
    # Allocate one (n, m) buffer upfront; each iteration is two in-place
    # operations with no temporaries: dky *= s; dky += yh[:,j].
    # Initialising from a view of yh would corrupt the stored Nordsieck array.
    dky = np.empty((n, len(t)), dtype=yh.dtype)
    dky[:] = yh[:, nq, np.newaxis]
    for j in range(nq - 1, -1, -1):
        dky *= s
        dky += yh[:, j, np.newaxis]

    return dky[:, 0] if scalar else dky


def _resolve_miter(jac, lband, uband, meth, n, explicit_miter=None):
    """Validate Jacobian/band arguments and resolve the MITER iteration-method flag.

    Raises TypeError or ValueError for inconsistent or out-of-range inputs and
    warns when a banded method's bandwidth exceeds half the system size, then
    returns ``(miter, lband, uband)`` with ``None`` band values normalised to 0.
    """

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
        miter = explicit_miter
    elif is_banded:
        miter = 4 if jac else 5
    elif jac:
        miter = 1
    else:
        miter = (
            0 if meth == 1 else 2
        )  # Adams: functional; BDF: chord with generated Jacobian

    if miter in (4, 5):
        if lband >= n:
            raise ValueError(f"'lband' ({lband}) must be less than neq ({n}).")
        if uband >= n:
            raise ValueError(f"'uband' ({uband}) must be less than neq ({n}).")
        bandwidth = lband + uband + 1
        if bandwidth * 2 > n:
            warnings.warn(
                f"Bandwidth lband + uband + 1 = {bandwidth} exceeds half "
                f"the system size neq = {n}; verify that a banded "
                "solver is appropriate for this problem.",
                stacklevel=3,
            )

    return miter, lband, uband


def _validate_max_order(max_order, meth):
    """Validate `max_order` and cap it to the method's ceiling (12 Adams / 5 BDF).

    `meth` is the ZVODE method code (1 = Adams, 2 = BDF).  ``None`` passes
    through; a positive value above the ceiling is capped to it after warning.
    Returns the order to use.
    """
    if max_order is None:
        return None
    if max_order <= 0:
        raise ValueError("`max_order` must be a positive integer.")
    maxord_allowed = _METH_MAXORD[meth]
    if max_order > maxord_allowed:
        warnings.warn(
            f"`max_order` ({max_order}) exceeds the maximum allowed order "
            f"for the selected method; it will be reduced to {maxord_allowed}.",
            stacklevel=3,
        )
        return maxord_allowed
    return max_order


def _make_workspace(
    n,
    miter,
    ml,
    mu,
    mf,
    t0,
    t_bound,
    first_step=None,
    min_step=0.0,
    max_step=np.inf,
    max_order=None,
    max_num_steps=0,
):
    """Allocate and initialise ZVODE's three workspace arrays.

    Internal helper: it sizes the arrays and packs the (already-validated) user
    parameters into their ZVODE slots.  Callers validate the step-size and order
    arguments at the public boundary; the only checks here are the int32 length
    overflows, which are intrinsic to the sizing itself.

    Returns ``(zwork, rwork, iwork)`` as NumPy arrays.
    Note: zwork, rwork, and iwork are mutable; the integration drivers update
    them in place on every step and read diagnostic counters from them on return.
    """
    _INT32_MAX = 2**31 - 1

    if miter in (1, 2) and n**2 > _INT32_MAX:
        raise ValueError(
            f"neq={n} exceeds the maximum of 46340 for dense Jacobian methods: "
            "neq**2 overflows the 32-bit integer arithmetic used internally."
        )
    if miter in (4, 5):
        _lenwm_max = (3 * ml + mu + 1) * n
        if _lenwm_max > _INT32_MAX:
            raise ValueError(
                f"Banded workspace ({_lenwm_max:,}) overflows int32 arithmetic."
            )

    if miter == 0:
        lwm = 0
    elif miter in (1, 2):
        lwm = 2 * n**2 if mf > 0 else n**2
    elif miter == 3:
        lwm = n
    elif miter in (4, 5):
        lwm = (3 * ml + 2 * mu + 2) * n if mf > 0 else (2 * ml + mu + 1) * n
    else:
        # Unreachable: _resolve_miter guarantees miter is 0..5.  Assert only in
        # the fallthrough, so the normal path pays nothing for the check.
        assert False, f"unexpected miter={miter!r} — bug in zvode"

    meth = abs(mf) // 10
    maxord = _METH_MAXORD[meth]
    lzw = n * (maxord + 1) + 2 * n + lwm
    zwork = np.zeros(lzw, dtype=np.complex128)

    lrw = 20 + n
    rwork = np.zeros(lrw, dtype=np.float64)

    liw = 30 if miter in (0, 3) else 30 + n
    iwork = np.zeros(liw, dtype=np.int32)

    if miter in (4, 5):
        iwork[0] = ml
        iwork[1] = mu

    rwork[0] = float(t_bound)  # TCRIT; required when ITASK=4 or 5
    if first_step is not None:
        # ZVODE requires H0 to carry the sign of the integration direction.
        rwork[4] = float(first_step) * np.sign(t_bound - t0)
    rwork[5] = float(max_step)
    rwork[6] = float(min_step)
    if max_order is not None:
        iwork[4] = int(max_order)
    if max_num_steps:
        iwork[5] = int(max_num_steps)  # MXSTEP: max internal steps per output point

    return zwork, rwork, iwork
