"""Procedural interface to the ZVODE ODE solver.

This module provides a single-call integration function in the spirit of
:func:`scipy.integrate.odeint` and the MATLAB ODE suite (``ode45``,
``ode15s``, ...).  The goal is to hide the stateful workspace management
of the underlying ZVODE Fortran library and expose a clean, Pythonic API
that is familiar to users of those tools while still allowing access to
ZVODE-specific options such as banded Jacobians and step-size controls.

Routines
--------
solve_complex_ivp
    Integrate a complex-valued initial value problem using variable-order
    Adams or BDF multistep methods.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from threading import Lock
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from . import _zvode
from ._helpers import (
    MESSAGES,
    _check_tolerances,
    _resolve_miter,
    _validate_max_step,
    _validate_min_step,
    _validate_first_step,
    _validate_fun_shape,
    _validate_jac_shape,
    _wrapped_fun,
    _wrapped_jac,
)

# ZVODE stores solver state in Fortran COMMON blocks that are global to the
# process.  Only one integration can be active at a time across all threads.
ZVODE_LOCK = Lock()

# ---------------------------------------------------------------------------
# Canonical ctypes CFUNCTYPE descriptors for compiled callbacks
# ---------------------------------------------------------------------------
# These serve two purposes:
#   1. As decorators for ctypes callbacks: @ZVODE_FUN_CTYPE
#   2. As documentation of the expected C-level calling convention.

ZVODE_FUN_CTYPE = ctypes.CFUNCTYPE(
    None,             # void return
    ctypes.c_int,     # neq
    ctypes.c_double,  # t
    ctypes.c_void_p,  # const double complex *y  (passed as opaque pointer)
    ctypes.c_void_p,  # double complex *dy        (passed as opaque pointer)
    ctypes.c_void_p,  # void *ctx
)

ZVODE_JAC_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,     # neq
    ctypes.c_double,  # t
    ctypes.c_void_p,  # const double complex *y
    ctypes.c_int,     # ml
    ctypes.c_int,     # mu
    ctypes.c_void_p,  # double complex *pd  (column-major)
    ctypes.c_int,     # nrowpd
    ctypes.c_void_p,  # void *ctx
)

# Set ZVODE_BACKEND=python to fall back to the pure-Python knot loop.
# Any other value (including unset) uses the C-level drive_knots entry point.
_USE_C_KNOTS: bool = os.environ.get("ZVODE_BACKEND", "C").upper() != "PYTHON"


class ZVODEResult(dict):
    """Result of :func:`solve_complex_ivp`; a dict with attribute access.

    All fields are accessible both as ``result['key']`` and ``result.key``.
    """

    def __getattr__(self, name):
        """Return self[name], raising AttributeError if the key is absent."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self):
        """Return a concise string showing t/y shapes and solver counters."""
        t = self.get("t")
        y = self.get("y")
        t_s = f"ndarray(shape={t.shape})" if isinstance(t, np.ndarray) else repr(t)
        y_s = (
            f"ndarray(shape={y.shape}, dtype={y.dtype})"
            if isinstance(y, np.ndarray)
            else repr(y)
        )
        return (
            f"ZVODEResult(t={t_s}, y={y_s}, "
            f"nfev={self.get('nfev')}, njev={self.get('njev')}, "
            f"nlu={self.get('nlu')})"
        )


# ---------------------------------------------------------------------------
# C function-pointer detection
# ---------------------------------------------------------------------------


def _cfunc_address(fun):
    """Return the C function pointer address (int) for compiled callbacks.

    Recognises ctypes ``CFUNCTYPE`` instances (including ``numba_cfunc.ctypes``).
    Returns ``None`` for ordinary Python callables.
    """
    if isinstance(fun, ctypes._CFuncPtr):
        return ctypes.cast(fun, ctypes.c_void_p).value
    return None


# ---------------------------------------------------------------------------
# Workspace allocation
# ---------------------------------------------------------------------------


def _make_workspace(
    n,
    miter,
    ml,
    mu,
    mf,
    maxord_allowed,
    first_step,
    min_step,
    max_step,
    max_order,
    max_num_steps,
    t0,
    t_bound,
):
    """Allocate and initialise ZVODE's three workspace arrays.

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
        raise RuntimeError(f"Unhandled miter={miter}")

    lzw = n * (maxord_allowed + 1) + 2 * n + lwm
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
    if max_step > 0:
        rwork[5] = float(max_step)
    if min_step:
        rwork[6] = float(min_step)
    if max_order is not None:
        iwork[4] = int(max_order)
    iwork[5] = int(max_num_steps)  # MXSTEP: max internal steps per output point

    return zwork, rwork, iwork


# ---------------------------------------------------------------------------
# Integration drivers
# ---------------------------------------------------------------------------


def _zvode_adaptive(
    fun,
    jac,
    y0,
    t0,
    t_bound,
    itol,
    rtol,
    atol,
    mf,
    iopt,
    zwork,
    rwork,
    iwork,
    refine=1,
    allow_overshoot=False,
):
    """Drive ZVODE in single-step mode, collecting every accepted step.

    Uses ITASK=5 by default (step must not overshoot TCRIT = rwork[0] = t_bound).
    When allow_overshoot=True, uses ITASK=2 instead (tout is ignored; ZVODE
    may step past t_bound).

    When refine > 1, inserts (refine - 1) evenly-spaced interpolated points
    inside each accepted step using ZVINDY before appending the step endpoint.

    Returns
    -------
    ts : ndarray, shape (m,)
    ys : ndarray, shape (n, m), complex128, Fortran order
    istate : int   (2 = success, negative = solver error)
    """
    ITASK = 2 if allow_overshoot else 5
    istate = 1  # initial call

    n = len(y0)
    t = float(t0)
    direction = np.sign(float(t_bound) - t)
    ytmp = y0.copy()

    ts = [t]
    ys = [y0]

    # TODO: replace this Python loop with a call to _zvode.drive() once the
    # C entry point is implemented.  Moving the loop into compiled code drops
    # the per-step Python/C boundary crossing AND, when fun/jac are compiled
    # cfuncs, eliminates argument tuple packing/unpacking on every RHS
    # evaluation — making the entire integration run without re-entering the
    # Python interpreter.
    with ZVODE_LOCK:
        while direction * (float(t_bound) - t) > 0:
            t_old = t
            t, istate = _zvode.zvode(
                fun,
                ytmp,
                t,
                t_bound,
                itol,
                rtol,
                atol,
                ITASK,
                istate,
                iopt,
                zwork,
                rwork,
                iwork,
                jac,
                mf,
            )

            if istate < 0:
                break

            if refine > 1:
                # After an accepted step the Nordsieck array in zwork[0:n*(nq+1)]
                # is valid for interpolation over [t_old, t].  ZVINDY is called
                # before the next zvode call overwrites zwork.
                nq = int(iwork[13])  # IWORK(14) = NQU: order last used
                hu = float(rwork[10])  # RWORK(11) = HU: step size last used
                yh = zwork[: n * (nq + 1)].reshape((n, nq + 1), order="F")
                dky = np.empty(n, dtype=np.complex128)
                for i in range(1, refine):
                    t_i = t_old + i * (t - t_old) / refine
                    # h == hu immediately after an accepted step; both are passed
                    # because zvindy uses h for normalisation and hu for the
                    # interval check.
                    _zvode.zvindy(t_i, 0, yh, hu, t, hu, dky)
                    ts.append(t_i)
                    ys.append(dky.copy())

            ts.append(t)
            ys.append(ytmp.copy())

    ts = np.asarray(ts)
    ys = np.asfortranarray(np.vstack(ys).T)  # (n, m)

    return ts, ys, istate


def _zvode_knots(fun, jac, y0, tspan, itol, rtol, atol, mf, iopt, zwork, rwork, iwork):
    """Drive ZVODE to each requested output knot using ITASK=1.

    ZVODE takes as many internal steps as needed to reach each knot and
    returns once per knot — no Python overhead between internal steps.

    On failure (istate < 0) the arrays are truncated to only the successfully
    completed knots; no uninitialized data is ever returned.

    Note: zwork, rwork, and iwork are updated in place by each call.

    Returns
    -------
    tspan : ndarray   (truncated to completed knots on failure)
    ys    : ndarray, shape (n, m), complex128, Fortran order
    istate : int
    """
    ITASK = 1  # normal: step to tout, taking as many steps as needed
    istate = 1  # initial call

    n = len(y0)
    ytmp = y0.copy()  # mutable work buffer; only this is passed to Fortran

    ys = np.empty((n, len(tspan)), dtype=np.complex128, order="F")
    ys[:, 0] = y0
    t = float(tspan[0])

    # TODO: same as _zvode_adaptive — replace with _zvode.drive() to move the
    # knot loop into C and fully eliminate Python overhead in the inner loop.
    with ZVODE_LOCK:
        for i in range(1, len(tspan)):
            t, istate = _zvode.zvode(
                fun,
                ytmp,
                t,
                float(tspan[i]),
                itol,
                rtol,
                atol,
                ITASK,
                istate,
                iopt,
                zwork,
                rwork,
                iwork,
                jac,
                mf,
            )

            if istate < 0:
                # Do not store ytmp: ZVODE's output is not meaningful on error.
                # Return only the knots that completed successfully.
                return tspan[:i], ys[:, :i], istate

            ys[:, i] = ytmp
            # istate == 2: carry the ZVODE continuation state into the next
            # segment; do not reset to 1.

    return tspan, ys, istate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_complex_ivp(
    fun: Callable[..., Any] | ctypes._CFuncPtr,
    tspan: ArrayLike,
    y0: ArrayLike,
    *,
    rtol: float | ArrayLike = 1.0e-3,
    atol: float | ArrayLike = 1.0e-6,
    jac: Callable[..., Any] | ctypes._CFuncPtr | None = None,
    ctx: ctypes.c_void_p | None = None,
    method: Literal["BDF", "Adams"] = "BDF",
    lband: int | None = None,
    uband: int | None = None,
    save_steps: bool = True,
    refine: int = 1,
    allow_overshoot: bool = False,
    first_step: float | None = None,
    min_step: float = 0.0,
    max_step: float = np.inf,
    max_num_steps: int = 1_000_000,
    max_order: int | None = None,
    miter: int | None = None,
    save_jac: bool = True,
) -> ZVODEResult:
    """Integrate a complex-valued ODE initial value problem.

    Solves::

        dy/dt = f(t, y),   y(t0) = y0,   y, f ∈ ℂⁿ

    The solver automatically selects and adjusts its order and step size at
    each step to meet the requested tolerances.  Use ``method='BDF'``
    (default) for stiff systems and ``method='Adams'`` for smooth, non-stiff
    ones.  Providing a Jacobian via ``jac`` improves efficiency for BDF since
    it avoids finite-difference approximation of the derivative matrix; for
    large or banded systems supplying the sparsity structure through ``lband``
    and ``uband`` reduces both memory and work per step.

    Parameters
    ----------
    fun : callable or ctypes._CFuncPtr
        Right-hand side of the system.

        * **Python callable**: ``fun(t, y) -> array_like`` (SciPy-compatible).
        * **Compiled callback** (``ctypes.CFUNCTYPE`` instance or
          ``numba_cfunc.ctypes``): called directly as a C function pointer,
          bypassing the Python interpreter on every RHS evaluation.  The
          C-level signature is::

              void fun(int neq, double t,
                       const double complex *y,
                       double complex       *dy,
                       void                 *ctx);

          Use ``ZVODE_FUN_CTYPE`` from this module as the ``CFUNCTYPE``
          decorator.  For numba, use ``zvode_fun_sig`` as the ``@cfunc``
          signature and pass ``my_rhs.ctypes``.

    tspan : array_like
        Integration times.

        * Two elements ``[t0, tf]`` and ``save_steps=True`` (default) →
          every accepted internal step is collected and returned.
        * Two elements ``[t0, tf]`` and ``save_steps=False`` →
          endpoint-only mode: returns a scalar ``t`` and a 1-D ``y``.
        * Three or more elements ``[t0, t1, …, tf]`` → output returned only
          at the requested knots (``save_steps`` is ignored).  The solver
          uses its own internal steps to advance between knots and evaluates
          the solution at each requested time; the accuracy at those points
          equals the accuracy at internal steps.  Providing many intermediate
          knots has little effect on computational efficiency.
    y0 : array_like, shape (n,)
        Initial state; cast to ``complex128``.
    rtol, atol : float or array_like, optional
        Relative and absolute local error tolerances.  The solver keeps the
        local error roughly below ``rtol * |y(i)| + atol`` for each component.
        Scalar or per-component arrays are accepted.  Defaults are
        ``rtol=1e-3``, ``atol=1e-6``.
    jac : callable, ctypes._CFuncPtr, or None, optional
        Jacobian of ``fun`` w.r.t. ``y``.

        * **Python callable**, full (no ``lband``/``uband``):
          ``jac(t, y) -> (n, n)`` array with ``J[i, j] = df(i)/dy(j)``.
        * **Python callable**, banded (``lband``/``uband`` set):
          ``jac(t, y) -> (lband + uband + 1, n)`` array where element
          ``J[i - j + uband, j]`` holds ``df(i)/dy(j)``.
        * **Compiled callback**: C-level signature::

              void jac(int neq, double t,
                       const double complex *y,
                       int ml, int mu,
                       double complex       *pd,
                       int nrowpd,
                       void                 *ctx);

        Mixed mode is supported: ``fun`` can be a Python callable while
        ``jac`` is a compiled callback, or vice versa.
    ctx : ctypes.c_void_p or None, optional
        Optional shared user-data pointer passed as the last argument to
        **both** compiled callbacks on every invocation.  ``None`` (default)
        passes a NULL pointer.  Ignored (with a ``UserWarning``) when all
        callbacks are plain Python callables.  The caller is responsible for
        keeping the referent alive for the duration of the integration.

    method : {'BDF', 'Adams'}, optional
        Linear multistep method.  ``'BDF'`` (default) for stiff problems
        (max order 5); ``'Adams'`` for non-stiff (max order 12).
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian.  Must be
        non-negative integers.  When either is set, the banded Jacobian path
        is used and the other defaults to 0.  The full band has width
        ``lband + uband + 1``.

    Returns
    -------
    result : ZVODEResult
        Dict-like object with attribute access.  Always contains:

        result.t : float or ndarray, shape (m,)
            Output time(s).  A scalar float in endpoint-only mode
            (``len(tspan) == 2`` and ``save_steps=False``); a 1-D array
            otherwise.
        result.y : ndarray, shape (n,) or (n, m), complex128
            Solution state(s).  A 1-D array in endpoint-only mode; a 2-D
            Fortran-order array with ``result.y[:, k]`` the state at
            ``result.t[k]`` otherwise.
        result.nfev : int
            Number of right-hand side evaluations.
        result.njev : int
            Number of Jacobian evaluations.
        result.nlu : int
            Number of LU decompositions.

    Other Parameters
    ----------------
    save_steps : bool, optional
        When ``tspan`` has exactly two elements, controls whether every
        accepted internal step is stored.  ``True`` (default) collects all
        steps; ``False`` returns only the endpoint.  Note: ZVODE always uses
        adaptive time-stepping regardless of this flag — it only governs what
        output is captured.
    refine : int, optional
        Number of output points per accepted step when ``save_steps=True``.
        ``refine=1`` (default) records only the step endpoints.
        ``refine=N`` inserts ``N - 1`` additional interpolated points inside
        each step for smoother plots; this does not improve the accuracy of
        the integration.  Ignored when ``save_steps=False`` or
        ``len(tspan) > 2``.
    allow_overshoot : bool, optional
        When ``save_steps=True``, allow the solver to step past the endpoint
        ``tspan[1]``.  ``False`` (default) ensures the last output point is
        exactly ``tspan[1]``.  ``True`` lets the solver choose its step size
        freely, which can occasionally be more efficient, but the last output
        point may lie slightly beyond ``tspan[1]``.  Ignored when
        ``save_steps=False`` or ``len(tspan) > 2``.
    first_step : float or None, optional
        Initial step size.  Chosen automatically if not given.
    min_step : float, optional
        Minimum allowed step size.  Default 0.
    max_step : float, optional
        Maximum allowed step size.  Default ``np.inf``.
    max_num_steps : int, optional
        Maximum number of internal steps between two consecutive output
        points.  Default 1 000 000.  Lower this to cap computational work
        when function evaluations are expensive; an error is raised if the
        budget is exhausted before the next output point.
    max_order : int or None, optional
        Maximum integration order.  Capped at 12 for Adams and 5 for BDF.
    miter : {0, 1, 2, 3, 4, 5} or None, optional
        Iteration method used by the corrector.  Normally inferred from
        ``method``, ``jac``, and the band arguments.  Without ``jac``,
        ``method='Adams'`` defaults to ``0`` (functional iteration) and
        ``method='BDF'`` defaults to ``2`` (internally generated Jacobian).
        Providing ``jac`` selects ``1`` (dense) or ``4`` (banded).  Pass
        this argument only to override the automatic selection — for instance
        to force diagonal (``3``) or finite-difference Jacobian generation
        even when a ``jac`` callable is supplied.  Use with care: an
        inconsistent combination (e.g. ``miter=4`` without band arguments)
        will raise a ``ValueError`` or cause a solver failure.
    save_jac : bool, optional
        If ``True`` (default), the solver retains a copy of the Jacobian to
        reuse when rebuilding the Newton iteration matrix, reducing Jacobian
        evaluations at the cost of extra memory.  If ``False``, no copy is
        kept and the Jacobian is recomputed whenever the iteration matrix
        needs updating.  Ignored for functional iteration (``miter=0``) or
        diagonal approximation (``miter=3``).

    Raises
    ------
    ValueError
        On invalid arguments.
    RuntimeError
        When the solver cannot reach the requested endpoint.

    Notes
    -----
    **Thread safety** — ``solve_complex_ivp`` holds a process-wide lock for
    the entire integration.  Concurrent calls from multiple threads will
    queue rather than run in parallel.  Use ``multiprocessing`` for parallel
    independent integrations.

    **C-level callbacks** — compiled callbacks (``ctypes.CFUNCTYPE`` instances
    or ``numba_cfunc.ctypes``) are called directly as C function pointers
    through the ``drive_knots`` / ``drive_adaptive`` integration loops,
    bypassing the Python interpreter on every RHS or Jacobian evaluation.

    References
    ----------
    .. [Brown1989] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE: A
       Variable-Coefficient ODE Solver," *SIAM J. Sci. Stat. Comput.*,
       10(5), pp. 1038-1051, 1989. https://doi.org/10.1137/0910062

    Examples
    --------
    Trace the unit circle: ``dy/dt = i*y``, ``y(0) = 1``, analytic solution
    ``y(t) = exp(i*t)``.  After one full revolution the state returns to 1:

    >>> import math
    >>> from zvode import solve_complex_ivp
    >>> sol = solve_complex_ivp(lambda t, y: 1j*y, [0, 2*math.pi], [1+0j])
    >>> bool(abs(sol.y[0, -1] - 1.0) < 1e-2)   # back near start after one loop
    True
    """

    # ------------------------------------------------------------------
    # 1.  Validate tspan and y0
    # ------------------------------------------------------------------
    tspan = np.asarray(tspan, dtype=float)
    if tspan.ndim != 1 or len(tspan) < 2:
        raise ValueError("`tspan` must be a 1-D array with at least two elements.")
    diffs = np.diff(tspan)
    # Python `or` short-circuits: the second np.all is skipped when the first is True.
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "`tspan` must be strictly monotonic (all increasing or all decreasing)."
        )

    if np.isrealobj(y0):
        warnings.warn(
            "y0 has a real dtype and will be cast to complex128. "
            "Pass a complex array to suppress this warning.",
            stacklevel=2,
        )
    y0 = np.array(y0, dtype=np.complex128)
    if y0.ndim != 1:
        raise ValueError("`y0` must be a 1-D array.")
    n = y0.size

    # ------------------------------------------------------------------
    # 2.  Tolerances
    # ------------------------------------------------------------------
    itol, rtol, atol = _check_tolerances(rtol, atol, n)

    # ------------------------------------------------------------------
    # 3.  Method flag
    # ------------------------------------------------------------------
    if method == "Adams":
        meth, maxord_allowed = 1, 12
    elif method == "BDF":
        meth, maxord_allowed = 2, 5
    else:
        raise ValueError(f"Invalid method {method!r}; choose 'Adams' or 'BDF'.")

    _miter, ml, mu = _resolve_miter(jac, lband, uband, meth, n, miter)

    if jac is not None and _miter in (1, 4) and _cfunc_address(jac) is None:
        _validate_jac_shape(jac, _miter, ml, mu, n, tspan[0], y0)

    jsv = 1 if save_jac else -1
    mf = jsv * (10 * meth + _miter)

    _validate_max_step(max_step)
    _validate_min_step(min_step)
    if first_step is not None:
        _validate_first_step(first_step, tspan[0], tspan[-1])
    if max_num_steps < 0:
        raise ValueError("`max_num_steps` must be non-negative.")

    # ------------------------------------------------------------------
    # 4.  Workspace
    # ------------------------------------------------------------------
    iopt = 1  # optional inputs present (rwork / iwork slots populated below)
    # Cap max_step at the largest interval in tspan so the solver cannot
    # overshoot a knot in a single step.  Honour a tighter user-supplied limit.
    _max_interval = float(np.max(np.abs(diffs)))
    _effective_max_step = min(_max_interval, max_step)
    zwork, rwork, iwork = _make_workspace(
        n,
        _miter,
        ml,
        mu,
        mf,
        maxord_allowed,
        first_step,
        min_step,
        _effective_max_step,
        max_order,
        max_num_steps,
        t0=float(tspan[0]),
        t_bound=float(tspan[-1]),
    )

    # ------------------------------------------------------------------
    # 5.  Normalize callbacks
    # ------------------------------------------------------------------
    #
    # Path A (plain Python callable): wrap SciPy-style fun(t,y)->array to the
    #   in-place form that _zvode.zvode and the C drivers expect.
    #
    # Path B (compiled cfunc — ctypes._CFuncPtr or numba @cfunc.ctypes):
    #   extract the raw C function pointer address (int) and pass it to
    #   drive_knots / drive_adaptive, which call it directly without entering
    #   the Python interpreter.  Mixed mode (Python fun + compiled jac, or
    #   vice versa) is supported.

    fun_addr = _cfunc_address(fun)
    jac_addr = _cfunc_address(jac) if jac is not None else None

    # ------------------------------------------------------------------
    # 5a. Validate and extract ctx
    # ------------------------------------------------------------------
    if ctx is not None and not isinstance(ctx, ctypes.c_void_p):
        raise TypeError(
            f"'ctx' must be a ctypes.c_void_p or None, got {type(ctx).__name__!r}."
        )
    # ctypes.c_void_p(0).value is None (null pointer); treat that as 0.
    ctx_addr: int = (ctx.value or 0) if ctx is not None else 0

    # Warn when ctx is provided but all callbacks are Python callables
    # (ctx is meaningless in that case; it's not passed to Python callbacks).
    if ctx is not None and fun_addr is None and jac_addr is None:
        warnings.warn(
            "ctx is ignored when all callbacks are plain Python callables.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 5b. Build the fun/jac objects to pass to the C drivers
    # ------------------------------------------------------------------
    if fun_addr is not None:
        # Path B fun: pass integer address; C layer calls it directly
        _fun = fun_addr
    else:
        # Path A fun: SciPy-compatible; validate shape and wrap to in-place
        _validate_fun_shape(fun, n, tspan[0], y0)
        _fun = _wrapped_fun(fun)

    if jac is None:
        _jac = None
    elif jac_addr is not None:
        # Path B jac: pass integer address
        _jac = jac_addr
    else:
        # Path A jac: SciPy-compatible; wrap to in-place
        _jac = _wrapped_jac(jac, banded=(_miter == 4))

    # ------------------------------------------------------------------
    # 6.  Validate refine; check backend compatibility
    # ------------------------------------------------------------------
    if refine < 1:
        raise ValueError("`refine` must be a positive integer.")

    # Compiled callbacks require the C integration loop (drive_knots /
    # drive_adaptive).  The Python fallback (ZVODE_BACKEND=python) only
    # supports Python callables.
    if not _USE_C_KNOTS and (fun_addr is not None or jac_addr is not None):
        raise RuntimeError(
            "Compiled callbacks (ctypes/numba) require the C integration loop. "
            "Unset the ZVODE_BACKEND environment variable (currently set to 'python')."
        )

    # ------------------------------------------------------------------
    # 7.  Integrate
    # ------------------------------------------------------------------
    if len(tspan) == 2 and save_steps:
        # Collect every accepted step (optionally with ZVINDY interpolation).
        if _USE_C_KNOTS:
            ytmp = y0.copy()
            with ZVODE_LOCK:
                t_out, y_out, istate = _zvode.drive_adaptive(
                    _fun,
                    _jac,
                    ctx_addr,
                    mf,
                    float(tspan[0]),
                    float(tspan[1]),
                    ytmp,
                    itol,
                    rtol,
                    atol,
                    iopt,
                    zwork,
                    rwork,
                    iwork,
                    int(refine),
                    int(allow_overshoot),
                )
        else:
            t_out, y_out, istate = _zvode_adaptive(
                _fun,
                _jac,
                y0,
                tspan[0],
                tspan[1],
                itol,
                rtol,
                atol,
                mf,
                iopt,
                zwork,
                rwork,
                iwork,
                refine=refine,
                allow_overshoot=allow_overshoot,
            )
    elif len(tspan) == 2:
        # Endpoint-only: ZVODE steps freely to t_bound; returns scalar t
        # and 1-D y — no intermediate storage.
        if _USE_C_KNOTS:
            ytmp = y0.copy()
            ts_out = np.empty(2, dtype=np.float64)
            ys_out = np.empty((n, 2), dtype=np.complex128, order="F")
            with ZVODE_LOCK:
                istate, knots_completed = _zvode.drive_knots(
                    _fun,
                    _jac,
                    ctx_addr,
                    mf,
                    tspan,
                    ytmp,
                    ts_out,
                    ys_out,
                    itol,
                    rtol,
                    atol,
                    iopt,
                    zwork,
                    rwork,
                    iwork,
                )
            if istate != 2:
                ts_out, ys_out = ts_out[:knots_completed], ys_out[:, :knots_completed]
            t_out, y_out = float(ts_out[-1]), ys_out[:, -1]
        else:
            t_out, y_out, istate = _zvode_knots(
                _fun, _jac, y0, tspan, itol, rtol, atol, mf, iopt, zwork, rwork, iwork
            )
            t_out = float(t_out[-1])
            y_out = y_out[:, -1]
    else:
        # Knots: output at each element of tspan.
        if _USE_C_KNOTS:
            ytmp = y0.copy()
            ts_out = np.empty(len(tspan), dtype=np.float64)
            ys_out = np.empty((n, len(tspan)), dtype=np.complex128, order="F")
            with ZVODE_LOCK:
                istate, knots_completed = _zvode.drive_knots(
                    _fun,
                    _jac,
                    ctx_addr,
                    mf,
                    tspan,
                    ytmp,
                    ts_out,
                    ys_out,
                    itol,
                    rtol,
                    atol,
                    iopt,
                    zwork,
                    rwork,
                    iwork,
                )
            if istate != 2:
                t_out, y_out = ts_out[:knots_completed], ys_out[:, :knots_completed]
            else:
                t_out, y_out = ts_out, ys_out
        else:
            t_out, y_out, istate = _zvode_knots(
                _fun, _jac, y0, tspan, itol, rtol, atol, mf, iopt, zwork, rwork, iwork
            )

    # ------------------------------------------------------------------
    # 8.  Error reporting
    # ------------------------------------------------------------------
    if istate < 0:
        _msg = MESSAGES.get(istate, "Unknown error.")
        _where = (
            f"at t={t_out[-1]}, before reaching t={tspan[-1]}"
            if len(tspan) == 2 and save_steps
            else f"after {len(t_out)} of {len(tspan)} requested output point(s)"
            if len(tspan) > 2
            else f"before reaching t={tspan[-1]}"
        )
        raise RuntimeError(
            f"solve_complex_ivp: integration failed {_where}. "
            f"ZVODE ISTATE={istate}: {_msg}"
        )

    # Indices follow the ZVODE user documentation (Fortran 1-based → Python 0-based):
    #   IWORK(11)=NST, IWORK(12)=NFE, IWORK(13)=NJE,
    #   IWORK(20)=NLU, IWORK(21)=NNI, IWORK(22)=NCFN, IWORK(23)=NETF.
    return ZVODEResult(
        {
            "t": t_out,
            "y": y_out,
            "nsteps": int(iwork[10]),
            "nfev": int(iwork[11]),
            "njev": int(iwork[12]),
            "nlu": int(iwork[19]),
            "nni": int(iwork[20]),
            "ncfn": int(iwork[21]),
            "netf": int(iwork[22]),
        }
    )
