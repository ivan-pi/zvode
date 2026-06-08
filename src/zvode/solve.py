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
)

# ZVODE stores solver state in Fortran COMMON blocks that are global to the
# process.  Only one integration can be active at a time across all threads.
ZVODE_LOCK = Lock()


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


def _get_cfunc_address(fun):
    """Return the integer C function pointer address if *fun* is a compiled callback.

    Accepts ``ctypes.CFUNCTYPE`` instances (including numba ``@cfunc`` objects
    exposed via their ``.ctypes`` property).  Returns ``None`` for plain Python
    callables.
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


def _zvode_drive_knots(fun, jac, ctx_int, y0, tspan, itol, rtol, atol, mf, iopt,
                       zwork, rwork, iwork):
    """Drive ZVODE to each output knot via the C-level drive_knots entry point.

    *fun* and *jac* are either Python callables or Python ints (compiled
    callback addresses).  *ctx_int* is an integer user-data pointer (0 = NULL).

    Returns ``(tspan_out, ys, istate)``, truncating on failure.
    """
    n = len(y0)
    nknots = len(tspan)
    ytmp = y0.copy()
    ts_out = np.empty(nknots, dtype=np.float64)
    ys_out = np.empty((n, nknots), dtype=np.complex128, order="F")

    _jac = jac if jac is not None else None

    with ZVODE_LOCK:
        istate, knots_completed = _zvode.drive_knots(
            fun, _jac, ctx_int,
            mf, tspan, ytmp, ts_out, ys_out,
            itol, rtol, atol,
            iopt, zwork, rwork, iwork,
        )

    if istate != 2:
        return ts_out[:knots_completed], ys_out[:, :knots_completed], istate
    return ts_out, ys_out, istate


def _zvode_drive_adaptive(fun, jac, ctx_int, y0, t0, t_bound, itol, rtol, atol,
                           mf, iopt, zwork, rwork, iwork,
                           refine=1, allow_overshoot=False):
    """Drive ZVODE in single-step mode via the C-level drive_adaptive entry point.

    Returns ``(ts, ys, istate)``.
    """
    _jac = jac if jac is not None else None

    with ZVODE_LOCK:
        ts, ys, istate = _zvode.drive_adaptive(
            fun, _jac, ctx_int,
            y0, rtol, atol,
            float(t0), float(t_bound),
            itol, iopt, mf,
            zwork, rwork, iwork,
            refine, int(allow_overshoot),
        )
    return ts, ys, istate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_complex_ivp(
    fun: Callable[..., Any],
    tspan: ArrayLike,
    y0: ArrayLike,
    *,
    rtol: float | ArrayLike = 1.0e-3,
    atol: float | ArrayLike = 1.0e-6,
    jac: Callable[..., Any] | None = None,
    ctx: Any | None = None,
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

        * Python callable: ``fun(t, y) -> array_like`` (SciPy-compatible
          return-value form).  The solver copies the result into its internal
          buffer on every evaluation.
        * Compiled callback (``ctypes.CFUNCTYPE`` instance or numba
          ``@cfunc`` object exposed via ``.ctypes``): in-place mutating
          form ``fun(neq, t, y_ptr, dy_ptr, ctx)`` — fills ``dy`` through
          a pointer, bypassing the Python interpreter on every evaluation.

    tspan : array_like
        Integration times.

        * Two elements ``[t0, tf]`` and ``save_steps=True`` (default) →
          every accepted internal step is collected and returned.
        * Two elements ``[t0, tf]`` and ``save_steps=False`` →
          endpoint-only mode: returns a scalar ``t`` and a 1-D ``y``.
        * Three or more elements ``[t0, t1, …, tf]`` → output returned only
          at the requested knots (``save_steps`` is ignored).
    y0 : array_like, shape (n,)
        Initial state; cast to ``complex128``.
    rtol, atol : float or array_like, optional
        Relative and absolute local error tolerances.  Defaults are
        ``rtol=1e-3``, ``atol=1e-6``.
    jac : callable, ctypes._CFuncPtr, or None, optional
        Jacobian of ``fun`` w.r.t. ``y``.

        * Python callable: ``jac(t, y) -> array_like``.  For a dense
          Jacobian return shape ``(n, n)`` with ``J[i, j] = df(i)/dy(j)``.
          For a banded Jacobian return shape ``(lband + uband + 1, n)``
          using ZVODE's banded-storage convention.
        * Compiled callback: in-place mutating form
          ``jac(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx)``.
    ctx : ctypes.c_void_p or None, optional
        Optional shared user-data pointer passed as the ``ctx`` argument to
        **compiled** callbacks on every invocation.  Ignored (with a warning)
        when both ``fun`` and ``jac`` are plain Python callables.

        * ``None`` (default) — NULL is passed as ``ctx``.
        * ``ctypes.c_void_p`` — its ``.value`` is forwarded.

        The caller is responsible for keeping the referent alive for the
        duration of the integration.
    method : {'BDF', 'Adams'}, optional
        Linear multistep method.  ``'BDF'`` (default) for stiff problems;
        ``'Adams'`` for non-stiff.
    lband, uband : int or None, optional
        Lower and upper half-bandwidths of a banded Jacobian.
    save_steps : bool, optional
        When ``tspan`` has exactly two elements, controls whether every
        accepted internal step is stored.
    refine : int, optional
        Number of output points per accepted step when ``save_steps=True``.
    allow_overshoot : bool, optional
        When ``save_steps=True``, allow the solver to step past the endpoint.
    first_step : float or None, optional
        Initial step size hint.
    min_step : float, optional
        Minimum allowed step size.
    max_step : float, optional
        Maximum allowed step size.
    max_num_steps : int, optional
        Maximum internal steps between two consecutive output points.
    max_order : int or None, optional
        Maximum integration order (capped at 12 for Adams and 5 for BDF).
    miter : {0, 1, 2, 3, 4, 5} or None, optional
        Corrector iteration method.  Normally inferred automatically.
    save_jac : bool, optional
        Whether to retain a saved copy of the Jacobian between steps.

    Returns
    -------
    result : ZVODEResult
        Dict-like object with attribute access.  Always contains:

        result.t : float or ndarray, shape (m,)
        result.y : ndarray, shape (n,) or (n, m), complex128
        result.nfev : int
        result.njev : int
        result.nlu : int

    Raises
    ------
    ValueError
        On invalid arguments.
    RuntimeError
        When the solver cannot reach the requested endpoint.

    Examples
    --------
    Trace the unit circle: ``dy/dt = i*y``, ``y(0) = 1``:

    >>> import math
    >>> from zvode import solve_complex_ivp
    >>> sol = solve_complex_ivp(lambda t, y: 1j*y, [0, 2*math.pi], [1+0j])
    >>> bool(abs(sol.y[0, -1] - 1.0) < 1e-2)
    True
    """

    # ------------------------------------------------------------------
    # 1.  Validate tspan and y0
    # ------------------------------------------------------------------
    tspan = np.asarray(tspan, dtype=float)
    if tspan.ndim != 1 or len(tspan) < 2:
        raise ValueError("`tspan` must be a 1-D array with at least two elements.")
    diffs = np.diff(tspan)
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

    # For _resolve_miter, jac may be a ctypes._CFuncPtr (which is callable)
    # or a plain callable.  Both pass the callable() check.
    _miter, ml, mu = _resolve_miter(jac, lband, uband, meth, n, miter)

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
    iopt = 1
    _max_interval = float(np.max(np.abs(diffs)))
    _effective_max_step = min(_max_interval, max_step)
    zwork, rwork, iwork = _make_workspace(
        n, _miter, ml, mu, mf, maxord_allowed,
        first_step, min_step, _effective_max_step,
        max_order, max_num_steps,
        t0=float(tspan[0]), t_bound=float(tspan[-1]),
    )

    # ------------------------------------------------------------------
    # 5.  Validate refine
    # ------------------------------------------------------------------
    if refine < 1:
        raise ValueError("`refine` must be a positive integer.")

    # ------------------------------------------------------------------
    # 6.  Normalize callbacks and ctx
    # ------------------------------------------------------------------
    fun_addr = _get_cfunc_address(fun)
    jac_addr = _get_cfunc_address(jac) if jac is not None else None

    # _fun / _jac: compiled → Python int (address); Python callable → as-is
    _fun = fun_addr if fun_addr is not None else fun
    _jac = (jac_addr if jac_addr is not None else jac) if jac is not None else None

    # Validate ctx
    if ctx is None:
        ctx_int = 0
    elif isinstance(ctx, ctypes.c_void_p):
        ctx_int = ctx.value if ctx.value is not None else 0
    else:
        raise TypeError(
            "`ctx` must be a ctypes.c_void_p or None; "
            f"got {type(ctx).__name__!r}."
        )

    if ctx is not None and fun_addr is None and jac_addr is None:
        warnings.warn(
            "`ctx` is ignored when both `fun` and `jac` are plain Python "
            "callables (no compiled callback detected).",
            stacklevel=2,
        )

    # Validate Python callable shapes before the integration starts.
    if fun_addr is None:
        _validate_fun_shape(fun, n, tspan[0], y0)
    if jac is not None and jac_addr is None and _miter in (1, 4):
        _validate_jac_shape(jac, _miter, ml, mu, n, tspan[0], y0)

    # ------------------------------------------------------------------
    # 7.  Integrate
    # ------------------------------------------------------------------
    if len(tspan) == 2 and save_steps:
        t_out, y_out, istate = _zvode_drive_adaptive(
            _fun, _jac, ctx_int, y0, tspan[0], tspan[1],
            itol, rtol, atol, mf, iopt, zwork, rwork, iwork,
            refine=refine, allow_overshoot=allow_overshoot,
        )
    elif len(tspan) == 2:
        t_out, y_out, istate = _zvode_drive_knots(
            _fun, _jac, ctx_int, y0, tspan,
            itol, rtol, atol, mf, iopt, zwork, rwork, iwork,
        )
        t_out = float(t_out[-1])
        y_out = y_out[:, -1]
    else:
        t_out, y_out, istate = _zvode_drive_knots(
            _fun, _jac, ctx_int, y0, tspan,
            itol, rtol, atol, mf, iopt, zwork, rwork, iwork,
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
