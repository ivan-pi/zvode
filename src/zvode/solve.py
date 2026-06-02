"""Procedural ZVODE bindings for complex-valued ODE systems."""

import ctypes
import warnings
from threading import Lock

import numpy as np

from . import _zvode
from .zvode_impl import (
    MESSAGES,
    _check_tolerances,
    _determine_miter,
    _wrapped_fun,
    _wrapped_jac,
)

# ZVODE stores solver state in Fortran COMMON blocks that are global to the
# process.  Only one integration can be active at a time across all threads.
ZVODE_LOCK = Lock()


class ZVODEStats(dict):
    """Integration statistics returned when ``ret_stats=True``.

    Subclasses :class:`dict`; fields are also accessible as attributes.

    Attributes
    ----------
    nsteps : int   Total number of steps taken.
    nfev   : int   Number of right-hand side evaluations.
    njev   : int   Number of Jacobian evaluations.
    nlu    : int   Number of LU decompositions.
    """

    def __init__(self, stats):
        super().__init__(zip(('nsteps', 'nfev', 'njev', 'nlu'), stats))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self):
        return (f"ZVODEStats(nsteps={self['nsteps']}, nfev={self['nfev']}, "
                f"njev={self['njev']}, nlu={self['nlu']})")


# ---------------------------------------------------------------------------
# C function-pointer detection
# ---------------------------------------------------------------------------

def _cfunc_address(fun):
    """Return the C function pointer address (int) for compiled callbacks.

    Recognises:
    * numba ``@cfunc`` objects — via the ``.address`` attribute (same approach
      as the numbalsoda package)
    * ctypes ``CFUNCTYPE`` instances — via ``ctypes.cast``

    Returns ``None`` for ordinary Python callables.
    """
    if hasattr(fun, 'address'):                 # numba @cfunc
        return int(fun.address)
    if isinstance(fun, ctypes._CFuncPtr):       # ctypes CFUNCTYPE
        return ctypes.cast(fun, ctypes.c_void_p).value
    return None


# ---------------------------------------------------------------------------
# Workspace allocation
# ---------------------------------------------------------------------------

def _make_workspace(n, miter, ml, mu, mf, maxord_allowed,
                    first_step, min_step, max_step, max_order, max_num_steps,
                    t_bound):
    """Allocate and initialise ZVODE's three workspace arrays.

    Returns ``(zwork, rwork, iwork)`` as numpy arrays.
    Note: zwork, rwork, and iwork are mutable; the integration drivers update
    them in place on every step and read diagnostic counters from them on return.
    """
    _INT32_MAX = 2**31 - 1

    if miter in (1, 2) and n**2 > _INT32_MAX:
        raise ValueError(
            f"neq={n} exceeds the maximum of 46340 for dense Jacobian methods: "
            "neq**2 overflows the 32-bit integer arithmetic used internally.")
    if miter in (4, 5):
        _lenwm_max = (3 * ml + mu + 1) * n
        if _lenwm_max > _INT32_MAX:
            raise ValueError(
                f"Banded workspace ({_lenwm_max:,}) overflows int32 arithmetic.")

    if miter == 0:
        lwm = 0
    elif miter in (1, 2):
        lwm = 2 * n**2 if mf > 0 else n**2
    elif miter == 3:
        lwm = n
    elif miter in (4, 5):
        lwm = (3*ml + 2*mu + 2)*n if mf > 0 else (2*ml + mu + 1)*n
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

    rwork[0] = float(t_bound)          # TCRIT; required when ITASK=4 or 5
    if first_step is not None:
        rwork[4] = float(first_step)
    if max_step > 0:
        rwork[5] = float(max_step)
    if min_step:
        rwork[6] = float(min_step)
    if max_order is not None:
        iwork[4] = int(max_order)
    iwork[5] = int(max_num_steps)           # MXSTEP: max internal steps per output point

    return zwork, rwork, iwork


# ---------------------------------------------------------------------------
# Integration drivers
# ---------------------------------------------------------------------------

def _zvode_adaptive(fun, jac, y0, t0, t_bound,
                    itol, rtol, atol, mf, iopt,
                    zwork, rwork, iwork,
                    refine=1, allow_overshoot=False):
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
    ys = [ytmp.copy()]

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
                fun, ytmp,
                t, t_bound,
                itol, rtol, atol,
                ITASK, istate, iopt,
                zwork, rwork, iwork,
                jac, mf)

            if istate < 0:
                break

            if refine > 1:
                # After an accepted step the Nordsieck array in zwork[0:n*(nq+1)]
                # is valid for interpolation over [t_old, t].  ZVINDY is called
                # before the next zvode call overwrites zwork.
                nq  = int(iwork[14])         # NQCUR: current order
                hu  = float(rwork[10])       # HU: step size just used
                tn  = t                       # TCUR: end of the current step
                yh  = zwork[:n * (nq + 1)].reshape((n, nq + 1), order='F')
                dky = np.empty(n, dtype=np.complex128)
                for i in range(1, refine):
                    t_i = t_old + i * (t - t_old) / refine
                    _zvode.zvindy(t_i, 0, yh, hu, tn, hu, dky)
                    ts.append(t_i)
                    ys.append(dky.copy())

            ts.append(t)
            ys.append(ytmp.copy())

    ts = np.asarray(ts)
    ys = np.asfortranarray(np.vstack(ys).T)  # (n, m)

    return ts, ys, istate


def _zvode_knots(fun, jac, y0, tspan,
                 itol, rtol, atol, mf, iopt,
                 zwork, rwork, iwork):
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
    ITASK = 1   # normal: step to tout, taking as many steps as needed
    istate = 1  # initial call

    n = len(y0)
    ytmp = y0.copy()

    ys = np.empty((n, len(tspan)), dtype=np.complex128, order='F')
    ys[:, 0] = ytmp
    t = float(tspan[0])

    # TODO: same as _zvode_adaptive — replace with _zvode.drive() to move the
    # knot loop into C and fully eliminate Python overhead in the inner loop.
    with ZVODE_LOCK:
        for i in range(1, len(tspan)):
            t, istate = _zvode.zvode(
                fun, ytmp,
                t, float(tspan[i]),
                itol, rtol, atol,
                ITASK, istate, iopt,
                zwork, rwork, iwork,
                jac, mf)

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

def solve_complex_ivp(fun, tspan, y0, *,
                      rtol=1.0e-3,
                      atol=1.0e-6,
                      jac=None,
                      method="BDF",
                      lband=None,
                      uband=None,
                      in_place=False,
                      ret_stats=False,
                      save_steps=True,
                      refine=1,
                      allow_overshoot=False,
                      first_step=None,
                      min_step=0.0,
                      max_step=np.inf,
                      max_num_steps=1_000_000,
                      max_order=None,
                      miter=None,
                      save_jac=True):
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
    fun : callable
        Right-hand side of the system.

        * ``in_place=False`` (default): ``fun(t, y) -> array_like``, SciPy
          compatible.  A return-value copy is performed on every call.
        * ``in_place=True``: ``fun(t, y, dy)`` — must fill ``dy`` in place.
          Accepted forms:

          - plain Python callable
          - numba ``@cfunc`` object — pass the decorated function directly;
            its ``.address`` attribute is used to route calls through a native
            C function pointer, bypassing the Python interpreter on every RHS
            evaluation.
          - ctypes ``CFUNCTYPE`` instance — same principle.

          The expected C-level signature (using numba types) is::

              @cfunc(types.void(
                  types.int32,                           # neq
                  types.float64,                         # t
                  types.CPointer(types.complex128),      # y[neq]  (read-only)
                  types.CPointer(types.complex128),      # dy[neq] (write)
                  types.voidptr,                         # ctx (pass 0 for now)
              ))
              def my_rhs(neq, t, y, dy, ctx): ...

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
        Relative and absolute tolerances.  Scalar or per-component arrays.
    jac : callable or None, optional
        Jacobian of ``fun`` w.r.t. ``y``.  Follows the same ``in_place``
        convention as ``fun``:

        * ``in_place=False``: ``jac(t, y) -> (n, n)`` array.
        * ``in_place=True``, full: ``jac(t, y, pd)`` — fill ``pd`` in place.
        * ``in_place=True``, banded: ``jac(t, y, pd, ml, mu)`` — fill the
          user band of ``pd`` in place.

    method : {'BDF', 'Adams'}, optional
        Linear multistep method.  ``'BDF'`` (default) for stiff problems
        (max order 5); ``'Adams'`` for non-stiff (max order 12).
    lband, uband : int or None, optional
        Lower / upper half-bandwidths of a banded Jacobian.
    in_place : bool, optional
        Selects the callback convention for ``fun`` and ``jac``.
        Default ``False`` (SciPy-compatible return-value form).
        Compiled callbacks (numba ``@cfunc``, ctypes ``CFUNCTYPE``) always
        use the in-place convention; ``in_place=True`` is required for them.
    ret_stats : bool, optional
        If ``True``, append a :class:`ZVODEStats` object to the return tuple.

    Returns
    -------
    t : float or ndarray, shape (m,)
        Output time(s).  A scalar float in endpoint-only mode
        (``len(tspan) == 2`` and ``save_steps=False``); a 1-D array otherwise.
    y : ndarray, shape (n,) or (n, m), complex128
        Solution state(s).  A 1-D array in endpoint-only mode; a 2-D
        Fortran-order array with ``y[:, k]`` the state at ``t[k]``
        otherwise.
    stats : ZVODEStats, only when ``ret_stats=True``
        Integration statistics (nsteps, nfev, njev, nlu).

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
    first_step, min_step, max_step : float, optional
        Step-size controls.
    max_num_steps : int, optional
        Maximum number of internal steps between two consecutive output
        points.  Default 1 000 000.  Lower this to cap computational work
        when function evaluations are expensive; an error is raised if the
        budget is exhausted before the next output point.
    max_order : int or None, optional
        Maximum integration order (capped at the method limit if exceeded).
    miter : {0, 1, 2, 3, 4, 5} or None, optional
        Iteration method; inferred from ``jac`` / band arguments when ``None``.
    save_jac : bool, optional
        If ``True`` (default), the Jacobian is evaluated once and reused
        across multiple steps, trading extra memory for fewer Jacobian
        evaluations.  Set to ``False`` to recompute the Jacobian on every
        step.

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

    **C-level callbacks** — the native function-pointer path
    (``in_place=True`` with a numba ``@cfunc`` or ctypes function) requires
    a C-level integration loop (``_zvode.drive``) that is not yet
    implemented.  Once available, the full integration will run in compiled
    code with no Python involvement in the inner loop.
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
        raise ValueError("`tspan` must be strictly monotonic (all increasing or all decreasing).")

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

    _miter, ml, mu = _determine_miter(jac, lband, uband, miter)

    jsv = 1 if save_jac else -1
    mf = jsv * (10 * meth + _miter)

    # ------------------------------------------------------------------
    # 4.  Workspace
    # ------------------------------------------------------------------
    iopt = 1  # optional inputs present (rwork / iwork slots populated below)
    # Cap max_step at the largest interval in tspan so the solver cannot
    # overshoot a knot in a single step.  Honour a tighter user-supplied limit.
    _max_interval = float(np.max(np.abs(diffs)))
    _effective_max_step = min(_max_interval, max_step) if max_step < np.inf else _max_interval
    zwork, rwork, iwork = _make_workspace(
        n, _miter, ml, mu, mf, maxord_allowed,
        first_step, min_step, _effective_max_step, max_order, max_num_steps,
        t_bound=float(tspan[-1]))

    # ------------------------------------------------------------------
    # 5.  Normalise callbacks
    # ------------------------------------------------------------------
    #
    # Path A (in_place=False)
    #   Wrap SciPy-style fun(t,y)->array to the in-place form that
    #   _zvode.zvode expects.
    #
    # Path B (in_place=True, plain Python callable)
    #   Pass through unchanged; fun must already accept (t, y, dy).
    #
    # Path C (in_place=True, compiled cfunc — numba or ctypes)
    #   Extract the raw C function pointer address (int).  The future
    #   _zvode.drive() C entry point will accept this integer and run the
    #   complete integration loop in C/Fortran without ever re-entering the
    #   Python interpreter.  Until _zvode.drive() is implemented this path
    #   raises NotImplementedError.

    fun_addr = _cfunc_address(fun)
    jac_addr = _cfunc_address(jac) if jac is not None else None

    if fun_addr is not None and not in_place:
        raise ValueError(
            "Compiled callbacks (numba @cfunc / ctypes) use the in-place calling "
            "convention and are incompatible with `in_place=False`.  "
            "Pass `in_place=True`, or use a plain Python callable with `in_place=False`."
        )

    if fun_addr is not None:
        # Path C — compiled callback
        # TODO: call _zvode.drive(fun_addr, jac_addr, y0, tspan, ...) once
        #       the C entry point is implemented.
        raise NotImplementedError(
            "C function-pointer callbacks (numba @cfunc / ctypes) require "
            "_zvode.drive(), which is not yet implemented.  "
            "Use a plain Python callable with in_place=True for now."
        )
    elif in_place:
        # Path B — Python in-place callable; use as-is
        _fun = fun
        _jac = jac
    else:
        # Path A — SciPy-compatible; wrap to in-place
        _fun = _wrapped_fun(fun)
        _jac = _wrapped_jac(jac, banded=(_miter == 4)) if jac is not None else None

    # ------------------------------------------------------------------
    # 6.  Validate refine
    # ------------------------------------------------------------------
    if refine < 1:
        raise ValueError("`refine` must be a positive integer.")

    # ------------------------------------------------------------------
    # 7.  Integrate
    # ------------------------------------------------------------------
    if len(tspan) == 2 and save_steps:
        # Collect every accepted step (optionally with ZVINDY interpolation).
        t_out, y_out, istate = _zvode_adaptive(
            _fun, _jac, y0, tspan[0], tspan[1],
            itol, rtol, atol, mf, iopt,
            zwork, rwork, iwork,
            refine=refine,
            allow_overshoot=allow_overshoot)
    elif len(tspan) == 2:
        # Endpoint-only: ZVODE steps freely to t_bound; returns scalar t
        # and 1-D y — no intermediate storage.
        t_out, y_out, istate = _zvode_knots(
            _fun, _jac, y0, tspan,
            itol, rtol, atol, mf, iopt,
            zwork, rwork, iwork)
        t_out = float(t_out[-1])
        y_out = y_out[:, -1]
    else:
        # Knots: output at each element of tspan.
        t_out, y_out, istate = _zvode_knots(
            _fun, _jac, y0, tspan,
            itol, rtol, atol, mf, iopt,
            zwork, rwork, iwork)

    # ------------------------------------------------------------------
    # 8.  Error reporting
    # ------------------------------------------------------------------
    if istate < 0:
        _msg = MESSAGES.get(istate, 'Unknown error.')
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

    if ret_stats:
        return t_out, y_out, ZVODEStats(
            (int(iwork[10]), int(iwork[11]), int(iwork[12]), int(iwork[19]))
        )

    return t_out, y_out
