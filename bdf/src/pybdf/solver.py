"""Friendly NumPy interface to the Fortran BDF integrator.

This is the top of the three-level stack:

    Fortran core (bdf_module)  ->  CPython bindings (_bdf)  ->  this module.

The public surface is intentionally small: a stateful :class:`BDF` integrator
and a one-call :func:`solve_bdf` driver in the spirit of
``scipy.integrate.solve_ivp`` (no dense output -- output is produced by
stepping exactly onto the requested times).
"""

from __future__ import annotations

import numpy as np

try:                      # installed as a package-local extension
    from . import _bdf
except ImportError:       # or as a top-level module (e.g. manual builds)
    import _bdf


__all__ = ["BDF", "BdfResult", "solve_bdf"]


def _resolve_jac(jac, band, n):
    """Translate the (jac, band) arguments into (jac_mode, callback, ml, mu).

    * ``jac=None``                -> finite differences
    * ``jac`` callable            -> user Jacobian, evaluated at (t, y)
    * ``jac`` array-like          -> constant user Jacobian
    * ``band=(ml, mu)``           -> banded Jacobian with those half-widths
    """
    ml = mu = -1
    if band is not None:
        ml, mu = int(band[0]), int(band[1])
        if ml < 0 or mu < 0:
            raise ValueError("band half-widths must be non-negative")

    if jac is None:
        return _bdf.JAC_FD, None, ml, mu
    if callable(jac):
        return _bdf.JAC_USER, jac, ml, mu

    # Constant Jacobian supplied as an array.
    J = np.ascontiguousarray(jac, dtype=float)
    return _bdf.JAC_CONSTANT, (lambda t, y, _J=J: _J), ml, mu


class BDF:
    """Stateful variable-order (1..5) BDF integrator.

    Parameters
    ----------
    fun : callable
        Right-hand side ``fun(t, y) -> dy/dt``.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Integration boundary; also fixes the integration direction.
    rtol, atol : float or array_like
        Relative and (scalar or per-component) absolute tolerances.
    jac : None, callable, or array_like, optional
        Jacobian source (see :func:`_resolve_jac`).  Defaults to finite
        differences.
    band : (ml, mu), optional
        Lower/upper half-bandwidths for a banded Jacobian.
    max_step : float, optional
        Largest allowed step (``np.inf`` for unbounded, the default).
    first_step : float, optional
        Initial step size (chosen automatically when ``None``).
    reuse_jac : bool, optional
        Cache and reuse the Jacobian between steps (default ``True``).
    """

    def __init__(self, fun, t0, y0, t_bound, *, rtol=1e-3, atol=1e-6,
                 jac=None, band=None, max_step=np.inf, first_step=None,
                 reuse_jac=True):
        y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        if y0.ndim != 1:
            raise ValueError("y0 must be one-dimensional")
        n = y0.size

        atol = np.broadcast_to(np.asarray(atol, dtype=float), (n,)).astype(float)
        jac_mode, jac_cb, ml, mu = _resolve_jac(jac, band, n)

        self.n = n
        self.fun = fun
        self._solver = _bdf.Solver()
        self._solver.setup(
            fun, float(t0), y0, float(t_bound), float(rtol), atol,
            jac_mode, jac_cb, ml, mu,
            -1.0 if not np.isfinite(max_step) else float(max_step),
            -1.0 if first_step is None else float(first_step),
            int(bool(reuse_jac)),
        )
        self.status = "running"

    @property
    def t(self):
        return self._solver.get_t()

    @property
    def y(self):
        return self._solver.get_y()

    @property
    def stats(self):
        return self._solver.get_stats()

    def set_t_bound(self, t_bound):
        """Move the integration boundary (direction unchanged)."""
        self._solver.set_t_bound(float(t_bound))

    def step(self):
        """Take one internal step.  Returns the status code from ``_bdf``."""
        code = self._solver.step()
        self._update_status(code)
        return code

    def integrate(self):
        """Integrate to the current ``t_bound``.  Returns the status code."""
        code = self._solver.integrate()
        self._update_status(code)
        return code

    def _update_status(self, code):
        if code < 0:
            self.status = "failed"
        elif code == _bdf.FINISHED:
            self.status = "finished"
        else:
            self.status = "running"


_MESSAGES = {
    _bdf.FINISHED: "The solver successfully reached the end of the interval.",
    _bdf.TOO_SMALL_STEP: "Required step size became too small.",
    _bdf.TOO_MANY_STEPS: "Maximum number of internal steps exceeded.",
}


class BdfResult:
    """Container for :func:`solve_bdf` output (a light Bunch)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = ", ".join(sorted(self.__dict__))
        return f"BdfResult({keys})"


def solve_bdf(fun, t_span, y0, *, t_eval=None, rtol=1e-3, atol=1e-6,
              jac=None, band=None, max_step=np.inf, first_step=None,
              reuse_jac=True):
    """Integrate an ODE system with the BDF method.

    Parameters
    ----------
    fun : callable
        Right-hand side ``fun(t, y) -> dy/dt``.
    t_span : (t0, tf)
        Integration interval.
    y0 : array_like, shape (n,)
        Initial state.
    t_eval : array_like, optional
        Times at which to store the solution.  When ``None`` (default), every
        internal step is stored.  Output times are reached exactly by stepping
        onto them (there is no dense output / interpolation).
    Other parameters are forwarded to :class:`BDF`.

    Returns
    -------
    BdfResult
        With fields ``t`` (shape ``(m,)``), ``y`` (shape ``(n, m)``),
        ``success``, ``status``, ``message`` and the evaluation counters
        ``nfev``, ``njev``, ``nlu``, ``nsteps``.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    solver = BDF(fun, t0, y0, tf, rtol=rtol, atol=atol, jac=jac, band=band,
                 max_step=max_step, first_step=first_step, reuse_jac=reuse_jac)

    ts = [t0]
    ys = [y0.copy()]
    code = _bdf.OK

    if t_eval is None:
        # Record every internal step until the boundary is reached.
        while True:
            code = solver.step()
            if code < 0:
                break
            ts.append(solver.t)
            ys.append(solver.y)
            if code == _bdf.FINISHED:
                break
    else:
        t_eval = np.asarray(t_eval, dtype=float)
        for tout in t_eval:
            solver.set_t_bound(float(tout))
            code = solver.integrate()
            if code < 0:
                break
            ts.append(solver.t)
            ys.append(solver.y)

    nfev, njev, nlu, nsteps = (solver.stats[k]
                               for k in ("nfev", "njev", "nlu", "nsteps"))
    return BdfResult(
        t=np.asarray(ts),
        y=np.asarray(ys).T,
        success=code >= 0,
        status=int(code),
        message=_MESSAGES.get(code, "Integration in progress."),
        nfev=nfev, njev=njev, nlu=nlu, nsteps=nsteps,
    )
