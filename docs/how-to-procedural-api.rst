.. _how-to-procedural-api:

Procedural API — ``solve_complex_ivp``
======================================

:func:`~zvode.solve_complex_ivp` is the main entry point to ZVODE.  Pass the
right-hand side, a time span, and an initial condition; get back a
:class:`~zvode.ZVODEResult` object with the solution and solver statistics as
attributes (also accessible as dict keys).

.. code-block:: python

   from zvode import solve_complex_ivp

   sol = solve_complex_ivp(fun, tspan, y0, **options)
   # sol.t    — 1-D float array of output times, shape (m,)
   # sol.y    — complex array, shape (n, m)
   # sol.nfev — RHS evaluations (int)
   # sol.njev — Jacobian evaluations (int)
   # sol.nlu  — LU decompositions (int)

The complete parameter reference is in the :func:`~zvode.solve_complex_ivp`
docstring.

----

Output modes
------------

**Collect every accepted step** (default) — pass a 2-element ``tspan``:

.. code-block:: python

   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j])
   # sol.t[0] == 0.0, sol.t[-1] == 10.0; number of points depends on tolerances

**Output at specific times** — pass three or more values as ``tspan``:

.. code-block:: python

   import numpy as np

   t_out = np.linspace(0.0, 10.0, 101)
   sol = solve_complex_ivp(rhs, tspan=t_out, y0=[1.0 + 0j])
   # sol.t is t_out, sol.y.shape == (1, 101)

**Endpoint only** — ``save_steps=False`` returns a scalar ``sol.t`` and a
1-D ``sol.y``:

.. code-block:: python

   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j],
                           save_steps=False)
   # float(sol.t), sol.y.shape == (1,)

Smoothing output with ``refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``refine`` parameter inserts additional interpolated output points inside
each accepted step using ZVODE's built-in dense output (Nordsieck
interpolation).  ``refine=N`` produces ``N − 1`` intermediate points per step,
yielding smoother plots without extra function evaluations:

.. code-block:: python

   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j], refine=4)

``refine`` is only effective when ``save_steps=True``; it is ignored in knot
and endpoint-only modes.

When ``allow_overshoot=True``, the solver may step slightly past the endpoint
``tspan[-1]`` to avoid constraining its step size; the last output time may
then lie just beyond ``tspan[-1]``.

----

Choosing a method
-----------------

``method='BDF'`` (default) is for stiff problems; ``method='Adams'`` for
non-stiff.

.. note::

   **Analyticity** — BDF uses complex Newton iteration, which requires every
   component of ``f(t, y)`` to be an analytic function of the complex state.
   If your RHS uses ``abs``, ``conj``, or real/imaginary-part splitting, use a
   real-valued solver on the equivalent doubled real system instead.

----

Providing a Jacobian
--------------------

Supplying the Jacobian avoids finite differences and reduces RHS evaluations
for stiff problems.

**Dense** — return an ``(n, n)`` array from ``jac(t, y)``:

.. code-block:: python

   import numpy as np

   def rhs(t, y):
       return np.array([-100j * y[0] + y[1], -1j * y[1]])

   def jac(t, y):
       return np.array([[-100j, 1.0], [0.0, -1j]])

   sol = solve_complex_ivp(rhs, tspan=(0.0, 5.0),
                           y0=[1.0 + 0j, 0.0 + 1j],
                           method='BDF', jac=jac)

**Banded** — pass ``lband`` and ``uband``; return a
``(lband + uband + 1, n)`` array where entry ``[uband + i − j, j]`` holds
``df[i]/dy[j]``.  See :doc:`banded_jacobian` for the compact-storage layout
and a worked example:

.. code-block:: python

   sol = solve_complex_ivp(rhs, tspan=(0.0, 5.0), y0=y0,
                           method='BDF', jac=jac_banded,
                           lband=1, uband=1)

----

Backward integration
--------------------

Set ``tspan`` in decreasing order; for knot mode pass times strictly
decreasing:

.. code-block:: python

   sol = solve_complex_ivp(rhs, tspan=(10.0, 0.0), y0=y_at_t10)

----

Compiled callbacks
------------------

For performance-critical work, ``fun`` and ``jac`` can be compiled C function
pointers (ctypes or numba ``@cfunc``), called directly without re-entering the
Python interpreter on each evaluation.  Pass the optional ``ctx`` argument to
share a user-data pointer between both compiled callbacks.  See
:doc:`how-to-compiled-callbacks` for setup and full examples.
