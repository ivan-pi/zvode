.. _tutorial-complex-ode:

Tutorial: solving and visualizing a complex ODE
===============================================

This tutorial walks through solving a single complex-valued ODE with
:func:`~zvode.solve_complex_ivp` and visualizing the result, reproducing an
example from the Wolfram Language documentation.

.. admonition:: Attribution

   The example problem and the idea of visualizing its solution come from
   `Solutions of Complex ODEs
   <https://www.wolfram.com/language/12/complex-visualization/solutions-of-complex-odes.html>`_
   in the Wolfram Language 12 documentation, where it is solved with
   ``NDSolveValue``:

   .. code-block:: text

      z = NDSolveValue[{y'[t] == 10 E^(2 Pi I t) y[t], y[0] == 1},
                       y, {t, 0, 5}];

The problem
-----------

We want to integrate the scalar, time-dependent complex ODE

.. math::

   y'(t) = 10\, e^{2\pi i t}\, y(t), \qquad y(0) = 1, \qquad t \in [0, 5].

The right-hand side is a (time-dependent) complex coefficient multiplying
``y``, so it is holomorphic in ``y``.  That is exactly the setting ZVODE is
built for: it advances the complex state directly, with no need to split the
problem into real and imaginary parts.

A reference solution
~~~~~~~~~~~~~~~~~~~~~~

This problem is separable and has a closed-form solution, which is handy for
checking accuracy.  From ``y'/y = 10\, e^{2\pi i t}`` we integrate to obtain

.. math::

   y(t) = \exp\!\left( \frac{10}{2\pi i}\,\bigl(e^{2\pi i t} - 1\bigr) \right).

Since :math:`e^{2\pi i t} = 1` whenever ``t`` is an integer, the exponent
vanishes there and the solution returns to ``y = 1`` at every integer time; in
particular ``y(5) = 1``.

Step 1 — define the right-hand side
-----------------------------------

The right-hand side takes the time ``t`` and the state vector ``y`` (here of
length 1) and returns ``y'``.  We write it to return a one-element list:

.. code-block:: python

   import numpy as np

   def rhs(t, y):
       # y'(t) = 10 e^{2πi t} y(t)
       return [10.0 * np.exp(2j * np.pi * t) * y[0]]

Because the coefficient does not depend on ``y``, the Jacobian
:math:`\partial f/\partial y = 10\, e^{2\pi i t}` is trivial, and for this
non-stiff problem we do not need to supply it.

Step 2 — integrate
------------------

:func:`~zvode.solve_complex_ivp` is the recommended entry point.  The problem
is non-stiff (a smooth, oscillatory coefficient), so we select the Adams
method.  Passing ``tspan`` as a full array of times requests the solution on
that grid, which makes the parametric plot below smooth:

.. code-block:: python

   from zvode import solve_complex_ivp

   t_eval = np.linspace(0.0, 5.0, 501)

   sol = solve_complex_ivp(
       fun=rhs,
       tspan=t_eval,
       y0=[1.0 + 0.0j],
       method="Adams",
       rtol=1e-10,
       atol=1e-12,
   )

   y = sol.y[0]   # complex array, shape (501,)

We can confirm the result against the closed form:

.. code-block:: python

   def exact(t):
       t = np.asarray(t, dtype=float)
       return np.exp((10.0 / (2j * np.pi)) * (np.exp(2j * np.pi * t) - 1.0))

   err = np.max(np.abs(y - exact(t_eval)))
   print(f"Max absolute error: {err:.2e}")   # ~9e-09

Step 3 — visualize
------------------

The Wolfram example draws three views of the solution.  Each is a few lines of
Matplotlib.

**Real and imaginary parts versus the real variable** ``t``:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.plot(t_eval, y.real, label=r"$\mathrm{Re}\,y$")
   plt.plot(t_eval, y.imag, "--", label=r"$\mathrm{Im}\,y$")
   plt.xlabel("$t$")
   plt.legend()

**Modulus** ``|y|`` versus ``t``:

.. code-block:: python

   plt.plot(t_eval, np.abs(y))
   plt.xlabel("$t$")
   plt.ylabel("$|y|$")

**The trajectory drawn parametrically in the complex plane.**  Plotting
``y.imag`` against ``y.real`` traces the path the solution follows; colouring
the curve by ``t`` shows the direction and speed of travel.  Because the
solution is periodic with period 1, the curve is retraced five times over
``t ∈ [0, 5]``:

.. code-block:: python

   from matplotlib.collections import LineCollection

   points = np.column_stack([y.real, y.imag]).reshape(-1, 1, 2)
   segments = np.concatenate([points[:-1], points[1:]], axis=1)
   lc = LineCollection(segments, cmap="viridis", linewidth=2)
   lc.set_array(t_eval)

   ax = plt.gca()
   ax.add_collection(lc)
   ax.set_aspect("equal")
   ax.autoscale()
   plt.colorbar(lc, ax=ax, label="$t$")

Putting all three panels together gives:

.. image:: complex_ode.png
   :width: 100%
   :alt: Real/imaginary parts, modulus, and complex-plane trajectory of the solution.

The complete, runnable script is ``demo_complex_ode.py``.

Next steps
----------

- Output modes, method selection, and supplying a Jacobian are covered in
  :doc:`how-to-procedural-api`.
- For larger or stiff systems, see :doc:`banded_jacobian` and
  :doc:`how-to-compiled-callbacks`.
