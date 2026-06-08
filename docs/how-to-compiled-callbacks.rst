.. _how-to-compiled-callbacks:

Compiled callbacks (ctypes and numba)
======================================

Python callbacks incur interpreter overhead on every right-hand side and
Jacobian evaluation.  For problems where the RHS is called millions of times,
passing compiled C function pointers can yield a substantial speedup.

:func:`~zvode.solve_complex_ivp` accepts compiled callbacks as ``fun`` and
``jac``.  It detects them by type (``ctypes._CFuncPtr`` instances) and calls
them directly through the C integration loop, bypassing the Python interpreter
on each evaluation.  Mixed mode is supported: one callback can be a Python
callable while the other is compiled.

Two approaches are covered here:

1. :ref:`numba-cfunc` — JIT-compile a Python function with Numba.
2. :ref:`dll-callback` — load a pre-compiled shared library (``*.so``).

The optional ``ctx`` pointer lets both approaches pass parameters without
global variables; see :ref:`ctx-parameter`.

.. note::

   The compiled-callback interface was inspired by the `numbalsoda
   <https://github.com/Nicholaswogan/numbalsoda>`_ project, which applies the
   same ctypes/numba technique to the LSODA solver.

----

C-level calling conventions
----------------------------

Both callbacks receive raw C pointers.  The expected signatures are:

**RHS** (``fun``):

.. code-block:: c

   void fun(int neq, double t,
            const double complex *y,
            double complex *dy,
            void *ctx);

**Jacobian** (``jac``):

.. code-block:: c

   void jac(int neq, double t,
            const double complex *y,
            int ml, int mu,
            double complex *pd,
            int nrowpd,
            void *ctx);

``pd`` is column-major (Fortran order).  For a banded Jacobian the
compact-storage convention applies: element ``df[i]/dy[j]`` goes to
``pd[mu + i - j + j*nrowpd]``.  See :doc:`banded_jacobian` for a full
explanation of the banded layout.

----

.. _numba-cfunc:

Numba ``@cfunc``
----------------

See the `Numba cfunc documentation <https://numba.readthedocs.io/en/stable/user/cfunc.html>`_
for a full introduction to compiled C callbacks in Numba.

Install numba (``pip install numba``) then use the signature objects exported
by this package.  :data:`~zvode.zvode_fun_sig` and
:data:`~zvode.zvode_jac_sig` are constructed on first access so that numba is
never imported at module load time.

RHS only
~~~~~~~~~

.. code-block:: python

   from numba import cfunc
   import zvode
   from zvode import solve_complex_ivp

   @cfunc(zvode.zvode_fun_sig)
   def rhs(neq, t, y, dy, ctx):
       dy[0] = -1j * y[0]

   sol = solve_complex_ivp(rhs.ctypes, tspan=(0.0, 10.0), y0=[1.0 + 0j])

Pass ``rhs.ctypes`` — the ctypes wrapper exposed by numba — not ``rhs``
itself.

Dense Jacobian
~~~~~~~~~~~~~~~

.. code-block:: python

   import numba as nb

   @cfunc(zvode.zvode_jac_sig)
   def jac(neq, t, y, ml, mu, pd, nrowpd, ctx):
       J = nb.farray(pd, (nrowpd, neq))   # 2-D Fortran-order view
       J[0, 0] = -1j                       # df[0]/dy[0]

   sol = solve_complex_ivp(rhs.ctypes, tspan=(0.0, 10.0), y0=[1.0 + 0j],
                           jac=jac.ctypes)

:func:`numba.farray` creates a 2-D view over the flat ``pd`` pointer with
Fortran (column-major) order, which is cleaner than manual index arithmetic.

Banded Jacobian
~~~~~~~~~~~~~~~~

The 1-D heat equation ``du/dt = α u_xx`` discretised on a uniform grid with
spacing ``h`` gives a tridiagonal Jacobian with weights ``α/h²``,
``-2α/h²``, ``α/h²`` (``lband = uband = 1``):

.. code-block:: python

   import numba as nb
   from numba import cfunc
   import numpy as np
   import zvode
   from zvode import solve_complex_ivp

   n = 50               # interior grid points
   h = 1.0 / (n + 1)   # spacing on [0, 1] with Dirichlet BCs
   alpha = 1.0 + 0.5j  # complex thermal diffusivity

   @cfunc(zvode.zvode_jac_sig)
   def heat_jac(neq, t, y, ml, mu, pd, nrowpd, ctx):
       J = nb.farray(pd, (nrowpd, neq))
       a = alpha / (h * h)
       for j in range(neq):
           J[mu, j] = -2.0 * a            # main diagonal
           if j > 0:
               J[mu - 1, j] = a           # superdiagonal
           if j < neq - 1:
               J[mu + 1, j] = a           # subdiagonal

   y0 = np.sin(np.pi * np.linspace(h, 1.0 - h, n)).astype(complex)
   sol = solve_complex_ivp(rhs.ctypes, tspan=(0.0, 0.1), y0=y0,
                           jac=heat_jac.ctypes, lband=1, uband=1)

----

.. _dll-callback:

Loading from a shared library
------------------------------

See the `ctypes callback functions documentation <https://docs.python.org/3/library/ctypes.html#callback-functions>`_
for background on ``CFUNCTYPE`` and wrapping C function pointers in Python.

When your RHS is already compiled as a C function, load it with
:mod:`ctypes` and cast it to :data:`~zvode.ZVODE_FUN_CTYPE`:

.. code-block:: c

   /* my_model.c — compile with:
      gcc -shared -fPIC -O2 -o my_model.so my_model.c  */
   #include <complex.h>

   void my_rhs(int neq, double t,
               const double complex *y,
               double complex *dy,
               void *ctx)
   {
       dy[0] = -1.0 * I * y[0];
   }

.. code-block:: python

   import ctypes
   from zvode import solve_complex_ivp, ZVODE_FUN_CTYPE

   lib = ctypes.CDLL("./my_model.so")

   # Wrap in ZVODE_FUN_CTYPE so solve_complex_ivp detects it as a compiled
   # callback and calls it directly without Python overhead.
   rhs = ctypes.cast(lib.my_rhs, ZVODE_FUN_CTYPE)

   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j])

The :func:`ctypes.cast` call wraps the raw DLL function pointer in the correct
``CFUNCTYPE``.  Without it, ``lib.my_rhs`` would be treated as a Python
callable and incur Python-layer overhead on each evaluation.

Similarly for the Jacobian:

.. code-block:: python

   from zvode import ZVODE_JAC_CTYPE

   jac = ctypes.cast(lib.my_jac, ZVODE_JAC_CTYPE)
   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j], jac=jac)

----

.. _ctx-parameter:

Passing parameters with ``ctx``
---------------------------------

The ``ctx`` argument to :func:`~zvode.solve_complex_ivp` is a
``ctypes.c_void_p`` forwarded as the last argument to **both** compiled
callbacks on every invocation.  It lets you parameterise the problem without
global variables.

**Pattern 1 — NumPy array of parameters:**

.. code-block:: python

   import ctypes
   import numpy as np
   from numba import cfunc, carray
   import numba as nb
   import zvode

   params = np.array([-1.0+2.0j, -2.0+1.0j], dtype=np.complex128)
   ctx = ctypes.cast(params.ctypes.data, ctypes.c_void_p)

   @cfunc(zvode.zvode_fun_sig)
   def rhs(neq, t, y, dy, ctx_ptr):
       p = carray(ctx_ptr, (2,), dtype=nb.complex128)
       dy[0] = p[0] * y[0]
       dy[1] = p[1] * y[1]

   sol = zvode.solve_complex_ivp(rhs.ctypes, tspan=(0.0, 5.0),
                                 y0=[1.0+0j, 0.0+1j],
                                 ctx=ctx)

Keep ``params`` alive for the entire duration of the integration — do not let
it be garbage-collected while :func:`~zvode.solve_complex_ivp` is running.

**Pattern 2 — ctypes Structure:**

.. code-block:: python

   class Params(ctypes.Structure):
       _fields_ = [("lambda1", ctypes.c_double * 2),   # complex128 as 2 doubles
                   ("lambda2", ctypes.c_double * 2)]

   p = Params()
   p.lambda1[0], p.lambda1[1] = -1.0, 2.0   # real, imag
   p.lambda2[0], p.lambda2[1] = -2.0, 1.0

   ctx = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)
   sol = zvode.solve_complex_ivp(rhs.ctypes, tspan, y0, ctx=ctx)

**Pattern 3 — numba compiled closure (no ``ctx`` needed):**

When parameters are known at compile time, numba can capture them from the
enclosing Python scope, avoiding the need for ``ctx`` entirely:

.. code-block:: python

   def make_rhs(lam1, lam2):
       @cfunc(zvode.zvode_fun_sig)
       def rhs(neq, t, y, dy, ctx):
           dy[0] = lam1 * y[0]
           dy[1] = lam2 * y[1]
       return rhs

   my_rhs = make_rhs(-1.0+2.0j, -2.0+1.0j)
   sol = zvode.solve_complex_ivp(my_rhs.ctypes, tspan, y0)

Each call to ``make_rhs`` triggers a fresh numba JIT compilation; cache the
result if the same parameters are reused across multiple integrations.

.. note::

   ``ctx`` is silently ignored (with a ``UserWarning``) when all callbacks are
   plain Python callables — it is only forwarded to compiled callbacks.
