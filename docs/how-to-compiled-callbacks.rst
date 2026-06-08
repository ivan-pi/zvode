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

Three approaches are covered here:

1. :ref:`numba-cfunc` — JIT-compile a Python function with Numba.
2. :ref:`ctypes-cfunc` — wrap a Python function via ctypes (useful for
   prototyping the interface before writing C).
3. :ref:`dll-callback` — load a pre-compiled shared library
   (``*.so`` / ``*.dll``).

The optional ``ctx`` pointer lets all three pass parameters without global
variables; see :ref:`ctx-parameter`.

.. note::

   Compiled callbacks require the C integration loop (the default).  Setting
   the environment variable ``ZVODE_BACKEND=python`` disables the C loop and
   causes a ``RuntimeError`` if a compiled callback is passed.

----

C-level calling conventions
----------------------------

Both callbacks receive raw C pointers.  The expected signatures are:

**RHS** (``fun``):

.. code-block:: c

   void fun(int neq, double t,
            const double complex *y,
            double complex       *dy,
            void                 *ctx);

**Jacobian** (``jac``):

.. code-block:: c

   void jac(int neq, double t,
            const double complex *y,
            int ml, int mu,
            double complex       *pd,
            int nrowpd,
            void                 *ctx);

``pd`` is column-major (Fortran order).  For a banded Jacobian the
compact-storage convention applies: element ``df[i]/dy[j]`` goes to
``pd[mu + i - j + j*nrowpd]``.  See :doc:`banded_jacobian` for a full
explanation of the banded layout.

----

.. _numba-cfunc:

Numba ``@cfunc``
----------------

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
Fortran (column-major) order, which is often cleaner than manual index
arithmetic.

Banded Jacobian
~~~~~~~~~~~~~~~~

For a banded system with lower half-bandwidth ``ml`` and upper half-bandwidth
``mu``, element ``df[i]/dy[j]`` goes to row ``mu + i - j``:

.. code-block:: python

   @cfunc(zvode.zvode_jac_sig)
   def jac_banded(neq, t, y, ml, mu, pd, nrowpd, ctx):
       J = nb.farray(pd, (nrowpd, neq))
       for j in range(neq):
           J[mu, j] = alpha                   # main diagonal
           if j > 0:
               J[mu - 1, j] = delta           # superdiagonal
           if j < neq - 1:
               J[mu + 1, j] = beta            # first subdiagonal

   sol = solve_complex_ivp(rhs.ctypes, tspan=(0.0, 5.0), y0=y0,
                           jac=jac_banded.ctypes, lband=1, uband=1)

----

.. _ctypes-cfunc:

ctypes callback
---------------

Use :data:`~zvode.ZVODE_FUN_CTYPE` and :data:`~zvode.ZVODE_JAC_CTYPE` as
decorators.  The decorated Python function is invoked via ctypes on each
evaluation; there is no GIL crossing, but ctypes itself adds some overhead.
This is most useful for prototyping or wrapping a small helper:

.. code-block:: python

   import ctypes
   from zvode import solve_complex_ivp, ZVODE_FUN_CTYPE

   @ZVODE_FUN_CTYPE
   def rhs(neq, t, y_ptr, dy_ptr, ctx):
       # complex128 is two consecutive doubles (real, imag)
       y  = (ctypes.c_double * (2 * neq)).from_address(y_ptr)
       dy = (ctypes.c_double * (2 * neq)).from_address(dy_ptr)
       # dy[0]/dt = i * y[0]  =>  real part = -im(y[0]), imag part = re(y[0])
       dy[0] = -y[1]
       dy[1] =  y[0]

   sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j])

.. note::

   Inside a ``ZVODE_FUN_CTYPE`` callback, ``y_ptr`` and ``dy_ptr`` are raw
   memory addresses.  NumPy views are also possible:

   .. code-block:: python

      import numpy as np
      y  = np.frombuffer((ctypes.c_double * (2 * neq)).from_address(y_ptr),
                         dtype=np.float64).view(np.complex128)
      dy = np.frombuffer((ctypes.c_double * (2 * neq)).from_address(dy_ptr),
                         dtype=np.float64).view(np.complex128)

----

.. _dll-callback:

Loading from a shared library
------------------------------

When your RHS is already compiled as a C function, load it with
:mod:`ctypes` and cast it to :data:`~zvode.ZVODE_FUN_CTYPE`:

.. code-block:: c

   /* my_model.c — compile with:
      gcc -shared -fPIC -O2 -o my_model.so my_model.c  */
   #include <complex.h>

   void my_rhs(int neq, double t,
               const double complex *y,
               double complex       *dy,
               void                 *ctx)
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

On Windows, replace ``ctypes.CDLL`` with ``ctypes.WinDLL`` for DLLs that use
the ``__stdcall`` calling convention, or keep ``CDLL`` for ``__cdecl`` (the
default for C code).

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
   import zvode

   params = np.array([-1.0+2.0j, -2.0+1.0j], dtype=np.complex128)
   ctx = ctypes.cast(params.ctypes.data, ctypes.c_void_p)

   @cfunc(zvode.zvode_fun_sig)
   def rhs(neq, t, y, dy, ctx_ptr):
       import numba as nb
       p = nb.carray(ctx_ptr, (2,), dtype=nb.complex128)
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
