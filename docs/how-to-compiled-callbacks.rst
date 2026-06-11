.. _how-to-compiled-callbacks:

Compiled callbacks
==================

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
2. :ref:`dll-callback` — load a pre-compiled shared library (``*.so``).
3. :ref:`gfort2py-callback` — write the callback in Fortran and compile it on the fly with gfort2py.

The optional ``ctx`` pointer lets all three approaches pass parameters
without global variables; see :ref:`ctx-parameter`.

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

.. _gfort2py-callback:

Fortran callback with gfort2py
-------------------------------

`gfort2py <https://github.com/rjfarmer/gfort2py>`_ lets you write the
RHS (or Jacobian) directly in Fortran and compile it from a Python
string using gfortran.  Because the subroutine is declared ``bind(c)``
and uses C-interoperable types, its calling convention exactly matches
the C interface expected by :func:`~zvode.solve_complex_ivp`.  (The
``iso_c_binding`` module merely supplies the interoperable kind
parameters such as ``c_double``; it is not a type itself.)

Install with ``pip install gfort2py``; consult the gfort2py
documentation for the supported platforms and gfortran versions.

The example below is a three-species reaction with a fast forward
rate, a non-linear back-reaction, and a slow decay.  Write the RHS as
a ``bind(c)`` subroutine, embedding the rate constants as ``parameter``
literals:

.. code-block:: python

   import numpy as np
   import gfort2py as gf
   from zvode import solve_complex_ivp

   t0, tf = 0.0, 3.0
   y0 = np.array([1.0 + 0.0j, 0.0 + 0.2j, 0.0 + 0.0j], dtype=np.complex128)

   src = """
   subroutine reaction_rhs(neq, t, y, dy, par) bind(c)
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value    :: neq
       real(c_double), value    :: t
       complex(c_double_complex), intent(in)  :: y(neq)
       complex(c_double_complex), intent(out) :: dy(neq)
       type(c_ptr), value :: par   ! unused here; t is also unused (autonomous system)

       complex(c_double_complex), parameter :: a = (1000.0d0, 200.0d0)
       complex(c_double_complex), parameter :: b = (1000.0d0,   0.0d0)
       complex(c_double_complex), parameter :: c = (   1.0d0,  50.0d0)

       dy(1) = -a*y(1) + b*y(2)*y(3)
       dy(2) =  a*y(1) - b*y(2)*y(3) - c*y(2)
       dy(3) =  c*y(2)
   end subroutine reaction_rhs
   """

   kinetics = gf.compile(string=src)

   sol = solve_complex_ivp(kinetics.reaction_rhs.ctype, (t0, tf), y0,
                           method="BDF", rtol=1e-8, atol=1e-8)

``gf.compile`` returns an object whose attributes are the compiled
Fortran procedures.  ``kinetics.reaction_rhs.ctype`` is the raw
``ctypes`` function pointer, which :func:`~zvode.solve_complex_ivp`
recognises as a compiled callback and invokes directly without Python
overhead.

.. warning::

   ``gf.compile`` launches gfortran in a subprocess and is therefore
   expensive.  Compile once and reuse the returned callback — never put
   it inside an integration loop or any other hot path.

Varying parameters at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Baking the constants in as ``parameter`` literals means changing a rate
requires recompiling.  To vary them at runtime instead, pass the
parameters through the ``ctx`` pointer and recover them in Fortran with
``c_f_pointer``:

.. code-block:: python

   import ctypes

   params = np.array([1000.0 + 200.0j, 1000.0 + 0.0j, 1.0 + 50.0j],
                     dtype=np.complex128)

   src = """
   subroutine reaction_rhs(neq, t, y, dy, par) bind(c)
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value    :: neq
       real(c_double), value    :: t
       complex(c_double_complex), intent(in)  :: y(neq)
       complex(c_double_complex), intent(out) :: dy(neq)
       type(c_ptr), value :: par

       complex(c_double_complex), pointer :: p(:) => null()

       call c_f_pointer(par, p, [3])
       associate (a => p(1), b => p(2), c => p(3))
           dy(1) = -a*y(1) + b*y(2)*y(3)
           dy(2) =  a*y(1) - b*y(2)*y(3) - c*y(2)
           dy(3) =  c*y(2)
       end associate
   end subroutine reaction_rhs
   """

   kinetics = gf.compile(string=src)

   sol = solve_complex_ivp(kinetics.reaction_rhs.ctype, (t0, tf), y0,
                           method="BDF", rtol=1e-8, atol=1e-8,
                           ctx=ctypes.c_void_p(params.ctypes.data))

Keep ``params`` alive for the whole integration so it is not
garbage-collected while the solver is running.  See :ref:`ctx-parameter`
for the general ``ctx`` conventions shared with the numba and
shared-library approaches.

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
