API reference
=============

Procedural API
--------------

.. autofunction:: zvode.solve_complex_ivp

Result type
-----------

.. autoclass:: zvode.solve.ZVODEResult
   :members:

Compiled callback types
-----------------------

These ctypes descriptors define the C-level calling convention for compiled
callbacks.  Use them as decorators (ctypes) or for casting DLL function
pointers.  See :doc:`how-to-compiled-callbacks` for usage examples.

.. autodata:: zvode.ZVODE_FUN_CTYPE

.. autodata:: zvode.ZVODE_JAC_CTYPE

.. data:: zvode.zvode_fun_sig

   Numba ``@cfunc`` signature for the RHS callback.  Constructed on first
   access so that numba is never imported at package load time.

   Equivalent C signature::

      void fun(int neq, double t,
               const double complex *y,
               double complex *dy,
               void *ctx);

.. data:: zvode.zvode_jac_sig

   Numba ``@cfunc`` signature for the Jacobian callback.  Constructed on first
   access so that numba is never imported at package load time.

   Equivalent C signature::

      void jac(int neq, double t,
               const double complex *y,
               int ml, int mu,
               double complex *pd,
               int nrowpd,
               void *ctx);

OdeSolver API
-------------

.. autoclass:: zvode.ZVODE
   :show-inheritance:

.. autoclass:: zvode.ZVODE_BDF
   :show-inheritance:

.. autoclass:: zvode.ZVODE_Adams
   :show-inheritance:
