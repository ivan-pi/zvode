# Fortran source modifications

This directory contains modified copies of the Fortran sources used by the `zvode` package.

## `zvode.f` — changes from the Netlib release

The [upstream ZVODE source](https://netlib.org/ode/zvode.f) (2006 LLNL release) has been changed in the following ways:

- **Module wrapping** — the entire code is enclosed in a Fortran 90 `MODULE ZVODE_MOD`
  with `IMPLICIT NONE` and explicit `PUBLIC`/`PRIVATE` declarations.
- **Abstract class callbacks** — the right-hand side function `F` and the Jacobian `JAC`
  were external subroutines in the original; here they are polymorphic `CLASS(ZVODE_FUN)`
  and `CLASS(ZVODE_JAC)` objects with deferred `EVAL` procedures (Fortran 2003 abstract
  classes). This is the most significant departure from the original design.
- **`RPAR`/`IPAR` removed** — the original interface passes user context through a real/complex
  array `RPAR` and an integer array `IPAR`. In the functor design, context is carried by the
  class object itself, so these arguments are no longer present.
- **Error-weight callback (`ZVODE_EWT`)** — the error-weight setter `ZEWSET`, which the upstream
  documentation invites users to replace by a link-time override, is exposed as a functor: a
  `ZVODE_EWT` class with an `NEQ` component and an `EVAL` binding that defaults to the historical
  `ZEWSET` weighting. Unlike `ZVODE_FUN` / `ZVODE_JAC`, a sensible default exists, so this is a
  concrete type (usable directly as `ZVODE_EWT()`) rather than an abstract type with a deferred
  binding; users override the policy by extending it and re-binding `EVAL`. `ZVODE` takes it through a new *optional*
  trailing argument `EWTFUN`; when absent the default is used, so existing positional calls (and
  the `c_zvode.f90` bind(c) layer) are unchanged. This is the ZVODE analogue of CVODE's
  `CVodeWFtolerances` / `CVEwtFn`. Following CVODE, only the error-weight setter is user-replaceable;
  the internal weighted-RMS norm `ZVNORM` is left fixed (it stays `PURE` and off the functor-dispatch
  path in the step inner loops). Like `CVEwtFn`, `EWTF % EVAL` is called just before every internal
  step, not once.
- The helper functions ZACOPY, DZSCAL, DZAXPY have been moved to a separate module `ZVODE_LINALG_MOD`.
- **`COMMON` blocks removed** — the internal state formerly held in the labeled `COMMON` blocks
  `/ZVOD01/` and `/ZVOD02/` is now declared as module variables of `ZVODE_MOD`, shared between the
  package routines by host association. Module variables carry the `SAVE` attribute implicitly, so
  the values persist between calls exactly as the `COMMON` blocks did. The one routine that relied on
  `COMMON` storage association, `ZVSRCO` (save/restore of the internal state), was rewritten to pack
  and unpack the module variables explicitly, preserving the original `RSAV`/`ISAV` layout (51 reals,
  41 integers). These module variables are intended to move into a derived type in a later step.

The internal numerics — the Adams and BDF stepping logic — are unchanged.

## `linpack/` — changes from the Netlib release

The LINPACK routines (`zgefa.f`, `zgesl.f`, `zgbfa.f`, `zgbsl.f`) are taken from the
[Netlib LINPACK distribution](https://netlib.org/linpack/).
The following change has been applied to each file:

- **Assumed-size array argument declarations** — dummy array arguments that were declared with an explicit length (e.g. `DIMENSION A(LDA,1)`) have been updated to use the standard assumed-size notation (`DIMENSION A(LDA,*)`). This silences warnings from strict Fortran compilers about incorrect array bound declarations.
