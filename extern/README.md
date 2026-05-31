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

The internal numerics — the `ZVOD01`/`ZVOD02` Fortran `COMMON` blocks, and the Adams and BDF
stepping logic — are unchanged.

## `linpack/` — changes from the Netlib release

The LINPACK routines (`zgefa.f`, `zgesl.f`, `zgbfa.f`, `zgbsl.f`) are taken from the
[Netlib LINPACK distribution](https://netlib.org/linpack/). The following change has been
applied to each file:

- **Assumed-size array argument declarations** — dummy array arguments that were declared
  with an explicit length (e.g. `DIMENSION A(LDA,1)`) have been updated to use the standard
  assumed-size notation (`DIMENSION A(LDA,*)`). This silences warnings from strict Fortran
  compilers about incorrect array bound declarations.
