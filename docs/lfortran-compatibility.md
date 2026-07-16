# LFortran compatibility

Status of compiling the vendored ZVODE Fortran sources
(`extern/zvode.F`, `extern/zvode_linalg.f90`, `extern/c_zvode.f90`) with the
[LFortran](https://lfortran.org/) compiler.

**Tested compiler:** LFortran **0.63.0** (LLVM 19.1.1), the current
`conda-forge` release, installed with:

```bash
conda install conda-forge::lfortran      # or: micromamba install -c conda-forge lfortran
```

**Bottom line:** as of 0.63.0 the code **does not** compile with LFortran yet.

- **Fixed-form** (`zvode.F` as written today): blocked immediately. LFortran's
  fixed-form parser does not implement the Fortran 2003 object-oriented
  features that the functor redesign is built on — the file fails on its own
  module header (line 9) and never reaches the numerics.
- **Free-form** (the eventual target): the parser accepts the whole file, and
  the 3900-line ZVODE core compiles with `--legacy-array-sections`. But a
  **code-generation bug for COMMON blocks shared across module procedures**
  (which is exactly how `ZVOD01`/`ZVOD02` are used) then blocks every consumer
  of `zvode_mod`.

None of this is caused by the ZVODE source — each item below is a gap in
LFortran that reproduces in a handful of lines. gfortran compiles all of it.
Minimal reproducers live in [`lfortran/reproducers/`](lfortran/reproducers);
run them with:

```bash
LFORTRAN=$(command -v lfortran) docs/lfortran/reproducers/run.sh
```

---

## 1. Fixed-form parser gaps (the current blocker)

`zvode.F` is fixed-form and is compiled by the build as:

```bash
lfortran -c --fixed-form --cpp extern/zvode.F
```

It fails at the very first accessibility statement:

```
tokenizer error: Expecting terminating symbol for module
  --> extern/zvode.F:9:9
   |
 9 |         PRIVATE
```

Working through the header reveals that **LFortran's fixed-form parser does not
support any of the Fortran 2003 OO constructs** the modernised ZVODE depends
on. Each fails in isolation (gfortran `-ffixed-form` compiles every one; the
identical code in free-form compiles under LFortran):

| # | Construct (used by `zvode.F`)                     | LFortran `--fixed-form` error                          |
|---|---------------------------------------------------|--------------------------------------------------------|
| 1 | `PUBLIC` / `PRIVATE` accessibility statements     | `tokenizer error: Expecting terminating symbol for module` |
| 2 | `ABSTRACT INTERFACE` blocks                       | `tokenizer error: Expecting terminating symbol for module` |
| 3 | `TYPE, ABSTRACT :: …`                             | `tokenizer error: ICE: Cannot recognize global scope entity` |
| 4 | Type-bound procedures (`CONTAINS` inside a type)  | `tokenizer error: Expecting terminating symbol for module` |
| 5 | `TYPE, EXTENDS(…)`                                | `tokenizer error: ICE: Cannot recognize global scope entity` |
| 6 | `CLASS(…)` polymorphic entities                   | `tokenizer error: ICE: Cannot recognize global scope entity` |

These are the core of the design (`ZVODE_FUN`/`ZVODE_JAC` abstract types with a
deferred `EVAL`, `CLASS(ZVODE_FUN)` arguments threaded through every routine,
the `ABSTRACT INTERFACE` blocks, and `TYPE, EXTENDS` in `c_zvode.f90`), so
there is no source-level workaround that keeps both fixed-form and the functor
interface. A plain `INTERFACE` block, `COMMON`, `SAVE`, `DATA`, `EQUIVALENCE`,
computed `GO TO`, and labelled `DO` loops all parse fine in fixed-form
(`00_baseline_f77_fixed.f` passes), so the gap is specifically F2003 OO in
fixed-form.

No compiler flag changes this: `--std=legacy`, `--std=f23`,
`--fixed-form-infer`, `--implicit-typing`, and `--implicit-interface` all give
the same result.

### Minimal reproducer (feature #4, type-bound procedure)

`04_type_bound_proc_fixed.f` — fails; `04_type_bound_proc_free.f90` — compiles:

```fortran
      MODULE M
        TYPE T
          INTEGER :: N
        CONTAINS
          PROCEDURE :: EVAL => MYEVAL
        END TYPE
      CONTAINS
        SUBROUTINE MYEVAL(SELF)
          CLASS(T) :: SELF
          SELF%N = 0
        END SUBROUTINE
      END MODULE M
```

---

## 2. Free-form experiment (the eventual target)

To see what lies beyond the parser gaps, `zvode.F` was mechanically converted
to free-form (with `findent`) and compiled — **this is an experiment, the
repository stays fixed-form for now.** Results:

- **`extern/zvode_linalg.f90`** — compiles as-is (`lfortran -c`). No issues.
- **`zvode.F` (converted to free-form)** — the parser accepts the full file,
  including all the OO above. It then needs one flag:

  ```
  semantic error: Passing a scalar argument to an array dummy argument is not
  allowed. Use --legacy-array-sections to enable sequence association
  ```

  This is ordinary F77 sequence association (ZVODE passes `RWORK(LWM)` etc. as
  array arguments). With `--legacy-array-sections` the **entire ZVODE core
  compiles** to an object file and a `zvode_mod.mod`.

- **Any consumer of `zvode_mod`** (the native test drivers, `c_zvode.f90`) then
  hits an internal compiler error rooted in the COMMON blocks:

  ```
  LCompilersException: ExternalSymbol cannot be resolved, the module
  'file_common_block_zvod01' was not found, so the symbol
  'struct_instance_zvod01' could not be resolved
  ```

### Minimal reproducer (feature #7, COMMON across module procedures)

This is the blocker that matters most for ZVODE, whose `ZVOD01`/`ZVOD02`
COMMON blocks are shared across dozens of module procedures.
`07_common_module_a.f90` + `07_common_use_b.f90`:

```fortran
! 07_common_module_a.f90
module a_mod
contains
  subroutine seta(v)
    real, intent(in) :: v
    real :: shared
    common /blk/ shared
    shared = v
  end subroutine
  function geta() result(r)
    real :: r
    real :: shared
    common /blk/ shared
    r = shared
  end function
end module a_mod
```

```fortran
! 07_common_use_b.f90
program b
  use a_mod, only: seta, geta
  call seta(3.5)
  print *, geta()      ! => Internal Compiler Error
end program b
```

Boundary of the bug: a COMMON block touched by a **single** module procedure
compiles and runs; sharing it across two procedures triggers a code-generation
error in the single-file case and the ICE above under separate compilation. A
COMMON block is what ZVODE uses to carry solver state between its routines, so
this must be fixed upstream before the free-form path can produce a working
library.

- **`extern/c_zvode.f90`** (the `bind(c)` layer for the Python extension) —
  independent of the above, its procedure-pointer / `type(c_ptr)` code trips a
  separate `Internal Compiler Error: Unhandled exception`. Only relevant to the
  Python wheel, not the pure-Fortran path.

---

## 3. Summary

| Component                          | fixed-form (today)          | free-form (target)                         |
|------------------------------------|-----------------------------|--------------------------------------------|
| `extern/zvode_linalg.f90`          | n/a (already free-form) ✅  | ✅ compiles                                |
| `extern/zvode.F` core              | ❌ F2003 OO parser gaps     | ✅ compiles (`--legacy-array-sections`)    |
| consuming `zvode_mod` (tests)      | ❌ (blocked above)          | ❌ COMMON-block codegen ICE                |
| `extern/c_zvode.f90` (C/Python)    | ❌ (blocked above)          | ❌ separate `bind(c)` ICE                  |

## 4. Recommended next steps

1. **File the reproducers upstream.** The fixed-form F2003-OO gaps (§1) and the
   COMMON-block codegen bug (§2) are the two that block ZVODE; the compiler
   itself asks for these at <https://github.com/lfortran/lfortran/issues>.
2. **Track a newer LFortran.** 0.63.0 is alpha ("expected to fail on
   third-party codes"); re-run `docs/lfortran/reproducers/run.sh` against each
   release to see when the gaps close.
3. **Fixed-form stays the source of truth for now.** There is no source change
   that makes the functor design compile in fixed-form under 0.63.0 without
   abandoning either fixed-form or the OO interface.
4. When moving to free-form later, `--legacy-array-sections` will be required,
   and the COMMON-block bug (§2) must be resolved first — otherwise the core
   compiles but nothing can link against it.
