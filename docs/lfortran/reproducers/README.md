# LFortran reproducers

Minimal cases behind [`../../lfortran-compatibility.md`](../../lfortran-compatibility.md),
against LFortran 0.63.0. gfortran compiles all of them.

Run the whole matrix:

```bash
LFORTRAN=$(command -v lfortran) ./run.sh
```

| File(s)                                                   | Feature LFortran rejects                              |
|-----------------------------------------------------------|------------------------------------------------------|
| `00_baseline_f77_fixed.f`                                 | control — F77 in fixed-form that **does** compile    |
| `01_access_{fixed.f,free.f90}`                            | `PUBLIC` / `PRIVATE` statements (fixed-form)          |
| `02_abstract_interface_{fixed.f,free.f90}`                | `ABSTRACT INTERFACE` (fixed-form)                    |
| `03_abstract_type_{fixed.f,free.f90}`                     | `TYPE, ABSTRACT` (fixed-form)                        |
| `04_type_bound_proc_{fixed.f,free.f90}`                   | type-bound procedures (fixed-form)                   |
| `05_type_extends_{fixed.f,free.f90}`                      | `TYPE, EXTENDS(...)` (fixed-form)                    |
| `06_class_polymorphic_{fixed.f,free.f90}`                 | `CLASS(...)` polymorphic entities (fixed-form)      |
| `07_common_module_a.f90` + `07_common_use_b.f90`          | COMMON block shared across module procedures (codegen) |

`*_fixed.f` fail under `lfortran --fixed-form`; the `*_free.f90` counterparts
compile, showing each gap is specific to the fixed-form parser. Case 07 is a
free-form code-generation bug (the module compiles; a separate unit that
`use`s it triggers an internal compiler error).
