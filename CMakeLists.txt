cmake_minimum_required(VERSION 3.17)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C Fortran)

# Only request the module-building components. Do NOT request the full
# `Development` package: that pulls in Development.Embed, which needs the
# Python library for linking and is wrong for extension modules.
find_package(Python REQUIRED
  COMPONENTS
    Interpreter
    Development.Module
    "${SKBUILD_SABI_COMPONENT}"
    NumPy
)

set(ZVODE_LINALG_BACKEND "LAPACK" CACHE STRING
  "Linear algebra backend for LU factorization and triangular solves")
set_property(CACHE ZVODE_LINALG_BACKEND PROPERTY STRINGS LAPACK LINPACK)

if(ZVODE_LINALG_BACKEND STREQUAL "LAPACK")
  find_package(LAPACK REQUIRED)
  message(STATUS "ZVODE linalg backend: LAPACK (${LAPACK_LIBRARIES})")
elseif(ZVODE_LINALG_BACKEND STREQUAL "LINPACK")
  find_package(BLAS REQUIRED)
  message(STATUS "ZVODE linalg backend: LINPACK (BLAS: ${BLAS_LIBRARIES})")
else()
  message(FATAL_ERROR
    "Unknown ZVODE_LINALG_BACKEND \"${ZVODE_LINALG_BACKEND}\". "
    "Valid choices are LAPACK or LINPACK.")
endif()

# Build the C extension. Compiling the Fortran straight into the module
# target lets CMake pull in the Fortran runtime (libgfortran) automatically
# at link time, which is the most foolproof option for a mixed C/Fortran module.
python_add_library(_zvode MODULE
    src/_zvode.c
    extern/zvode.F
    extern/zvode_linalg.f90
    extern/c_zvode.f90
  WITH_SOABI
    ${SKBUILD_SABI_VERSION})

if(ZVODE_LINALG_BACKEND STREQUAL "LAPACK")
  target_link_libraries(_zvode PRIVATE LAPACK::LAPACK)
else()
  target_sources(_zvode PRIVATE
    extern/linpack/zgbfa.f
    extern/linpack/zgbsl.f
    extern/linpack/zgefa.f
    extern/linpack/zgesl.f)
  target_compile_definitions(_zvode PRIVATE ZVODE_USE_LINPACK)
  target_link_libraries(_zvode PRIVATE BLAS::BLAS)
  # In the LINPACK ZGBFA routine, abd(m+1,k) is passed to zscal with lm=0
  # (BLAS no-op), but m+1 > lda trips gfortran's bounds checker even though
  # the call is harmless (lm=0 means zscal does nothing).
  # Originally surfaced in the banded test_oscillator_miter test case.
  set_source_files_properties(
    extern/linpack/zgbfa.f
    PROPERTIES
      COMPILE_OPTIONS "$<$<Fortran_COMPILER_ID:GNU>:-fcheck=no-bounds>")
endif()

# Developer-only diagnostics for the hand-written C binding layer
# (src/_zvode.c).  Both are OFF by default and are exercised by the
# memory-safety CI workflow; they are intentionally scoped to the C source
# so the upstream Fortran is left untouched.
option(ZVODE_STRICT_WARNINGS
  "Compile src/_zvode.c with -Wall -Wextra -Wpedantic -Werror" OFF)
option(ZVODE_SANITIZE
  "Build the C extension with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# These flags are GCC/Clang spellings; guard them with a generator expression
# on the C compiler id so a different C compiler (MSVC, Intel, ...) simply
# ignores them instead of failing on an unknown option.
set(_zvode_gnulike "$<C_COMPILER_ID:GNU,Clang,AppleClang>")

if(ZVODE_STRICT_WARNINGS)
  # Scope the strict flags to the C binding layer only; the vendored Fortran
  # is upstream code and is not held to these warnings.
  #
  # -Wno-error=pedantic keeps -Wpedantic emitting diagnostics (so they stay
  # visible in the build log) but does not let them fail the build: the only
  # pedantic finding is the void *->function-pointer cast used to accept raw
  # ctypes/numba callback addresses, which is unavoidable and harmless on
  # every platform we target.  Every other warning remains a hard error.
  set_property(SOURCE src/_zvode.c APPEND PROPERTY COMPILE_OPTIONS
    "$<${_zvode_gnulike}:-Wall;-Wextra;-Wpedantic;-Werror;-Wno-error=pedantic>")
endif()

if(ZVODE_SANITIZE)
  # Instrument the C source and link the sanitizer runtimes into the module.
  # -fno-sanitize-recover=all turns UBSan findings into hard failures so the
  # CI job goes red instead of merely printing a diagnostic.
  set_property(SOURCE src/_zvode.c APPEND PROPERTY COMPILE_OPTIONS
    "$<${_zvode_gnulike}:-fsanitize=address,undefined;-fno-sanitize-recover=all;-fno-omit-frame-pointer>")
  target_link_options(_zvode PRIVATE
    "$<${_zvode_gnulike}:-fsanitize=address,undefined>")
endif()

target_include_directories(
  _zvode
  PRIVATE src)

target_link_libraries(
  _zvode
  PRIVATE Python::NumPy)

# Install the compiled module INTO the package directory, so it imports
# as `zvode._zvode`.
install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})
