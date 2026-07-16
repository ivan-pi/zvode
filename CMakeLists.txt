cmake_minimum_required(VERSION 3.18)

# Two configuration paths share this file:
#   * the scikit-build-core wheel build (SKBUILD_* variables defined), which
#     builds the `_zvode` Python extension module, and
#   * a plain `cmake -S . -B build` standalone configure used in CI to build
#     and run the Fortran-native test programs via CTest.
# Both consume the common `zvode` Fortran core library defined below; the
# Python-extension pieces are guarded with SKBUILD so the standalone path does
# not require scikit-build's injected variables.
if(SKBUILD)
  project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES C Fortran)
else()
  project(zvode LANGUAGES C Fortran)
endif()

# ---------------------------------------------------------------------------
# Linear algebra backend
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Common ZVODE Fortran core library (libzvode)
# ---------------------------------------------------------------------------
# Built once from the vendored Fortran in extern/ and consumed by both the
# `_zvode` Python extension and the native test programs, so the sources,
# backend selection, and link dependencies live in a single place.
#
# STATIC (an OBJECT library would be equivalent) keeps the core baked into the
# consumers: `_zvode` ends up self-contained, so the wheel ships no extra
# shared library to locate at runtime.  POSITION_INDEPENDENT_CODE is then
# required so this archive can be linked into the `_zvode` shared module.
add_library(zvode STATIC
  extern/zvode.F
  extern/zvode_linalg.f90
  extern/c_zvode.f90)

set(_zvode_moddir ${CMAKE_CURRENT_BINARY_DIR}/zvode_modules)
set_target_properties(zvode PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${_zvode_moddir})

# Expose the library's interface to consumers (the `_zvode` Python extension
# and the native C/Fortran tests):
#   * extern/ holds the public C API header zvode.h, next to the c_zvode.f90
#     that implements it;
#   * the module directory holds the generated zvode_mod.mod for the Fortran
#     callers.
# Plain paths are fine here because `zvode` is internal to this build and is
# never installed/exported, so no BUILD_INTERFACE/INSTALL_INTERFACE genexes
# are needed.
target_include_directories(zvode PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/extern
  ${_zvode_moddir})

if(ZVODE_LINALG_BACKEND STREQUAL "LAPACK")
  # PRIVATE: LAPACK is called only inside this library's own objects, never by
  # consumers.  As a static archive, `zvode` still propagates this onto the
  # final link line (CMake wraps it in $<LINK_ONLY:>), so symbols resolve in
  # the module and the tests without leaking LAPACK's usage requirements.
  target_link_libraries(zvode PRIVATE LAPACK::LAPACK)
else()
  target_sources(zvode PRIVATE
    extern/linpack/zgbfa.f
    extern/linpack/zgbsl.f
    extern/linpack/zgefa.f
    extern/linpack/zgesl.f)
  target_compile_definitions(zvode PRIVATE ZVODE_USE_LINPACK)
  target_link_libraries(zvode PRIVATE BLAS::BLAS)
  # In the LINPACK ZGBFA routine, abd(m+1,k) is passed to zscal with lm=0
  # (BLAS no-op), but m+1 > lda trips gfortran's bounds checker even though
  # the call is harmless (lm=0 means zscal does nothing).
  # Originally surfaced in the banded test_oscillator_miter test case.
  set_source_files_properties(
    extern/linpack/zgbfa.f
    PROPERTIES
      COMPILE_OPTIONS "$<$<Fortran_COMPILER_ID:GNU>:-fcheck=no-bounds>")
endif()

# ---------------------------------------------------------------------------
# Python extension module (_zvode)
# ---------------------------------------------------------------------------
if(SKBUILD)

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

  # Only the hand-written C binding layer lives in the module target; the
  # Fortran core comes in through the `zvode` library.  CMake propagates the
  # Fortran runtime (libgfortran) onto the link line automatically because the
  # linked archive was built from Fortran sources.
  python_add_library(_zvode MODULE
      src/_zvode.c
    WITH_SOABI
      ${SKBUILD_SABI_VERSION})

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

  # zvode.h comes in via the `zvode` library's PUBLIC include directories.
  target_link_libraries(_zvode PRIVATE zvode Python::NumPy)
  target_compile_features(_zvode PRIVATE c_std_11)

  # Install the compiled module INTO the package directory, so it imports
  # as `zvode._zvode`.
  install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})

endif() # SKBUILD

# ---------------------------------------------------------------------------
# Native test programs (CTest)
# ---------------------------------------------------------------------------
# These exercise the compiled ZVODE core directly, independent of the Python
# bindings, by linking the common `zvode` library.  The Fortran drivers use
# the functor-based public API (zvode_mod, picked up via its PUBLIC module
# directory); the C driver uses the C API in extern/zvode.h, covering the
# bind(c) layer in extern/c_zvode.f90 (callbacks, user Jacobian, and ctx
# forwarding) that is otherwise only reached through Python.  Built only in
# the standalone CMake configure, not the wheel build.
# Each new test MUST be added to one of the lists below.
if(SKBUILD)
  set(_zvode_tests_default OFF)
else()
  set(_zvode_tests_default ON)
endif()
option(ZVODE_BUILD_TESTS
  "Build and register the native ZVODE test programs"
  ${_zvode_tests_default})

if(ZVODE_BUILD_TESTS)
  enable_testing()

  # Fortran-native drivers (zvode_mod functor API).
  foreach(_test IN ITEMS
      test_zvode_decay
      test_zvode_complex_oscillator
      test_zvode_public_api
      test_zvode_ewt_functor)
    add_executable(${_test} test/${_test}.f90)
    target_link_libraries(${_test} PRIVATE zvode)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()

  # C-native drivers (C API in extern/zvode.h, inherited from `zvode`).
  foreach(_test IN ITEMS
      test_zvode_coupled)
    add_executable(${_test} test/${_test}.c)
    target_link_libraries(${_test} PRIVATE zvode)
    target_compile_features(${_test} PRIVATE c_std_11)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()
endif()
