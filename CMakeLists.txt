cmake_minimum_required(VERSION 3.17)

# Two configuration paths share this file:
#   * the scikit-build-core wheel build (SKBUILD_* variables defined), which
#     builds the `_zvode` Python extension module, and
#   * a plain `cmake -S . -B build` standalone configure used in CI to build
#     and run the Fortran-native test programs via CTest.
# Both consume the common `zvode` Fortran core library defined below; the
# Python-extension pieces are guarded with SKBUILD so the standalone path does
# not require scikit-build's injected variables.
if(DEFINED SKBUILD_PROJECT_NAME)
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
# `_zvode` Python extension and the Fortran-native test programs, so the
# sources, backend selection, and link dependencies live in a single place.
#
# POSITION_INDEPENDENT_CODE is required because the LAPACK/LINPACK choice and
# the generated .mod files are PUBLIC: linking this static archive into the
# `_zvode` shared module needs PIC objects, and consumers find zvode_mod.mod
# via the module directory exported as a PUBLIC include directory.
add_library(zvode STATIC
  extern/zvode.F
  extern/zvode_linalg.f90
  extern/c_zvode.f90)

set(_zvode_moddir ${CMAKE_CURRENT_BINARY_DIR}/zvode_modules)
set_target_properties(zvode PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${_zvode_moddir})
target_include_directories(zvode PUBLIC ${_zvode_moddir})

if(ZVODE_LINALG_BACKEND STREQUAL "LAPACK")
  target_link_libraries(zvode PUBLIC LAPACK::LAPACK)
else()
  target_sources(zvode PRIVATE
    extern/linpack/zgbfa.f
    extern/linpack/zgbsl.f
    extern/linpack/zgefa.f
    extern/linpack/zgesl.f)
  target_compile_definitions(zvode PRIVATE ZVODE_USE_LINPACK)
  target_link_libraries(zvode PUBLIC BLAS::BLAS)
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

  target_include_directories(_zvode PRIVATE src)
  target_link_libraries(_zvode PRIVATE zvode Python::NumPy)

  # Install the compiled module INTO the package directory, so it imports
  # as `zvode._zvode`.
  install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})

endif() # SKBUILD

# ---------------------------------------------------------------------------
# Fortran-native test programs (CTest)
# ---------------------------------------------------------------------------
# These exercise the ZVODE Fortran core directly through its functor-based
# public API (zvode_mod), independent of the Python bindings.  They link the
# common `zvode` library (picking up zvode_mod.mod via its PUBLIC module
# directory) and are built only in the standalone CMake configure, not in the
# wheel build.  Each new Fortran test MUST be added to the list below.
if(DEFINED SKBUILD)
  set(_zvode_tests_default OFF)
else()
  set(_zvode_tests_default ON)
endif()
option(ZVODE_BUILD_TESTS
  "Build and register the Fortran-native ZVODE test programs"
  ${_zvode_tests_default})

if(ZVODE_BUILD_TESTS)
  enable_testing()

  foreach(_test IN ITEMS
      test_zvode_constant
      test_zvode_decay
      test_zvode_complex_oscillator)
    add_executable(${_test} test/${_test}.f90)
    target_link_libraries(${_test} PRIVATE zvode)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()
endif()
