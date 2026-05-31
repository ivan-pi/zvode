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

find_package(BLAS REQUIRED)

# Build the C extension. Compiling the Fortran straight into the module
# target lets CMake pull in the Fortran runtime (libgfortran) automatically
# at link time, which is the most foolproof option for a mixed C/Fortran module.
python_add_library(_zvode MODULE
    src/_zvode.c
    extern/zvode.f
    extern/c_zvode.f90
    extern/lapack.f90
    extern/linpack/zgbfa.f
    extern/linpack/zgbsl.f
    extern/linpack/zgefa.f
    extern/linpack/zgesl.f
  WITH_SOABI
    ${SKBUILD_SABI_VERSION})

# In the LINPACK ZGBFA routine, abd(m+1,k) is passed to zscal with lm=0
# (BLAS no-op), however m+1 > lda which trips gfortran's bound checker.
# See my chat with Claude for a full description of the problem
#   https://claude.ai/code/session_01P6jXN7Mrzxv28HirHe8UhG
# Originally surfaced in the banded test_oscillator_miter
set_source_files_properties(
  extern/linpack/zgbfa.f
  PROPERTIES
    COMPILE_OPTIONS "$<$<Fortran_COMPILER_ID:GNU>:-fcheck=no-bounds>")

# Disable bounds checking on the LINPACK routines, because it
# produces false positives in some corner cases (e.g. ml = 0)


target_include_directories(
  _zvode
  PRIVATE src)

target_link_libraries(
  _zvode
  PRIVATE Python::NumPy BLAS::BLAS)

# Install the compiled module INTO the package directory, so it imports
# as `zvode._zvode`.
install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})
