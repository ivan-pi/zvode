cmake_minimum_required(VERSION 3.17.2...3.30)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C Fortran)

set(CMAKE_Fortran_COMPILER gfortran)

find_package(BLAS REQUIRED)

# Only request the module-building components. Do NOT request the full
# `Development` package: that pulls in Development.Embed, which needs the
# Python library for linking and is wrong for extension modules.
find_package(Python REQUIRED
  COMPONENTS
    Interpreter
    Development.Module
    ${SKBUILD_SABI_COMPONENT}
    NumPy
)

# Build the C extension. Compiling the Fortran straight into the module
# target lets CMake pull in the Fortran runtime (libgfortran) automatically
# at link time, which is the most foolproof option for a mixed C/Fortran module.
python_add_library(_zvode MODULE
    src/_zvode.c
    extern/zvode.f
    extern/zgbfa.f
    extern/zgbsl.f
  WITH_SOABI
    ${SKBUILD_SABI_VERSION})

target_include_directories(
  _zvode
  PRIVATE src)

target_link_libraries(
  _zvode
  PRIVATE Python::NumPy BLAS::BLAS)

# Install the compiled module INTO the package directory, so it imports
# as `zvode._zvode`.
install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})
