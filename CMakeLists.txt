cmake_minimum_required(VERSION 3.17.2...3.30)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C Fortran)

# Only request the module-building components. Do NOT request the full
# `Development` package: that pulls in Development.Embed, which needs the
# Python library for linking and is wrong for extension modules.
find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

# Gather the vendored legacy ZVODE Fortran sources.
file(GLOB ZVODE_FORTRAN_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/extern/zvode/*.f")

# Build the C extension. Compiling the Fortran straight into the module
# target lets CMake pull in the Fortran runtime (libgfortran) automatically
# at link time, which is the most foolproof option for a mixed C/Fortran module.
python_add_library(
  _zvode MODULE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/_zvode.c"
  ${ZVODE_FORTRAN_SOURCES}
  WITH_SOABI)

target_include_directories(
  _zvode
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(
  _zvode
  PRIVATE Python::NumPy)

# Install the compiled module INTO the package directory, so it imports
# as `zvode._zvode`.
install(TARGETS _zvode DESTINATION ${SKBUILD_PROJECT_NAME})
