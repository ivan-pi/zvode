# Build for the experimental BDF integrator subproject.
#
# Two paths share this file, mirroring the parent zvode build:
#   * a scikit-build-core wheel build (SKBUILD defined) producing the `_bdf`
#     CPython extension, and
#   * a plain `cmake -S . -B build` configure used to build and run the native
#     Fortran test program via CTest.
# Both consume the common `bdf` Fortran core library defined below.

cmake_minimum_required(VERSION 3.17)

if(SKBUILD)
  project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION}
          LANGUAGES C Fortran)
else()
  project(pybdf LANGUAGES C Fortran)
  enable_testing()
endif()

find_package(LAPACK REQUIRED)

# ---------------------------------------------------------------------------
# Common Fortran core (built once, consumed by the extension and the tests).
# ---------------------------------------------------------------------------
add_library(bdf STATIC
  src/fortran/bdf.f90
  src/fortran/c_bdf.f90)

set(_bdf_moddir ${CMAKE_CURRENT_BINARY_DIR}/bdf_modules)
set_target_properties(bdf PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY ${_bdf_moddir})
target_include_directories(bdf PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fortran
  ${_bdf_moddir})
# PRIVATE: LAPACK is called only inside this archive; CMake still propagates it
# (wrapped in $<LINK_ONLY:>) onto consumers' link lines.
target_link_libraries(bdf PRIVATE LAPACK::LAPACK)

# ---------------------------------------------------------------------------
# CPython extension `_bdf` (wheel build only).
# ---------------------------------------------------------------------------
if(SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
  python_add_library(_bdf MODULE src/_bdf.c WITH_SOABI)
  target_link_libraries(_bdf PRIVATE bdf Python::NumPy)
  install(TARGETS _bdf DESTINATION pybdf)
endif()

# ---------------------------------------------------------------------------
# Native Fortran test (standalone configure only).
# ---------------------------------------------------------------------------
if(NOT SKBUILD)
  add_executable(test_bdf test/test_bdf.f90)
  target_link_libraries(test_bdf PRIVATE bdf)
  add_test(NAME test_bdf COMMAND test_bdf)
endif()
