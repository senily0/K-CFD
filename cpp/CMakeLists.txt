cmake_minimum_required(VERSION 3.16)
project(twofluid_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MinGW -O2 causes stack-related crashes with Eigen; use -O1 for Release
if(MINGW AND CMAKE_BUILD_TYPE STREQUAL "Release")
    string(REPLACE "-O3" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-O2" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Eigen: use the local MinGW include path directly
set(EIGEN3_INCLUDE_DIR "C:/mingw64/include/eigen-3.4.0")

# Core static library
add_library(twofluid_core STATIC
    src/mesh.cpp
    src/fields.cpp
    src/fvm_operators.cpp
    src/gradient.cpp
    src/linear_solver.cpp
    src/interpolation.cpp
    src/simple_solver.cpp
    src/turbulence.cpp
    src/turbulence_sst.cpp
    src/closure.cpp
    src/two_fluid_solver.cpp
    src/vtk_writer.cpp
    src/mesh_generator.cpp
    src/preconditioner.cpp
    src/time_control.cpp
    src/phase_change.cpp
    src/solid_conduction.cpp
    src/conjugate_ht.cpp
    src/radiation.cpp
    src/amr.cpp
    src/mesh_generator_3d.cpp
    src/chemistry.cpp
    src/hybrid_mesh_generator.cpp
    src/steam_tables.cpp
    src/mesh_reader.cpp
)
target_include_directories(twofluid_core PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
)

# Standalone solver executable
add_executable(twofluid_solver src/main.cpp)
target_link_libraries(twofluid_solver PRIVATE twofluid_core)
# Increase stack size to 8MB (MinGW default 1MB can cause crashes with Eigen)
if(MINGW)
    target_link_options(twofluid_solver PRIVATE -Wl,--stack,8388608)
endif()

# Native test executables (no Python dependency)
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE twofluid_core)

add_executable(test_simple tests/test_simple.cpp)
target_link_libraries(test_simple PRIVATE twofluid_core)

add_executable(test_new_modules tests/test_new_modules.cpp)
target_link_libraries(test_new_modules PRIVATE twofluid_core)
if(MINGW)
    target_link_options(test_new_modules PRIVATE -Wl,--stack,8388608)
endif()

add_executable(verification_cases tests/verification_cases.cpp)
target_link_libraries(verification_cases PRIVATE twofluid_core)
if(MINGW)
    target_link_options(verification_cases PRIVATE -Wl,--stack,8388608)
endif()

# pybind11 module (optional -- only if pybind11 is found)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(twofluid_cpp src/bindings.cpp)
    target_link_libraries(twofluid_cpp PRIVATE twofluid_core)
    message(STATUS "pybind11 found -- building Python module")
else()
    message(STATUS "pybind11 not found -- skipping Python module (install with: pip install pybind11)")
endif()
