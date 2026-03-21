cmake_minimum_required(VERSION 3.16)
project(twofluid_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MinGW + Eigen: -O2 causes segfaults in solver with Rhie-Chow. Use -O1.
if(MINGW AND CMAKE_BUILD_TYPE STREQUAL "Release")
    string(REPLACE "-O3" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-O2" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Try system Eigen first, fall back to local path
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    # Fallback: check common locations
    if(EXISTS "C:/mingw64/include/eigen-3.4.0")
        set(EIGEN3_INCLUDE_DIR "C:/mingw64/include/eigen-3.4.0")
    elseif(EXISTS "/usr/include/eigen3")
        set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
    else()
        message(FATAL_ERROR "Eigen3 not found. Install with: apt install libeigen3-dev (Linux) or specify -DEIGEN3_INCLUDE_DIR=...")
    endif()
    message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
endif()

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
    src/checkpoint.cpp
    src/hybrid_mesh_generator.cpp
    src/steam_tables.cpp
    src/mesh_reader.cpp
    src/polymesh_reader.cpp
    src/surface_tension.cpp
    src/mpi_comm.cpp
    src/partitioning.cpp
    src/mpi_solver.cpp
    src/distributed_mesh.cpp
    src/distributed_solver.cpp
    src/json_input.cpp
    src/wall_boiling.cpp
)
target_include_directories(twofluid_core PUBLIC
    include
    $<$<NOT:$<BOOL:${Eigen3_FOUND}>>:${EIGEN3_INCLUDE_DIR}>
)
if(Eigen3_FOUND)
    target_link_libraries(twofluid_core PUBLIC Eigen3::Eigen)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(twofluid_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found -- parallel execution enabled")
else()
    message(STATUS "OpenMP not found -- serial execution only")
endif()

option(USE_MPI "Enable MPI distributed parallel" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(twofluid_core PUBLIC USE_MPI)
    target_link_libraries(twofluid_core PUBLIC MPI::MPI_CXX)
    message(STATUS "MPI found -- distributed parallel enabled")
endif()

option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(twofluid_core PRIVATE src/gpu_solver.cu)
    target_compile_definitions(twofluid_core PUBLIC USE_CUDA)
    target_link_libraries(twofluid_core PUBLIC CUDA::cusparse CUDA::cublas CUDA::cudart)
    set_source_files_properties(src/gpu_solver.cu PROPERTIES LANGUAGE CUDA)
    message(STATUS "CUDA found -- GPU acceleration enabled")
else()
    target_sources(twofluid_core PRIVATE src/gpu_solver_stub.cpp)
    message(STATUS "CUDA disabled -- GPU stubs only")
endif()

# Standalone solver executable
add_executable(twofluid_solver src/main.cpp)
target_link_libraries(twofluid_solver PRIVATE twofluid_core)
# Increase stack size to 8MB (MinGW default 1MB can cause crashes with Eigen)
if(MINGW)
    target_link_options(twofluid_solver PRIVATE -Wl,--stack,33554432)
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
    target_link_options(verification_cases PRIVATE -Wl,--stack,16777216)
endif()

add_executable(benchmark_mpi tests/benchmark_mpi.cpp)
target_link_libraries(benchmark_mpi PRIVATE twofluid_core)
if(MINGW)
    target_link_options(benchmark_mpi PRIVATE -Wl,--stack,16777216)
endif()

# Standalone CUDA benchmark (only with CUDA enabled)
if(USE_CUDA)
    add_executable(benchmark_gpu tests/benchmark_gpu.cu)
    set_target_properties(benchmark_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(benchmark_gpu PRIVATE CUDA::cusparse CUDA::cublas CUDA::cudart)
    message(STATUS "CUDA benchmark target: benchmark_gpu")
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
