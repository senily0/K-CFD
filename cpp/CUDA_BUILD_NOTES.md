# CUDA Native Build Investigation

## System Configuration
- GPU: RTX 3050 4GB
- CUDA: 12.3 (nvcc available at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc`)
- Host compiler: MinGW g++ 15.2.0 (at `C:\mingw64\bin\g++`)
- MSVC (cl.exe): NOT installed

## Attempted Build

Minimal test (`test_cuda.cu`):
```c
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("CUDA devices: %d\n", count);
    return 0;
}
```

Command tried:
```
nvcc -ccbin g++ -o test_cuda.exe test_cuda.cu
```

## Result

**FAILED**: `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`

Even with `-ccbin g++` explicitly set, nvcc on Windows unconditionally requires `cl.exe`
(MSVC) as part of its toolchain initialization before delegating to the specified host
compiler. This is a known Windows-only nvcc limitation.

Alternative compilers checked:
- `cl.exe` (MSVC): not found
- `clang`: not found

## Conclusion

**Native CUDA build requires MSVC on Windows.** MinGW g++ cannot substitute for cl.exe
in nvcc's Windows build pipeline.

## Options to Enable Native CUDA Build

1. **Install Visual Studio Build Tools** (recommended, free):
   - Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
   - Select "Desktop development with C++" workload
   - After install, use the "Developer Command Prompt" or add cl.exe to PATH

2. **Use WSL2** (Linux environment):
   - nvcc on Linux works with g++ without requiring cl.exe
   - `sudo apt install cuda-toolkit-12-3` or use NVIDIA's WSL2 CUDA support

3. **Use pre-built CuPy** (current working approach):
   - GPU acceleration via CuPy is already working (5.4x speedup at 128K cells)
   - No native CUDA compilation required
   - See `verification/case13_gpu.py`

## Current GPU Status

GPU acceleration is implemented and tested via CuPy (Python):
- 5.4x speedup at 128,000 cells
- RTX 3050 confirmed working for cuSPARSE BiCGSTAB
- C++ GPU stub (`src/gpu_solver_stub.cpp`) compiles without CUDA for cross-platform builds
- Full CUDA implementation (`src/gpu_solver.cu`) ready for MSVC/Linux build
