# Case 13: GPU Acceleration

## Problem Description

Verification that GPU-accelerated linear algebra produces identical results to the
CPU solver. A 2D lid-driven cavity at Re=400 is solved using CuPy-backed sparse
matrix operations on the GPU, and results are compared against the CPU (SciPy)
reference solution. This validates the GPU code path including sparse matrix
assembly, preconditioner application, and iterative solver convergence.

## Geometry

```
  lid: u = 1.0 m/s -->
  ========================
  |                      |
  |                      |
  |   Re = 400 cavity   |
  |   64x64 mesh        |
  |                      |
  |                      |
  ========================

  Lx = Ly = 1.0 m
```

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.0025 | Pa.s |
| Reynolds number (Re) | 400 | - |
| Lid velocity | 1.0 | m/s |
| Grid size | 64 x 64 | cells |

## GPU Configuration

| Parameter | Value |
|-----------|-------|
| Backend | CuPy (CUDA) |
| Sparse format | CSR |
| Linear solver | BiCGSTAB |
| Preconditioner | Diagonal (Jacobi) |
| Fallback | SciPy CPU if no GPU available |

## Verification Method

GPU acceleration is verified by `verification_cases.exe` (Case 13) and the
Python implementation in `verification/case13_gpu.py`. The test compares
GPU and CPU solutions:

1. Assemble identical linear systems on CPU and GPU
2. Solve with identical parameters
3. Compare solution vectors element-wise

## Expected Results

- GPU vs. CPU solution L2 difference < 1e-6
- GPU speedup > 2x on supported hardware (problem-size dependent)
- Identical convergence history (iteration count within +/- 1)
- Correct handling of GPU memory allocation and transfer

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 13

# Via Python verification suite (requires CuPy)
cd VV
python run_all.py --case 13
```

## Reference

- Nishida, A., "Experience in developing an open source scalable software
  infrastructure in Japan", *LNCS*, 6017, 448-462 (2010).
- Bell, N. & Garland, M., "Efficient sparse matrix-vector multiplication on
  CUDA", NVIDIA Technical Report NVR-2008-004 (2008).
