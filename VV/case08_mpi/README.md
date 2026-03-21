# Case 08: MPI Parallel Decomposition

## Problem Description

Verification of the MPI-parallel domain decomposition solver. A 2D channel flow
problem is partitioned across multiple MPI ranks and solved using the distributed
SIMPLE algorithm. Results are compared against the serial solution to verify that
domain decomposition introduces no numerical artifacts at partition boundaries.

## Geometry

```
  wall_top (no-slip)
  ================================================
  |     Rank 0      |      Rank 1               |
  |   (left half)   |   (right half)            |
  |  inlet -->      |               --> outlet  |
  |                  |                          |
  ================================================
  wall_bottom (no-slip)

  Lx = 1.0 m, Ly = 0.1 m
  Decomposition: 2 ranks in x-direction
```

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 100 | - |
| Grid size | 20 x 10 | cells |
| MPI ranks | 2 | - |

## MPI Configuration

| Parameter | Value |
|-----------|-------|
| Decomposition method | 1D strip (x-direction) |
| Halo exchange | 1-cell overlap |
| Global reduction | MPI_Allreduce for residuals |

## Verification Method

The MPI benchmark executable (`benchmark_mpi.exe`) runs the parallel solver.
The case verifies:
1. Solution matches serial result within machine precision
2. Residual convergence is independent of partition count
3. Halo exchange correctness at partition boundaries

```bash
mpiexec -n 2 cpp/build/benchmark_mpi.exe 20 10 10 50
```

Note: `benchmark_mpi.exe` reports timing and residual data but does not produce
VTU output. VTU generation from MPI runs is available via the Python solver
(`verification/case8_mpi.py`).

## Expected Results

- Serial vs. parallel L2 difference < 1e-10
- Linear speedup efficiency > 80% on 2 ranks
- Converged residual < 1e-5

## How to Run

```bash
# Via MPI benchmark
cd cpp/build
mpiexec -n 2 ./benchmark_mpi.exe 20 10 10 50

# Via verification_cases.exe (internal test)
./verification_cases.exe    # Runs all cases including Case 8

# Via Python verification suite
cd VV
python run_all.py --case 08
```

## Reference

- Gropp, W., Lusk, E. & Skjellum, A., *Using MPI*, 3rd ed., MIT Press (2014).
- Smith, B., Bjorstad, P. & Gropp, W., *Domain Decomposition: Parallel
  Multilevel Methods for Elliptic PDEs*, Cambridge University Press (1996).
