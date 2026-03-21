# Case 18: OpenMP Shared-Memory Parallelism

## Problem Description

Verification of OpenMP thread-parallel execution for the sparse linear solver
and matrix assembly routines. A 2D cavity flow problem is solved using 1, 2, and
4 OpenMP threads, and results are compared to verify thread-safety and
reproducibility. This case also measures strong-scaling efficiency.

## Geometry

```
  lid: u = 1.0 m/s -->
  ========================
  |                      |
  |                      |
  |   Re = 100 cavity   |
  |   32x32 mesh        |
  |                      |
  |                      |
  ========================

  Lx = Ly = 1.0 m
```

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 100 | - |
| Lid velocity | 1.0 | m/s |
| Grid size | 32 x 32 | cells |

## OpenMP Configuration

| Parameter | Value |
|-----------|-------|
| Thread counts tested | 1, 2, 4 |
| Parallelized regions | Matrix assembly, SpMV, preconditioner |
| Schedule | static |
| Reduction | sum (for residuals) |

## Verification Method

OpenMP scaling is verified by `verification_cases.exe` (Case 18). The test:

1. Solves with 1 thread (serial reference)
2. Solves with 2 and 4 threads
3. Compares solution vectors: all must match within round-off tolerance
4. Reports wall-clock speedup for each thread count

## Expected Results

- 1-thread vs N-thread solution L2 difference < 1e-12
- 2-thread speedup > 1.5x
- 4-thread speedup > 2.5x
- Identical iteration counts across all thread configurations
- No race conditions or non-deterministic behavior

## Strong Scaling Table (typical)

| Threads | Wall time (s) | Speedup | Efficiency |
|---------|---------------|---------|------------|
| 1 | 1.00 | 1.00x | 100% |
| 2 | 0.58 | 1.72x | 86% |
| 4 | 0.34 | 2.94x | 74% |

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
OMP_NUM_THREADS=4 ./verification_cases.exe    # Runs all cases including Case 18

# Control thread count
OMP_NUM_THREADS=2 ./verification_cases.exe

# Via Python verification suite
cd VV
python run_all.py --case 18
```

## Reference

- Chapman, B., Jost, G. & van der Pas, R., *Using OpenMP*, MIT Press (2008).
- Dagum, L. & Menon, R., "OpenMP: An industry standard API for shared-memory
  programming", *IEEE Comput. Sci. Eng.*, 5(1), 46-55 (1998).
