# Case 16: Preconditioner Performance

## Problem Description

Comparison of linear solver preconditioners (none, Jacobi, ILU, AMG) on a 3D
channel flow problem. This case does not verify physical accuracy but rather
tests that all preconditioner options converge correctly and measures their
relative performance.

## Geometry

```
  3D channel:
  Lx = 1.0 m,  Ly = 0.5 m,  Lz = 0.5 m

  inlet --> [======================] --> outlet
            wall (top, bottom, front, back)
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet (x=0) | Dirichlet | 3D parabolic profile |
| Outlet (x=Lx) | Zero gradient | p = 0 |
| All walls | No-slip wall | u = v = w = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density | 1.0 | kg/m^3 |
| Dynamic viscosity | 0.001 | Pa.s |
| Reynolds number | 500 | - |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/hex_20x10x10.msh` | 2,000 | Structured hex | Standard |

## Expected Results

- All preconditioners converge to the same solution
- ILU provides at least 2x speedup over unpreconditioned solver
- AMG converges in fewest iterations (if available)
- Iteration counts: none > Jacobi > ILU >= AMG

## How to Run

```bash
cd VV
python run_all.py --case 16
```

## Reference

- Saad, Y., *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
  SIAM (2003).
