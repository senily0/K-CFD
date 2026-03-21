# Case 06: MUSCL Method of Manufactured Solutions

## Problem Description

Grid convergence study for the MUSCL (Monotone Upstream-centered Schemes for
Conservation Laws) convection scheme using the Method of Manufactured Solutions
(MMS). A known analytical solution is substituted into the governing equation to
derive a source term, then the discretization error convergence rate is measured
across three grid levels.

## Geometry

```
  ========================
  |                      |
  |   phi = sin(pi*x)   |
  |       * sin(pi*y)   |
  |                      |
  |   u = (1.0, 0.5)    |
  |                      |
  ========================

  Lx = Ly = 1.0 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| All | Dirichlet | MMS exact solution |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density | 1.0 | kg/m^3 |
| Diffusivity (mu) | 0.01 | m^2/s |
| Velocity | (1.0, 0.5) | m/s |
| Limiter | van Leer | - |

## Meshes

| File | Cells | h | Type |
|------|-------|---|------|
| `mesh/quad_10x10.msh` | 100 | 0.1 | Structured quad |
| `mesh/quad_20x20.msh` | 400 | 0.05 | Structured quad |
| `mesh/quad_40x40.msh` | 1,600 | 0.025 | Structured quad |

## Expected Results

- Observed convergence order >= 1.8 (nominally 2nd order)
- L2 error on finest mesh < 1%
- Richardson extrapolation applicable

### Convergence Table (typical)

| Grid | h | L2 error | Order |
|------|---|----------|-------|
| 10x10 | 0.100 | ~0.04 | - |
| 20x20 | 0.050 | ~0.01 | ~2.0 |
| 40x40 | 0.025 | ~0.003 | ~2.0 |

## How to Run

```bash
cd VV
python run_all.py --case 06
```

## Reference

- Roache, P.J., *Verification and Validation in Computational Science and
  Engineering*, Hermosa Publishers (2002).
- Roache, P.J., "Code verification by the Method of Manufactured Solutions",
  *J. Fluids Eng.*, 124(1), 4-10 (2002).
