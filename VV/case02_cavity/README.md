# Case 02: Lid-Driven Cavity

## Problem Description

Steady-state flow in a square cavity driven by a moving top lid at Re = 100.
This is the standard benchmark for incompressible flow solvers, with well-known
reference data from Ghia et al. (1982).

## Geometry

```
     u = 1.0, v = 0
  ------>------>------>
  |                   |
  |                   |
  |   Re = 100        |
  |                   |
  |                   |
  =====================
  wall (no-slip)

  Lx = Ly = 1.0 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Top (lid) | Dirichlet | u = 1.0, v = 0 |
| Bottom | No-slip wall | u = v = 0 |
| Left | No-slip wall | u = v = 0 |
| Right | No-slip wall | u = v = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Lid velocity | 1.0 | m/s |
| Reynolds number (Re) | 100 | - |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/quad_32x32.msh` | 1,024 | Structured quad | Coarse |
| `mesh/quad_64x64.msh` | 4,096 | Structured quad | Fine |

## Expected Results

- Primary vortex centered near (0.6172, 0.7344) at Re = 100
- Velocity profiles along centerlines match Ghia et al. reference data
- u-velocity at vertical centerline: min ~ -0.2109 at y ~ 0.4531
- v-velocity at horizontal centerline: min ~ -0.2453 at x ~ 0.8047

## How to Run

```bash
cd VV
python run_all.py --case 02
```

## Reference

- Ghia, U., Ghia, K.N., Shin, C.T., "High-Re solutions for incompressible
  flow using the Navier-Stokes equations and a multigrid method",
  *J. Comput. Phys.*, 48, 387-411 (1982).
