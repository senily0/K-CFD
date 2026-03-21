# Case 01: Poiseuille Flow

## Problem Description

Steady-state laminar flow between two infinite parallel plates driven by a
constant pressure gradient. This is the most fundamental verification case for
any CFD solver, as the analytical solution is known exactly.

## Geometry

```
  wall_top (no-slip)
  ================================================
  -->  -->  -->  -->  -->  -->  -->  -->  -->  -->
  inlet                                    outlet
  -->  -->  -->  -->  -->  -->  -->  -->  -->  -->
  ================================================
  wall_bottom (no-slip)

  Lx = 1.0 m,  Ly = 0.1 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | Parabolic velocity profile |
| Outlet | Zero gradient | p = 0 |
| Top wall | No-slip wall | u = v = 0 |
| Bottom wall | No-slip wall | u = v = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 10 | - |

## Analytical Solution

The velocity profile is parabolic:

```
u(y) = (dp/dx) / (2*mu) * y * (H - y)
u_max = (dp/dx) * H^2 / (8*mu)
```

where `H = Ly` is the channel height.

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/quad_50x20.msh` | 1,000 | Structured quad | Coarse |
| `mesh/quad_100x40.msh` | 4,000 | Structured quad | Fine |

## Expected Results

- Parabolic velocity profile at outlet matching analytical solution
- L2 velocity error < 5% on coarse mesh
- Maximum velocity error < 1%
- Observed convergence order: ~2 (second-order FVM)

## How to Run

```bash
cd VV
python run_all.py --case 01
```

## Reference

- Analytical solution (textbook, e.g., Kundu & Cohen, *Fluid Mechanics*)
