# Case 14: 3D Lid-Driven Cavity

## Problem Description

Three-dimensional lid-driven cavity flow at Re = 100. The top lid moves in the
x-direction, driving recirculating flow in a unit cube. This extends the 2D
cavity benchmark to 3D, testing the solver's 3D hex mesh handling and 3D
momentum coupling.

## Geometry

```
        lid (u=1, v=0, w=0)
       ___________________
      /                  /|
     /   Re = 100       / |
    /                  /  |
   /_________________ /   |
   |                 |    |
   |                 |    |  Lz = 1.0
   |                 |    |
   |                 |   /
   |                 |  /
   |_________________| /
                        Ly = 1.0
      Lx = 1.0
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Top (y_max) | Dirichlet | u = 1.0, v = w = 0 |
| All others | No-slip wall | u = v = w = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density | 1.0 | kg/m^3 |
| Dynamic viscosity | 0.01 | Pa.s |
| Lid velocity | 1.0 | m/s |
| Reynolds number | 100 | - |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/hex_8x8x8.msh` | 512 | Structured hex | Coarse |

## Expected Results

- Primary vortex in the symmetry plane (z = 0.5) matches 2D cavity data
- Residual reduction of at least 4 orders of magnitude
- Centerline velocity profiles within 10% of reference

## How to Run

```bash
cd VV
python run_all.py --case 14
```

## Reference

- Ku, H.C., Hirsh, R.S., Taylor, T.D., "A pseudospectral method for solution
  of the three-dimensional incompressible Navier-Stokes equations",
  *J. Comput. Phys.*, 70, 439-462 (1987).
