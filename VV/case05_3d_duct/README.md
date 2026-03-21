# Case 05: 3D Poiseuille Duct Flow

## Problem Description

Fully developed laminar flow in a three-dimensional rectangular duct driven by
a constant pressure gradient. This verifies the 3D solver implementation against
the known series solution for duct flow. The velocity profile is parabolic in
both the y- and z-directions.

## Geometry

```
          wall_top
  ========================
  /                      /|
 /    3D Rectangular    / |
/       Duct           /  | Lz = 0.1 m
========================  |
|  inlet -->   outlet  |  /
|                      | / Ly = 0.1 m
========================
  wall_bottom

  Lx = 1.0 m, Ly = 0.1 m, Lz = 0.1 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | Developed duct profile |
| Outlet | Zero gradient | p = 0 |
| All walls | No-slip | u = v = w = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 10 | - |

## Analytical Solution

The velocity in a rectangular duct of height 2a and width 2b is given by the
infinite series:

```
u(y,z) = (16 * a^2 / (mu * pi^3)) * dp/dx
         * sum_{n=1,3,5,...} (-1)^((n-1)/2) / n^3
           * [1 - cosh(n*pi*z/(2*a)) / cosh(n*pi*b/(2*a))]
           * cos(n*pi*y/(2*a))
```

## Verification Method

The CLI solver (`twofluid_solver.exe`) supports 2D cases only. 3D duct flow is
verified internally by `verification_cases.exe` (Case 5). The Python solver
(`verification/case15_3d_convection.py`) also tests 3D discretization.

## Expected Results

- Centerline velocity within 5% of series solution
- L2 velocity error < 3% on the 20x10x10 mesh
- Observed second-order spatial convergence

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 5

# Via Python verification suite
cd VV
python run_all.py --case 05
```

## Reference

- White, F.M., *Viscous Fluid Flow*, 3rd ed., McGraw-Hill (2006), Section 3-3.
- Shah, R.K. & London, A.L., *Laminar Flow Forced Convection in Ducts*,
  Academic Press (1978).
