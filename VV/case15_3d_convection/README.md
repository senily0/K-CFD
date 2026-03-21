# Case 15: 3D Natural Convection

## Problem Description

Three-dimensional natural convection in a differentially heated cubic cavity.
One vertical wall is heated and the opposite wall is cooled, driving buoyancy
flow via the Boussinesq approximation. This verifies the coupled momentum-energy
solver in 3D with the buoyancy source term.

## Geometry

```
         insulated top
         ==================
        /                 /|
       /   insulated     / |
      /    front/back   /  |
     ==================    |
     |                 |   |
  T_h|                 |T_c
  hot|   Ra = 10^4     |cold
  wall                 |wall
     |                 |  /
     |                 | /
     ==================
         insulated bottom

  Lx = Ly = Lz = 1.0 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Hot wall (x=0) | Dirichlet | T = 1.0 (dimensionless) |
| Cold wall (x=1) | Dirichlet | T = 0.0 (dimensionless) |
| Top / Bottom | Insulated | dT/dn = 0 |
| Front / Back | Insulated | dT/dn = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Rayleigh number (Ra) | 10,000 | - |
| Prandtl number (Pr) | 0.71 | - |
| Density (rho) | 1.0 | kg/m^3 |
| Thermal expansion (beta) | 0.01 | 1/K |
| Reference temperature | 0.5 | - |
| Gravity | (0, -9.81, 0) | m/s^2 |

## Analytical/Reference Solution

For Ra = 10^4 in a cubic cavity, benchmark values (Fusegi et al. 1991):

```
Nu_avg (hot wall) ~ 2.054
u_max (mid-height) ~ 0.198
v_max (mid-width)  ~ 0.222
```

## Verification Method

3D natural convection is verified by `verification_cases.exe` (Case 15) and
the Python implementation in `verification/case15_3d_convection.py`. The CLI
solver does not support buoyancy or energy equation coupling.

## Expected Results

- Average Nusselt number on hot wall within 5% of benchmark (2.054)
- Peak velocity components within 10% of reference
- Symmetric flow pattern about the horizontal mid-plane
- Energy conservation: Q_hot = Q_cold within 2%

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 15

# Via Python verification suite
cd VV
python run_all.py --case 15
```

## Reference

- Fusegi, T., Hyun, J.M., Kuwahara, K. & Farouk, B., "A numerical study of
  three-dimensional natural convection in a differentially heated cubical
  enclosure", *Int. J. Heat Mass Transfer*, 34(6), 1543-1557 (1991).
- de Vahl Davis, G., "Natural convection of air in a square cavity: A benchmark
  numerical solution", *Int. J. Numer. Meth. Fluids*, 3, 249-264 (1983).
