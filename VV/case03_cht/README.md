# Case 03: Conjugate Heat Transfer

## Problem Description

Steady-state conjugate heat transfer between a heated solid plate and an adjacent
fluid channel. The solid conducts heat internally, and the fluid carries it away
by forced convection. This case verifies the thermal coupling between solid and
fluid domains and the temperature continuity at the interface.

## Geometry

```
  insulated
  ================================================
  |               FLUID DOMAIN                   |
  |   rho=1.0, cp=1000, k_f=0.6                |
  |   u = parabolic inlet                       |
  ------------------------------------------------
  |               SOLID DOMAIN                   |
  |   k_s = 50.0 W/(m.K)                        |
  |   q_dot = 1000 W/m^2 (bottom)               |
  ================================================
  heated wall (constant heat flux)

  Lx = 0.5 m, Ly_fluid = 0.02 m, Ly_solid = 0.01 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | Parabolic velocity, T = 300 K |
| Outlet | Zero gradient | p = 0, dT/dx = 0 |
| Top wall | Insulated | dT/dy = 0 |
| Bottom wall | Heat flux | q = 1000 W/m^2 |
| Solid-fluid interface | Coupled | T and q continuous |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Fluid density | 1.0 | kg/m^3 |
| Fluid viscosity | 0.001 | Pa.s |
| Fluid conductivity (k_f) | 0.6 | W/(m.K) |
| Fluid specific heat (cp) | 1000.0 | J/(kg.K) |
| Solid conductivity (k_s) | 50.0 | W/(m.K) |
| Heat flux (q) | 1000.0 | W/m^2 |
| Inlet temperature | 300.0 | K |

## Analytical Solution

For a fully developed thermal profile with constant heat flux, the Nusselt number
approaches:

```
Nu = 7.54  (parallel plates, constant heat flux, one side insulated)
```

The bulk temperature rises linearly along the channel:

```
T_bulk(x) = T_inlet + q * x / (rho * cp * u_mean * H)
```

## Verification Method

This case is verified internally by `verification_cases.exe` (Case 3: CHT).
The CLI solver (`twofluid_solver.exe`) does not currently support multi-domain
CHT simulations. Full VTU output will be available when CLI integration is
completed.

## Expected Results

- Interface temperature continuity (jump < 0.1 K)
- Nusselt number within 5% of analytical value (7.54)
- Linear bulk temperature rise along the channel
- Energy conservation: Q_in = Q_out within 1%

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 3

# Via Python verification suite
cd VV
python run_all.py --case 03
```

## Reference

- Incropera, F.P. & DeWitt, D.P., *Fundamentals of Heat and Mass Transfer*,
  7th ed., Wiley (2011).
- Shah, R.K. & London, A.L., *Laminar Flow Forced Convection in Ducts*,
  Academic Press (1978).
