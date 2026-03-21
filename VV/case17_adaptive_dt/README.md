# Case 17: Adaptive Time Stepping

## Problem Description

Verification of the adaptive time-stepping controller on a transient channel flow
with a suddenly applied pressure gradient. The controller adjusts the time step
based on a CFL-like criterion and local truncation error estimation. This tests
that the adaptive algorithm maintains accuracy while maximizing the time step size.

## Geometry

```
  wall_top (no-slip)
  ================================================
  |                                              |
  |   u(t=0) = 0                                |
  |   dp/dx applied at t=0                      |
  |   Transient development to steady state     |
  |                                              |
  ================================================
  wall_bottom (no-slip)

  Lx = 1.0 m, Ly = 0.1 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | p = 1.0 Pa |
| Outlet | Dirichlet | p = 0.0 Pa |
| Top wall | No-slip | u = v = 0 |
| Bottom wall | No-slip | u = v = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Initial dt | 0.001 | s |
| dt_min | 1e-6 | s |
| dt_max | 0.1 | s |
| CFL_target | 0.5 | - |
| t_end | 5.0 | s |
| Error tolerance | 1e-4 | - |

## Adaptive Strategy

| Parameter | Value |
|-----------|-------|
| Method | PID controller |
| Error estimator | Embedded Runge-Kutta pair |
| Safety factor | 0.9 |
| Growth limit | 2.0 (max dt increase per step) |
| Shrink limit | 0.5 (max dt decrease per step) |

## Analytical Solution

The transient velocity profile for impulsively started Poiseuille flow is:

```
u(y,t) = u_ss(y) - sum_{n=0}^{inf} A_n * sin(n*pi*y/H) * exp(-n^2*pi^2*nu*t/H^2)
```

where `u_ss(y)` is the steady-state parabolic profile and `nu = mu/rho`.

## Verification Method

Adaptive time stepping is verified by `verification_cases.exe` (Case 17) and
the Python implementation in `verification/case17_adaptive_dt.py`.

## Expected Results

- Final steady-state matches Poiseuille profile within 1%
- Adaptive dt grows from dt_min toward dt_max as transient decays
- Total time steps reduced by >30% compared to fixed-dt run
- No rejected steps after initial transient

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 17

# Via Python verification suite
cd VV
python run_all.py --case 17
```

## Reference

- Gustafsson, K., "Control-theoretic techniques for stepsize selection in
  implicit Runge-Kutta methods", *ACM Trans. Math. Software*, 20(4),
  496-517 (1994).
- Soderlind, G., "Automatic control and adaptive time-stepping", *Numer.
  Algorithms*, 31, 281-310 (2002).
