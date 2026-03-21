# Case 21: Virtual Mass Force

## Problem Description

Verification of the virtual (added) mass force implementation in the two-fluid
model. A single gas bubble accelerates from rest under buoyancy, and the
transient acceleration is compared against the analytical solution that accounts
for the added mass effect (C_vm = 0.5 for a sphere).

## Geometry

```
  ========================
  |       outlet (top)   |
  |                      |
  |                      |
  |                      |
  |                      |
  |                      |
  |        O  <- bubble  |
  |      (gas)           |
  |                      |
  ========================
  wall_bottom

  Lx = 0.1 m,  Ly = 0.3 m
  Bubble radius: 0.01 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Bottom | No-slip wall | u = v = 0 |
| Left | No-slip wall | u = v = 0 |
| Right | No-slip wall | u = v = 0 |
| Top | Outlet | zero gradient |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Liquid density | 1000.0 | kg/m^3 |
| Gas density | 1.0 | kg/m^3 |
| Virtual mass coefficient | 0.5 | - |
| Gravity | -9.81 | m/s^2 |
| Bubble radius | 0.01 | m |

## Analytical Solution

For a spherical bubble accelerating from rest under buoyancy with virtual mass:

```
m_eff * dv/dt = (rho_l - rho_g) * V_b * g - drag
m_eff = (rho_g + C_vm * rho_l) * V_b
```

At early times (before drag dominates), the initial acceleration is:

```
a_0 = (rho_l - rho_g) * g / (rho_g + C_vm * rho_l)
```

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/quad_20x60.msh` | 1,200 | Structured quad | Standard |

## Expected Results

- Initial bubble acceleration matches analytical value within 10%
- Energy balance error < 5%
- Virtual mass force visibly slows initial acceleration compared to no-VM case

## How to Run

```bash
cd VV
python run_all.py --case 21
```

## Reference

- Zuber, N., "On the dispersed two-phase flow in the laminar flow regime",
  *Chem. Eng. Sci.*, 19(11), 897-917 (1964).
- Drew, D.A., Lahey, R.T., "The virtual mass and lift force on a sphere in
  rotating and straining inviscid flow", *Int. J. Multiphase Flow*, 13(1),
  113-121 (1987).
