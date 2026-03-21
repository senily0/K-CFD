# Case 20: RPI Wall Boiling Model

## Problem Description

Subcooled nucleate boiling in a vertical heated channel using the RPI (Rensselaer
Polytechnic Institute) wall boiling model. The wall heat flux is partitioned into
single-phase convection, quenching, and evaporation components. This verifies the
two-fluid boiling framework at PWR-representative conditions.

## Geometry

```
  adiabatic wall (top)
  ================================================
  T_in = 543K -->  subcooled water + vapor  --> out
  alpha_v = 0     P = 15.5 MPa
  ================================================
  heated wall (bottom), q = 500 kW/m^2 (RPI model)

  Lx = 0.5 m,  Ly = 0.02 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | T = 543.15 K, alpha_v = 0 |
| Outlet | Zero gradient | - |
| Bottom wall | RPI boiling | q = 500 kW/m^2 |
| Top wall | Adiabatic | q = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| System pressure | 15.5 | MPa |
| Saturation temperature | 617.94 | K |
| Inlet temperature | 543.15 (270 C) | K |
| Inlet subcooling | 74.79 | K |
| Wall heat flux | 500,000 | W/m^2 |
| Mass flux | 3,000 | kg/(m^2.s) |
| Liquid density | 594.0 | kg/m^3 |
| Vapor density | 101.9 | kg/m^3 |
| Latent heat | 961,000 | J/kg |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/heated_channel_20x10.msh` | 200 | Structured quad | Standard |

## Expected Results

- Outlet void fraction between 0 and 0.3 (subcooled boiling regime)
- Wall temperature between 580 K and 650 K
- Energy balance error < 5%
- Vapor generation begins where liquid reaches near-saturation temperature

## How to Run

```bash
cd VV
python run_all.py --case 20
```

## Reference

- Kurul, N., Podowski, M.Z., "Multidimensional effects in forced convection
  subcooled boiling", *Proc. 9th Int. Heat Transfer Conf.*, Jerusalem (1990).
- Podowski, R.M., "Toward mechanistic modeling of boiling heat transfer",
  *Nuclear Eng. & Technology*, 44(8), 811-820 (2012).
