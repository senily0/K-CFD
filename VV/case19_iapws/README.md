# Case 19: IAPWS Steam Tables Verification

## Problem Description

Verification of the IAPWS-IF97 thermophysical property implementation. A heated
channel flow at PWR-representative conditions (15.5 MPa, 270 C inlet) is used
to exercise the steam tables across a range of temperatures. Computed properties
(density, viscosity, enthalpy, thermal conductivity) are compared against
tabulated IAPWS-IF97 reference values.

## Geometry

```
  adiabatic wall (top)
  ================================================
  T_in = 543K -->  heated subcooled water  --> out
  P = 15.5 MPa
  ================================================
  heated wall (bottom), q = 500 kW/m^2

  Lx = 0.5 m,  Ly = 0.02 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | T = 543.15 K, uniform velocity |
| Outlet | Zero gradient | - |
| Bottom wall | Heat flux | q = 500 kW/m^2 |
| Top wall | Adiabatic | q = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| System pressure | 15.5 | MPa |
| Inlet temperature | 543.15 (270 C) | K |
| Wall heat flux | 500,000 | W/m^2 |
| Mass flux | 3,000 | kg/(m^2.s) |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/channel_20x5.msh` | 100 | Structured quad | Standard |

## Expected Results

- Density matches IAPWS-IF97 within 0.1%
- Viscosity matches IAPWS-IF97 within 0.5%
- Enthalpy matches IAPWS-IF97 within 0.1%
- Smooth temperature profile along channel

## How to Run

```bash
cd VV
python run_all.py --case 19
```

## Reference

- Wagner, W., Kruse, A., *Properties of Water and Steam / Zustandsgrossen von
  Wasser und Wasserdampf*, Springer (1998).
- IAPWS-IF97: International Association for the Properties of Water and Steam,
  "Revised Release on the IAPWS Industrial Formulation 1997" (2012).
