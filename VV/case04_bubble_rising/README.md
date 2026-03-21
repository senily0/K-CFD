# Case 04: Single Bubble Rising

## Problem Description

A single gas bubble rises through a stagnant liquid column under buoyancy.
This is a canonical two-fluid benchmark testing the solver's ability to handle
large density ratios, surface tension, and interface tracking.

## Geometry

```
  ========================
  |          wall_top     |
  |                       |
  |                       |
  |                       |
  |                       |
  |                       |
  |                       |
  |        O  <- bubble   |
  |      (gas)            |
  |                       |
  ========================
  wall_bottom

  Lx = 0.1 m,  Ly = 0.3 m
  Bubble: center (0.05, 0.05), R = 0.025 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| All walls | No-slip wall | u = v = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Liquid density | 1000.0 | kg/m^3 |
| Gas density | 1.0 | kg/m^3 |
| Liquid viscosity | 0.001 | Pa.s |
| Gas viscosity | 1e-5 | Pa.s |
| Surface tension | 0.073 | N/m |
| Gravity | -9.81 | m/s^2 |
| Density ratio | 1000 | - |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/quad_20x60.msh` | 1,200 | Structured quad | Standard |
| `mesh/tri_unstructured.msh` | ~2,400 | Unstructured tri | Alternative |

## Expected Results

- Bubble rises with approximately spherical cap shape
- Terminal rise velocity within 15% of Stokes/Hysing benchmark
- Global mass conservation error < 1%
- Density ratio of 1000:1 handled stably

## How to Run

```bash
cd VV
python run_all.py --case 04
```

## Reference

- Hysing, S., Turek, S., Kuzmin, D., et al., "Quantitative benchmark
  computations of two-dimensional bubble dynamics",
  *Int. J. Numer. Methods Fluids*, 60, 1259-1288 (2009).
