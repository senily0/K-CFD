# Case 09: Stefan Phase Change

## Problem Description

One-dimensional Stefan problem for melting with a moving solid-liquid interface.
A hot boundary melts a cold solid; the interface position as a function of time
is known analytically via the Stefan condition. This verifies the phase change
model and enthalpy-porosity formulation.

## Geometry

```
  T_hot                              T_cold
  (373K)                             (273K)
  |========|/////////////////////|
  | liquid | <-- interface -->   solid  |
  |========|/////////////////////|
  0        s(t)                  L = 0.1 m
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Left (x=0) | Dirichlet | T = 373.15 K |
| Right (x=L) | Dirichlet | T = 273.15 K |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density | 1000.0 | kg/m^3 |
| Specific heat | 4186.0 | J/(kg.K) |
| Thermal conductivity | 0.6 | W/(m.K) |
| Latent heat | 334,000 | J/kg |
| Melting temperature | 273.15 | K |
| Stefan number | ~0.125 | - |

## Analytical Solution

The interface position follows:

```
s(t) = 2 * lambda * sqrt(alpha * t)
```

where `alpha = k/(rho*cp)` is thermal diffusivity and `lambda` is determined
from the transcendental Stefan condition.

## Meshes

This case uses an internally generated 1D mesh (no external .msh file).

## Expected Results

- Interface position error < 5% relative to analytical solution
- Temperature profile error (L2) < 2%
- Energy conservation satisfied

## How to Run

```bash
cd VV
python run_all.py --case 09
```

## Reference

- Alexiades, V., Solomon, A.D., *Mathematical Modeling of Melting and Freezing
  Processes*, Hemisphere Publishing (1993).
- Stefan, J., "Ueber die Theorie der Eisbildung", *Annalen der Physik*, 278(2),
  269-286 (1891).
