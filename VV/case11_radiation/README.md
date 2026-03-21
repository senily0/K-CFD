# Case 11: Radiation Transport (P1 Approximation)

## Problem Description

One-dimensional radiative transfer through a grey, absorbing, non-scattering
slab using the P1 (spherical harmonics) approximation. The analytical solution
for the incident radiation and temperature profile is known for this simplified
configuration.

## Geometry

```
  T_hot = 1000 K          T_cold = 300 K
  |                        |
  | --> q_rad -->          |
  |  absorbing medium      |
  |  kappa = 1.0 /m        |
  |                        |
  0                        L = 1.0 m
  (slab modeled as thin 2D strip: Lx = 0.01 m)
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Left (inlet) | Blackbody | T = 1000 K |
| Right (outlet) | Blackbody | T = 300 K |
| Top/Bottom | Symmetry | - |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Absorption coefficient | 1.0 | 1/m |
| Scattering coefficient | 0.0 | 1/m |
| Hot wall temperature | 1000 | K |
| Cold wall temperature | 300 | K |
| Emissivity | 1.0 | - |
| Optical thickness | 1.0 | - |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/slab_1x50.msh` | 50 | Structured quad | Coarse |
| `mesh/slab_1x100.msh` | 100 | Structured quad | Fine |

## Expected Results

- Incident radiation profile follows exponential decay
- Heat flux error < 5%
- Temperature profile L2 error < 3%

## How to Run

```bash
cd VV
python run_all.py --case 11
```

## Reference

- Modest, M.F., *Radiative Heat Transfer*, 3rd ed., Academic Press (2013).
