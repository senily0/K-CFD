# Case 12: Adaptive Mesh Refinement

## Problem Description

Verification of the adaptive mesh refinement (AMR) framework on a 2D lid-driven
cavity flow. Cells are refined based on a velocity gradient indicator near the
driven lid and corner singularities. This tests the AMR infrastructure: cell
splitting, solution interpolation, hanging-node treatment, and convergence on
adaptively refined meshes.

## Geometry

```
  lid: u = 1.0 m/s -->
  ========================
  |      |  |  | | ||   |
  |      |--+--+-+-++---|
  |      |  |  | | ||   |  AMR near lid
  |------+--+--+-+-++---|  and corners
  |      |  |  |        |
  |      |  |  |        |
  |      |  |  |        |  Coarse in center
  |      |  |  |        |
  ========================

  Lx = Ly = 1.0 m
  Base mesh: 16x16 quads
  Max refinement levels: 3
```

## AMR Strategy

| Parameter | Value |
|-----------|-------|
| Refinement indicator | Velocity gradient magnitude |
| Refinement threshold | 0.5 * max(indicator) |
| Coarsening threshold | 0.1 * max(indicator) |
| Max refinement level | 3 |
| Min cell size | Lx / (16 * 2^3) = 0.0078 m |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 100 | - |
| Lid velocity | 1.0 | m/s |

## Verification Method

AMR is verified internally by `verification_cases.exe` (Case 12) and by the
Python implementation in `verification/case12_amr.py`. The CLI solver does not
support adaptive mesh refinement.

## Expected Results

- AMR solution matches uniform fine-mesh solution within 5%
- Cell count reduced by >50% compared to uniformly refined mesh
- Refinement concentrated near lid and corner singularities
- Conservation maintained across refinement levels

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 12

# Via Python verification suite
cd VV
python run_all.py --case 12
```

## Reference

- Berger, M.J. & Oliger, J., "Adaptive mesh refinement for hyperbolic partial
  differential equations", *J. Comput. Phys.*, 53, 484-512 (1984).
- Berger, M.J. & Colella, P., "Local adaptive mesh refinement for shock
  hydrodynamics", *J. Comput. Phys.*, 82, 64-84 (1989).
