# Case 07: Unstructured Mesh Poiseuille Flow

## Problem Description

Poiseuille channel flow solved on an unstructured triangular mesh. This case
verifies that the finite volume discretization produces correct results on
non-orthogonal, unstructured meshes with skewed control volumes. The same
analytical parabolic profile from Case 01 serves as reference.

## Geometry

```
  wall_top (no-slip)
  ================================================
  -->  /\  /\  /\  /\  /\  /\  /\  /\  /\  -->
  --> /  \/  \/  \/  \/  \/  \/  \/  \/  \ -->
  inlet   (triangular mesh)             outlet
  --> \  /\  /\  /\  /\  /\  /\  /\  /\ / -->
  -->  \/  \/  \/  \/  \/  \/  \/  \/  \/  -->
  ================================================
  wall_bottom (no-slip)

  Lx = 1.0 m, Ly = 0.1 m
```

## Mesh Generation

A triangular mesh is generated using Gmsh:

```python
import gmsh
gmsh.initialize()
gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 0.1)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
gmsh.model.mesh.generate(2)
gmsh.write("mesh/tri_unstructured.msh")
gmsh.finalize()
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | Parabolic velocity profile |
| Outlet | Zero gradient | p = 0 |
| Top wall | No-slip wall | u = v = 0 |
| Bottom wall | No-slip wall | u = v = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Density (rho) | 1.0 | kg/m^3 |
| Dynamic viscosity (mu) | 0.01 | Pa.s |
| Reynolds number (Re) | 10 | - |

## Verification Method

The CLI solver (`twofluid_solver.exe`) uses internal structured mesh generators
and does not accept external `.msh` files. Unstructured mesh support is available
via the `read_gmsh_msh()` library function in the Python solver. The case is
verified internally by `verification_cases.exe` (Case 7).

## Expected Results

- Parabolic velocity profile at outlet
- L2 velocity error < 8% (higher tolerance due to mesh non-orthogonality)
- Pressure drop within 5% of analytical value
- Non-orthogonal corrector needed for skewed cells

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 7

# Via Python verification suite
cd VV
python run_all.py --case 07
```

## Reference

- Jasak, H., *Error Analysis and Estimation for the Finite Volume Method with
  Applications to Fluid Flows*, Ph.D. thesis, Imperial College London (1996).
- Ferziger, J.H. & Peric, M., *Computational Methods for Fluid Dynamics*,
  3rd ed., Springer (2002), Chapter 9.
