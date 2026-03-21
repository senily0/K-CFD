# Case 23: Hybrid Mesh Verification

## Problem Description

Poiseuille flow solved on hybrid (mixed-element) meshes to verify that the FVM
operators (gradient, divergence, Laplacian) work correctly when different cell
types coexist in the same mesh. The 2D mesh has structured quads on the left
half and unstructured triangles on the right half. The 3D mesh has structured
hexahedra on the left and tetrahedra on the right.

The analytical solution is the same as Case 01 (parabolic velocity profile),
but the non-uniform mesh topology exercises different code paths in face
interpolation and gradient reconstruction.

## Geometry (2D)

```
  wall_top
  ================================================
  |  quad (structured)  |  tri (unstructured)    |
  |  25 x 20 cells      |  ~500 triangles        |
  |                     |                         |
  inlet              interface                outlet
  |                     |                         |
  |  structured         |  unstructured           |
  ================================================
  wall_bottom

  Left:  [0, 0.5] x [0, 0.1]  -- structured quad
  Right: [0.5, 1.0] x [0, 0.1] -- unstructured tri
```

## Geometry (3D)

```
  Left:  [0, 0.5] x [0, 0.1] x [0, 0.1]  -- structured hex (10x5x5)
  Right: [0.5, 1.0] x [0, 0.1] x [0, 0.1] -- unstructured tet
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet (x=0) | Dirichlet | Parabolic velocity profile |
| Outlet (x=Lx) | Zero gradient | p = 0 |
| Top wall | No-slip wall | u = v = 0 |
| Bottom wall | No-slip wall | u = v = 0 |

## Physical Parameters

Same as Case 01:

| Parameter | Value | Unit |
|-----------|-------|------|
| Density | 1.0 | kg/m^3 |
| Dynamic viscosity | 0.01 | Pa.s |
| Reynolds number | 10 | - |

## Meshes

| File | Elements | Type | Notes |
|------|----------|------|-------|
| `mesh/hybrid_quad_tri.msh` | ~1,000 | Quad + tri | 2D hybrid |
| `mesh/hybrid_hex_tet.msh` | ~500 | Hex + tet | 3D hybrid |

## Key Verification Points

1. **Interface continuity**: Solution is smooth across the quad-tri boundary
2. **Mass conservation**: Global mass balance error < 0.1%
3. **Accuracy**: L2 velocity error < 5% (2D) and < 8% (3D)
4. **Operator consistency**: Gradient and Laplacian operators produce correct
   results on both element types

## Expected Results

- Parabolic velocity profile recoverable on both mesh halves
- No spurious oscillations at the element-type interface
- Convergence behaviour similar to pure-quad mesh

## How to Run

```bash
cd VV
python run_all.py --case 23
```

## Reference

- Analytical Poiseuille solution (same as Case 01).
- Jasak, H., "Error analysis and estimation for the finite volume method with
  applications to fluid flows", PhD thesis, Imperial College London (1996)
  (discusses FVM on arbitrary polyhedral meshes).
