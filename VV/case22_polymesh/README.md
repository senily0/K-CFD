# Case 22: Polyhedral Mesh I/O

## Problem Description

Minimal verification of mesh reader and finite volume operators on a simple
2-cell hexahedral mesh. This tests that cell volumes, face areas, face normals,
and owner/neighbour connectivity are computed correctly. No physics is solved;
this is purely a mesh infrastructure test.

## Geometry

```
  2 hex cells stacked vertically:

       ___________
      /           /|
     /   cell 1  / |
    /___________/  |  Ly = 2.0 (1.0 per cell)
    |           |  |
    |   cell 0  |  /
    |           | /   Lz = 1.0
    |___________|/
      Lx = 1.0
```

## Boundary Conditions

None (mesh validation only).

## Expected Results

| Quantity | Expected |
|----------|----------|
| Cell count | 2 |
| Volume per cell | 1.0 m^3 |
| Total volume | 2.0 m^3 |
| Internal faces | 1 |
| Boundary faces | 10 |
| Total faces | 11 |
| Face normals | Outward-pointing, unit magnitude |

## Meshes

| File | Cells | Type | Notes |
|------|-------|------|-------|
| `mesh/hex_2cell.msh` | 2 | Structured hex | Minimal |

## Pass Criteria

- Cell count equals exactly 2
- Volume error < 1e-10 (machine precision)
- All face normals consistently outward-pointing

## How to Run

```bash
cd VV
python run_all.py --case 22
```

## Reference

- Exact geometry (no external reference needed).
