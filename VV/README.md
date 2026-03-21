# K-CFD Verification & Validation Suite

This directory contains the complete V&V test suite for the K-CFD two-fluid
finite volume solver. Each subdirectory is a self-contained verification case
with its own mesh, configuration, and documentation.

## Quick Start

```bash
# Generate all meshes (requires Gmsh Python API)
python generate_meshes.py

# Run all verification cases
python run_all.py

# Run a single case
python run_all.py --case 01

# View results in ParaView
# Open VV/caseXX_*/results/*.vtu
```

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| Python 3.8+ | Runner scripts |
| `gmsh` (Python) | Mesh generation (`pip install gmsh`) |
| `numpy` | Numerical computations |
| `scipy` | Linear solvers, sparse matrices |
| `meshio` | Mesh I/O for VTU export |

Install all dependencies:

```bash
pip install gmsh numpy scipy meshio
```

## Cases

| # | Name | Type | Mesh | Reference |
|---|------|------|------|-----------|
| 01 | Poiseuille Flow | Single-phase | 2D quad | Analytical |
| 02 | Lid-Driven Cavity | Single-phase | 2D quad | Ghia et al. 1982 |
| 03 | Conjugate Heat Transfer | CHT | 2D multi-domain | Incropera & DeWitt 2011 |
| 04 | Single Bubble Rising | Two-fluid | 2D quad + tri | Hysing et al. 2009 |
| 05 | 3D Poiseuille Duct | Single-phase 3D | 3D hex | White 2006 |
| 06 | MUSCL MMS | Convection-diffusion | 2D quad (3 levels) | Roache 2002 |
| 07 | Unstructured Mesh | Single-phase | 2D tri | Jasak 1996 |
| 08 | MPI Parallel | Distributed | 2D quad (decomposed) | Serial vs. parallel |
| 09 | Phase Change | Two-fluid + phase change | (generated) | Stefan problem |
| 10 | Chemical Reaction | Scalar transport | 2D quad (quasi-1D) | Analytical decay |
| 11 | Radiation Transport | Radiation | 1D slab | Analytical P1 |
| 12 | Adaptive Mesh Refinement | AMR | 2D quad (adaptive) | Berger & Oliger 1984 |
| 13 | GPU Acceleration | GPU compute | 2D quad | CPU reference |
| 14 | 3D Lid-Driven Cavity | Single-phase 3D | 3D hex | Ku et al. 1987 |
| 15 | 3D Natural Convection | Buoyancy 3D | 3D hex | Fusegi et al. 1991 |
| 16 | Preconditioner | Linear solver | 3D hex | Iterative methods |
| 17 | Adaptive Time Stepping | Transient | 2D quad | Gustafsson 1994 |
| 18 | OpenMP Parallelism | Shared-memory | 2D quad | Serial vs. parallel |
| 19 | IAPWS Steam Tables | Thermophysical | 2D channel | IAPWS-IF97 |
| 20 | RPI Wall Boiling | Two-fluid boiling | 2D channel | Kurul & Podowski 1990 |
| 21 | Virtual Mass Force | Two-fluid | 2D quad | Analytical acceleration |
| 22 | Polyhedral Mesh | Mesh I/O | 3D hex (2-cell) | Exact geometry |
| 23 | Hybrid Mesh | Mixed elements | 2D quad+tri, 3D hex+tet | Analytical (Poiseuille) |

## Directory Layout

Each case follows this structure:

```
caseXX_name/
  README.md        # Problem description, references, expected results
  input.json       # Solver configuration and parameters
  mesh/            # Gmsh .msh files (pre-generated or from generate_meshes.py)
  results/         # Solver output: VTU files (created at runtime)
  figures/         # ParaView renders (created at runtime)
```

## Pass/Fail Criteria

Each `input.json` contains a `pass_criteria` section that defines quantitative
acceptance thresholds. The runner (`run_all.py`) evaluates these automatically
and reports PASS/FAIL for each case.

## Adding a New Case

1. Create `caseXX_name/` with `README.md`, `input.json`, and `mesh/` directory.
2. Add mesh generation to `generate_meshes.py`.
3. Register the case in `run_all.py` by adding it to `CASES`.
4. Update this README table.
