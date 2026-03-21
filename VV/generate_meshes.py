#!/usr/bin/env python3
"""
Generate Gmsh meshes for all K-CFD V&V cases.

Usage:
    python generate_meshes.py           # Generate all meshes
    python generate_meshes.py --case 01 # Generate meshes for case 01 only

Requires: pip install gmsh
"""
import argparse
import os
import sys

try:
    import gmsh
except ImportError:
    print("ERROR: gmsh Python package not found. Install with: pip install gmsh")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Reusable mesh generators
# ---------------------------------------------------------------------------

def generate_2d_quad(Lx, Ly, nx, ny, filepath, bc_names=None):
    """Generate 2D structured quad mesh with physical groups for BCs."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mesh")

    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    gmsh.model.geo.mesh.setRecombine(2, s)

    gmsh.model.geo.synchronize()

    if bc_names is None:
        bc_names = {
            "bottom": "wall_bottom",
            "right": "outlet",
            "top": "wall_top",
            "left": "inlet",
        }

    gmsh.model.addPhysicalGroup(1, [l1], name=bc_names.get("bottom", "wall_bottom"))
    gmsh.model.addPhysicalGroup(1, [l2], name=bc_names.get("right", "outlet"))
    gmsh.model.addPhysicalGroup(1, [l3], name=bc_names.get("top", "wall_top"))
    gmsh.model.addPhysicalGroup(1, [l4], name=bc_names.get("left", "inlet"))
    gmsh.model.addPhysicalGroup(2, [s], name="fluid")

    gmsh.model.mesh.generate(2)
    ensure_dir(os.path.dirname(filepath))
    gmsh.write(filepath)

    n_cells = nx * ny
    print(f"  {os.path.relpath(filepath, SCRIPT_DIR)}: {n_cells} cells (quad)")
    gmsh.finalize()


def generate_2d_tri(Lx, Ly, cell_size, filepath, bc_names=None):
    """Generate 2D unstructured triangle mesh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mesh")

    p1 = gmsh.model.geo.addPoint(0, 0, 0, cell_size)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, cell_size)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, cell_size)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0, cell_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    if bc_names is None:
        bc_names = {
            "bottom": "wall_bottom",
            "right": "outlet",
            "top": "wall_top",
            "left": "inlet",
        }

    gmsh.model.addPhysicalGroup(1, [l1], name=bc_names.get("bottom", "wall_bottom"))
    gmsh.model.addPhysicalGroup(1, [l2], name=bc_names.get("right", "outlet"))
    gmsh.model.addPhysicalGroup(1, [l3], name=bc_names.get("top", "wall_top"))
    gmsh.model.addPhysicalGroup(1, [l4], name=bc_names.get("left", "inlet"))
    gmsh.model.addPhysicalGroup(2, [s], name="fluid")

    gmsh.model.mesh.generate(2)
    ensure_dir(os.path.dirname(filepath))
    gmsh.write(filepath)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_cells = sum(len(t) for t in elem_tags)
    print(f"  {os.path.relpath(filepath, SCRIPT_DIR)}: ~{n_cells} cells (tri)")
    gmsh.finalize()


def generate_3d_hex(Lx, Ly, Lz, nx, ny, nz, filepath, bc_names=None):
    """Generate 3D structured hex mesh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mesh")

    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    curves = gmsh.model.getEntities(1)

    for dim, tag in curves:
        bounds = gmsh.model.getBoundary([(dim, tag)])
        p1_coords = gmsh.model.getValue(0, bounds[0][1], [])
        p2_coords = gmsh.model.getValue(0, bounds[1][1], [])
        dx = abs(p2_coords[0] - p1_coords[0])
        dy = abs(p2_coords[1] - p1_coords[1])
        dz = abs(p2_coords[2] - p1_coords[2])
        if dx > max(dy, dz):
            gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
        elif dy > max(dx, dz):
            gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(tag, nz + 1)

    for dim, tag in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)

    volumes = gmsh.model.getEntities(3)
    for dim, tag in volumes:
        gmsh.model.mesh.setTransfiniteVolume(tag)

    if bc_names is None:
        bc_names = {}

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[0]) < 1e-6:
            name = bc_names.get("x_min", "inlet")
        elif abs(com[0] - Lx) < 1e-6:
            name = bc_names.get("x_max", "outlet")
        elif abs(com[1]) < 1e-6:
            name = bc_names.get("y_min", "wall_bottom")
        elif abs(com[1] - Ly) < 1e-6:
            name = bc_names.get("y_max", "wall_top")
        elif abs(com[2]) < 1e-6:
            name = bc_names.get("z_min", "wall_front")
        elif abs(com[2] - Lz) < 1e-6:
            name = bc_names.get("z_max", "wall_back")
        else:
            name = f"surface_{tag}"
        gmsh.model.addPhysicalGroup(2, [tag], name=name)

    for dim, tag in volumes:
        gmsh.model.addPhysicalGroup(3, [tag], name="fluid")

    gmsh.model.mesh.generate(3)
    ensure_dir(os.path.dirname(filepath))
    gmsh.write(filepath)

    n_cells = nx * ny * nz
    print(f"  {os.path.relpath(filepath, SCRIPT_DIR)}: {n_cells} cells (hex)")
    gmsh.finalize()


def generate_hybrid_2d(Lx, Ly, nx_half, ny, filepath):
    """Generate 2D hybrid mesh: left half structured quad, right half unstructured tri."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hybrid")

    # Two rectangles side by side
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx / 2, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p4 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p5 = gmsh.model.geo.addPoint(Lx / 2, Ly, 0)
    p6 = gmsh.model.geo.addPoint(0, Ly, 0)

    l_bot_left = gmsh.model.geo.addLine(p1, p2)
    l_bot_right = gmsh.model.geo.addLine(p2, p3)
    l_right = gmsh.model.geo.addLine(p3, p4)
    l_top_right = gmsh.model.geo.addLine(p4, p5)
    l_mid = gmsh.model.geo.addLine(p5, p2)   # interface
    l_top_left = gmsh.model.geo.addLine(p5, p6)
    l_left = gmsh.model.geo.addLine(p6, p1)

    # Left surface (quad)
    cl_left = gmsh.model.geo.addCurveLoop([l_bot_left, -l_mid, l_top_left, l_left])
    s_left = gmsh.model.geo.addPlaneSurface([cl_left])

    # Right surface (tri)
    cl_right = gmsh.model.geo.addCurveLoop([l_bot_right, l_right, l_top_right, l_mid])
    s_right = gmsh.model.geo.addPlaneSurface([cl_right])

    # Transfinite for left (structured quad)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_bot_left, nx_half + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_top_left, nx_half + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_mid, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_left, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s_left)
    gmsh.model.geo.mesh.setRecombine(2, s_left)

    # Right: set transfinite on boundary curves for consistent spacing
    gmsh.model.geo.mesh.setTransfiniteCurve(l_bot_right, nx_half + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_top_right, nx_half + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l_right, ny + 1)

    gmsh.model.geo.synchronize()

    # Physical groups for BCs
    gmsh.model.addPhysicalGroup(1, [l_left], name="inlet")
    gmsh.model.addPhysicalGroup(1, [l_right], name="outlet")
    gmsh.model.addPhysicalGroup(1, [l_bot_left, l_bot_right], name="wall_bottom")
    gmsh.model.addPhysicalGroup(1, [l_top_left, l_top_right], name="wall_top")
    gmsh.model.addPhysicalGroup(2, [s_left, s_right], name="fluid")

    gmsh.model.mesh.generate(2)
    ensure_dir(os.path.dirname(filepath))
    gmsh.write(filepath)

    # Count cells
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_cells = sum(len(t) for t in elem_tags)
    print(f"  {os.path.relpath(filepath, SCRIPT_DIR)}: ~{n_cells} cells (hybrid quad+tri)")
    gmsh.finalize()


def generate_hybrid_3d(Lx, Ly, Lz, nx_half, ny, nz, filepath):
    """Generate 3D hybrid mesh: left half structured hex, right half unstructured tet."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hybrid_3d")

    # Left box: structured hex
    box_left = gmsh.model.occ.addBox(0, 0, 0, Lx / 2, Ly, Lz)
    # Right box: unstructured tet
    box_right = gmsh.model.occ.addBox(Lx / 2, 0, 0, Lx / 2, Ly, Lz)

    # Fragment to create shared interface
    gmsh.model.occ.fragment(
        [(3, box_left)], [(3, box_right)]
    )
    gmsh.model.occ.synchronize()

    # Set transfinite on the left volume for hex meshing
    # Identify left volume by center of mass
    volumes = gmsh.model.getEntities(3)
    left_vol = None
    right_vol = None
    for dim, tag in volumes:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if com[0] < Lx / 2:
            left_vol = tag
        else:
            right_vol = tag

    if left_vol is not None:
        # Get curves of the left volume and set transfinite
        left_boundary = gmsh.model.getBoundary([(3, left_vol)], combined=False)
        for surf_dim, surf_tag in left_boundary:
            surf_curves = gmsh.model.getBoundary([(2, abs(surf_tag))], combined=False)
            for c_dim, c_tag in surf_curves:
                ct = abs(c_tag)
                try:
                    bounds = gmsh.model.getBoundary([(1, ct)])
                    p1_coords = gmsh.model.getValue(0, abs(bounds[0][1]), [])
                    p2_coords = gmsh.model.getValue(0, abs(bounds[1][1]), [])
                    dx = abs(p2_coords[0] - p1_coords[0])
                    dy = abs(p2_coords[1] - p1_coords[1])
                    dz = abs(p2_coords[2] - p1_coords[2])
                    if dx > max(dy, dz):
                        gmsh.model.mesh.setTransfiniteCurve(ct, nx_half + 1)
                    elif dy > max(dx, dz):
                        gmsh.model.mesh.setTransfiniteCurve(ct, ny + 1)
                    else:
                        gmsh.model.mesh.setTransfiniteCurve(ct, nz + 1)
                except Exception:
                    pass
            try:
                gmsh.model.mesh.setTransfiniteSurface(abs(surf_tag))
                gmsh.model.mesh.setRecombine(2, abs(surf_tag))
            except Exception:
                pass
        try:
            gmsh.model.mesh.setTransfiniteVolume(left_vol)
        except Exception:
            pass

    # Physical groups
    surfaces = gmsh.model.getEntities(2)
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[0]) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="inlet")
        elif abs(com[0] - Lx) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="outlet")
        elif abs(com[1]) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="wall_bottom")
        elif abs(com[1] - Ly) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="wall_top")
        elif abs(com[2]) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="wall_front")
        elif abs(com[2] - Lz) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [tag], name="wall_back")

    for dim, tag in volumes:
        gmsh.model.addPhysicalGroup(3, [tag], name=f"fluid_{tag}")

    gmsh.model.mesh.generate(3)
    ensure_dir(os.path.dirname(filepath))
    gmsh.write(filepath)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    n_cells = sum(len(t) for t in elem_tags)
    print(f"  {os.path.relpath(filepath, SCRIPT_DIR)}: ~{n_cells} cells (hybrid hex+tet)")
    gmsh.finalize()


# ---------------------------------------------------------------------------
# Per-case mesh generation
# ---------------------------------------------------------------------------

def gen_case01():
    """Case 01: Poiseuille Flow"""
    print("Case 01: Poiseuille Flow")
    d = os.path.join(SCRIPT_DIR, "case01_poiseuille", "mesh")
    generate_2d_quad(1.0, 0.1, 50, 20, os.path.join(d, "quad_50x20.msh"))
    generate_2d_quad(1.0, 0.1, 100, 40, os.path.join(d, "quad_100x40.msh"))


def gen_case02():
    """Case 02: Lid-Driven Cavity"""
    print("Case 02: Lid-Driven Cavity")
    d = os.path.join(SCRIPT_DIR, "case02_cavity", "mesh")
    cavity_bc = {
        "bottom": "wall_bottom",
        "right": "wall_right",
        "top": "lid",
        "left": "wall_left",
    }
    generate_2d_quad(1.0, 1.0, 32, 32, os.path.join(d, "quad_32x32.msh"), cavity_bc)
    generate_2d_quad(1.0, 1.0, 64, 64, os.path.join(d, "quad_64x64.msh"), cavity_bc)


def gen_case04():
    """Case 04: Single Bubble Rising"""
    print("Case 04: Single Bubble Rising")
    d = os.path.join(SCRIPT_DIR, "case04_bubble_rising", "mesh")
    wall_bc = {
        "bottom": "wall_bottom",
        "right": "wall_right",
        "top": "wall_top",
        "left": "wall_left",
    }
    generate_2d_quad(0.1, 0.3, 20, 60, os.path.join(d, "quad_20x60.msh"), wall_bc)
    generate_2d_tri(0.1, 0.3, 0.005, os.path.join(d, "tri_unstructured.msh"), wall_bc)


def gen_case06():
    """Case 06: MUSCL MMS"""
    print("Case 06: MUSCL MMS")
    d = os.path.join(SCRIPT_DIR, "case06_muscl_mms", "mesh")
    for n in [10, 20, 40]:
        generate_2d_quad(1.0, 1.0, n, n, os.path.join(d, f"quad_{n}x{n}.msh"))


def gen_case11():
    """Case 11: Radiation (P1)"""
    print("Case 11: Radiation Transport")
    d = os.path.join(SCRIPT_DIR, "case11_radiation", "mesh")
    slab_bc = {
        "bottom": "wall_bottom",
        "right": "outlet",
        "top": "wall_top",
        "left": "inlet",
    }
    generate_2d_quad(0.01, 1.0, 1, 50, os.path.join(d, "slab_1x50.msh"), slab_bc)
    generate_2d_quad(0.01, 1.0, 1, 100, os.path.join(d, "slab_1x100.msh"), slab_bc)


def gen_case14():
    """Case 14: 3D Lid-Driven Cavity"""
    print("Case 14: 3D Cavity")
    d = os.path.join(SCRIPT_DIR, "case14_3d_cavity", "mesh")
    generate_3d_hex(1.0, 1.0, 1.0, 8, 8, 8, os.path.join(d, "hex_8x8x8.msh"),
                    {"y_max": "lid"})


def gen_case16():
    """Case 16: Preconditioner"""
    print("Case 16: Preconditioner")
    d = os.path.join(SCRIPT_DIR, "case16_preconditioner", "mesh")
    generate_3d_hex(1.0, 0.5, 0.5, 20, 10, 10, os.path.join(d, "hex_20x10x10.msh"))


def gen_case19():
    """Case 19: IAPWS"""
    print("Case 19: IAPWS Steam Tables")
    d = os.path.join(SCRIPT_DIR, "case19_iapws", "mesh")
    generate_2d_quad(0.5, 0.02, 20, 5, os.path.join(d, "channel_20x5.msh"))


def gen_case20():
    """Case 20: RPI Boiling"""
    print("Case 20: RPI Wall Boiling")
    d = os.path.join(SCRIPT_DIR, "case20_rpi_boiling", "mesh")
    generate_2d_quad(0.5, 0.02, 20, 10, os.path.join(d, "heated_channel_20x10.msh"),
                     {"bottom": "wall_heated", "right": "outlet",
                      "top": "wall_top", "left": "inlet"})


def gen_case21():
    """Case 21: Virtual Mass"""
    print("Case 21: Virtual Mass Force")
    d = os.path.join(SCRIPT_DIR, "case21_virtual_mass", "mesh")
    wall_bc = {
        "bottom": "wall_bottom",
        "right": "wall_right",
        "top": "wall_top",
        "left": "wall_left",
    }
    generate_2d_quad(0.1, 0.3, 20, 60, os.path.join(d, "quad_20x60.msh"), wall_bc)


def gen_case22():
    """Case 22: Polyhedral Mesh"""
    print("Case 22: Polyhedral Mesh I/O")
    d = os.path.join(SCRIPT_DIR, "case22_polymesh", "mesh")
    generate_3d_hex(1.0, 2.0, 1.0, 1, 2, 1, os.path.join(d, "hex_2cell.msh"))


def gen_case23():
    """Case 23: Hybrid Mesh"""
    print("Case 23: Hybrid Mesh")
    d = os.path.join(SCRIPT_DIR, "case23_hybrid_mesh", "mesh")
    generate_hybrid_2d(1.0, 0.1, 25, 20, os.path.join(d, "hybrid_quad_tri.msh"))
    generate_hybrid_3d(1.0, 0.1, 0.1, 10, 5, 5, os.path.join(d, "hybrid_hex_tet.msh"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "01": gen_case01,
    "02": gen_case02,
    "04": gen_case04,
    "06": gen_case06,
    "11": gen_case11,
    "14": gen_case14,
    "16": gen_case16,
    "19": gen_case19,
    "20": gen_case20,
    "21": gen_case21,
    "22": gen_case22,
    "23": gen_case23,
}


def main():
    parser = argparse.ArgumentParser(description="Generate meshes for K-CFD V&V cases")
    parser.add_argument("--case", type=str, default=None,
                        help="Generate meshes for a specific case (e.g., 01, 23)")
    args = parser.parse_args()

    print("=" * 60)
    print("K-CFD V&V Suite - Mesh Generation")
    print("=" * 60)
    print()

    if args.case:
        case_id = args.case.zfill(2)
        if case_id in GENERATORS:
            GENERATORS[case_id]()
        else:
            print(f"ERROR: Unknown case '{args.case}'. Available: {sorted(GENERATORS.keys())}")
            sys.exit(1)
    else:
        for case_id in sorted(GENERATORS.keys()):
            GENERATORS[case_id]()
            print()

    print("=" * 60)
    print("Mesh generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
