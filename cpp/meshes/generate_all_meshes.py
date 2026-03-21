#!/usr/bin/env python3
"""Generate Gmsh meshes for all K-CFD verification cases."""
import os
import gmsh


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def generate_2d_quad(Lx, Ly, nx, ny, filepath, bc_names=None):
    """Generate 2D structured quad mesh with physical groups for BCs."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mesh")

    # Rectangle
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

    # Transfinite for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    gmsh.model.geo.mesh.setRecombine(2, s)  # quads

    gmsh.model.geo.synchronize()

    # Physical groups for boundary conditions
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
    gmsh.write(filepath)

    n_cells = nx * ny
    print(f"  {filepath}: {n_cells} cells (quad)")

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
    gmsh.write(filepath)

    # Count cells
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_cells = sum(len(t) for t in elem_tags)
    print(f"  {filepath}: ~{n_cells} cells (tri)")

    gmsh.finalize()


def generate_3d_hex(Lx, Ly, Lz, nx, ny, nz, filepath, bc_names=None):
    """Generate 3D structured hex mesh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mesh")

    # Create box using OCC
    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    # Get surfaces for BCs
    surfaces = gmsh.model.getEntities(2)

    # Set transfinite on all curves
    curves = gmsh.model.getEntities(1)
    for dim, tag in curves:
        # Determine which direction this curve is
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

    # Transfinite surfaces and volume
    for dim, tag in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)

    volumes = gmsh.model.getEntities(3)
    for dim, tag in volumes:
        gmsh.model.mesh.setTransfiniteVolume(tag)

    # Physical groups - identify surfaces by their center of mass
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
    gmsh.write(filepath)

    n_cells = nx * ny * nz
    print(f"  {filepath}: {n_cells} cells (hex)")

    gmsh.finalize()


def main():
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = script_dir

    print("Generating meshes for K-CFD verification cases...\n")

    # Case 1: Poiseuille
    d = os.path.join(base, "case01_poiseuille")
    ensure_dir(d)
    generate_2d_quad(1.0, 0.1, 50, 20, os.path.join(d, "mesh_50x20.msh"))
    generate_2d_quad(1.0, 0.1, 100, 40, os.path.join(d, "mesh_100x40.msh"))

    # Case 2: Cavity
    d = os.path.join(base, "case02_cavity")
    ensure_dir(d)
    generate_2d_quad(
        1.0,
        1.0,
        32,
        32,
        os.path.join(d, "mesh_32x32.msh"),
        {
            "bottom": "wall_bottom",
            "right": "wall_right",
            "top": "lid",
            "left": "wall_left",
        },
    )
    generate_2d_quad(
        1.0,
        1.0,
        64,
        64,
        os.path.join(d, "mesh_64x64.msh"),
        {
            "bottom": "wall_bottom",
            "right": "wall_right",
            "top": "lid",
            "left": "wall_left",
        },
    )

    # Case 4: Bubble rising
    d = os.path.join(base, "case04_bubble_rising")
    ensure_dir(d)
    generate_2d_quad(
        0.1,
        0.3,
        20,
        60,
        os.path.join(d, "mesh_20x60.msh"),
        {
            "bottom": "wall_bottom",
            "right": "wall_right",
            "top": "wall_top",
            "left": "wall_left",
        },
    )
    generate_2d_tri(
        0.1,
        0.3,
        0.005,
        os.path.join(d, "mesh_tri.msh"),
        {
            "bottom": "wall_bottom",
            "right": "wall_right",
            "top": "wall_top",
            "left": "wall_left",
        },
    )

    # Case 6: MUSCL MMS
    d = os.path.join(base, "case06_muscl")
    ensure_dir(d)
    for n in [10, 20, 40]:
        generate_2d_quad(1.0, 1.0, n, n, os.path.join(d, f"mesh_{n}x{n}.msh"))

    # Case 11: Radiation 1D slab
    d = os.path.join(base, "case11_radiation")
    ensure_dir(d)
    generate_2d_quad(
        0.01,
        1.0,
        1,
        50,
        os.path.join(d, "mesh_1x50.msh"),
        {
            "bottom": "wall_bottom",
            "right": "outlet",
            "top": "wall_top",
            "left": "inlet",
        },
    )
    generate_2d_quad(
        0.01,
        1.0,
        1,
        100,
        os.path.join(d, "mesh_1x100.msh"),
        {
            "bottom": "wall_bottom",
            "right": "outlet",
            "top": "wall_top",
            "left": "inlet",
        },
    )

    # Case 14: 3D Cavity
    d = os.path.join(base, "case14_3d_cavity")
    ensure_dir(d)
    generate_3d_hex(
        1.0, 1.0, 1.0, 8, 8, 8, os.path.join(d, "mesh_8x8x8.msh"), {"y_max": "lid"}
    )

    # Case 16: Preconditioner 3D
    d = os.path.join(base, "case16_preconditioner")
    ensure_dir(d)
    generate_3d_hex(
        1.0, 0.5, 0.5, 20, 10, 10, os.path.join(d, "mesh_20x10x10.msh")
    )

    # Case 19: IAPWS heated channel
    d = os.path.join(base, "case19_iapws")
    ensure_dir(d)
    generate_2d_quad(0.5, 0.02, 20, 5, os.path.join(d, "mesh_20x5.msh"))

    # Case 20: RPI boiling channel
    d = os.path.join(base, "case20_rpi_boiling")
    ensure_dir(d)
    generate_2d_quad(
        0.5,
        0.02,
        20,
        10,
        os.path.join(d, "mesh_20x10.msh"),
        {
            "bottom": "wall_heated",
            "right": "outlet",
            "top": "wall_top",
            "left": "inlet",
        },
    )

    # Case 22: Simple 2-cell for polymesh testing
    d = os.path.join(base, "case22_polymesh")
    ensure_dir(d)
    generate_3d_hex(1.0, 2.0, 1.0, 1, 2, 1, os.path.join(d, "mesh_2cell.msh"))

    print("\nAll meshes generated.")


if __name__ == "__main__":
    main()
