"""Test SIMPLE solver via Python bindings."""
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, os.path.abspath(build_dir))

import twofluid_cpp as tc
import numpy as np


def make_channel_mesh(Lx, Ly, Nx, Ny):
    """Build a 2D channel mesh."""
    mesh = tc.FVMesh(2)
    dx = Lx / Nx
    dy = Ly / Ny

    # Nodes
    n_nodes = (Nx + 1) * (Ny + 1)
    nodes = np.zeros((n_nodes, 3))
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            nid = j * (Nx + 1) + i
            nodes[nid] = [i * dx, j * dy, 0]
    mesh.nodes = nodes

    # Cells
    mesh.n_cells = Nx * Ny
    for j in range(Ny):
        for i in range(Nx):
            c = tc.Cell()
            c.center = np.array([(i + 0.5) * dx, (j + 0.5) * dy, 0])
            c.volume = dx * dy
            c.nodes = [j * (Nx + 1) + i, j * (Nx + 1) + i + 1,
                        (j + 1) * (Nx + 1) + i + 1, (j + 1) * (Nx + 1) + i]
            c.faces = []
            mesh.add_cell(c)

    face_id = [0]
    inlet_faces, outlet_faces, bottom_faces, top_faces = [], [], [], []

    def add_face(owner, neighbour, center, normal, area, fnodes, tag=""):
        f = tc.Face()
        f.owner = owner
        f.neighbour = neighbour
        f.center = np.array(center)
        f.normal = np.array(normal)
        f.area = area
        f.nodes = fnodes
        f.boundary_tag = tag
        fid = face_id[0]
        mesh.add_face(f)
        # Cannot modify cell.faces through binding easily; they were set empty
        face_id[0] += 1
        return fid

    # Internal horizontal faces
    for j in range(Ny):
        for i in range(Nx - 1):
            left = j * Nx + i
            right = j * Nx + i + 1
            add_face(left, right, [(i + 1) * dx, (j + 0.5) * dy, 0],
                     [1, 0, 0], dy, [j * (Nx + 1) + i + 1, (j + 1) * (Nx + 1) + i + 1])

    # Internal vertical faces
    for j in range(Ny - 1):
        for i in range(Nx):
            bot = j * Nx + i
            top_cell = (j + 1) * Nx + i
            add_face(bot, top_cell, [(i + 0.5) * dx, (j + 1) * dy, 0],
                     [0, 1, 0], dx, [(j + 1) * (Nx + 1) + i, (j + 1) * (Nx + 1) + i + 1])

    # Inlet
    for j in range(Ny):
        fid = add_face(j * Nx, -1, [0, (j + 0.5) * dy, 0],
                       [-1, 0, 0], dy, [j * (Nx + 1), (j + 1) * (Nx + 1)], "inlet")
        inlet_faces.append(fid)

    # Outlet
    for j in range(Ny):
        fid = add_face(j * Nx + Nx - 1, -1, [Lx, (j + 0.5) * dy, 0],
                       [1, 0, 0], dy, [j * (Nx + 1) + Nx, (j + 1) * (Nx + 1) + Nx], "outlet")
        outlet_faces.append(fid)

    # Bottom
    for i in range(Nx):
        fid = add_face(i, -1, [(i + 0.5) * dx, 0, 0],
                       [0, -1, 0], dx, [i, i + 1], "bottom")
        bottom_faces.append(fid)

    # Top
    for i in range(Nx):
        fid = add_face((Ny - 1) * Nx + i, -1, [(i + 0.5) * dx, Ly, 0],
                       [0, 1, 0], dx, [Ny * (Nx + 1) + i, Ny * (Nx + 1) + i + 1], "top")
        top_faces.append(fid)

    mesh.n_faces = face_id[0]
    mesh.n_internal_faces = Ny * (Nx - 1) + (Ny - 1) * Nx
    mesh.n_boundary_faces = mesh.n_faces - mesh.n_internal_faces
    mesh.boundary_patches = {
        "inlet": inlet_faces,
        "outlet": outlet_faces,
        "bottom": bottom_faces,
        "top": top_faces,
    }

    return mesh


def test_simple_poiseuille():
    """Poiseuille flow test using C++ SIMPLE solver via Python."""
    Lx, Ly = 2.0, 1.0
    Nx, Ny = 10, 10
    rho, mu = 1.0, 0.01

    mesh = make_channel_mesh(Lx, Ly, Nx, Ny)
    print(f"Mesh: {mesh.n_cells} cells, {mesh.n_faces} faces")

    solver = tc.SIMPLESolver(mesh, rho, mu)
    solver.alpha_u = 0.7
    solver.alpha_p = 0.3
    solver.max_iter = 1000
    solver.tol = 1e-5

    # Parabolic inlet
    dy = Ly / Ny
    n_inlet = len(mesh.boundary_patches["inlet"])
    inlet_U = np.zeros((n_inlet, 2))
    for j in range(n_inlet):
        y = (j + 0.5) * dy
        inlet_U[j, 0] = 6.0 * y * (Ly - y) / (Ly * Ly)

    solver.set_inlet("inlet", inlet_U)
    solver.set_outlet("outlet", 0.0)
    solver.set_wall("top")
    solver.set_wall("bottom")

    result = solver.solve_steady()
    print(f"Converged: {result.converged}, iterations: {result.iterations}, "
          f"wall_time: {result.wall_time:.3f}s")

    U = solver.velocity()
    p = solver.pressure()

    # Check parabolic profile at midpoint
    x_mid = Lx / 2
    dx_mesh = Lx / Nx
    for ci in range(mesh.n_cells):
        cell = mesh.get_cell(ci)
        if abs(cell.center[0] - x_mid) < dx_mesh * 0.6:
            y = cell.center[1]
            u = U.values[ci, 0]
            if abs(y - 0.5 * Ly) < dy * 0.6:
                print(f"  Center velocity: u({y:.2f}) = {u:.4f}")
                assert u > 1.0, f"Center velocity too low: {u}"

    # Pressure gradient check
    p_vals = p.values
    p_in = np.mean([p_vals[ci] for ci in range(mesh.n_cells)
                     if mesh.get_cell(ci).center[0] < dx_mesh])
    p_out = np.mean([p_vals[ci] for ci in range(mesh.n_cells)
                      if mesh.get_cell(ci).center[0] > Lx - dx_mesh])
    print(f"  Pressure: inlet={p_in:.4f}, outlet={p_out:.4f}")
    assert p_in > p_out, "Pressure should drop from inlet to outlet"

    print("Python SIMPLE test PASSED")


if __name__ == "__main__":
    test_simple_poiseuille()
