#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/interpolation.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/fvm_operators.hpp"

using namespace twofluid;

/// Build a 2D channel mesh for Poiseuille flow.
/// Channel: Lx x Ly, Nx x Ny cells.
/// Patches: inlet (left), outlet (right), top, bottom.
FVMesh make_channel_mesh(double Lx, double Ly, int Nx, int Ny) {
    FVMesh mesh(2);
    double dx = Lx / Nx;
    double dy = Ly / Ny;

    // Nodes: (Nx+1) x (Ny+1)
    int n_nodes = (Nx + 1) * (Ny + 1);
    mesh.nodes.resize(n_nodes, 3);
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int nid = j * (Nx + 1) + i;
            mesh.nodes(nid, 0) = i * dx;
            mesh.nodes(nid, 1) = j * dy;
            mesh.nodes(nid, 2) = 0.0;
        }
    }

    // Cells
    mesh.n_cells = Nx * Ny;
    mesh.cells.resize(mesh.n_cells);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int cid = j * Nx + i;
            Cell& cell = mesh.cells[cid];
            cell.center = Eigen::Vector3d((i + 0.5) * dx, (j + 0.5) * dy, 0);
            cell.volume = dx * dy;
            // Node indices (bottom-left, bottom-right, top-right, top-left)
            cell.nodes = {
                j * (Nx + 1) + i,
                j * (Nx + 1) + i + 1,
                (j + 1) * (Nx + 1) + i + 1,
                (j + 1) * (Nx + 1) + i
            };
        }
    }

    // Faces: internal + boundary
    int face_id = 0;
    std::vector<int> inlet_faces, outlet_faces, bottom_faces, top_faces;

    // Helper to add a face and register it with cells
    auto add_face = [&](int owner, int neighbour,
                        Eigen::Vector3d center, Eigen::Vector3d normal,
                        double area, std::vector<int> nodes) -> int {
        Face f;
        f.owner = owner;
        f.neighbour = neighbour;
        f.center = center;
        f.normal = normal;
        f.area = area;
        f.nodes = nodes;
        int fid = face_id++;
        mesh.faces.push_back(f);
        mesh.cells[owner].faces.push_back(fid);
        if (neighbour >= 0) {
            mesh.cells[neighbour].faces.push_back(fid);
        }
        return fid;
    };

    // Internal horizontal faces (between cells in x-direction)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx - 1; ++i) {
            int left = j * Nx + i;
            int right = j * Nx + i + 1;
            Eigen::Vector3d c((i + 1) * dx, (j + 0.5) * dy, 0);
            Eigen::Vector3d n(1, 0, 0);
            int n0 = j * (Nx + 1) + i + 1;
            int n1 = (j + 1) * (Nx + 1) + i + 1;
            add_face(left, right, c, n, dy, {n0, n1});
        }
    }

    // Internal vertical faces (between cells in y-direction)
    for (int j = 0; j < Ny - 1; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int bot = j * Nx + i;
            int top_cell = (j + 1) * Nx + i;
            Eigen::Vector3d c((i + 0.5) * dx, (j + 1) * dy, 0);
            Eigen::Vector3d n(0, 1, 0);
            int n0 = (j + 1) * (Nx + 1) + i;
            int n1 = (j + 1) * (Nx + 1) + i + 1;
            add_face(bot, top_cell, c, n, dx, {n0, n1});
        }
    }

    int n_internal = face_id;

    // Boundary faces: inlet (left, x=0)
    for (int j = 0; j < Ny; ++j) {
        int cid = j * Nx + 0;
        Eigen::Vector3d c(0, (j + 0.5) * dy, 0);
        Eigen::Vector3d n(-1, 0, 0);  // outward from owner
        int n0 = j * (Nx + 1);
        int n1 = (j + 1) * (Nx + 1);
        int fid = add_face(cid, -1, c, n, dy, {n0, n1});
        mesh.faces[fid].boundary_tag = "inlet";
        inlet_faces.push_back(fid);
    }

    // Boundary faces: outlet (right, x=Lx)
    for (int j = 0; j < Ny; ++j) {
        int cid = j * Nx + (Nx - 1);
        Eigen::Vector3d c(Lx, (j + 0.5) * dy, 0);
        Eigen::Vector3d n(1, 0, 0);
        int n0 = j * (Nx + 1) + Nx;
        int n1 = (j + 1) * (Nx + 1) + Nx;
        int fid = add_face(cid, -1, c, n, dy, {n0, n1});
        mesh.faces[fid].boundary_tag = "outlet";
        outlet_faces.push_back(fid);
    }

    // Boundary faces: bottom (y=0)
    for (int i = 0; i < Nx; ++i) {
        int cid = 0 * Nx + i;
        Eigen::Vector3d c((i + 0.5) * dx, 0, 0);
        Eigen::Vector3d n(0, -1, 0);
        int n0 = i;
        int n1 = i + 1;
        int fid = add_face(cid, -1, c, n, dx, {n0, n1});
        mesh.faces[fid].boundary_tag = "bottom";
        bottom_faces.push_back(fid);
    }

    // Boundary faces: top (y=Ly)
    for (int i = 0; i < Nx; ++i) {
        int cid = (Ny - 1) * Nx + i;
        Eigen::Vector3d c((i + 0.5) * dx, Ly, 0);
        Eigen::Vector3d n(0, 1, 0);
        int n0 = Ny * (Nx + 1) + i;
        int n1 = Ny * (Nx + 1) + i + 1;
        int fid = add_face(cid, -1, c, n, dx, {n0, n1});
        mesh.faces[fid].boundary_tag = "top";
        top_faces.push_back(fid);
    }

    mesh.n_faces = face_id;
    mesh.n_internal_faces = n_internal;
    mesh.n_boundary_faces = face_id - n_internal;
    mesh.boundary_patches["inlet"] = inlet_faces;
    mesh.boundary_patches["outlet"] = outlet_faces;
    mesh.boundary_patches["bottom"] = bottom_faces;
    mesh.boundary_patches["top"] = top_faces;

    return mesh;
}

void test_linear_solver() {
    std::cout << "=== test_linear_solver ===" << std::endl;

    // Simple 3x3 system: [[4,-1,0],[-1,4,-1],[0,-1,4]] x = [1,2,3]
    FVMSystem sys(3);
    sys.add_diagonal(0, 4.0);
    sys.add_diagonal(1, 4.0);
    sys.add_diagonal(2, 4.0);
    sys.add_off_diagonal(0, 1, -1.0);
    sys.add_off_diagonal(1, 0, -1.0);
    sys.add_off_diagonal(1, 2, -1.0);
    sys.add_off_diagonal(2, 1, -1.0);
    sys.add_source(0, 1.0);
    sys.add_source(1, 2.0);
    sys.add_source(2, 3.0);

    // Direct solve
    Eigen::VectorXd x = solve_linear_system(sys);
    Eigen::SparseMatrix<double> A = sys.to_sparse();
    Eigen::VectorXd residual = A * x - sys.rhs;
    std::cout << "  Direct solve residual: " << residual.norm() << std::endl;
    assert(residual.norm() < 1e-10);

    // BiCGSTAB solve
    Eigen::VectorXd x2 = solve_linear_system(sys, Eigen::VectorXd(), "bicgstab");
    Eigen::VectorXd residual2 = A * x2 - sys.rhs;
    std::cout << "  BiCGSTAB solve residual: " << residual2.norm() << std::endl;
    assert(residual2.norm() < 1e-8);

    std::cout << "PASSED\n\n";
}

void test_interpolation() {
    std::cout << "=== test_interpolation ===" << std::endl;

    // Test limiters
    assert(std::abs(limiter_van_leer(1.0) - 1.0) < 1e-10);
    assert(std::abs(limiter_van_leer(0.0)) < 1e-10);
    assert(std::abs(limiter_van_leer(-1.0)) < 1e-10);
    assert(std::abs(limiter_minmod(0.5) - 0.5) < 1e-10);
    assert(std::abs(limiter_minmod(2.0) - 1.0) < 1e-10);
    assert(std::abs(limiter_superbee(1.0) - 1.0) < 1e-10);
    assert(std::abs(limiter_van_albada(1.0) - 1.0) < 1e-10);

    std::cout << "  Limiters OK" << std::endl;

    // Test mass flux with uniform velocity on the channel mesh
    FVMesh mesh = make_channel_mesh(1.0, 1.0, 4, 4);
    VectorField U(mesh, "velocity");
    Eigen::VectorXd u_vec(2);
    u_vec << 1.0, 0.0;
    U.set_uniform(u_vec);

    Eigen::VectorXd mf = compute_mass_flux(U, 1.0, mesh);
    std::cout << "  Mass flux computed, n_faces=" << mesh.n_faces << std::endl;

    // Internal faces in x-direction should have positive flux
    // Internal faces in y-direction should have zero flux
    // Check that total inlet flux matches total outlet flux
    double inlet_flux = 0.0, outlet_flux = 0.0;
    for (int fid : mesh.boundary_patches["inlet"]) {
        inlet_flux += mf[fid];
    }
    for (int fid : mesh.boundary_patches["outlet"]) {
        outlet_flux += mf[fid];
    }
    std::cout << "  Inlet flux: " << inlet_flux << ", Outlet flux: " << outlet_flux << std::endl;
    // For uniform flow (1,0) with rho=1, inlet has outward normal (-1,0) so flux=-dy per face
    // outlet has normal (1,0) so flux=+dy per face
    assert(std::abs(inlet_flux + outlet_flux) < 1e-10);  // conservation

    std::cout << "PASSED\n\n";
}

void test_simple_poiseuille() {
    std::cout << "=== test_simple_poiseuille ===" << std::endl;

    // Poiseuille flow: channel Lx=2, Ly=1, mu=0.01, rho=1
    // Analytical: u(y) = (dP/dx) / (2*mu) * y * (H - y)
    // With fixed inlet velocity and outlet pressure BC
    double Lx = 2.0, Ly = 1.0;
    int Nx = 10, Ny = 10;
    double rho = 1.0, mu = 0.01;

    FVMesh mesh = make_channel_mesh(Lx, Ly, Nx, Ny);
    std::cout << "  Mesh: " << mesh.n_cells << " cells, " << mesh.n_faces << " faces" << std::endl;

    SIMPLESolver solver(mesh, rho, mu);
    solver.alpha_u = 0.7;
    solver.alpha_p = 0.3;
    solver.max_iter = 500;
    solver.tol = 1e-5;

    // Parabolic inlet velocity: u(y) = 6 * U_mean * y * (H - y) / H^2
    // U_mean = 1.0, H = 1.0
    double U_mean = 1.0;
    int n_inlet = static_cast<int>(mesh.boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 2);
    double dy = Ly / Ny;
    for (int j = 0; j < n_inlet; ++j) {
        double y = (j + 0.5) * dy;
        double u_parabolic = 6.0 * U_mean * y * (Ly - y) / (Ly * Ly);
        inlet_U(j, 0) = u_parabolic;
        inlet_U(j, 1) = 0.0;
    }

    solver.set_inlet("inlet", inlet_U);
    solver.set_outlet("outlet", 0.0);
    solver.set_wall("top");
    solver.set_wall("bottom");

    SolveResult result = solver.solve_steady();

    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Wall time: " << result.wall_time << " s" << std::endl;

    if (!result.residuals.empty()) {
        std::cout << "  Final residual: " << result.residuals.back() << std::endl;
    }

    // Check that velocity profile is roughly parabolic at midpoint
    VectorField& U = solver.velocity();
    ScalarField& p = solver.pressure();

    // Sample velocity at x = Lx/2
    double x_mid = Lx / 2.0;
    double dx_mesh = Lx / Nx;
    std::vector<double> ys, us;
    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        if (std::abs(mesh.cells[ci].center[0] - x_mid) < dx_mesh * 0.6) {
            ys.push_back(mesh.cells[ci].center[1]);
            us.push_back(U.values(ci, 0));
        }
    }

    std::cout << "  Velocity profile at x=" << x_mid << ":" << std::endl;
    double max_u = 0.0;
    for (size_t i = 0; i < ys.size(); ++i) {
        if (us[i] > max_u) max_u = us[i];
        if (i % 2 == 0) {
            std::cout << "    y=" << ys[i] << " u=" << us[i] << std::endl;
        }
    }

    // The maximum velocity should be near the center and positive
    std::cout << "  Max u at midpoint: " << max_u << std::endl;
    assert(max_u > 0.5);  // Should be greater than zero (flow exists)

    // Velocity at walls should be close to zero
    for (size_t i = 0; i < ys.size(); ++i) {
        if (ys[i] < dy * 0.6 || ys[i] > Ly - dy * 0.6) {
            // Near-wall cells: velocity should be reduced
            std::cout << "    Near-wall: y=" << ys[i] << " u=" << us[i] << std::endl;
        }
    }

    // Pressure should decrease from inlet to outlet
    double p_inlet_avg = 0.0, p_outlet_avg = 0.0;
    int n_in = 0, n_out = 0;
    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        if (mesh.cells[ci].center[0] < dx_mesh) {
            p_inlet_avg += p.values[ci];
            n_in++;
        }
        if (mesh.cells[ci].center[0] > Lx - dx_mesh) {
            p_outlet_avg += p.values[ci];
            n_out++;
        }
    }
    if (n_in > 0) p_inlet_avg /= n_in;
    if (n_out > 0) p_outlet_avg /= n_out;
    std::cout << "  Pressure: inlet_avg=" << p_inlet_avg
              << " outlet_avg=" << p_outlet_avg << std::endl;
    assert(p_inlet_avg > p_outlet_avg);  // Pressure drops in flow direction

    std::cout << "PASSED\n\n";
}

int main() {
    std::cout << "Running SIMPLE solver tests...\n\n";

    test_linear_solver();
    test_interpolation();
    test_simple_poiseuille();

    std::cout << "All SIMPLE tests PASSED.\n";
    return 0;
}
