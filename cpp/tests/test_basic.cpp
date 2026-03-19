#include <iostream>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/fvm_operators.hpp"

using namespace twofluid;

/// Build a minimal 2-cell 1D-like mesh:
///
///   Cell 0          Cell 1
///  [0,0]-[0.5,0]-[1,0]
///    |     face1     |
///  face0           face2
///  (boundary)      (boundary)
///
/// Three nodes along x, two quad cells (top nodes at y=1).
/// face0: left boundary, face1: internal, face2: right boundary
FVMesh make_two_cell_mesh() {
    FVMesh mesh(2);

    // Nodes: 6 nodes forming two quads
    //  3---4---5
    //  |   |   |
    //  0---1---2
    mesh.nodes.resize(6, 3);
    mesh.nodes << 0, 0, 0,
                  0.5, 0, 0,
                  1, 0, 0,
                  0, 1, 0,
                  0.5, 1, 0,
                  1, 1, 0;

    // Cells
    Cell c0;
    c0.center = Eigen::Vector3d(0.25, 0.5, 0);
    c0.volume = 0.5;
    c0.nodes = {0, 1, 4, 3};
    c0.faces = {0, 1, 2, 3};

    Cell c1;
    c1.center = Eigen::Vector3d(0.75, 0.5, 0);
    c1.volume = 0.5;
    c1.nodes = {1, 2, 5, 4};
    c1.faces = {1, 4, 5, 6};

    mesh.cells = {c0, c1};
    mesh.n_cells = 2;

    // Faces (7 faces total for two quads sharing one internal face)
    // face 0: left boundary (x=0 edge, nodes 0-3)
    Face f0;
    f0.owner = 0; f0.neighbour = -1;
    f0.center = Eigen::Vector3d(0, 0.5, 0);
    f0.normal = Eigen::Vector3d(-1, 0, 0);
    f0.area = 1.0;
    f0.nodes = {0, 3};
    f0.boundary_tag = "left";

    // face 1: internal (x=0.5, nodes 1-4)
    Face f1;
    f1.owner = 0; f1.neighbour = 1;
    f1.center = Eigen::Vector3d(0.5, 0.5, 0);
    f1.normal = Eigen::Vector3d(1, 0, 0);
    f1.area = 1.0;
    f1.nodes = {1, 4};

    // face 2: bottom of cell 0 (y=0, nodes 0-1)
    Face f2;
    f2.owner = 0; f2.neighbour = -1;
    f2.center = Eigen::Vector3d(0.25, 0, 0);
    f2.normal = Eigen::Vector3d(0, -1, 0);
    f2.area = 0.5;
    f2.nodes = {0, 1};
    f2.boundary_tag = "bottom";

    // face 3: top of cell 0 (y=1, nodes 3-4)
    Face f3;
    f3.owner = 0; f3.neighbour = -1;
    f3.center = Eigen::Vector3d(0.25, 1, 0);
    f3.normal = Eigen::Vector3d(0, 1, 0);
    f3.area = 0.5;
    f3.nodes = {3, 4};
    f3.boundary_tag = "top";

    // face 4: right boundary (x=1, nodes 2-5)
    Face f4;
    f4.owner = 1; f4.neighbour = -1;
    f4.center = Eigen::Vector3d(1, 0.5, 0);
    f4.normal = Eigen::Vector3d(1, 0, 0);
    f4.area = 1.0;
    f4.nodes = {2, 5};
    f4.boundary_tag = "right";

    // face 5: bottom of cell 1 (y=0, nodes 1-2)
    Face f5;
    f5.owner = 1; f5.neighbour = -1;
    f5.center = Eigen::Vector3d(0.75, 0, 0);
    f5.normal = Eigen::Vector3d(0, -1, 0);
    f5.area = 0.5;
    f5.nodes = {1, 2};
    f5.boundary_tag = "bottom";

    // face 6: top of cell 1 (y=1, nodes 4-5)
    Face f6;
    f6.owner = 1; f6.neighbour = -1;
    f6.center = Eigen::Vector3d(0.75, 1, 0);
    f6.normal = Eigen::Vector3d(0, 1, 0);
    f6.area = 0.5;
    f6.nodes = {4, 5};
    f6.boundary_tag = "top";

    mesh.faces = {f0, f1, f2, f3, f4, f5, f6};
    mesh.n_faces = 7;
    mesh.n_internal_faces = 1;
    mesh.n_boundary_faces = 6;

    mesh.boundary_patches["left"] = {0};
    mesh.boundary_patches["right"] = {4};
    mesh.boundary_patches["bottom"] = {2, 5};
    mesh.boundary_patches["top"] = {3, 6};

    return mesh;
}

void test_mesh_summary() {
    std::cout << "=== test_mesh_summary ===" << std::endl;
    FVMesh mesh = make_two_cell_mesh();
    std::string s = mesh.summary();
    std::cout << s;
    assert(s.find("2 cells") != std::string::npos);
    assert(s.find("7 faces") != std::string::npos);
    std::cout << "PASSED\n\n";
}

void test_scalar_field() {
    std::cout << "=== test_scalar_field ===" << std::endl;
    FVMesh mesh = make_two_cell_mesh();
    ScalarField phi(mesh, "temperature", 300.0);

    assert(phi.values.size() == 2);
    assert(std::abs(phi.values(0) - 300.0) < 1e-12);
    assert(std::abs(phi.values(1) - 300.0) < 1e-12);

    phi.set_uniform(500.0);
    assert(std::abs(phi.max() - 500.0) < 1e-12);

    phi.values(0) = 100.0;
    phi.values(1) = 200.0;
    assert(std::abs(phi.mean() - 150.0) < 1e-12);

    phi.store_old();
    assert(phi.old_values.has_value());
    assert(std::abs(phi.old_values.value()(0) - 100.0) < 1e-12);

    phi.set_boundary("left", 0.0);
    phi.set_boundary("right", 1000.0);
    auto it_left = phi.boundary_values.find("left");
    assert(it_left != phi.boundary_values.end());
    assert(std::abs(it_left->second(0) - 0.0) < 1e-12);

    std::cout << "PASSED\n\n";
}

void test_vector_field() {
    std::cout << "=== test_vector_field ===" << std::endl;
    FVMesh mesh = make_two_cell_mesh();
    VectorField vel(mesh, "velocity");

    assert(vel.values.rows() == 2);
    assert(vel.values.cols() == 2);

    Eigen::VectorXd u(2);
    u << 1.0, 0.0;
    vel.set_uniform(u);
    assert(std::abs(vel.values(0, 0) - 1.0) < 1e-12);
    assert(std::abs(vel.values(1, 0) - 1.0) < 1e-12);

    Eigen::VectorXd mag = vel.magnitude();
    assert(std::abs(mag(0) - 1.0) < 1e-12);

    std::cout << "PASSED\n\n";
}

void test_green_gauss_gradient() {
    std::cout << "=== test_green_gauss_gradient ===" << std::endl;
    FVMesh mesh = make_two_cell_mesh();

    // Linear field phi = x (at cell centers: 0.25, 0.75)
    ScalarField phi(mesh, "phi");
    phi.values(0) = 0.25;
    phi.values(1) = 0.75;
    // Set boundary values matching phi = x
    phi.set_boundary("left", 0.0);
    phi.set_boundary("right", 1.0);
    // bottom/top: use cell center x-values
    Eigen::VectorXd bottom_vals(2);
    bottom_vals << 0.25, 0.75;
    phi.set_boundary("bottom", bottom_vals);
    Eigen::VectorXd top_vals(2);
    top_vals << 0.25, 0.75;
    phi.set_boundary("top", top_vals);

    Eigen::MatrixXd grad = green_gauss_gradient(phi);
    std::cout << "Gradient:\n" << grad << std::endl;

    // For phi=x, grad should be approximately (1, 0)
    assert(std::abs(grad(0, 0) - 1.0) < 0.2);
    assert(std::abs(grad(1, 0) - 1.0) < 0.2);
    assert(std::abs(grad(0, 1)) < 0.2);
    assert(std::abs(grad(1, 1)) < 0.2);

    std::cout << "PASSED\n\n";
}

void test_fvm_system() {
    std::cout << "=== test_fvm_system ===" << std::endl;
    FVMSystem sys(3);
    sys.add_diagonal(0, 4.0);
    sys.add_diagonal(1, 5.0);
    sys.add_diagonal(2, 6.0);
    sys.add_off_diagonal(0, 1, -1.0);
    sys.add_source(0, 10.0);

    auto mat = sys.to_sparse();
    assert(mat.rows() == 3);
    assert(mat.cols() == 3);
    assert(std::abs(mat.coeff(0, 0) - 4.0) < 1e-12);
    assert(std::abs(mat.coeff(0, 1) - (-1.0)) < 1e-12);
    assert(std::abs(sys.rhs(0) - 10.0) < 1e-12);
    assert(std::abs(sys.diag(0) - 4.0) < 1e-12);

    sys.reset();
    assert(std::abs(sys.rhs(0)) < 1e-12);
    assert(std::abs(sys.diag(1)) < 1e-12);
    assert(sys.rows.empty());

    std::cout << "PASSED\n\n";
}

int main() {
    std::cout << "Running twofluid C++ tests...\n\n";

    test_mesh_summary();
    test_scalar_field();
    test_vector_field();
    test_green_gauss_gradient();
    test_fvm_system();

    std::cout << "All tests PASSED.\n";
    return 0;
}
