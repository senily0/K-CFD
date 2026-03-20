#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/distributed_mesh.hpp"
#include "twofluid/simple_solver.hpp"

namespace twofluid {

/// Distributed BiCGSTAB that uses MPI ghost exchange inside SpMV.
///
/// The matrix A_local has (n_owned + n_ghost) rows/cols.
/// After every SpMV (A*v), ghost values in the result are exchanged.
/// Dot products and norms use global_dot/global_norm (MPI_Allreduce).
///
/// Only owned rows of x are updated by the solve. Ghost rows are
/// always overwritten by exchange.
///
/// This makes the distributed solve mathematically equivalent to
/// solving the full global system.
Eigen::VectorXd distributed_bicgstab(
    const Eigen::SparseMatrix<double>& A_local,
    const Eigen::VectorXd& b_local,
    const Eigen::VectorXd& x0,
    const DistributedMesh& dmesh,
    double tol = 1e-6,
    int maxiter = 1000);

/// Distributed SIMPLE solver that uses DistributedMesh.
///
/// The SIMPLE algorithm doesn't know about MPI -- it just operates
/// on the local mesh (which includes ghost cells as real internal cells).
/// The linear solves use distributed_bicgstab with MPI ghost exchange,
/// and field updates are followed by ghost exchange.
///
/// This matches the OpenFOAM/ANSYS approach where the outer algorithm
/// is agnostic to parallelism.
class DistributedSIMPLESolver {
public:
    DistributedSIMPLESolver(DistributedMesh& dmesh, double rho, double mu);

    // SIMPLE parameters
    double alpha_u = 0.7;   // velocity under-relaxation
    double alpha_p = 0.3;   // pressure under-relaxation
    int max_iter = 500;
    double tol = 1e-5;

    // Boundary conditions (applied only to real boundary faces, not ghost interfaces)
    void set_inlet(const std::string& patch, const Eigen::MatrixXd& U_vals);
    void set_wall(const std::string& patch);
    void set_outlet(const std::string& patch, double p_val = 0.0);

    // Solve
    SolveResult solve_steady();

    // Access fields
    VectorField& velocity() { return U_; }
    ScalarField& pressure() { return p_; }

private:
    DistributedMesh& dmesh_;
    FVMesh& mesh_;  // alias for dmesh_.local_mesh
    double rho_, mu_;

    VectorField U_;
    ScalarField p_;

    // Per-component aP coefficients for pressure correction
    std::unordered_map<int, Eigen::VectorXd> aP_;

    // Cell-center pressure gradient for Rhie-Chow
    Eigen::MatrixXd grad_p_;

    // BC storage
    struct BCInfo { std::string type; };
    std::unordered_map<std::string, BCInfo> bc_u_, bc_p_;
    std::unordered_map<int, std::pair<std::string, int>> face_bc_cache_;

    void build_face_bc_cache();
    double gc(int fid) const;  // geometric interpolation weight
    Eigen::VectorXd compute_face_mass_flux();
    double solve_momentum(int comp, const Eigen::VectorXd& mass_flux);
    double solve_pressure_correction(const Eigen::VectorXd& mass_flux);
};

} // namespace twofluid
